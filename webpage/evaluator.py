import re
import json
import math
import logging
import requests
import nest_asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.callbacks.base import BaseCallbackHandler
try:
    import tiktoken
except ImportError:
    tiktoken = None

class TPSCallbackHandler(BaseCallbackHandler):
    def __init__(self, update_callback):
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_input_tokens = 0
        self.start_time = time.time()
        self.update_callback = update_callback
        self.last_update = time.time()
        self.encoder = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        
        # Save main thread's Streamlit context
        try:
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
            self.st_ctx = get_script_run_ctx()
        except ImportError:
            self.st_ctx = None
        
    def _count(self, text):
        if not text: return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        return len(str(text)) // 4

    def on_llm_start(self, serialized, prompts, **kwargs):
        current_batch_input = 0
        for p in prompts:
            c = self._count(p)
            current_batch_input += c
            self.input_tokens += c
            self.total_tokens += c
        
        if current_batch_input > self.max_input_tokens:
            self.max_input_tokens = current_batch_input
            
        self._report()

    def on_llm_end(self, response, **kwargs):
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, 'text'):
                        c = self._count(gen.text)
                        self.output_tokens += c
                        self.total_tokens += c
        self._report()

    def _report(self):
        now = time.time()
        if now - self.last_update > 0.5:
            # Inject context into background thread if missing
            if self.st_ctx:
                try:
                    import threading
                    from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx, add_script_run_ctx
                    if not get_script_run_ctx():
                        add_script_run_ctx(threading.current_thread(), self.st_ctx)
                except Exception:
                    pass
            
            elapsed = now - self.start_time
            tps = self.total_tokens / elapsed if elapsed > 0 else 0
            if self.update_callback:
                try:
                    self.update_callback(self.total_tokens, self.input_tokens, self.output_tokens, self.max_input_tokens, tps)
                except Exception:
                    pass
            self.last_update = now

from bs4 import BeautifulSoup

nest_asyncio.apply()

from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
    context_recall,
)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util as st_util

logger = logging.getLogger("evaluator")

ROOT_DIR = Path(__file__).resolve().parents[1]

MAX_CONTEXT_CHARS = 180_000
_embedding_model = None
sys.path.append(str(ROOT_DIR))

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model

def clean_html_text(text):
    """Clean HTML tags and entities using BeautifulSoup."""
    if not text or ("<" not in text and "&" not in text):
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def parse_llm_endpoint_or_key(endpoint_or_key: str, default_base_url: str = "http://localhost:11434", api_key: str = "") -> tuple[str, dict]:
    """
    Parses an LLM Judge endpoint or API key input.
    """
    val = (endpoint_or_key or "").strip()
    
    # If API key is provided directly as an argument, use the URL as base_url
    if api_key:
        base_url = val if val else default_base_url
        if base_url.endswith("/api"): base_url = base_url[:-4]
        auth_header = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
        return base_url, {"headers": {"Authorization": auth_header}}
        
    if not val:
        return default_base_url, {}
    if val.startswith("http://") or val.startswith("https://"):
        base_url = val.split("/api")[0]
        return base_url, {}
    if val.startswith("localhost") or val.startswith("127.0.0.1"):
        base_url = f"http://{val}".split("/api")[0]
        return base_url, {}
    # Otherwise treat as API key
    auth_header = val if val.lower().startswith("bearer ") else f"Bearer {val}"
    return default_base_url, {"headers": {"Authorization": auth_header}}
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def split_context_in_docs(context):
    if isinstance(context, list):
        return [clean_html_text(str(c)) for c in context if str(c).strip()]
    text = str(context or "").strip()
    if not text: return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines: return [clean_html_text(text)]

    docs, current = [], []
    for line in lines:
        if line.startswith("[Doc ") or line.startswith("[Fonte:") or line.startswith("[Documento "):
            if current:
                docs.append(" ".join(current).strip())
                current = []
        current.append(line)
    if current: docs.append(" ".join(current).strip())
    return [clean_html_text(doc) for doc in docs]

def truncate_text(text, max_chars):
    """Truncates the text respecting sentence boundaries (. ! ?) or, alternatively,
    word boundaries. Never cuts in the middle of a word."""
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    # Don't go back more than 30% to look for a sentence end
    min_threshold = int(max_chars * 0.7)
    pos = -1
    for terminator in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
        idx = candidate.rfind(terminator, min_threshold)
        if idx > pos:
            pos = idx + 1  # include the punctuation mark
    if pos > min_threshold:
        return candidate[:pos].rstrip() + "\n[... truncated ...]"
    # Fallback: word boundary
    last_space = candidate.rfind(' ', min_threshold)
    if last_space > min_threshold:
        return candidate[:last_space].rstrip() + "\n[... truncated ...]"
    # Last resort (text without spaces): truncate at the limit
    return candidate.rstrip() + "\n[... truncated ...]"

def truncate_context(texts, max_chars=MAX_CONTEXT_CHARS):
    """Truncates a list of context strings so that the total length does not
    exceed max_chars, avoiding 'prompt too long' errors in RAGAS.
    Distributes space evenly among documents and truncates at sentence boundaries."""
    if not texts:
        return texts
    total = sum(len(t) for t in texts)
    if total <= max_chars:
        return texts
    logger.warning(
        "Context too long (%d chars, limit %d). Truncating...",
        total, max_chars
    )
    # Distribute space evenly, with a minimum of 200 chars per doc
    quota_per_doc = max(200, max_chars // len(texts))
    truncated = []
    remaining_chars = max_chars
    for t in texts:
        limit = min(len(t), quota_per_doc, remaining_chars)
        if limit <= 0:
            break
        piece = truncate_text(t, limit)
        truncated.append(piece)
        remaining_chars -= len(piece)
    return truncated

def _extract_doc_text(d):
    """Helper to robustly extract text from a retrieved document (Local or DAVE)."""
    if isinstance(d, str): return d
    if isinstance(d, dict):
        if 'text' in d: return str(d['text'])
        if 'chunks' in d: return " ".join(c.get('text', '') for c in d['chunks'])
        return str(d)
    return str(d)

def filter_docs_by_threshold(docs, query, threshold, model):
    """Filter retrieved documents by semantic similarity threshold using batch encoding."""
    if not docs:
        return docs

    doc_texts = []
    valid_docs = []
    for d in docs:
        doc_text = _extract_doc_text(d)
        if doc_text.strip():
            doc_texts.append(doc_text.strip())
            valid_docs.append(d)

    if not doc_texts:
        return docs[:1]

    # Single batch encode: query + all docs at once
    all_embeddings = model.encode([query] + doc_texts)
    scores = st_util.cos_sim(all_embeddings[0], all_embeddings[1:])[0]

    filtered = [d for d, score in zip(valid_docs, scores) if float(score) >= threshold]
    return filtered if filtered else [docs[0]]

def format_retrieved(docs):
    if not docs: return "No documents retrieved."
    return "".join(f"[Doc {i}]\n{_extract_doc_text(d)}\n\n" for i, d in enumerate(docs, 1))

# RAG ENGINES
def ingest_corpus_to_rag(benchmark_files, run_dir, rag_base_url, progress_callback=None):
    """Loads ALL benchmark documents into the RAG server via /ingest.
    Works with any RAG system that exposes a POST /ingest endpoint.
    If the endpoint does not exist or fails, logs a warning and continues."""
    ingest_endpoint = rag_base_url.rstrip("/") + "/ingest"

    total_files = len(benchmark_files)
    for idx, file_name in enumerate(benchmark_files):
        path = run_dir / file_name
        if not path.exists():
            logger.warning("Benchmark file not found: %s", path)
            continue

        # Load ALL elements from the benchmark file (.jsonl or .json)
        all_items = []
        with open(path, "r", encoding="utf-8-sig") as f:
            if str(path).endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        all_items.append(json.loads(line))
            else:
                all_items = json.load(f)

        # Identify the source from the first element
        source = all_items[0].get("source", "default") if all_items else "default"

        # Extract and deduplicate context documents
        unique_docs = {}
        for item in all_items:
            context = item.get("context", "")
            if not context or not str(context).strip():
                continue
            doc_texts = split_context_in_docs(context)
            for doc_text in doc_texts:
                stripped = doc_text.strip()
                if stripped:
                    content_hash = hash(stripped[:500])
                    if content_hash not in unique_docs:
                        unique_docs[content_hash] = stripped

        if not unique_docs:
            logger.warning("No documents found in %s", file_name)
            continue

        doc_list = [{"id": f"{source}_doc{i}", "text": v} for i, v in enumerate(unique_docs.values())]

        logger.info("Ingestion: sending %d unique documents for benchmark '%s' (from %d total queries)",
                     len(doc_list), source, len(all_items))
        if progress_callback:
            progress_callback("ingest", idx, total_files,
                              f"Loading corpus into RAG: {source} ({len(doc_list)} unique docs from {len(all_items)} queries)...")

        try:
            r = requests.post(ingest_endpoint, json={
                "benchmark_name": source,
                "documents": doc_list
            }, timeout=300)
            if r.status_code == 200:
                logger.info("Ingestion OK for '%s': %s", source, r.json().get("message", ""))
            else:
                logger.warning("Ingestion failed for '%s': HTTP %d", source, r.status_code)
        except Exception as e:
            logger.warning("Ingestion failed for '%s': %s", source, e)

    if progress_callback:
        progress_callback("ingest", total_files, total_files, "Corpus loading completed.")


def get_dave_client_and_collections():
    from dave_client import DaveClient
    collections = {}
    col_path = ROOT_DIR / ".dave_collections.json"
    if col_path.exists():
        with open(col_path, "r") as f:
            collections = json.load(f)
    return DaveClient(), collections

def query_dave(client, collections, question, source, actual_top_k, context_docs, search_query):
    if source == "OHRBench":
        formatted = [{"chunks": [{"text": d}]} if isinstance(d, str) else d for d in context_docs]
        sys_prompt = client.build_system_prompt(formatted, question, max_docs=actual_top_k)
        return client.generate_answer(sys_prompt, question), context_docs[:actual_top_k]
        
    collection_name = {"MultiHop-RAG": "MultiHop", "LegalBench-RAG": "LegalBench", "SQuAD-it": "SQuAD"}.get(source, source)
    col_id = collections.get(collection_name)
    
    if not col_id:
        return f"Error: No collection for {collection_name}", []
        
    retrieved_docs = client.retrieve_chunks(search_query, col_id, top_k=actual_top_k)
    sys_prompt = client.build_system_prompt(retrieved_docs, question, max_docs=actual_top_k)
    return client.generate_answer(sys_prompt, question, collection_id=col_id), retrieved_docs

def query_local_rag(endpoint, question, actual_top_k, source, context_docs, search_query):
    payload = {"question": question, "top_k": actual_top_k, "benchmark_name": source, "search_query": search_query}
    if source == "OHRBench": payload["documents"] = context_docs
        
    try:
        r = requests.post(endpoint, json=payload, timeout=120)
        data = r.json()
        return (data.get("answer", ""), data.get("contexts", [])) if r.status_code == 200 else (f"Error {r.status_code}", [])
    except Exception as e:
        return f"Exception: {e}", []

# EVALUATION PIPELINE

def process_single_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, threshold, conversation_memory):
    query_id, source, gold, raw_question = item['id'], item['source'], item['gold_answer'], item["question"]
    conv_id = query_id.split("_t")[0] if source == "MTRAG" else None
    
    search_query = raw_question
    if source == "MTRAG":
        history = conversation_memory.get(conv_id, "")
        sent_question = f"[HISTORY]\n{history}\n[NEW QUESTION]\n{raw_question}" if history else raw_question
        search_query = f"{history.replace('User:', '').replace('Assistant:', '')} {raw_question}".strip()
    else:
        sent_question = raw_question

    context_docs = split_context_in_docs(item.get("context", ""))
    
    # Initial top_k is fixed and high (e.g. 10)
    fetch_k = 10 

    if "DAVE" in motore_rag:
        rag_answer_temp, retrieved_docs = query_dave(dave_client, dave_collections, sent_question, source, fetch_k, context_docs, search_query)
    else:
        rag_answer_temp, retrieved_docs = query_local_rag(ask_endpoint, sent_question, fetch_k, source, context_docs, search_query)

    # THRESHOLD — batch-encoded semantic filtering
    model = _get_embedding_model()
    filtered_docs = filter_docs_by_threshold(retrieved_docs, search_query, threshold, model)
                    
    # Regenerate the answer ONLY on filtered documents
    if "DAVE" in motore_rag:
        if source == "OHRBench":
            formatted = [{"chunks": [{"text": d}]} if isinstance(d, str) else d for d in filtered_docs]
            sys_prompt = dave_client.build_system_prompt(formatted, sent_question, max_docs=fetch_k)
            rag_answer = dave_client.generate_answer(sys_prompt, sent_question)
        else:
            sys_prompt = dave_client.build_system_prompt(filtered_docs, sent_question, max_docs=fetch_k)
            col_name = {"MultiHop-RAG": "MultiHop", "LegalBench-RAG": "LegalBench", "SQuAD-it": "SQuAD"}.get(source, source)
            rag_answer = dave_client.generate_answer(sys_prompt, sent_question, collection_id=dave_collections.get(col_name))
    else:
        rag_answer = rag_answer_temp if source != "OHRBench" else query_local_rag(ask_endpoint, sent_question, fetch_k, source, filtered_docs, search_query)[0]

    if source == "MTRAG":
        conversation_memory[conv_id] = conversation_memory.get(conv_id, "") + f"User: {raw_question}\nAssistant: {rag_answer}\n"

    doc_strings = [_extract_doc_text(d) for d in filtered_docs] or ["No relevant document found above the threshold."]
    
    return {
        "query_id": query_id, "source": source, "domanda": sent_question, "risposta_rag": rag_answer, "gold": gold,
        "docs": format_retrieved(filtered_docs)
    }, {
        "user_input": sent_question, "response": rag_answer, "retrieved_contexts": doc_strings, "reference": gold
    }

def run_evaluation(dataset, rag_base_url, threshold, ollama_url, ollama_model, llm_api_key="", progress_callback=None, tps_callback=None, motore_rag="Local RAG (BM25)", benchmark_files=None, selected_metrics=None, max_workers=1):
    # --- Automatic ingestion: send the corpus to the RAG under test ---
    if benchmark_files and "DAVE" not in motore_rag:
        if progress_callback: progress_callback("ingest", 0, 1, "Loading corpus into the RAG system...")
        ingest_corpus_to_rag(benchmark_files, ROOT_DIR / "Run", rag_base_url, progress_callback)

    if progress_callback: progress_callback("fetch", 0, len(dataset), "Querying RAG in progress...")

    dave_client, dave_collections = get_dave_client_and_collections() if "DAVE" in motore_rag else (None, None)
    ask_endpoint = rag_base_url.rstrip("/") + "/ask"
    conversation_memory = {}
    ui_reports = []
    ragas_samples = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    for idx, item in enumerate(dataset):
        ui_rep, ragas_samp = process_single_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, threshold, conversation_memory)
        ui_reports.append(ui_rep)
        for k in ragas_samples: ragas_samples[k].append(ragas_samp[k])
        if progress_callback: progress_callback("fetch", idx + 1, len(dataset), "Querying RAG in progress...")

    # --- Truncation of contexts that are too long for RAGAS ---
    truncated_contexts = 0
    for i, ctx_list in enumerate(ragas_samples["retrieved_contexts"]):
        total_chars = sum(len(c) for c in ctx_list)
        if total_chars > MAX_CONTEXT_CHARS:
            ragas_samples["retrieved_contexts"][i] = truncate_context(ctx_list)
            truncated_contexts += 1
    if truncated_contexts > 0:
        logger.info(
            "Truncated %d/%d contexts that exceeded the %d character limit.",
            truncated_contexts, len(ragas_samples["retrieved_contexts"]), MAX_CONTEXT_CHARS
        )

    if progress_callback: progress_callback("ragas", 0, 1, "Computing RAGAS metrics...")

    base_url, client_kwargs = parse_llm_endpoint_or_key(ollama_url, api_key=llm_api_key)
    is_openai = "openai" in base_url.lower() or bool(llm_api_key)
    if not is_openai and ("v1" in base_url or "azure" in base_url.lower()):
        is_openai = True
        
    callbacks = [TPSCallbackHandler(tps_callback)] if tps_callback else []
        
    if is_openai:
        from langchain_openai import ChatOpenAI
        ragas_llm = ChatOpenAI(
            model=ollama_model,
            base_url=base_url if base_url and "localhost" not in base_url else None,
            api_key=llm_api_key or "sk-no-key",
            temperature=0,
            callbacks=callbacks
        )
    else:
        from langchain_ollama import ChatOllama
        ragas_llm = ChatOllama(
            model=ollama_model,
            base_url=base_url,
            temperature=0,
            callbacks=callbacks,
            **({"client_kwargs": client_kwargs} if client_kwargs else {})
        )

    ragas_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    # Ottimizzazioni: usiamo i worker scelti dall'utente
    retries = 3 if is_openai else 2

    # Metriche di default se non specificate
    if not selected_metrics:
        selected_metrics = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness", "context_recall"]
        
    metric_objects = []
    if "faithfulness" in selected_metrics: metric_objects.append(faithfulness)
    if "answer_relevancy" in selected_metrics: metric_objects.append(answer_relevancy)
    if "context_precision" in selected_metrics: metric_objects.append(context_precision)
    if "answer_correctness" in selected_metrics: metric_objects.append(answer_correctness)
    if "context_recall" in selected_metrics: metric_objects.append(context_recall)

    try:
        df_ragas = evaluate(
            dataset=Dataset.from_dict(ragas_samples),
            metrics=metric_objects,
            llm=ragas_llm, embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=max_workers, max_retries=retries, timeout=120),
            raise_exceptions=False
        ).to_pandas()

        # Count queries with failed metrics (NaN)
        metric_cols = ["context_precision", "faithfulness", "answer_correctness", "answer_relevancy", "context_recall"]
        failed_queries = int(df_ragas[metric_cols].isna().any(axis=1).sum())
        total_queries = len(df_ragas)
        successful_queries = total_queries - failed_queries

        if failed_queries > 0:
            logger.warning(
                "RAGAS: %d/%d queries have NaN metrics (failed). "
                "Averages will be calculated only on the %d successful queries.",
                failed_queries, total_queries, successful_queries
            )

        for idx, rep in enumerate(ui_reports):
            row = df_ragas.iloc[idx]
            # Use 0.0 as fallback for NaN, but mark the query as failed
            values = {
                "retrieval_voto": float(row.get("context_precision", 0)),
                "faithfulness_voto": float(row.get("faithfulness", 0)),
                "correctness_voto": float(row.get("answer_correctness", 0)),
                "relevancy_voto": float(row.get("answer_relevancy", 0)),
                "recall_voto": float(row.get("context_recall", 0)),
            }
            # Replace NaN with 0.0 and mark as failed
            has_nan = False
            for k, v in values.items():
                if math.isnan(v):
                    values[k] = 0.0
                    has_nan = True
            values["ragas_fallita"] = has_nan
            rep.update(values)

        # Add coverage statistics to the first report for visibility
        if ui_reports:
            ui_reports[0]["_ragas_stats"] = {
                "totali": total_queries,
                "riuscite": successful_queries,
                "fallite": failed_queries,
                "contesti_troncati": truncated_contexts,
            }

    except Exception as e:
        for rep in ui_reports: rep.update({
            "retrieval_voto": 0, "faithfulness_voto": 0, "correctness_voto": 0,
            "relevancy_voto": 0, "recall_voto": 0, "ragas_fallita": True
        })
        raise RuntimeError(f"RAGAS Error: {e}") from e

    if progress_callback: progress_callback("done", 1, 1, "Completed.")
    return ui_reports
