import os
import json
import math
import re
import logging
import requests
import nest_asyncio
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.callbacks.base import BaseCallbackHandler
# error classification rules live in taxonomy.py (re-exported for app.py)
from taxonomy import classify_query_verdict, METRIC_KEYS as _METRIC_KEYS
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

logger = logging.getLogger("evaluator")

# a RAGAS job (one metric of one query) gets this long; slow judges need minutes for Answer Correctness
# on long gold answers. Metrics that still come back empty are computed again, only for those queries.
RAGAS_TIMEOUT = 300
RAGAS_RETRY_PASSES = 2


class RagError(RuntimeError):
    """The RAG under test failed to answer (HTTP error, empty answer, exception):
    the query is an infrastructure failure, not an answer to evaluate."""

ROOT_DIR = Path(__file__).resolve().parents[1]

MAX_CONTEXT_CHARS = 180_000
FETCH_K = 10  # documents requested from the RAG for each query
sys.path.append(str(ROOT_DIR))
# benchmark corpora shared with ingest_dave.py, so both RAG engines index the same documents
from corpus_sources import Corpus, benchmark_source, DAVE_COLLECTIONS

# Ollama native API paths: stripped from the judge URL, ChatOllama wants the bare host
_OLLAMA_API_PATH = re.compile(r"/api(/(chat|generate|tags|embed|embeddings))?/?$")

def _normalize_llm_url(url: str) -> str:
    """Full URL of the judge endpoint: adds http:// when missing and drops an Ollama
    /api/... path, keeping any other path (e.g. /v1 of an OpenAI-compatible server)."""
    parsed = urlparse(url if "://" in url else f"http://{url}")
    path = _OLLAMA_API_PATH.sub("", parsed.path).rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"

def parse_llm_endpoint_or_key(endpoint_or_key: str, default_base_url: str = "http://localhost:11434", api_key: str = "") -> tuple[str, dict]:
    """
    Parses an LLM Judge endpoint or API key input.
    Returns (base_url, client_kwargs); base_url is "" when only an API key is given,
    meaning the official OpenAI endpoint.
    """
    val = (endpoint_or_key or "").strip()

    # If API key is provided directly as an argument, use the URL as base_url
    if api_key:
        auth_header = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
        return (_normalize_llm_url(val) if val else ""), {"headers": {"Authorization": auth_header}}

    if not val:
        return default_base_url, {}
    if "://" in val or val.startswith(("localhost", "127.0.0.1")):
        return _normalize_llm_url(val), {}
    # Otherwise treat as API key
    auth_header = val if val.lower().startswith("bearer ") else f"Bearer {val}"
    return default_base_url, {"headers": {"Authorization": auth_header}}

_JUDGE_OPENAI_CLASS = None


def _rejects_temperature(error):
    """True for the 400 of models that only accept their default temperature (OpenAI o-series, gpt-5...)."""
    text = str(error).lower()
    return "temperature" in text and "unsupported" in text


def _openai_judge_class():
    """ChatOpenAI that asks for temperature 0 (comparable runs) but, if the model refuses any other
    temperature than its default, repeats the call without it and keeps doing so."""
    global _JUDGE_OPENAI_CLASS
    if _JUDGE_OPENAI_CLASS is None:
        from langchain_openai import ChatOpenAI

        class JudgeChatOpenAI(ChatOpenAI):
            # RAGAS also passes temperature (0.01, 0.3) as a per-call kwarg, so the field alone is not enough
            fixed_temperature: bool = False

            def _generate(self, *args, **kwargs):
                if self.fixed_temperature:
                    kwargs.pop("temperature", None)
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as e:
                    if self.fixed_temperature or not _rejects_temperature(e):
                        raise
                    self.fixed_temperature, self.temperature = True, None
                    kwargs.pop("temperature", None)
                    return super()._generate(*args, **kwargs)

            async def _agenerate(self, *args, **kwargs):
                if self.fixed_temperature:
                    kwargs.pop("temperature", None)
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as e:
                    if self.fixed_temperature or not _rejects_temperature(e):
                        raise
                    self.fixed_temperature, self.temperature = True, None
                    kwargs.pop("temperature", None)
                    return await super()._agenerate(*args, **kwargs)

        _JUDGE_OPENAI_CLASS = JudgeChatOpenAI
    return _JUDGE_OPENAI_CLASS


def build_judge_llm(endpoint: str, model: str, api_key: str = "", callbacks=None, timeout=None):
    """LangChain chat model for the judge, shared by RAGAS and the dashboard's expert report.
    OpenAI-compatible (ChatOpenAI) with an API key or a /v1, openai or azure URL,
    otherwise Ollama. Returns (llm, is_openai)."""
    base_url, client_kwargs = parse_llm_endpoint_or_key(endpoint, api_key=api_key)
    is_openai = bool(api_key) or "openai" in base_url.lower() or "azure" in base_url.lower()         or "/v1" in urlparse(base_url).path
    if is_openai:
        # always the configured URL: None would send a local server's calls to api.openai.com
        return _openai_judge_class()(
            model=model,
            base_url=base_url or None,
            api_key=api_key or "sk-no-key",
            temperature=0,
            callbacks=callbacks,
            **({"timeout": timeout} if timeout else {}),
        ), True
    if timeout:
        client_kwargs = {**client_kwargs, "timeout": timeout}
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        callbacks=callbacks,
        **({"client_kwargs": client_kwargs} if client_kwargs else {})
    ), False

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

def format_retrieved(docs):
    if not docs: return "No documents retrieved."
    return "".join(f"[Doc {i}]\n{_extract_doc_text(d)}\n\n" for i, d in enumerate(docs, 1))

# RAG ENGINES

INGEST_BATCH_DOCS = 500
INGEST_BATCH_BYTES = 20 * 1024 * 1024

def _missing_doc_ids(ingest_endpoint, source, doc_ids):
    """Asks /ingest (check_only) which corpus documents the RAG does not hold yet.
    Returns the set of missing ids; a RAG that does not support check_only (error or
    no "complete"/"missing" field) gets the whole corpus."""
    try:
        r = requests.post(ingest_endpoint, json={
            "benchmark_name": source, "check_only": True,
            "expected_count": len(doc_ids), "ids": doc_ids,
        }, timeout=120)
        data = r.json() if r.status_code == 200 else {}
    except Exception as e:
        logger.warning("Ingestion check failed for '%s': %s", source, e)
        data = {}
    if data.get("complete") is True:
        return set()
    if isinstance(data.get("missing"), list):
        return set(map(str, data["missing"])) & set(doc_ids)
    return set(doc_ids)

def ingest_corpus_to_rag(benchmark_files, run_dir, rag_base_url, progress_callback=None):
    """Loads the benchmark corpus into the RAG server via POST /ingest.
    First asks /ingest (check_only, with the corpus ids) which documents the RAG is
    missing: if none the benchmark is skipped, otherwise only the missing documents are
    sent in batches and the RAG's /ingest must add them to the existing index.
    If the check_only request fails the RAG gets the whole corpus; if sending the
    documents fails (HTTP != 200 or exception) raises RuntimeError, so the evaluation
    never runs on an incomplete index.
    Returns the seconds spent on each benchmark (check plus upload), keyed by source."""
    ingest_endpoint = rag_base_url.rstrip("/") + "/ingest"
    timings = {}

    total_files = len(benchmark_files)
    for idx, file_name in enumerate(benchmark_files):
        path = run_dir / file_name
        if not path.exists():
            logger.warning("Benchmark file not found: %s", path)
            continue

        start = time.perf_counter()
        source = benchmark_source(path)
        corpus = Corpus(source)
        missing = corpus.missing_paths()
        if missing:
            raise RuntimeError(
                f"Corpus for '{source}' not found: {', '.join(str(p) for p in missing)}. "
                "Download the benchmarks with the 'Download Benchmark' button and extract them into Benchmarks/."
            )

        if progress_callback:
            progress_callback("ingest", idx, total_files, f"Checking whether the RAG already holds the {source} corpus...")
        doc_ids = corpus.ids()
        missing_ids = _missing_doc_ids(ingest_endpoint, source, doc_ids)
        if not missing_ids:
            logger.info("Ingestion skipped for '%s': the RAG already holds all %d documents", source, len(doc_ids))
            if progress_callback:
                progress_callback("ingest", idx, total_files, f"Corpus {source} already in the RAG, skipping ingestion.")
            timings[source] = time.perf_counter() - start
            continue
        to_send = len(missing_ids)

        def _send(batch):
            r = requests.post(ingest_endpoint, json={"benchmark_name": source, "documents": batch}, timeout=300)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}".rstrip(": "))

        sent = 0
        batch, batch_bytes = [], 0
        try:
            for doc_id, text in corpus.docs(include=missing_ids):
                batch.append({"id": doc_id, "text": text})
                batch_bytes += len(text.encode("utf-8"))
                if len(batch) >= INGEST_BATCH_DOCS or batch_bytes >= INGEST_BATCH_BYTES:
                    _send(batch)
                    sent += len(batch)
                    batch, batch_bytes = [], 0
                    if progress_callback:
                        progress_callback("ingest", idx, total_files,
                                          f"Loading corpus into RAG: {source} ({sent}/{to_send} missing docs sent)...")
            if batch:
                _send(batch)
                sent += len(batch)
            timings[source] = time.perf_counter() - start
            logger.info("Ingestion OK for '%s': %d documents sent in %.1f s", source, sent, timings[source])
        except Exception as e:
            # an incomplete index would make every later answer unreliable: stop here
            # (the next evaluation resumes from the documents still missing)
            logger.error("Ingestion failed for '%s' after %d documents: %s", source, sent, e)
            raise RuntimeError(
                f"Ingestion of the {source} corpus into the RAG failed after {sent}/{to_send} documents "
                f"({ingest_endpoint}): {e}. Evaluation stopped: check the RAG /ingest endpoint and retry."
            ) from e

    if progress_callback:
        progress_callback("ingest", total_files, total_files, "Corpus loading completed.")
    return timings


def get_dave_client_and_collections():
    from dave_client import DaveClient
    collections = {}
    col_path = Path(os.environ.get("STATE_DIR") or ROOT_DIR) / ".dave_collections.json"
    if col_path.exists():
        with open(col_path, "r") as f:
            collections = json.load(f)
    from settings import dave_settings
    return DaveClient(**dave_settings()), collections

# benchmark -> DAVE collection name, from the single mapping used by ingest_dave.py
DAVE_COLLECTION_NAMES = {source: name for name, source in DAVE_COLLECTIONS.items()}

def _dave_collection_id(collections, source):
    """DAVE collection id of a benchmark."""
    collection_name = DAVE_COLLECTION_NAMES.get(source, source)
    col_id = (collections or {}).get(collection_name)
    if not col_id:
        raise RagError(f"No DAVE collection for '{collection_name}' in .dave_collections.json (run ingest_dave.py)")
    return col_id

def query_dave(client, collections, source, actual_top_k, search_query):
    """DAVE retrieval only (no generation): at most actual_top_k chunks, like the local RAG."""
    col_id = _dave_collection_id(collections, source)
    return client.retrieve_chunks(search_query, col_id, top_k=actual_top_k)

def generate_dave(client, collections, source, question, docs):
    """One DAVE generation on exactly `docs` (the retrieved documents RAGAS will judge)."""
    formatted = [{"chunks": [{"text": d}]} if isinstance(d, str) else d for d in docs]
    sys_prompt = client.build_system_prompt(formatted, question)
    return client.generate_answer(sys_prompt, question, collection_id=_dave_collection_id(collections, source))

def query_local_rag(endpoint, question, actual_top_k, source, search_query):
    """Ask the local RAG, which retrieves from its own index and answers."""
    payload = {"question": question, "top_k": actual_top_k, "benchmark_name": source, "search_query": search_query}

    # an HTTP error, an exception or a missing answer is a failure of the RAG, never an
    # answer to evaluate: raise RagError and let run_evaluation mark the query as RAG_ERROR
    try:
        r = requests.post(endpoint, json=payload, timeout=120)
    except Exception as e:
        raise RagError(f"/ask request failed: {type(e).__name__}: {e}") from e
    try:
        data = r.json()
    except ValueError:
        data = None
    if r.status_code != 200:
        detail = data.get("error", "") if isinstance(data, dict) else r.text[:300]
        raise RagError(f"/ask returned HTTP {r.status_code}: {detail}".strip().rstrip(":"))
    if not isinstance(data, dict):
        raise RagError(f"/ask returned a non-JSON response: {r.text[:300]}")
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise RagError("/ask returned no answer (missing or empty 'answer' field)")
    return answer, data.get("contexts") or []

# EVALUATION PIPELINE

def process_single_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, conversation_memory):
    query_id, source, gold, raw_question = item['id'], item['source'], item['gold_answer'], item["question"]
    conv_id = query_id.split("_t")[0] if source == "MTRAG" else None
    
    search_query = raw_question
    if source == "MTRAG":
        history = conversation_memory.get(conv_id, "")
        sent_question = f"[HISTORY]\n{history}\n[NEW QUESTION]\n{raw_question}" if history else raw_question
        search_query = f"{history.replace('User:', '').replace('Assistant:', '')} {raw_question}".strip()
    else:
        sent_question = raw_question

    # Same top_k for both engines. The answer is judged exactly as returned, together with
    # the documents retrieved for it (the retrieved_contexts given to RAGAS)
    if "DAVE" in motore_rag:
        retrieved_docs = query_dave(dave_client, dave_collections, source, FETCH_K, search_query)
        rag_answer = generate_dave(dave_client, dave_collections, source, sent_question, retrieved_docs)
    else:
        rag_answer, retrieved_docs = query_local_rag(ask_endpoint, sent_question, FETCH_K, source, search_query)

    if source == "MTRAG":
        conversation_memory[conv_id] = conversation_memory.get(conv_id, "") + f"User: {raw_question}\nAssistant: {rag_answer}\n"

    doc_strings = [_extract_doc_text(d) for d in retrieved_docs] or ["No document retrieved."]
    
    return {
        "query_id": query_id, "source": source, "domanda": sent_question, "risposta_rag": rag_answer, "gold": gold,
        "docs": format_retrieved(retrieved_docs)
    }, {
        "user_input": sent_question, "response": rag_answer, "retrieved_contexts": doc_strings, "reference": gold
    }

def _as_rag_error(e):
    """Any failure of the RAG call as a RagError (DAVE raises HTTPError, TRPCError, DaveEmptyAnswerError...)."""
    return e if isinstance(e, RagError) else RagError(f"{type(e).__name__}: {e}")

def _rag_error_report(item, err):
    """Report of a query the RAG failed to answer: no answer, no RAGAS metrics (all marked
    as not computed, so every average skips it), classified as RAG_ERROR."""
    rep = {
        "query_id": item["id"], "source": item["source"], "domanda": item["question"],
        "risposta_rag": "", "gold": item["gold_answer"], "docs": "", "rag_error": str(err),
        "ragas_fallita": False, "metriche_non_valide": list(_METRIC_KEYS),
        **{key: 0.0 for key in _METRIC_KEYS},
    }
    rep.update(classify_query_verdict(rep))
    return rep

def run_evaluation(dataset, rag_base_url, ollama_url, ollama_model, llm_api_key="", progress_callback=None, tps_callback=None, motore_rag="Local RAG (BM25)", benchmark_files=None, selected_metrics=None, max_workers=1):
    run_start = time.perf_counter()
    # --- Automatic ingestion: send the corpus to the RAG under test ---
    ingest_times = {}
    if benchmark_files and "DAVE" not in motore_rag:
        if progress_callback: progress_callback("ingest", 0, 1, "Loading corpus into the RAG system...")
        ingest_times = ingest_corpus_to_rag(benchmark_files, ROOT_DIR / "Run", rag_base_url, progress_callback)

    if progress_callback: progress_callback("fetch", 0, len(dataset), "Querying RAG in progress...")

    dave_client, dave_collections = get_dave_client_and_collections() if "DAVE" in motore_rag else (None, None)
    ask_endpoint = rag_base_url.rstrip("/") + "/ask"
    conversation_memory = {}
    ui_reports = []
    ragas_samples = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    answered_reports = []  # queries the RAG answered, aligned with the RAGAS rows
    query_times = {}  # seconds spent querying the RAG, per benchmark
    for idx, item in enumerate(dataset):
        query_start = time.perf_counter()
        try:
            ui_rep, ragas_samp = process_single_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, conversation_memory)
        except Exception as e:
            # infrastructure failure (HTTP error, empty answer, DAVE/tRPC error...): the query
            # is recorded as RAG_ERROR, kept out of RAGAS and of the averages, and the run goes on
            err = _as_rag_error(e)
            logger.warning("RAG error on query %s: %s", item.get("id"), err)
            ui_reports.append(_rag_error_report(item, err))
        else:
            ui_reports.append(ui_rep)
            answered_reports.append(ui_rep)
            for k in ragas_samples: ragas_samples[k].append(ragas_samp[k])
        query_times[item["source"]] = query_times.get(item["source"], 0.0) + time.perf_counter() - query_start
        if progress_callback: progress_callback("fetch", idx + 1, len(dataset), "Querying RAG in progress...")

    rag_errors = len(ui_reports) - len(answered_reports)
    if not answered_reports:
        first_error = next((r["rag_error"] for r in ui_reports if r.get("rag_error")), "no queries")
        raise RuntimeError(
            f"The RAG did not answer any of the {len(ui_reports)} queries, nothing to evaluate. "
            f"First error: {first_error}"
        )
    if rag_errors:
        logger.warning("%d/%d queries failed on the RAG side (RAG_ERROR): excluded from RAGAS and the averages.",
                       rag_errors, len(ui_reports))

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

    callbacks = [TPSCallbackHandler(tps_callback)] if tps_callback else []
    ragas_llm, is_openai = build_judge_llm(ollama_url, ollama_model, llm_api_key, callbacks=callbacks)

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
        # RAGAS rows follow the answered queries only (RAG_ERROR queries were never sent)
        ragas_start = time.perf_counter()
        run_config = RunConfig(max_workers=max_workers, max_retries=retries, timeout=RAGAS_TIMEOUT)

        def _run_ragas(row_ids, metrics):
            samples = {key: [values[i] for i in row_ids] for key, values in ragas_samples.items()}
            return evaluate(
                dataset=Dataset.from_dict(samples), metrics=metrics,
                llm=ragas_llm, embeddings=ragas_embeddings,
                run_config=run_config, raise_exceptions=False,
            ).to_pandas()

        df_ragas = _run_ragas(range(len(answered_reports)), metric_objects)

        # retry the metrics that came back empty (judge timeout or bad output), only for their queries
        for attempt in range(1, RAGAS_RETRY_PASSES + 1):
            retry = [(m, [i for i in range(len(df_ragas)) if math.isnan(float(df_ragas.at[i, m.name]))])
                     for m in metric_objects if m.name in df_ragas.columns]
            retry = [(m, rows) for m, rows in retry if rows]
            if not retry:
                break
            logger.info("RAGAS retry %d/%d: %s", attempt, RAGAS_RETRY_PASSES,
                        ", ".join(f"{m.name} x{len(rows)}" for m, rows in retry))
            if progress_callback:
                progress_callback("ragas", 0, 1, f"Retrying {sum(len(rows) for _, rows in retry)} empty metrics "
                                                 f"({attempt}/{RAGAS_RETRY_PASSES})...")
            for metric, rows in retry:
                df_retry = _run_ragas(rows, [metric])
                if metric.name in df_retry.columns:
                    df_ragas.loc[rows, metric.name] = df_retry[metric.name].to_numpy()
        ragas_seconds = time.perf_counter() - ragas_start

        for idx, rep in enumerate(answered_reports):
            row = df_ragas.iloc[idx]
            # Use 0.0 as fallback for NaN or not selected metrics, but keep track of them
            # in 'metriche_non_valide' so the classifier ignores them
            values, invalid, has_nan = {}, [], False
            for key, col in _METRIC_KEYS.items():
                v = float(row[col]) if col in df_ragas.columns else math.nan
                if math.isnan(v):
                    invalid.append(key)
                    has_nan = has_nan or col in df_ragas.columns
                    v = 0.0
                values[key] = v
            values["ragas_fallita"] = has_nan
            values["metriche_non_valide"] = invalid
            rep.update(values)

        # Count queries with failed metrics (NaN), only among the metrics actually computed
        total_queries = len(answered_reports)
        failed_queries = sum(1 for r in answered_reports if r["ragas_fallita"])
        successful_queries = total_queries - failed_queries

        if failed_queries > 0:
            logger.warning(
                "RAGAS: %d/%d queries have NaN metrics (failed). "
                "Averages will be calculated only on the %d successful queries.",
                failed_queries, total_queries, successful_queries
            )

        # Classify verdict and error pattern for each query
        for rep in ui_reports:
            rep.update(classify_query_verdict(rep))

        # Add coverage statistics to the first report for visibility
        if ui_reports:
            ui_reports[0]["_ragas_stats"] = {
                "totali": total_queries,
                "riuscite": successful_queries,
                "fallite": failed_queries,
                "contesti_troncati": truncated_contexts,
                "errori_rag": rag_errors,
                # seconds per phase: ingestion and queries per benchmark, RAGAS for the whole run
                "tempi": {
                    "ingestion": ingest_times,
                    "query": query_times,
                    "ragas": ragas_seconds,
                    "totale": time.perf_counter() - run_start,
                },
            }
        logger.info("Timings: ingestion %.1f s, RAG queries %.1f s, RAGAS %.1f s",
                    sum(ingest_times.values()), sum(query_times.values()), ragas_seconds)

    except Exception as e:
        for rep in answered_reports:
            rep.update({
                "retrieval_voto": 0, "faithfulness_voto": 0, "correctness_voto": 0,
                "relevancy_voto": 0, "recall_voto": 0, "ragas_fallita": True,
                "metriche_non_valide": list(_METRIC_KEYS),
            })
            rep.update(classify_query_verdict(rep))
        raise RuntimeError(f"RAGAS Error: {e}") from e

    if progress_callback: progress_callback("done", 1, 1, "Completed.")
    return ui_reports
