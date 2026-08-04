import re
import json
import logging
import requests
import nest_asyncio
import sys
from collections import defaultdict
from html import unescape
from pathlib import Path

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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    if not text or ("<" not in text and "&" not in text):
        return text
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", unescape(cleaned)).strip()

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

def _tronca_testo(testo, max_chars):
    """Tronca il testo rispettando i confini di frase (. ! ?) o, in subordine,
    i confini di parola. Non taglia mai a metà parola."""
    if len(testo) <= max_chars:
        return testo
    candidato = testo[:max_chars]
    # Non tornare indietro più del 30% per cercare una fine di frase
    soglia_min = int(max_chars * 0.7)
    pos = -1
    for terminatore in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
        idx = candidato.rfind(terminatore, soglia_min)
        if idx > pos:
            pos = idx + 1  # include il segno di punteggiatura
    if pos > soglia_min:
        return candidato[:pos].rstrip() + "\n[... troncato ...]"
    # Fallback: confine di parola
    ultimo_spazio = candidato.rfind(' ', soglia_min)
    if ultimo_spazio > soglia_min:
        return candidato[:ultimo_spazio].rstrip() + "\n[... troncato ...]"
    # Ultima risorsa (testo senza spazi): tronca al limite
    return candidato.rstrip() + "\n[... troncato ...]"

def tronca_contesto(testi, max_chars=MAX_CONTEXT_CHARS):
    """Tronca una lista di stringhe di contesto in modo che la lunghezza
    totale non superi max_chars, evitando errori 'prompt too long' in RAGAS.
    Distribuisce lo spazio equamente tra i documenti e tronca ai confini di frase."""
    if not testi:
        return testi
    totale = sum(len(t) for t in testi)
    if totale <= max_chars:
        return testi
    logger.warning(
        "Contesto troppo lungo (%d chars, limite %d). Troncamento in corso...",
        totale, max_chars
    )
    # Distribuisci lo spazio equamente, con un minimo di 200 chars per doc
    quota_per_doc = max(200, max_chars // len(testi))
    troncati = []
    chars_rimasti = max_chars
    for t in testi:
        limite = min(len(t), quota_per_doc, chars_rimasti)
        if limite <= 0:
            break
        pezzo = _tronca_testo(t, limite)
        troncati.append(pezzo)
        chars_rimasti -= len(pezzo)
    return troncati

def calcola_similarita_semantica(gold, pred):
    if not gold.strip() and not pred.strip(): return 1.0
    if not gold.strip() or not pred.strip(): return 0.0
    model = _get_embedding_model()
    embeddings = model.encode([gold, pred])
    return max(0.0, float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]))

def formatta_retrieved(docs):
    if not docs: return "Nessun documento recuperato."
    return "".join(f"[Doc {i}]\n{d.get('text', d) if isinstance(d, dict) else str(d)}\n\n" for i, d in enumerate(docs, 1))

# MOTORI RAG
def ingest_corpus_to_rag(benchmark_files, run_dir, rag_base_url, progress_callback=None):
    """Carica TUTTI i documenti dei benchmark nel server RAG via /ingest.
    Funziona con qualsiasi sistema RAG che espone un endpoint POST /ingest.
    Se l'endpoint non esiste o fallisce, logga un warning e continua."""
    ingest_endpoint = rag_base_url.rstrip("/") + "/ingest"

    total_files = len(benchmark_files)
    for idx, nome_file in enumerate(benchmark_files):
        path = run_dir / nome_file
        if not path.exists():
            logger.warning("File benchmark non trovato: %s", path)
            continue

        # Carica TUTTI gli elementi dal file JSON del benchmark
        with open(path, "r", encoding="utf-8-sig") as f:
            tutti_elementi = json.load(f)

        # Identifica il source dal primo elemento
        source = tutti_elementi[0].get("source", "default") if tutti_elementi else "default"

        # Estrai e deduplica i documenti di contesto
        docs_unici = {}
        for item in tutti_elementi:
            context = item.get("context", "")
            if not context or not str(context).strip():
                continue
            doc_texts = split_context_in_docs(context)
            for doc_text in doc_texts:
                stripped = doc_text.strip()
                if stripped:
                    content_hash = hash(stripped[:500])
                    if content_hash not in docs_unici:
                        docs_unici[content_hash] = stripped

        if not docs_unici:
            logger.warning("Nessun documento trovato in %s", nome_file)
            continue

        doc_list = [{"id": f"{source}_doc{i}", "text": v} for i, v in enumerate(docs_unici.values())]

        logger.info("Ingestion: invio %d documenti unici per benchmark '%s' (da %d query totali)",
                     len(doc_list), source, len(tutti_elementi))
        if progress_callback:
            progress_callback("ingest", idx, total_files,
                              f"Caricamento corpus nel RAG: {source} ({len(doc_list)} doc unici da {len(tutti_elementi)} query)...")

        try:
            r = requests.post(ingest_endpoint, json={
                "benchmark_name": source,
                "documents": doc_list
            }, timeout=300)
            if r.status_code == 200:
                logger.info("Ingestion OK per '%s': %s", source, r.json().get("message", ""))
            else:
                logger.warning("Ingestion fallita per '%s': HTTP %d", source, r.status_code)
        except Exception as e:
            logger.warning("Ingestion fallita per '%s': %s", source, e)

    if progress_callback:
        progress_callback("ingest", total_files, total_files, "Caricamento corpus completato.")


def get_dave_client_and_collections():
    from dave_client import DaveClient
    collections = {}
    col_path = ROOT_DIR / ".dave_collections.json"
    if col_path.exists():
        with open(col_path, "r") as f:
            collections = json.load(f)
    return DaveClient(), collections

def interroga_dave(client, collections, domanda, source, actual_top_k, docs_contesto, query_di_ricerca):
    if source == "OHRBench":
        formattati = [{"chunks": [{"text": d}]} if isinstance(d, str) else d for d in docs_contesto]
        sys_prompt = client.build_system_prompt(formattati, domanda, max_docs=actual_top_k)
        return client.generate_answer(sys_prompt, domanda), docs_contesto[:actual_top_k]
        
    nome_collection = {"MultiHop-RAG": "MultiHop", "LegalBench-RAG": "LegalBench"}.get(source, source)
    col_id = collections.get(nome_collection)
    
    if not col_id:
        return f"Errore: Nessuna collection per {nome_collection}", []
        
    doc_recuperati = client.retrieve_chunks(query_di_ricerca, col_id, top_k=actual_top_k)
    sys_prompt = client.build_system_prompt(doc_recuperati, domanda, max_docs=actual_top_k)
    return client.generate_answer(sys_prompt, domanda, collection_id=col_id), doc_recuperati

def interroga_rag_locale(endpoint, domanda, actual_top_k, source, docs_contesto, query_di_ricerca):
    payload = {"question": domanda, "top_k": actual_top_k, "benchmark_name": source, "search_query": query_di_ricerca}
    if source == "OHRBench": payload["documents"] = docs_contesto
        
    try:
        r = requests.post(endpoint, json=payload, timeout=120)
        data = r.json()
        return (data.get("answer", ""), data.get("contexts", [])) if r.status_code == 200 else (f"Errore {r.status_code}", [])
    except Exception as e:
        return f"Eccezione: {e}", []

# PIPELINE DI VALUTAZIONE

def processa_singola_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, threshold, memoria_conversazioni):
    query_id, source, gold, domanda_pura = item['id'], item['source'], item['gold_answer'], item["question"]
    conv_id = query_id.split("_t")[0] if source == "MTRAG" else None
    
    search_query = domanda_pura
    if source == "MTRAG":
        storia = memoria_conversazioni.get(conv_id, "")
        domanda_inviata = f"[STORIA]\n{storia}\n[NUOVA DOMANDA]\n{domanda_pura}" if storia else domanda_pura
        search_query = f"{storia.replace('Utente:', '').replace('Assistente:', '')} {domanda_pura}".strip()
    else:
        domanda_inviata = domanda_pura

    docs_contesto = split_context_in_docs(item.get("context", ""))
    
    # Ora il top_k iniziale è fisso e alto (es. 10)
    fetch_k = 10 

    if "DAVE" in motore_rag:
        risposta_rag_temp, doc_recuperati = interroga_dave(dave_client, dave_collections, domanda_inviata, source, fetch_k, docs_contesto, search_query)
    else:
        risposta_rag_temp, doc_recuperati = interroga_rag_locale(ask_endpoint, domanda_inviata, fetch_k, source, docs_contesto, search_query)

    # THRESHOLD
    doc_filtrati = []
    if doc_recuperati:
        model = _get_embedding_model()
        query_emb = model.encode([search_query])[0]
        
        for d in doc_recuperati:
            testo_doc = d.get('text', d) if isinstance(d, dict) else str(d)
            if non_vuoto := testo_doc.strip():
                doc_emb = model.encode([non_vuoto])[0]
                sim = float(cosine_similarity([query_emb], [doc_emb])[0][0])
                if sim >= threshold:
                    doc_filtrati.append(d)
                    
    # Se il filtro svuota tutto, teniamo il migliore
    if not doc_filtrati and doc_recuperati:
        doc_filtrati = [doc_recuperati[0]]
        
    # Rigeneriamo la risposta SOLO sui documenti filtrati
    if "DAVE" in motore_rag:
        if source == "OHRBench":
            formattati = [{"chunks": [{"text": d}]} if isinstance(d, str) else d for d in doc_filtrati]
            sys_prompt = dave_client.build_system_prompt(formattati, domanda_inviata, max_docs=fetch_k)
            risposta_rag = dave_client.generate_answer(sys_prompt, domanda_inviata)
        else:
            sys_prompt = dave_client.build_system_prompt(doc_filtrati, domanda_inviata, max_docs=fetch_k)
            col_name = {"MultiHop-RAG": "MultiHop", "LegalBench-RAG": "LegalBench"}.get(source, source)
            risposta_rag = dave_client.generate_answer(sys_prompt, domanda_inviata, collection_id=dave_collections.get(col_name))
    else:
        risposta_rag = risposta_rag_temp if source != "OHRBench" else interroga_rag_locale(ask_endpoint, domanda_inviata, fetch_k, source, doc_filtrati, search_query)[0]

    if source == "MTRAG":
        memoria_conversazioni[conv_id] = memoria_conversazioni.get(conv_id, "") + f"Utente: {domanda_pura}\nAssistente: {risposta_rag}\n"

    doc_strings = [d.get('text', d) if isinstance(d, dict) else str(d) for d in doc_filtrati] or ["Nessun documento rilevante trovato oltre la soglia."]
    
    return {
        "query_id": query_id, "source": source, "domanda": domanda_inviata, "risposta_rag": risposta_rag, "gold": gold,
        "docs": formatta_retrieved(doc_filtrati), "semantic_sim": calcola_similarita_semantica(gold, risposta_rag)
    }, {
        "user_input": domanda_inviata, "response": risposta_rag, "retrieved_contexts": doc_strings, "reference": gold
    }

def run_evaluation(dataset, rag_base_url, threshold, ollama_url, ollama_model, progress_callback=None, motore_rag="Local RAG (BM25)", benchmark_files=None):
    # --- Ingestion automatica: invia il corpus al RAG sotto test ---
    if benchmark_files:
        if progress_callback: progress_callback("ingest", 0, 1, "Caricamento corpus nel sistema RAG...")
        ingest_corpus_to_rag(benchmark_files, ROOT_DIR / "Run", rag_base_url, progress_callback)

    if progress_callback: progress_callback("fetch", 0, len(dataset), "Interrogazione RAG in corso...")

    dave_client, dave_collections = get_dave_client_and_collections() if "DAVE" in motore_rag else (None, None)
    ask_endpoint = rag_base_url.rstrip("/") + "/ask"
    memoria_conversazioni = {}
    ui_reports = []
    ragas_samples = {"user_input": [], "response": [], "retrieved_contexts": [], "reference": []}

    for idx, item in enumerate(dataset):
        ui_rep, ragas_samp = processa_singola_query(item, motore_rag, dave_client, dave_collections, ask_endpoint, threshold, memoria_conversazioni)
        ui_reports.append(ui_rep)
        for k in ragas_samples: ragas_samples[k].append(ragas_samp[k])
        if progress_callback: progress_callback("fetch", idx + 1, len(dataset), "Interrogazione RAG in corso...")

    # --- Troncamento contesti troppo lunghi per RAGAS ---
    contesti_troncati = 0
    for i, ctx_list in enumerate(ragas_samples["retrieved_contexts"]):
        totale_chars = sum(len(c) for c in ctx_list)
        if totale_chars > MAX_CONTEXT_CHARS:
            ragas_samples["retrieved_contexts"][i] = tronca_contesto(ctx_list)
            contesti_troncati += 1
    if contesti_troncati > 0:
        logger.info(
            "Troncati %d/%d contesti che superavano il limite di %d caratteri.",
            contesti_troncati, len(ragas_samples["retrieved_contexts"]), MAX_CONTEXT_CHARS
        )

    if progress_callback: progress_callback("ragas", 0, 1, "Calcolo metriche Ragas in corso...")

    ragas_llm = ChatOllama(model=ollama_model, base_url=ollama_url.split("/api")[0], temperature=0, client_kwargs={"headers": {"Authorization": "Bearer 130e9216867047ceaccab3e8017c5123.joL1WY4giNPRo76atH5cJJOk"}})
    ragas_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    try:
        df_ragas = evaluate(
            dataset=Dataset.from_dict(ragas_samples),
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness, context_recall],
            llm=ragas_llm, embeddings=ragas_embeddings,
            run_config=RunConfig(max_workers=1, max_retries=2, timeout=120),
            raise_exceptions=False
        ).to_pandas()

        # Conta le query con metriche fallite (NaN)
        metric_cols = ["context_precision", "faithfulness", "answer_correctness", "answer_relevancy", "context_recall"]
        query_fallite = int(df_ragas[metric_cols].isna().any(axis=1).sum())
        query_totali = len(df_ragas)
        query_ok = query_totali - query_fallite

        if query_fallite > 0:
            logger.warning(
                "RAGAS: %d/%d query hanno metriche NaN (fallite). "
                "Le medie saranno calcolate solo sulle %d query riuscite.",
                query_fallite, query_totali, query_ok
            )

        for idx, rep in enumerate(ui_reports):
            row = df_ragas.iloc[idx]
            # Usa 0.0 come fallback per i NaN, ma segna la query come fallita
            import math
            valori = {
                "retrieval_voto": float(row.get("context_precision", 0)),
                "faithfulness_voto": float(row.get("faithfulness", 0)),
                "correctness_voto": float(row.get("answer_correctness", 0)),
                "relevancy_voto": float(row.get("answer_relevancy", 0)),
                "recall_voto": float(row.get("context_recall", 0)),
            }
            # Sostituisci NaN con 0.0 e segna come fallita
            ha_nan = False
            for k, v in valori.items():
                if math.isnan(v):
                    valori[k] = 0.0
                    ha_nan = True
            valori["ragas_fallita"] = ha_nan
            rep.update(valori)

        # Aggiungi statistiche di copertura al primo report per visibilità
        if ui_reports:
            ui_reports[0]["_ragas_stats"] = {
                "totali": query_totali,
                "riuscite": query_ok,
                "fallite": query_fallite,
                "contesti_troncati": contesti_troncati,
            }

    except Exception as e:
        for rep in ui_reports: rep.update({
            "retrieval_voto": 0, "faithfulness_voto": 0, "correctness_voto": 0,
            "relevancy_voto": 0, "recall_voto": 0, "ragas_fallita": True
        })
        raise RuntimeError(f"Errore Ragas: {e}") from e

    if progress_callback: progress_callback("done", 1, 1, "Completato.")
    return ui_reports
