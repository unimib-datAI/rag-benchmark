import re
import json
import random
import requests

from collections import defaultdict
from html import unescape
from pathlib import Path

from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# singleton per il modello di embedding, così non lo ricarica ogni volta
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model


# pulizia testo e split documenti

def clean_html_text(text):
    # se non c'è html o entità, ritorno subito
    if not text or ("<" not in text and "&" not in text):
        return text
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_context_in_docs(context):
    """Splitta il contesto in una lista di documenti."""
    if isinstance(context, list):
        docs = [str(c).strip() for c in context if str(c).strip()]
        return [clean_html_text(doc) for doc in docs]

    text = str(context or "").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [clean_html_text(text)]

    # qui cerco di capire dove inizia un nuovo documento
    # il pattern è tipo [Doc 1] oppure [Documento X]
    docs, current = [], []
    for line in lines:
        is_new_doc = line.startswith("[") and "]" in line
        if is_new_doc and current:
            docs.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        docs.append(" ".join(current).strip())

    return [clean_html_text(doc) for doc in docs]


# similarità semantica

def calcola_similarita_semantica(gold, pred):
    # caso banale: entrambi vuoti -> perfetto
    if not gold.strip() and not pred.strip():
        return 1.0
    if not gold.strip() or not pred.strip():
        return 0.0

    model = _get_embedding_model()
    embeddings = model.encode([gold, pred])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # print(f"DEBUG sim: {sim}")
    return max(0.0, float(sim))


# interazione con endpoint RAG

def interroga_rag(endpoint, query, top_k, documents):
    try:
        r = requests.post(endpoint, json={"question": query, "top_k": top_k, "documents": documents}, timeout=120)
        if r.status_code == 200:
            return r.json()
        return {"answer": f"Errore: {r.status_code}", "retrieved_docs": []}
    except Exception as e:
        return {"answer": f"Eccezione: {str(e)}", "retrieved_docs": []}


def formatta_retrieved(docs):
    if not docs:
        return "Nessun documento recuperato."
    return "".join(
        f"[Doc {i}]\n"
        f"{d.get('text', d) if isinstance(d, dict) else str(d)}\n\n"
        for i, d in enumerate(docs, 1)
    )


# funzione principale di valutazione

def run_evaluation(dataset, rag_endpoint, top_k, ollama_url, ollama_model,
                   progress_callback=None):
    """Esegue la valutazione RAG sul dataset. Ritorna lista di report per la UI."""

    total = len(dataset)
    memoria_conversazioni = {}

    # qui accumulo i dati: ui_reports per la pagina, ragas_samples per ragas
    ui_reports = []
    ragas_samples = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    if progress_callback:
        progress_callback("fetch", 0, total, "Fase 1/2: Interrogazione Endpoint RAG in corso...")

    for idx, item in enumerate(dataset):
        query_id = item['id']
        source = item['source']
        gold = item['gold_answer']
        domanda_pura = item["question"]

        conv_id = None
        if source == "MTRAG":
            # ricostruisco la storia per le conversazioni
            conv_id = query_id.split("_t")[0]
            storia_passata = memoria_conversazioni.get(conv_id, "")
            domanda_inviata = (
                f"[STORIA]\n{storia_passata}\n[NUOVA DOMANDA]\n{domanda_pura}"
                if storia_passata
                else domanda_pura
            )
        else:
            domanda_inviata = domanda_pura

        docs_contesto = split_context_in_docs(item.get("context", ""))
        rag_res = interroga_rag(rag_endpoint, domanda_inviata, top_k, docs_contesto)
        risposta_rag = rag_res.get("answer", str(rag_res))
        doc_recuperati = rag_res.get("retrieved_docs", rag_res.get("contexts", []))

        # converto i doc in stringhe semplici per ragas
        # NB: ragas vuole i nomi colonna ESATTI altrimenti crasha
        doc_strings = [
            d.get('text', d) if isinstance(d, dict) else str(d)
            for d in doc_recuperati
        ]
        if not doc_strings:
            doc_strings = ["Nessun documento recuperato."]

        testo_doc_recuperati = formatta_retrieved(doc_recuperati)

        if source == "MTRAG":
            memoria_conversazioni[conv_id] = (
                memoria_conversazioni.get(conv_id, "")
                + f"Utente: {domanda_pura}\nAssistente: {risposta_rag}\n"
            )

        semantic_sim = calcola_similarita_semantica(gold, risposta_rag)

        # dettagli per la UI
        rep = {
            "query_id": query_id,
            "source": source,
            "domanda": domanda_inviata,
            "risposta_rag": risposta_rag,
            "gold": gold,
            "docs": testo_doc_recuperati,
            "semantic_sim": semantic_sim,
        }
        ui_reports.append(rep)

        ragas_samples["user_input"].append(domanda_inviata)
        ragas_samples["response"].append(risposta_rag)
        ragas_samples["retrieved_contexts"].append(doc_strings)
        ragas_samples["reference"].append(gold)

        if progress_callback:
            progress_callback("fetch", idx + 1, total,
                              "Fase 1/2: Interrogazione Endpoint RAG in corso...")

    # fase 2: metriche ragas
    if progress_callback:
        progress_callback("ragas", 0, 1,
                          "Fase 2/2: Calcolo metriche in blocco tramite LLM (potrebbe richiedere tempo)...")

    hf_dataset = Dataset.from_dict(ragas_samples)

    # pulisco l'url nel caso l'utente abbia lasciato /api/generate
    base_url_pulito = ollama_url.split("/api")[0]
    ragas_llm = ChatOllama(model=ollama_model, base_url=base_url_pulito, temperature=0)
    ragas_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    try:
        # max_workers=1 sennò ollama va in timeout (provato con 2 e 4, crasha)
        run_config = RunConfig(max_workers=1, max_retries=10, timeout=180)
        risultati = evaluate(
            dataset=hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
            raise_exceptions=True,
        )
        df_ragas = risultati.to_pandas()

        # unisco i risultati ragas ai report
        for idx, rep in enumerate(ui_reports):
            row = df_ragas.iloc[idx]
            rep["retrieval_voto"] = float(row.get("context_precision", 0))
            rep["faithfulness_voto"] = float(row.get("faithfulness", 0))
            rep["correctness_voto"] = float(row.get("answer_correctness", 0))
            rep["relevancy_voto"] = float(row.get("answer_relevancy", 0))

    except Exception as e:
        # se ragas fallisce metto tutto a zero e rilancio l'errore
        for rep in ui_reports:
            rep["retrieval_voto"] = 0.0
            rep["faithfulness_voto"] = 0.0
            rep["correctness_voto"] = 0.0
            rep["relevancy_voto"] = 0.0
        raise RuntimeError(f"Errore durante l'esecuzione di Ragas: {e}") from e

    if progress_callback:
        progress_callback("done", 1, 1, "Valutazione completata.")

    return ui_reports
