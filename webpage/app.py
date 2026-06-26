import json
import os
import random
import re
import requests
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from html import unescape

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SuperBenchmark · RAG Evaluation",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load external CSS
_CSS_PATH = Path(__file__).resolve().parent / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.error("⚠️ style.css non trovato!")

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "Run"

BENCHMARK_MAPPING = {
    "LegalBench-RAG (Precisione Contratti)": "Tutto_LEGALBENCH.json",
    "MultiHop-RAG (Sintesi Multi-Documento)": "Tutto_MULTIHOP.json",
    "CRAG (Parsing HTML & Trabocchetti)": "Tutto_CRAG_DAVE.json",
    "MTRAG (Conversazioni Multi-Turno)": "Tutto_MTRAG_DAVE.json",
    "OHRBench (Rumore OCR / Scansioni)": "Tutto_OHRBench_DAVE.json"
}

BENCHMARK_SHORT = {
    "LegalBench-RAG": "LegalBench",
    "MultiHop-RAG": "MultiHop",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "OHRBench": "OHRBench",
}

# Funzione per tokenizzare il testo in una lista di token minuscoli, rimuovendo la punteggiatura.
def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())

# Funzione per pulire il testo HTML, rimuovendo tag e decodificando entità.
def _clean_html_text(text: str) -> str:
    if not text or ("<" not in text and "&" not in text):
        return text
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _split_context_in_docs(context):
    if isinstance(context, list):
        docs = [str(c).strip() for c in context if str(c).strip()]
        return [_clean_html_text(doc) for doc in docs]
    text = str(context or "").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [_clean_html_text(text)]
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
    return [_clean_html_text(doc) for doc in docs]

# 
def precision_recall_f1(gold, pred):
    gold_tokens = _tokenize(gold)
    pred_tokens = _tokenize(pred)
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    cg, cp = Counter(gold_tokens), Counter(pred_tokens)
    common = sum((cg & cp).values())
    precision = common / sum(cp.values()) if sum(cp.values()) > 0 else 0.0
    recall = common / sum(cg.values()) if sum(cg.values()) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def pulisci_json(testo_grezzo: str) -> str:
    match = re.search(r'\{.*\}', testo_grezzo, re.DOTALL)
    return match.group(0) if match else testo_grezzo

def invoca_ollama(prompt: str, url: str, model: str) -> str:
    try:
        r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False, "temperature": 0}, timeout=120)
    except requests.RequestException as exc:
        st.error(f"Richiesta fallita verso Ollama: {exc}")
        return ""
    if r.status_code != 200:
        st.error(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        return ""
    payload = r.json()
    return payload.get("response") or ""

def interroga_rag(endpoint: str, query: str, top_k: int, documents: list[str]) -> dict:
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
    return "".join(f"[Doc {i}]\n{d.get('text', d) if isinstance(d, dict) else str(d)}\n\n" for i, d in enumerate(docs, 1))

def _value_color(v: float) -> str:
    if v >= 0.7: return "green"
    if v >= 0.4: return "yellow"
    return "red"

def _short_source(source: str) -> str:
    for key, short in BENCHMARK_SHORT.items():
        if key in source:
            return short
    return source


PROMPT_JUDGE = """Agisci come un Giudice di sistemi AI. Analizza questi elementi:
1. DOMANDA (inclusa eventuale storia pregressa): {domanda}
2. DOCUMENTI RECUPERATI: {contesto}
3. RISPOSTA RAG: {risposta}
4. RISPOSTA CORRETTA (GOLD): {gold}

Valuta con un voto da 0.0 a 1.0 (usa i decimali, es. 0.8) e spiega brevemente:
- retrieval_voto: i documenti forniti contengono le informazioni utili per rispondere?
- faithfulness_voto: L'AI si è basata SOLO sui documenti o ha inventato dati?
- correctness_voto: La risposta dell'AI è corretta rispetto alla GOLD?

REGOLE TASSATIVE: Restituisci ESCLUSIVAMENTE un oggetto JSON valido. Niente testo prima o dopo.
{{
  "retrieval_voto": 0.0,
  "retrieval_perche": "...",
  "faithfulness_voto": 0.0,
  "faithfulness_perche": "...",
  "correctness_voto": 0.0,
  "correctness_perche": "..."
}}
"""

# Testo di benvenuto e intestazione
st.markdown("""
<div class="header-bar">
    <h1>SuperBenchmark Dashboard</h1>
    <p>Valutazione interattiva di sistemi RAG con LLM-as-a-Judge</p>
</div>
""", unsafe_allow_html=True)

# Stato
if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = []

# Sidebar
with st.sidebar:
    st.markdown("### Connessione")
    st.markdown('<p class="sidebar-label">Endpoint RAG</p>', unsafe_allow_html=True)
    rag_endpoint = st.text_input("endpoint_rag", "http://127.0.0.1:8000/ask", label_visibility="collapsed")
    
    st.markdown('<p class="sidebar-label">Top-K documenti</p>', unsafe_allow_html=True)
    top_k = st.slider("top_k", min_value=1, max_value=10, value=3, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Giudice LLM")
    st.markdown('<p class="sidebar-label">URL Ollama</p>', unsafe_allow_html=True)
    ollama_url = st.text_input("ollama_url", "http://localhost:11434/api/generate", label_visibility="collapsed")
    
    st.markdown('<p class="sidebar-label">Modello</p>', unsafe_allow_html=True)
    ollama_model = st.selectbox("model_select", options=["gemma4:31b-cloud", "qwen3.5:cloud"], index=0, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Benchmark")
    selected_names = st.multiselect(
        "bench_select",
        options=list(BENCHMARK_MAPPING.keys()),
        default=list(BENCHMARK_MAPPING.keys()),
        label_visibility="collapsed",
    )
    selected_files = [BENCHMARK_MAPPING[name] for name in selected_names]
    st.caption(f"{len(selected_names)} di {len(BENCHMARK_MAPPING)} selezionati")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    can_start = not st.session_state.running and len(selected_files) > 0
    if st.button("Avvia valutazione", type="primary", disabled=not can_start, use_container_width=True):
        st.session_state.running = True
        st.session_state.results = []
        st.session_state.selected_files = selected_files
        st.rerun()

# Schermata principale
if not st.session_state.running and not st.session_state.results:
    st.markdown("""
    <div class="welcome-box">
        <h3>Nessuna valutazione eseguita</h3>
        <p>Configura i parametri nella barra laterale e premi "Avvia valutazione" per iniziare.</p>
    </div>
    """, unsafe_allow_html=True)

# Valutazione
if st.session_state.running:
    st.markdown('<div class="section-title">Valutazione in corso</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    status_text = st.empty()

    dataset = []
    files_to_run = st.session_state.get("selected_files", list(BENCHMARK_MAPPING.values()))
    for fn in files_to_run:
        path = RUN_DIR / fn
        if path.exists():
            with open(path, 'r', encoding='utf-8-sig') as f:
                items = json.load(f)
                if "MTRAG" in fn:
                    conversations = defaultdict(list)
                    for item in items:
                        conv_id = item['id'].split('_t')[0]
                        conversations[conv_id].append(item)
                    chosen_conv = random.choice(list(conversations.keys()))
                    sorted_turns = sorted(conversations[chosen_conv], key=lambda x: x['id'])
                    dataset.extend(sorted_turns[:6])
                else:
                    sample_size = min(3, len(items))
                    dataset.extend(random.sample(items, sample_size))

    total = len(dataset)
    memoria_conversazioni = {}
    reports = []

    for idx, item in enumerate(dataset):
        query_id = item['id']
        source = item['source']
        gold = item['gold_answer']
        domanda_pura = item["question"]

        status_text.markdown(f"`{idx+1}/{total}` — **{query_id}** ({source})")

        conv_id = None
        if source == "MTRAG":
            conv_id = query_id.split("_t")[0]
            storia_passata = memoria_conversazioni.get(conv_id, "")
            domanda_inviata = f"[STORIA]\n{storia_passata}\n[NUOVA DOMANDA]\n{domanda_pura}" if storia_passata else domanda_pura
        else:
            domanda_inviata = domanda_pura

        docs_contesto = _split_context_in_docs(item.get("context", ""))
        rag_res = interroga_rag(rag_endpoint, domanda_inviata, top_k, docs_contesto)
        risposta_rag = rag_res.get("answer", str(rag_res))
        doc_recuperati = rag_res.get("retrieved_docs", rag_res.get("contexts", []))
        testo_doc_recuperati = formatta_retrieved(doc_recuperati)

        if source == "MTRAG":
            memoria_conversazioni[conv_id] = memoria_conversazioni.get(conv_id, "") + f"Utente: {domanda_pura}\nAssistente: {risposta_rag}\n"

        prompt_valutazione = PROMPT_JUDGE.format(domanda=domanda_inviata, contesto=testo_doc_recuperati, risposta=risposta_rag, gold=gold)
        giudizio_raw = invoca_ollama(prompt_valutazione, ollama_url, ollama_model)
        json_pulito = pulisci_json(giudizio_raw)

        try:
            voti = json.loads(json_pulito)
        except json.JSONDecodeError:
            voti = {"retrieval_voto": 0.0, "retrieval_perche": "Errore parsing", "faithfulness_voto": 0.0, "faithfulness_perche": "N/A", "correctness_voto": 0.0, "correctness_perche": "N/A"}

        precision, recall, f1 = precision_recall_f1(gold, risposta_rag)

        rep = {
            "query_id": query_id, "source": source, "domanda": domanda_inviata,
            "risposta_rag": risposta_rag, "gold": gold, "docs": testo_doc_recuperati,
            "retrieval_voto": float(voti.get("retrieval_voto", 0)),
            "faithfulness_voto": float(voti.get("faithfulness_voto", 0)),
            "correctness_voto": float(voti.get("correctness_voto", 0)),
            "precision": precision, "recall": recall, "f1": f1, "dettagli": voti
        }
        reports.append(rep)
        st.session_state.results = reports
        progress_bar.progress((idx + 1) / total)

    st.session_state.running = False
    status_text.empty()
    st.rerun()

# Risultati
if st.session_state.results:
    reports = st.session_state.results
    df = pd.DataFrame(reports)

    avg_c = df['correctness_voto'].mean()
    avg_f = df['faithfulness_voto'].mean()
    avg_r = df['retrieval_voto'].mean()
    avg_f1 = df['f1'].mean()

    # Metriche generali
    st.markdown('<div class="section-title">Panoramica</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in [(col1, "Correctness", avg_c), (col2, "Faithfulness", avg_f), (col3, "Retrieval", avg_r), (col4, "F1-Score", avg_f1)]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {_value_color(val)}">{val:.2f}</div>
            </div>""", unsafe_allow_html=True)

    sources = df['source'].unique()
    if len(sources) > 1:
        st.markdown('<div class="section-title">Per Benchmark</div>', unsafe_allow_html=True)
        cols = st.columns(len(sources))
        for i, src in enumerate(sources):
            src_df = df[df['source'] == src]
            sc = src_df['correctness_voto'].mean()
            with cols[i]:
                st.markdown(f"""
                <div class="breakdown-card">
                    <div class="src-name">{_short_source(src)}</div>
                    <div class="src-score c-{_value_color(sc)}">{sc:.2f}</div>
                    <div class="src-count">{len(src_df)} query</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risultati</div>', unsafe_allow_html=True)
    display_df = df[['query_id', 'source', 'retrieval_voto', 'faithfulness_voto', 'correctness_voto', 'f1']].copy()
    display_df.columns = ['Query', 'Benchmark', 'Retrieval', 'Faithfulness', 'Correctness', 'F1']
    st.dataframe(
        display_df.style.format({'Retrieval': '{:.2f}', 'Faithfulness': '{:.2f}', 'Correctness': '{:.2f}', 'F1': '{:.2f}'})
        .background_gradient(subset=['Retrieval', 'Faithfulness', 'Correctness', 'F1'], cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True, hide_index=True,
    )

    # Dettaglio query
    st.markdown('<div class="section-title">Dettaglio Query</div>', unsafe_allow_html=True)
    for rep in reports:
        corr = rep['correctness_voto']
        label = f"{rep['query_id']}  |  {rep['source']}  |  Correctness: {corr}"
        with st.expander(label):
            st.markdown("**Domanda**")
            st.code(rep['domanda'], language=None)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Risposta RAG**")
                st.info(rep['risposta_rag'])
            with c2:
                st.markdown("**Risposta Gold**")
                st.success(rep['gold'])

            st.markdown("---")
            st.markdown("**Giudizio LLM**")
            m1, m2, m3 = st.columns(3)
            for col, metric_key, metric_label in [(m1, 'retrieval', 'Retrieval'), (m2, 'faithfulness', 'Faithfulness'), (m3, 'correctness', 'Correctness')]:
                v = rep[f'{metric_key}_voto']
                reason = rep['dettagli'].get(f'{metric_key}_perche', '')
                with col:
                    st.markdown(f"""
                    <div class="judge-box">
                        <div class="jb-label">{metric_label}</div>
                        <div class="jb-value c-{_value_color(v)}">{v}</div>
                        <div class="jb-reason">{reason}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="token-bar">
                <b>Token:</b> Precision: <code>{rep['precision']:.2f}</code> · Recall: <code>{rep['recall']:.2f}</code> · F1: <code>{rep['f1']:.2f}</code>
            </div>""", unsafe_allow_html=True)

            with st.expander("Documenti recuperati"):
                st.text(rep['docs'])

    # Diagnostica
    st.markdown('<div class="section-title">Diagnostica e soluzioni</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        html = '<div class="diag-panel issues"><h3>Problematiche rilevate</h3>'
        issues = []
        if avg_r < 0.6:
            issues.append("<b>Retrieval debole (< 0.60):</b> I documenti recuperati non contengono le informazioni necessarie.")
        if avg_f < 0.7:
            issues.append("<b>Allucinazioni (Faithfulness < 0.70):</b> Il modello genera informazioni non presenti nel contesto fornito.")
        if avg_c < 0.5:
            issues.append("<b>Bassa correttezza (< 0.50):</b> Le risposte non corrispondono alla Ground Truth.")
        if avg_f1 < 0.3:
            issues.append("<b>Sovra-generazione (F1 < 0.30):</b> Le risposte sono troppo verbose o contengono testo grezzo.")
        if not issues:
            html += '<div class="diag-ok">Nessun problema critico rilevato.</div>'
        else:
            for iss in issues:
                html += f'<p>{iss}</p>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    with col_d2:
        html = '<div class="diag-panel solutions"><h3>Possibili soluzioni</h3>'
        solutions = []
        if avg_r < 0.6:
            solutions.append("<b>Chunking:</b> Sperimenta con chunk più piccoli o sovrapposti per catturare meglio le frasi chiave.")
            solutions.append("<b>Retriever ibrido:</b> Combina BM25 con embeddings densi (FAISS + sentence-transformers).")
            solutions.append("<b>Reranking:</b> Aggiungi un modello di re-ranking dopo il primo retrieval.")
        if avg_f < 0.7:
            solutions.append("<b>System prompt:</b> Aggiungi istruzioni esplicite per limitare le risposte al solo contesto fornito.")
            solutions.append("<b>Temperature 0.0:</b> Riduci la creatività per forzare risposte deterministiche.")
        if avg_c < 0.5 or avg_f1 < 0.3:
            solutions.append("<b>Risposte concise:</b> Istruisci l'LLM a rispondere in massimo 2 frasi estraendo solo l'informazione esatta.")
            solutions.append("<b>Pulizia contesto:</b> Implementa filtri per HTML/OCR noise prima della generazione.")
        if not solutions:
            html += '<div class="diag-ok">Prestazioni stabili.</div>'
        else:
            for sol in solutions:
                html += f'<p>{sol}</p>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="footer-bar">
        SuperBenchmark  ·  {len(reports)} query  ·  Giudice: {ollama_model}  ·  {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </div>""", unsafe_allow_html=True)
