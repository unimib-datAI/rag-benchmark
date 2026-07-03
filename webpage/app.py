import json
import os
import random
import re
import requests
import pandas as pd
import streamlit as st

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from html import unescape
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import dashboard as dash

st.set_page_config(
    page_title="SuperBenchmark · RAG Evaluation",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS caricato
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

def _clean_html_text(text: str) -> str:
    if not text or ("<" not in text and "&" not in text):
        return text
    # Rimuove script e style
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Rimuove tutti i tag HTML
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Decodifica entità HTML
    cleaned = unescape(cleaned)
    # Normalizza spazi bianchi
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _split_context_in_docs(context):
    # input in forma di lista
    if isinstance(context, list):
        docs = [str(c).strip() for c in context if str(c).strip()]
        return [_clean_html_text(doc) for doc in docs]
    text = str(context or "").strip()
    # input in forma di stringa
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Se non ci sono linee, restituisce il testo pulito come unico documento
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
    # Aggiunge l'ultimo documento se presente
    if current:
        docs.append(" ".join(current).strip())
    return [_clean_html_text(doc) for doc in docs]

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def calcola_similarita_semantica(gold, pred):
    if not gold.strip() and not pred.strip():
        return 1.0
    if not gold.strip() or not pred.strip():
        return 0.0
    model = get_embedding_model()
    embeddings = model.encode([gold, pred])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return max(0.0, float(sim))

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
    return "".join(
        f"[Doc {i}]\n"
        f"{d.get('text', d) if isinstance(d, dict) else str(d)}\n\n"
        for i, d in enumerate(docs, 1)
    )

def _value_color(v: float) -> str:
    if v >= 0.7: return "green"
    if v >= 0.4: return "yellow"
    return "red"

def _short_source(source: str) -> str:
    for key, short in BENCHMARK_SHORT.items():
        if key in source:
            return short
    return source


JUDGE_PROMPT = """Agisci come un Giudice di sistemi AI. Analizza questi elementi:
1. DOMANDA: {domanda}
2. DOCUMENTI RECUPERATI: {contesto}
3. RISPOSTA RAG: {risposta}
4. RISPOSTA CORRETTA (GOLD): {gold}

Valuta con un voto da 0.0 a 1.0 e spiega brevemente:
- retrieval_voto: i documenti forniti contengono le informazioni utili?
- faithfulness_voto: L'AI si è basata solo sui documenti?
- correctness_voto: La risposta dell'AI è corretta rispetto alla GOLD?

Rispondi ESCLUSIVAMENTE con un JSON:
{{
  "retrieval_voto": 0.0,
  "retrieval_perche": "...",
  "faithfulness_voto": 0.0,
  "faithfulness_perche": "...",
  "correctness_voto": 0.0,
  "correctness_perche": "..."
}}
"""

st.markdown(dash.get_header_html(), unsafe_allow_html=True)

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

    if not st.session_state.running:
        can_start = len(selected_files) > 0
        if st.button("Avvia valutazione", type="primary", disabled=not can_start, use_container_width=True):
            st.session_state.running = True
            st.session_state.results = []
            st.session_state.selected_files = selected_files
            if "llm_advice" in st.session_state:
                del st.session_state.llm_advice
            st.rerun()
    else:
        if st.button("Stop valutazione", type="secondary", use_container_width=True):
            st.session_state.running = False
            st.rerun()

if not st.session_state.running and not st.session_state.results:
    st.markdown(dash.get_welcome_html(), unsafe_allow_html=True)

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

        prompt_valutazione = JUDGE_PROMPT.format(domanda=domanda_inviata, contesto=testo_doc_recuperati, risposta=risposta_rag, gold=gold)
        giudizio_raw = invoca_ollama(prompt_valutazione, ollama_url, ollama_model)
        json_pulito = pulisci_json(giudizio_raw)

        try:
            voti = json.loads(json_pulito)
        except json.JSONDecodeError:
            voti = {"retrieval_voto": 0.0, "retrieval_perche": "Errore parsing", "faithfulness_voto": 0.0, "faithfulness_perche": "N/A", "correctness_voto": 0.0, "correctness_perche": "N/A"}

        semantic_sim = calcola_similarita_semantica(gold, risposta_rag)

        rep = {
            "query_id": query_id, "source": source, "domanda": domanda_inviata,
            "risposta_rag": risposta_rag, "gold": gold, "docs": testo_doc_recuperati,
            "retrieval_voto": float(voti.get("retrieval_voto", 0)),
            "faithfulness_voto": float(voti.get("faithfulness_voto", 0)),
            "correctness_voto": float(voti.get("correctness_voto", 0)),
            "semantic_sim": semantic_sim, "dettagli": voti
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
    avg_f1 = df['semantic_sim'].mean()

    # Metriche generali
    st.markdown('<div class="section-title">Panoramica</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in [(col1, "Correctness", avg_c), (col2, "Faithfulness", avg_f), (col3, "Retrieval", avg_r), (col4, "Semantic Sim", avg_f1)]:
        with col:
            st.markdown(dash.get_metric_card_html(label, val, _value_color(val)), unsafe_allow_html=True)

    sources = df['source'].unique()
    if len(sources) > 1:
        st.markdown('<div class="section-title">Per Benchmark</div>', unsafe_allow_html=True)
        cols = st.columns(len(sources))
        for i, src in enumerate(sources):
            src_df = df[df['source'] == src]
            sc = src_df['correctness_voto'].mean()
            with cols[i]:
                st.markdown(dash.get_breakdown_card_html(_short_source(src), sc, len(src_df), _value_color(sc)), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Risultati</div>', unsafe_allow_html=True)
    display_df = df[['query_id', 'source', 'retrieval_voto', 'faithfulness_voto', 'correctness_voto', 'semantic_sim']].copy()
    display_df.columns = ['Query', 'Benchmark', 'Retrieval', 'Faithfulness', 'Correctness', 'Semantic Sim']
    st.dataframe(
        display_df.style.format({'Retrieval': '{:.2f}', 'Faithfulness': '{:.2f}', 'Correctness': '{:.2f}', 'Semantic Sim': '{:.2f}'})
        .background_gradient(subset=['Retrieval', 'Faithfulness', 'Correctness', 'Semantic Sim'], cmap='RdYlGn', vmin=0, vmax=1),
        use_container_width=True, hide_index=True,
    )

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
                    st.markdown(dash.get_judge_box_html(metric_label, v, reason, _value_color(v)), unsafe_allow_html=True)

            st.markdown(dash.get_token_bar_html(rep['semantic_sim']), unsafe_allow_html=True)

            with st.expander("Documenti recuperati"):
                st.text(rep['docs'])

    st.markdown('<div class="section-title">Diagnostica e soluzioni</div>', unsafe_allow_html=True)
    
    worst_query = min(reports, key=lambda x: x['correctness_voto'] + x['retrieval_voto'], default=None)
    if worst_query and "llm_advice" not in st.session_state:
        with st.spinner("🧠 L'AI sta scrivendo il report diagnostico personalizzato..."):
            PROMPT_EXPERT = f"""Agisci come un Architetto AI esperto di RAG.
            Ho valutato il mio sistema RAG. Ecco le medie:
            - Retrieval: {avg_r:.2f}
            - Faithfulness: {avg_f:.2f}
            - Correctness: {avg_c:.2f}

            Ecco la query in cui il sistema ha performato peggio:
            DOMANDA: {worst_query['domanda']}
            RISPOSTA RAG: {worst_query['risposta_rag']}
            RISPOSTA IDEALE: {worst_query['gold']}

            Scrivi un'analisi di massimo 4 frasi evidenziando il problema principale e dando 2 consigli tecnici concreti. Scrivi in italiano usando la formattazione Markdown."""
            st.session_state.llm_advice = invoca_ollama(PROMPT_EXPERT, ollama_url, ollama_model)

    llm_advice_text = st.session_state.get("llm_advice", "")
    html_issues = dash.get_diagnostics_html(avg_r, avg_f, avg_c, avg_f1)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown(html_issues, unsafe_allow_html=True)

    with col_d2:
        st.markdown('<div class="diag-panel solutions" style="margin-bottom: 1rem;"><h3>🤖 Il Verdetto dell\'Esperto</h3></div>', unsafe_allow_html=True)
        if llm_advice_text:
            st.success(llm_advice_text)
        else:
            st.warning("Verdetto LLM non disponibile.")

    # Footer
    st.markdown(dash.get_footer_html(len(reports), ollama_model, datetime.now().strftime('%d/%m/%Y %H:%M')), unsafe_allow_html=True)
