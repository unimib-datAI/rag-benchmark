import json
import random
import re
import requests
import pandas as pd
import streamlit as st

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from html import unescape

from evaluator import run_evaluation
from persistence import save_evaluation, list_evaluations, load_evaluation, delete_evaluation

import dashboard as dash

st.set_page_config(
    page_title="SuperBenchmark · RAG Evaluation",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# carico il css custom
_CSS_PATH = Path(__file__).resolve().parent / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.error("⚠️ style.css non trovato!")

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "Run"

# mapping nomi benchmark -> file json nella cartella Run
BENCHMARK_MAPPING = {
    "LegalBench-RAG (Precisione Contratti)": "Tutto_LEGALBENCH.json",
    "MultiHop-RAG (Sintesi Multi-Documento)": "Tutto_MULTIHOP.json",
    "CRAG (Parsing HTML & Trabocchetti)": "Tutto_CRAG_DAVE.json",
    "MTRAG (Conversazioni Multi-Turno)": "Tutto_MTRAG_DAVE.json",
    "OHRBench (Rumore OCR / Scansioni)": "Tutto_OHRBench_DAVE.json",
    "SQuAD-it (Italian Q&A su Wikipedia)": "Tutto_SQUAD.json"
}

BENCHMARK_SHORT = {
    "LegalBench-RAG": "LegalBench",
    "MultiHop-RAG": "MultiHop",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "OHRBench": "OHRBench",
    "SQuAD-it": "SQuAD",
}

def invoca_ollama(prompt: str, url: str, model: str) -> str:
    """Chiama ollama e ritorna la risposta come stringa."""
    try:
        # Uso direttamente il secondo account per le diagnostiche per sicurezza
        headers = {"Authorization": "Bearer 130e9216867047ceaccab3e8017c5123.joL1WY4giNPRo76atH5cJJOk"}
        r = requests.post(url, json={"model": model, "prompt": prompt, "stream": False, "temperature": 0}, headers=headers, timeout=120)
    except requests.RequestException as exc:
        st.error(f"Richiesta fallita verso Ollama: {exc}")
        return ""
    if r.status_code != 200:
        st.error(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
        return ""
    payload = r.json()
    return payload.get("response") or ""

# colore in base al voto (verde/giallo/rosso)
def _value_color(v: float) -> str:
    if pd.isna(v): return "red"
    if v >= 0.7: return "green"
    if v >= 0.4: return "yellow"
    return "red"

def _short_source(source: str) -> str:
    # prendo il nome corto dal dict, se c'è
    for key, short in BENCHMARK_SHORT.items():
        if key in source:
            return short
    return source
# banner della pagina
st.markdown(dash.get_header_html(), unsafe_allow_html=True)

if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = []

# sidebar
with st.sidebar:
    st.header("⚙️ Impostazioni Motore RAG")
    motore_rag = st.radio("Seleziona il Motore di Ricerca & LLM", ["Local RAG (BM25)", "DAVE (Elasticsearch)"])

    st.markdown('<p class="sidebar-label">Base URL RAG</p>', unsafe_allow_html=True)
    rag_base_url = st.text_input("rag_base_url", "http://127.0.0.1:8000", label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Soglia Similarità (Threshold)</p>', unsafe_allow_html=True)
    threshold = st.slider("threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Giudice LLM")
    st.markdown('<p class="sidebar-label">URL Ollama Base</p>', unsafe_allow_html=True)
    ollama_url = st.text_input("ollama_url", "http://localhost:11434", label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Modello</p>', unsafe_allow_html=True)
    ollama_model = st.selectbox("model_select", options=["gemma4:31b-cloud"], index=0, label_visibility="collapsed")

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

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Query per Benchmark</p>', unsafe_allow_html=True)
    sample_size_param = st.slider("sample_size_param", min_value=1, max_value=50, value=3, step=1, label_visibility="collapsed")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Modalità Selezione</p>', unsafe_allow_html=True)
    query_selection_mode = st.radio("query_selection_mode", options=["Casuali", "Fisse"], horizontal=True, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # bottone avvia/stop
    if not st.session_state.running:
        can_start = len(selected_files) > 0
        if st.button("Avvia valutazione", type="primary", disabled=not can_start, width='stretch'):
            st.session_state.running = True
            st.session_state.results = []
            st.session_state.selected_files = selected_files
            st.session_state.sample_size_param = sample_size_param
            st.session_state.query_selection_mode = query_selection_mode
            if "llm_advice" in st.session_state:
                del st.session_state.llm_advice
            st.rerun()
    else:
        if st.button("Stop valutazione", type="secondary", width='stretch'):
            st.session_state.running = False
            st.rerun()

# se non ci sono risultati correnti, viene caricata l'ultima valutazione salvata
if not st.session_state.running and not st.session_state.results:
    valutazione_salvate = list_evaluations()
    if valutazione_salvate:
        ultima = valutazione_salvate[0]
        valutazioni_complete = load_evaluation(ultima["filepath"])
        st.session_state.results = valutazioni_complete.get("results", [])

# tabs principali
tab_valutazione, tab_explore = st.tabs(["🔬 Valutazione", "📊 Explore"])

with tab_valutazione:
    if not st.session_state.running and not st.session_state.results:
        st.markdown(dash.get_welcome_html(), unsafe_allow_html=True)

    # --- valutazione in corso ---
    if st.session_state.running:
        st.markdown('<div class="section-title">Valutazione in corso</div>', unsafe_allow_html=True)
        barra_progressiva = st.progress(0)
        status_text = st.empty()

        # vengono caricati i dataset dai file selezionati nella sidebar
        insieme_dati = []
        files_to_run = st.session_state.get("selected_files", list(BENCHMARK_MAPPING.values()))
        for nome_file in files_to_run:
            path = RUN_DIR / nome_file
            if path.exists():
                with open(path, 'r', encoding='utf-8-sig') as f:
                    elementi = json.load(f)
                    # caso particolare MTRAG: prendere solo un certo numero di conversazioni multi-turno, non query singole
                    if "MTRAG" in nome_file:
                        conversazioni = defaultdict(list)
                        for elemento in elementi:
                            id_conv = elemento['id'].split('_t')[0]
                            conversazioni[id_conv].append(elemento)

                        turni_necessari = st.session_state.get("sample_size_param", 3)
                        chiavi_conv = list(conversazioni.keys())
                        if st.session_state.get("query_selection_mode", "Casuali") == "Casuali":
                            random.shuffle(chiavi_conv)
                        else:
                            chiavi_conv.sort()

                        turni_aggiunti = 0
                        for chiave_conv in chiavi_conv:
                            if turni_aggiunti >= turni_necessari: break
                            for turno in sorted(conversazioni[chiave_conv], key=lambda x: x['id']):
                                if turni_aggiunti < turni_necessari:
                                    insieme_dati.append(turno)
                                    turni_aggiunti += 1
                    else:
                        dimensione = min(st.session_state.get("sample_size_param", 3), len(elementi))
                        if st.session_state.get("query_selection_mode", "Casuali") == "Casuali":
                            insieme_dati.extend(random.sample(elementi, dimensione))
                        else:
                            insieme_dati.extend(elementi[:dimensione])

        tot = len(insieme_dati)

        # callback per aggiornare la progress bar di streamlit
        def _streamlit_progress(phase, current, total_items, message):
            if phase == "ingest":
                barra_progressiva.progress(0.05 * (current / max(total_items, 1)))
                status_text.markdown(f"**📥 {message}**")
            elif phase == "fetch":
                barra_progressiva.progress(0.05 + (current / total_items) * 0.45 if total_items else 0.05)
                status_text.markdown(f"**{message}**")
            elif phase == "ragas":
                status_text.markdown(f"**{message}**")
            elif phase == "done":
                barra_progressiva.progress(1.0)

        with st.spinner("Valutazione LLM Ragas in corso... attendi la risposta dal modello"):
            try:
                ui_reports = run_evaluation(
                    dataset=insieme_dati,
                    rag_base_url=rag_base_url,
                    threshold=threshold,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    progress_callback=_streamlit_progress,
                    motore_rag=motore_rag,
                    benchmark_files=files_to_run
                )
            # se c'è un errore, mostro il messaggio e azzero i risultati
            except RuntimeError as e:
                st.error(str(e))
                ui_reports = []

        barra_progressiva.progress(1.0)
        st.session_state.results = ui_reports

        # salvo la valutazione automaticamente
        saved = save_evaluation(
            results=ui_reports,
            model=ollama_model,
            endpoint=rag_base_url,
            threshold=threshold,
            sample_size=st.session_state.get('sample_size_param', 3),
            benchmarks_used=files_to_run,
        )
        st.session_state.last_saved_id = saved['id']

        st.session_state.running = False
        status_text.empty()
        st.rerun()

    # mostra risultati
    if st.session_state.results:
        reports = st.session_state.results
        df = pd.DataFrame(reports)

        # medie delle metriche principali
        avg_c = df['correctness_voto'].mean()
        avg_f = df['faithfulness_voto'].mean()
        avg_r = df['retrieval_voto'].mean()
        avg_recall = df['recall_voto'].mean() if 'recall_voto' in df.columns else 0.0
        avg_f1 = df['semantic_sim'].mean()

        # panoramica metriche
        st.markdown('<div class="section-title">Panoramica (Ragas Metrics)</div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        for col, label, val in [(col1, "Answer Correctness", avg_c), (col2, "Faithfulness", avg_f), (col3, "Context Precision", avg_r), (col4, "Context Recall", avg_recall), (col5, "Semantic Sim", avg_f1)]:
            with col:
                st.markdown(dash.get_metric_card_html(label, val, _value_color(val)), unsafe_allow_html=True)

        # Avviso su query fallite e contesti troncati
        ragas_stats = reports[0].get("_ragas_stats", {}) if reports else {}
        query_fallite = sum(1 for r in reports if r.get("ragas_fallita", False))
        contesti_troncati = ragas_stats.get("contesti_troncati", 0)
        if query_fallite > 0 or contesti_troncati > 0:
            parti_msg = []
            if contesti_troncati > 0:
                parti_msg.append(f"**{contesti_troncati}** contesti troncati (superavano il limite del modello)")
            if query_fallite > 0:
                parti_msg.append(f"**{query_fallite}/{len(reports)}** query con metriche RAGAS fallite (impostate a 0.0)")
            st.warning("⚠️ " + " · ".join(parti_msg) + ". Le medie potrebbero non riflettere tutte le query.")

        # se ci sono piu benchmark, mostro il breakdown
        sources = df['source'].unique()
        if len(sources) > 1:
            st.markdown('<div class="section-title">Per Benchmark</div>', unsafe_allow_html=True)
            cols = st.columns(len(sources))
            for i, src in enumerate(sources):
                src_df = df[df['source'] == src]
                sc = src_df['correctness_voto'].mean()
                with cols[i]:
                    st.markdown(dash.get_breakdown_card_html(_short_source(src), sc, len(src_df), _value_color(sc)), unsafe_allow_html=True)

        # tabella risultati
        st.markdown('<div class="section-title">Risultati</div>', unsafe_allow_html=True)
        display_df = df[['query_id', 'source', 'retrieval_voto', 'recall_voto', 'faithfulness_voto', 'correctness_voto', 'relevancy_voto']].copy() if 'recall_voto' in df.columns else df[['query_id', 'source', 'retrieval_voto', 'faithfulness_voto', 'correctness_voto', 'relevancy_voto']].copy()
        if 'recall_voto' in df.columns:
            display_df.columns = ['Query', 'Benchmark', 'C. Precision', 'C. Recall', 'Faithfulness', 'Correctness', 'Relevancy']
            st.dataframe(
                display_df.style.format({'C. Precision': '{:.2f}', 'C. Recall': '{:.2f}', 'Faithfulness': '{:.2f}', 'Correctness': '{:.2f}', 'Relevancy': '{:.2f}'})
                .background_gradient(subset=['C. Precision', 'C. Recall', 'Faithfulness', 'Correctness', 'Relevancy'], cmap='RdYlGn', vmin=0, vmax=1),
                width='stretch', hide_index=True,
            )
        else:
            display_df.columns = ['Query', 'Benchmark', 'C. Precision', 'Faithfulness', 'Correctness', 'Relevancy']
            st.dataframe(
                display_df.style.format({'C. Precision': '{:.2f}', 'Faithfulness': '{:.2f}', 'Correctness': '{:.2f}', 'Relevancy': '{:.2f}'})
                .background_gradient(subset=['C. Precision', 'Faithfulness', 'Correctness', 'Relevancy'], cmap='RdYlGn', vmin=0, vmax=1),
                width='stretch', hide_index=True,
            )

        # dettaglio singole query con expander
        st.markdown('<div class="section-title">Dettaglio Query</div>', unsafe_allow_html=True)
        for rep in reports:
            corr = rep.get('correctness_voto', 0)
            label = f"{rep['query_id']}  |  {rep['source']}  |  Correctness: {corr:.2f}"
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
                st.markdown("**Metriche Ragas (LLM)**")
                m1, m2, m3, m4, m5 = st.columns(5)
                for col, metric_key, metric_label in [(m1, 'retrieval', 'Context Precis.'), (m2, 'recall', 'Context Recall'), (m3, 'faithfulness', 'Faithfulness'), (m4, 'correctness', 'Correctness'), (m5, 'relevancy', 'Relevancy')]:
                    raw_v = rep.get(f'{metric_key}_voto', 0)
                    v = round(float(raw_v), 3) if pd.notna(raw_v) else 0.0
                    reason = ""  # ragas restituisce solo punteggi, non i motivi purtroppo
                    with col:
                        st.markdown(dash.get_judge_box_html(metric_label, v, reason, _value_color(v)), unsafe_allow_html=True)

                st.markdown(dash.get_token_bar_html(rep['semantic_sim']), unsafe_allow_html=True)

                with st.expander("Documenti recuperati"):
                    st.text(rep['docs'])

        # diagnostica
        st.markdown('<div class="section-title">Diagnostica e soluzioni</div>', unsafe_allow_html=True)

        worst_query = min(reports, key=lambda x: x['correctness_voto'] + x['retrieval_voto'], default=None)
        if worst_query and "llm_advice" not in st.session_state:
            with st.spinner("🧠 L'AI sta scrivendo il report diagnostico personalizzato..."):
                # costruisco il prompt per l'esperto
                PROMPT_EXPERT = f"""Agisci come un Architetto AI esperto di RAG.
            Ho valutato il mio sistema RAG. Ecco le medie:
            - Context Precision: {avg_r:.2f}
            - Faithfulness: {avg_f:.2f}
            - Answer Correctness: {avg_c:.2f}

            Ecco la query in cui il sistema ha performato peggio:
            DOMANDA: {worst_query['domanda']}
            RISPOSTA RAG: {worst_query['risposta_rag']}
            RISPOSTA IDEALE: {worst_query['gold']}

            Scrivi un'analisi di massimo 4 frasi evidenziando il problema principale e dando 2 consigli tecnici concreti. Scrivi in italiano usando la formattazione Markdown."""

                url_generazione = ollama_url.split("/api")[0] + "/api/generate"
                st.session_state.llm_advice = invoca_ollama(PROMPT_EXPERT, url_generazione, ollama_model)

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

        # footer con info sulla valutazione
        st.markdown(dash.get_footer_html(len(reports), ollama_model, datetime.now().strftime('%d/%m/%Y %H:%M')), unsafe_allow_html=True)

# tab explore - storico valutazioni e confronto
with tab_explore:
    import plotly.graph_objects as go

    if "explore_msg" in st.session_state:
        st.success(st.session_state.explore_msg)
        del st.session_state.explore_msg

    val_salvate = list_evaluations()

    # lista valutazioni salvate
    st.markdown('<div class="section-title">Lista Valutazioni Salvate</div>', unsafe_allow_html=True)

    if not val_salvate:
        st.markdown(
            '<div class="welcome-box">'
            '<h3>Nessuna valutazione salvata</h3>'
            '<p>Esegui una valutazione dalla scheda "Valutazione" per vederla qui.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # checkbox per selezionare le val da confrontare
        sel_to_confront: list[dict] = []

        for idx, ev in enumerate(val_salvate):
            ts = datetime.fromisoformat(ev["timestamp"]).strftime("%d/%m/%Y %H:%M")
            avg_corr = ev.get("aggregate", {}).get("answer_correctness", 0.0)
            color = _value_color(avg_corr)
            color_hex = {"green": "#00E676", "yellow": "#FFC400", "red": "#FF1744"}.get(color, "#a2a2c0")

            # html della card per ogni valutazione
            card_html = (
                f'<div class="eval-card">'
                f'  <div class="eval-card-header">'
                f'    <span class="eval-card-name">{ev["name"]}</span>'
                f'    <span class="eval-card-ts">{ts}</span>'
                f'  </div>'
                f'  <div class="eval-card-body">'
                f'    <span class="eval-card-detail">Modello: <strong>{ev["model"]}</strong></span>'
                f'    <span class="eval-card-detail">Query: <strong style="font-family:JetBrains Mono,monospace">{ev["num_queries"]}</strong></span>'
                f'    <span class="eval-card-detail">Correctness: <strong style="color:{color_hex};font-family:JetBrains Mono,monospace">{avg_corr:.2f}</strong></span>'
                f'  </div>'
                f'</div>'
            )

            col_chk, col_card, col_load, col_del = st.columns([0.5, 8, 1.5, 1])

            with col_chk:
                checked = st.checkbox(" ", key=f"explore_cmp_{idx}", label_visibility="collapsed")
                if checked:
                    sel_to_confront.append(ev)

            with col_card:
                st.markdown(card_html, unsafe_allow_html=True)

            with col_load:
                if st.button("📂 Apri", key=f"explore_load_{idx}", width='stretch'):
                    full_eval = load_evaluation(ev["filepath"])
                    st.session_state.results = full_eval.get("results", [])
                    st.session_state.explore_msg = f"✅ Dati di '{ev['name']}' caricati! Spostati nella scheda '🔬 Valutazione' in alto per visualizzarli."
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"explore_del_{idx}", width='stretch'):
                    delete_evaluation(ev["filepath"])
                    st.rerun()

        # confronto valutazioni
        st.markdown('<div class="section-title">Confronto Valutazioni</div>', unsafe_allow_html=True)

        if len(sel_to_confront) == 0:
            st.info("Seleziona una o più valutazioni per confrontarle.")
        else:
            metric_keys = ["answer_correctness", "faithfulness", "context_precision", "context_recall", "semantic_sim"]
            metric_labels = ["Answer Correctness", "Faithfulness", "Context Precision", "Context Recall", "Semantic Sim"]

            # Tabella riassuntiva dei risultati
            st.markdown('<div class="section-title">Tabella Risultati Metriche</div>', unsafe_allow_html=True)
            table_agg_rows = []
            for ev in sel_to_confront:
                agg = ev.get("aggregate", {})
                row = {"Valutazione": ev["name"]}
                for k, label in zip(metric_keys, metric_labels):
                    row[label] = round(agg.get(k, 0.0), 2)
                table_agg_rows.append(row)
            
            st.dataframe(pd.DataFrame(table_agg_rows), use_container_width=True)

            # Tabella di dettaglio delle singole query a confronto
            st.markdown('<div class="section-title">Dettaglio Singole Query a Confronto</div>', unsafe_allow_html=True)
            query_map = {}
            for ev in sel_to_confront:
                full_eval = load_evaluation(ev["filepath"])
                eval_name = ev["name"]
                for res in full_eval.get("results", []):
                    qid = res.get("query_id", "")
                    if qid not in query_map:
                        query_map[qid] = {
                            ("Info", "Query ID"): qid,
                            ("Info", "Benchmark"): res.get("source", ""),
                            ("Info", "Domanda"): res.get("domanda", ""),
                        }
                    query_map[qid][("Answer Correctness", eval_name)] = round(res.get("correctness_voto", 0.0), 2)
                    query_map[qid][("Faithfulness", eval_name)] = round(res.get("faithfulness_voto", 0.0), 2)
                    query_map[qid][("Context Precision", eval_name)] = round(res.get("retrieval_voto", 0.0), 2)
                    query_map[qid][("Context Recall", eval_name)] = round(res.get("recall_voto", 0.0), 2)
                    query_map[qid][("Semantic Sim", eval_name)] = round(res.get("semantic_sim", 0.0), 2)
            
            if query_map:
                df_compare = pd.DataFrame(list(query_map.values()))
                df_compare.columns = pd.MultiIndex.from_tuples(df_compare.columns)
                st.dataframe(df_compare, use_container_width=True)

            # grafico radar o barre 
            st.markdown('<div class="compare-chart">', unsafe_allow_html=True)

            chart_type = st.radio(
                "Tipo di grafico",
                ["Radar", "Barre"],
                horizontal=True,
                key="explore_chart_type",
                label_visibility="collapsed",
            )

            # palette colori per i grafici
            palette = ["#6c5ce7", "#00E676", "#FFC400", "#FF1744", "#00B0FF", "#FF6D00", "#D500F9", "#76FF03"]

            if chart_type == "Radar":
                fig = go.Figure()
                # ciclo che aggiunge una traccia per ogni valutazione selezionata
                for i, ev in enumerate(sel_to_confront):
                    agg = ev.get("aggregate", {})
                    # serve per chiudere il poligono del radar, quindi aggiungo il primo valore alla fine della lista
                    values = [agg.get(k, 0) for k in metric_keys]
                    values_closed = values + [values[0]] 
                    labels_closed = metric_labels + [metric_labels[0]]
                    # aggiungo la traccia al grafico radar
                    fig.add_trace(go.Scatterpolar(
                        r=values_closed,
                        theta=labels_closed,
                        fill='toself',
                        name=ev["name"],
                        line=dict(color=palette[i % len(palette)]),
                        opacity=0.7,
                    ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(20,20,30,0.6)",
                        radialaxis=dict(visible=True, range=[0, 1], color="#5a5a75"),
                        angularaxis=dict(color="#a2a2c0"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e0e0e0"),
                    legend=dict(bgcolor="rgba(20,20,30,0.6)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
                    margin=dict(t=40, b=40),
                )
            else:
                # barre raggruppate
                fig = go.Figure()
                for i, ev in enumerate(sel_to_confront):
                    agg = ev.get("aggregate", {})
                    values = [agg.get(k, 0) for k in metric_keys]
                    fig.add_trace(go.Bar(
                        x=metric_labels,
                        y=values,
                        name=ev["name"],
                        marker_color=palette[i % len(palette)],
                        opacity=0.85,
                    ))
                fig.update_layout(
                    barmode='group',
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,30,0.6)",
                    font=dict(family="Inter", color="#e0e0e0"),
                    yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    legend=dict(bgcolor="rgba(20,20,30,0.6)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
                    margin=dict(t=40, b=40),
                )

            st.plotly_chart(fig, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            # tabella breakdown per benchmark
            st.markdown('<div class="section-title">Dettaglio per Benchmark</div>', unsafe_allow_html=True)

            # raccolgo tutti i nomi benchmark dalle valutazioni selezionate
            all_benchmarks: set[str] = set()
            for ev in sel_to_confront:
                all_benchmarks.update(ev.get("per_benchmark", {}).keys())

            if all_benchmarks:
                table_rows = []
                for bm in sorted(all_benchmarks):
                    row: dict[str, object] = {"Benchmark": _short_source(bm)}
                    for ev in sel_to_confront:
                        bm_data = ev.get("per_benchmark", {}).get(bm, {})
                        row[ev["name"]] = bm_data.get("answer_correctness", "—")
                    table_rows.append(row)
                compare_df = pd.DataFrame(table_rows)

                # formatto le colonne numeriche
                fmt = {}
                for c in compare_df.columns:
                    if c != "Benchmark":
                        fmt[c] = '{:.2f}'
                st.dataframe(
                    compare_df.style.format(fmt, na_rep="—")
                    .background_gradient(
                        subset=[c for c in compare_df.columns if c != "Benchmark"],
                        cmap='RdYlGn', vmin=0, vmax=1,
                    ),
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.info("Nessun dato per-benchmark disponibile.")

            # riassunto: migliore valutazione tra quelle selezionate
            best_ev = max(sel_to_confront, key=lambda e: e.get("aggregate", {}).get("answer_correctness", 0))
            best_score = best_ev.get("aggregate", {}).get("answer_correctness", 0)
            best_ts = datetime.fromisoformat(best_ev["timestamp"]).strftime("%d/%m/%Y %H:%M")

            summary_html = (
                '<div class="eval-card" style="border-left:4px solid #00E676;">'
                '  <div class="eval-card-header">'
                '    <span class="eval-card-name">🏆 Migliore Valutazione</span>'
                f'    <span class="eval-best-badge">{best_score:.2f}</span>'
                '  </div>'
                '  <div class="eval-card-body">'
                f'    <span class="eval-card-detail"><strong>{best_ev["name"]}</strong></span>'
                f'    <span class="eval-card-detail">Modello: {best_ev["model"]}</span>'
                f'    <span class="eval-card-detail">{best_ts}</span>'
                '  </div>'
                '</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)
