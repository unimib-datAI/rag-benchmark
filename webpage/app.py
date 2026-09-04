import json
import random
import requests
import pandas as pd
import streamlit as st

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama

from evaluator import run_evaluation
from persistence import save_evaluation, list_evaluations, load_evaluation, delete_evaluation

import dashboard as dash

st.set_page_config(
    page_title="SuperBenchmark · RAG Evaluation",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# load the custom css
_CSS_PATH = Path(__file__).resolve().parent / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.error("⚠️ style.css not found!")

ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "Run"

# benchmark names mapping -> json files in Run folder
BENCHMARK_MAPPING = {
    "LegalBench-RAG (Contract Precision)": "Tutto_LEGALBENCH.json",
    "MultiHop-RAG (Multi-Document Synthesis)": "Tutto_MULTIHOP.json",
    "CRAG (HTML Parsing & Traps)": "Tutto_CRAG_DAVE.json",
    "MTRAG (Multi-Turn Conversations)": "Tutto_MTRAG_DAVE.json",
    "OHRBench (OCR Noise / Scans)": "Tutto_OHRBench_DAVE.json",
    "SQuAD-it (Italian Q&A on Wikipedia)": "Tutto_SQUAD.json"
}

BENCHMARK_SHORT = {
    "LegalBench-RAG": "LegalBench",
    "MultiHop-RAG": "MultiHop",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "OHRBench": "OHRBench",
    "SQuAD-it": "SQuAD",
}

def invoke_ollama(prompt: str, ollama_url: str, model: str) -> str:
    """Call Ollama via langchain_ollama and return the response as a string."""
    try:
        llm = ChatOllama(model=model, base_url=ollama_url, temperature=0)
        return llm.invoke(prompt).content
    except Exception as exc:
        st.error(f"Ollama request failed: {exc}")
        return ""

# color based on score (green/yellow/red)
def _value_color(v: float) -> str:
    if pd.isna(v): return "red"
    if v >= 0.7: return "green"
    if v >= 0.4: return "yellow"
    return "red"

def _short_source(source: str) -> str:
    # get short name from dict, if any
    for key, short in BENCHMARK_SHORT.items():
        if key in source:
            return short
    return source

# page banner
st.markdown(dash.get_header_html(), unsafe_allow_html=True)

if "running" not in st.session_state:
    st.session_state.running = False
if "results" not in st.session_state:
    st.session_state.results = []

# sidebar
with st.sidebar:
    st.header("⚙️ RAG Engine Settings")
    motore_rag = st.radio("Select Search Engine & LLM", ["Local RAG (BM25)", "DAVE (Elasticsearch)"])

    st.markdown('<p class="sidebar-label">Base URL RAG</p>', unsafe_allow_html=True)
    rag_base_url = st.text_input("rag_base_url", "http://127.0.0.1:8000", label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Similarity Threshold</p>', unsafe_allow_html=True)
    threshold = st.slider("threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### LLM Judge")
    st.markdown('<p class="sidebar-label">Ollama Base URL</p>', unsafe_allow_html=True)
    ollama_url = st.text_input("ollama_url", "http://localhost:11434", label_visibility="collapsed")

    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
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
    st.caption(f"{len(selected_names)} of {len(BENCHMARK_MAPPING)} selected")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Queries per Benchmark</p>', unsafe_allow_html=True)
    sample_size_param = st.slider("sample_size_param", min_value=1, max_value=50, value=3, step=1, label_visibility="collapsed")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Selection Mode</p>', unsafe_allow_html=True)
    query_selection_mode = st.radio("query_selection_mode", options=["Random", "Fixed"], horizontal=True, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # start/stop button
    if not st.session_state.running:
        can_start = len(selected_files) > 0
        if st.button("Start Evaluation", type="primary", disabled=not can_start, width='stretch'):
            st.session_state.running = True
            st.session_state.results = []
            st.session_state.selected_files = selected_files
            st.session_state.sample_size_param = sample_size_param
            st.session_state.query_selection_mode = query_selection_mode
            if "llm_advice" in st.session_state:
                del st.session_state.llm_advice
            st.rerun()
    else:
        if st.button("Stop Evaluation", type="secondary", width='stretch'):
            st.session_state.running = False
            st.rerun()

# if there are no current results, load the last saved evaluation
if not st.session_state.running and not st.session_state.results:
    saved_evals = list_evaluations()
    if saved_evals:
        last_eval = saved_evals[0]
        complete_evals = load_evaluation(last_eval["filepath"])
        st.session_state.results = complete_evals.get("results", [])

# main tabs
tab_evaluation, tab_explore = st.tabs(["🔬 Evaluation", "📊 Explore"])

with tab_evaluation:
    if not st.session_state.running and not st.session_state.results:
        st.markdown(dash.get_welcome_html(), unsafe_allow_html=True)

    # --- evaluation in progress ---
    if st.session_state.running:
        st.markdown('<div class="section-title">Evaluation in progress</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        # datasets are loaded from the selected files in the sidebar
        dataset_items = []
        files_to_run = st.session_state.get("selected_files", list(BENCHMARK_MAPPING.values()))
        for file_name in files_to_run:
            path = RUN_DIR / file_name
            if path.exists():
                with open(path, 'r', encoding='utf-8-sig') as f:
                    elements = json.load(f)
                    # MTRAG special case: take only a certain number of multi-turn conversations, not single queries
                    if "MTRAG" in file_name:
                        conversations = defaultdict(list)
                        for element in elements:
                            conv_id = element['id'].split('_t')[0]
                            conversations[conv_id].append(element)

                        turns_needed = st.session_state.get("sample_size_param", 3)
                        conv_keys = list(conversations.keys())
                        if st.session_state.get("query_selection_mode", "Random") == "Random":
                            random.shuffle(conv_keys)
                        else:
                            conv_keys.sort()

                        turns_added = 0
                        for conv_key in conv_keys:
                            if turns_added >= turns_needed: break
                            for turn in sorted(conversations[conv_key], key=lambda x: x['id']):
                                if turns_added < turns_needed:
                                    dataset_items.append(turn)
                                    turns_added += 1
                    else:
                        size = min(st.session_state.get("sample_size_param", 3), len(elements))
                        if st.session_state.get("query_selection_mode", "Random") == "Random":
                            dataset_items.extend(random.sample(elements, size))
                        else:
                            dataset_items.extend(elements[:size])

        tot = len(dataset_items)

        # callback to update the streamlit progress bar
        def _streamlit_progress(phase, current, total_items, message):
            if phase == "ingest":
                progress_bar.progress(0.05 * (current / max(total_items, 1)))
                status_text.markdown(f"**📥 {message}**")
            elif phase == "fetch":
                progress_bar.progress(0.05 + (current / total_items) * 0.45 if total_items else 0.05)
                status_text.markdown(f"**{message}**")
            elif phase == "ragas":
                status_text.markdown(f"**{message}**")
            elif phase == "done":
                progress_bar.progress(1.0)

        with st.spinner("LLM Ragas evaluation in progress... waiting for model response"):
            try:
                ui_reports = run_evaluation(
                    dataset=dataset_items,
                    rag_base_url=rag_base_url,
                    threshold=threshold,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    progress_callback=_streamlit_progress,
                    motore_rag=motore_rag,
                    benchmark_files=files_to_run
                )
            # if there is an error, show the message and reset results
            except RuntimeError as e:
                st.error(str(e))
                ui_reports = []

        progress_bar.progress(1.0)
        st.session_state.results = ui_reports

        # save evaluation automatically
        saved = save_evaluation(
            results=ui_reports,
            model=ollama_model,
            endpoint=rag_base_url,
            threshold=threshold,
            sample_size=st.session_state.get('sample_size_param', 3),
            benchmarks_used=files_to_run,
            rag_engine=motore_rag,
        )
        st.session_state.last_saved_id = saved['id']

        st.session_state.running = False
        status_text.empty()
        st.rerun()

    # show results
    if st.session_state.results:
        reports = st.session_state.results
        df = pd.DataFrame(reports)

        # averages of main metrics
        avg_c = df['correctness_voto'].mean()
        avg_f = df['faithfulness_voto'].mean()
        avg_r = df['retrieval_voto'].mean()
        avg_recall = df['recall_voto'].mean() if 'recall_voto' in df.columns else 0.0

        # metrics overview (4 RAGAS metrics)
        st.markdown('<div class="section-title">Overview (Ragas Metrics)</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        for col, label, val in [(col1, "Answer Correctness", avg_c), (col2, "Faithfulness", avg_f), (col3, "Context Precision", avg_r), (col4, "Context Recall", avg_recall)]:
            with col:
                st.markdown(dash.get_metric_card_html(label, val, _value_color(val)), unsafe_allow_html=True)

        # Warning on failed queries and truncated contexts
        ragas_stats = reports[0].get("_ragas_stats", {}) if reports else {}
        failed_queries = sum(1 for r in reports if r.get("ragas_fallita", False))
        truncated_contexts = ragas_stats.get("contesti_troncati", 0)
        if failed_queries > 0 or truncated_contexts > 0:
            msg_parts = []
            if truncated_contexts > 0:
                msg_parts.append(f"**{truncated_contexts}** truncated contexts (exceeded model limit)")
            if failed_queries > 0:
                msg_parts.append(f"**{failed_queries}/{len(reports)}** queries with failed RAGAS metrics (set to 0.0)")
            st.warning("⚠️ " + " · ".join(msg_parts) + ". The averages might not reflect all queries.")

        # if there are multiple benchmarks, show breakdown
        sources = df['source'].unique()
        if len(sources) > 1:
            st.markdown('<div class="section-title">Per Benchmark</div>', unsafe_allow_html=True)
            cols = st.columns(len(sources))
            for i, src in enumerate(sources):
                src_df = df[df['source'] == src]
                sc = src_df['correctness_voto'].mean()
                with cols[i]:
                    st.markdown(dash.get_breakdown_card_html(_short_source(src), sc, len(src_df), _value_color(sc)), unsafe_allow_html=True)

        # results table
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
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

        # single query details with expander
        st.markdown('<div class="section-title">Query Details</div>', unsafe_allow_html=True)
        for rep in reports:
            corr = rep.get('correctness_voto', 0)
            label = f"{rep['query_id']}  |  {rep['source']}  |  Correctness: {corr:.2f}"
            with st.expander(label):
                st.markdown("**Question**")
                st.code(rep['domanda'], language=None)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**RAG Answer**")
                    st.info(rep['risposta_rag'])
                with c2:
                    st.markdown("**Gold Answer**")
                    st.success(rep['gold'])

                st.markdown("---")
                st.markdown("**Ragas Metrics (LLM)**")
                m1, m2, m3, m4, m5 = st.columns(5)
                for col, metric_key, metric_label in [(m1, 'retrieval', 'Context Precis.'), (m2, 'recall', 'Context Recall'), (m3, 'faithfulness', 'Faithfulness'), (m4, 'correctness', 'Correctness'), (m5, 'relevancy', 'Relevancy')]:
                    raw_v = rep.get(f'{metric_key}_voto', 0)
                    v = round(float(raw_v), 3) if pd.notna(raw_v) else 0.0
                    reason = ""  # ragas returns only scores, not reasons unfortunately
                    with col:
                        st.markdown(dash.get_judge_box_html(metric_label, v, reason, _value_color(v)), unsafe_allow_html=True)

                with st.expander("Retrieved Documents"):
                    st.text(rep['docs'])

        # diagnostics
        st.markdown('<div class="section-title">Diagnostics & Solutions</div>', unsafe_allow_html=True)

        worst_query = min(reports, key=lambda x: x['correctness_voto'] + x['retrieval_voto'], default=None)
        if worst_query and "llm_advice" not in st.session_state:
            with st.spinner("🧠 AI is writing the custom diagnostic report..."):
                # build the prompt for the expert
                PROMPT_EXPERT = f"""Act as an AI Architect expert in RAG.
            I have evaluated my RAG system. Here are the averages:
            - Context Precision: {avg_r:.2f}
            - Faithfulness: {avg_f:.2f}
            - Answer Correctness: {avg_c:.2f}

            Here is the query where the system performed worst:
            QUESTION: {worst_query['domanda']}
            RAG ANSWER: {worst_query['risposta_rag']}
            GOLD ANSWER: {worst_query['gold']}

            Write an analysis of maximum 4 sentences highlighting the main problem and giving 2 concrete technical advice. Write in English using Markdown formatting."""

                st.session_state.llm_advice = invoke_ollama(PROMPT_EXPERT, ollama_url, ollama_model)

        llm_advice_text = st.session_state.get("llm_advice", "")
        html_issues = dash.get_diagnostics_html(avg_r, avg_f, avg_c)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown(html_issues, unsafe_allow_html=True)

        with col_d2:
            st.markdown('<div class="diag-panel solutions" style="margin-bottom: 1rem;"><h3>🤖 Expert Verdict</h3></div>', unsafe_allow_html=True)
            if llm_advice_text:
                st.success(llm_advice_text)
            else:
                st.warning("LLM Verdict not available.")

        # footer with evaluation info
        st.markdown(dash.get_footer_html(len(reports), ollama_model, datetime.now().strftime('%d/%m/%Y %H:%M')), unsafe_allow_html=True)

# explore tab - evaluation history and comparison
with tab_explore:
    import plotly.graph_objects as go

    if "explore_msg" in st.session_state:
        st.success(st.session_state.explore_msg)
        del st.session_state.explore_msg

    saved_evals = list_evaluations()

    # saved evaluations list
    st.markdown('<div class="section-title">Saved Evaluations</div>', unsafe_allow_html=True)

    if not saved_evals:
        st.markdown(
            '<div class="welcome-box">'
            '<h3>No saved evaluations</h3>'
            '<p>Run an evaluation from the "Evaluation" tab to see it here.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # checkbox to select evaluations to compare
        selected_for_compare: list[dict] = []

        for idx, ev in enumerate(saved_evals):
            ts = datetime.fromisoformat(ev["timestamp"]).strftime("%d/%m/%Y %H:%M")
            avg_corr = ev.get("aggregate", {}).get("answer_correctness", 0.0)
            color = _value_color(avg_corr)
            color_hex = {"green": "#00E676", "yellow": "#FFC400", "red": "#FF1744"}.get(color, "#a2a2c0")

            # html of the card for each evaluation
            card_html = (
                f'<div class="eval-card">'
                f'  <div class="eval-card-header">'
                f'    <span class="eval-card-name">{ev["name"]}</span>'
                f'    <span class="eval-card-ts">{ts}</span>'
                f'  </div>'
                f'  <div class="eval-card-body">'
                f'    <span class="eval-card-detail">RAG: <strong>{ev.get("rag_engine", "Unknown")}</strong></span>'
                f'    <span class="eval-card-detail">Model: <strong>{ev.get("model", "Unknown")}</strong></span>'
                f'    <span class="eval-card-detail">Queries: <strong style="font-family:JetBrains Mono,monospace">{ev["num_queries"]}</strong></span>'
                f'    <span class="eval-card-detail">Correctness: <strong style="color:{color_hex};font-family:JetBrains Mono,monospace">{avg_corr:.2f}</strong></span>'
                f'  </div>'
                f'</div>'
            )

            col_chk, col_card, col_load, col_del = st.columns([0.5, 8, 1.5, 1])

            with col_chk:
                checked = st.checkbox(" ", key=f"explore_cmp_{idx}", label_visibility="collapsed")
                if checked:
                    selected_for_compare.append(ev)

            with col_card:
                st.markdown(card_html, unsafe_allow_html=True)

            with col_load:
                if st.button("📂 Open", key=f"explore_load_{idx}", width='stretch'):
                    full_eval = load_evaluation(ev["filepath"])
                    st.session_state.results = full_eval.get("results", [])
                    st.session_state.explore_msg = f"✅ Data for '{ev['name']}' loaded! Switch to the '🔬 Evaluation' tab above to view it."
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"explore_del_{idx}", width='stretch'):
                    delete_evaluation(ev["filepath"])
                    st.rerun()

        # evaluation comparison
        st.markdown('<div class="section-title">Evaluation Comparison</div>', unsafe_allow_html=True)

        if len(selected_for_compare) == 0:
            st.info("Select one or more evaluations to compare them.")
        else:
            metric_keys = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]
            metric_labels = ["Answer Correctness", "Faithfulness", "Context Precision", "Context Recall"]

            # Results summary table
            st.markdown('<div class="section-title">Metrics Results Table</div>', unsafe_allow_html=True)
            table_agg_rows = []
            for ev in selected_for_compare:
                agg = ev.get("aggregate", {})
                row = {"Evaluation": ev["name"]}
                for k, label in zip(metric_keys, metric_labels):
                    row[label] = round(agg.get(k, 0.0), 2)
                table_agg_rows.append(row)
            
            st.dataframe(pd.DataFrame(table_agg_rows), use_container_width=True)

            # Detailed table of single queries compared
            st.markdown('<div class="section-title">Compared Single Queries Detail</div>', unsafe_allow_html=True)
            query_map = {}
            for ev in selected_for_compare:
                full_eval = load_evaluation(ev["filepath"])
                eval_name = ev["name"]
                for res in full_eval.get("results", []):
                    qid = res.get("query_id", "")
                    if qid not in query_map:
                        query_map[qid] = {
                            ("Info", "Query ID"): qid,
                            ("Info", "Benchmark"): res.get("source", ""),
                            ("Info", "Question"): res.get("domanda", ""),
                        }
                    query_map[qid][("Answer Correctness", eval_name)] = round(res.get("correctness_voto", 0.0), 2)
                    query_map[qid][("Faithfulness", eval_name)] = round(res.get("faithfulness_voto", 0.0), 2)
                    query_map[qid][("Context Precision", eval_name)] = round(res.get("retrieval_voto", 0.0), 2)
                    query_map[qid][("Context Recall", eval_name)] = round(res.get("recall_voto", 0.0), 2)
            
            if query_map:
                df_compare = pd.DataFrame(list(query_map.values()))
                df_compare.columns = pd.MultiIndex.from_tuples(df_compare.columns)
                st.dataframe(df_compare, use_container_width=True)

            # radar or bar chart 
            st.markdown('<div class="compare-chart">', unsafe_allow_html=True)

            chart_type = st.radio(
                "Chart Type",
                ["Radar", "Bars"],
                horizontal=True,
                key="explore_chart_type",
                label_visibility="collapsed",
            )

            # color palette for charts
            palette = ["#6c5ce7", "#00E676", "#FFC400", "#FF1744", "#00B0FF", "#FF6D00", "#D500F9", "#76FF03"]

            if chart_type == "Radar":
                fig = go.Figure()
                # loop adding a trace for each selected evaluation
                for i, ev in enumerate(selected_for_compare):
                    agg = ev.get("aggregate", {})
                    # used to close the radar polygon, so add the first value at the end of the list
                    values = [agg.get(k, 0) for k in metric_keys]
                    values_closed = values + [values[0]] 
                    labels_closed = metric_labels + [metric_labels[0]]
                    # add trace to radar chart
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
                # grouped bars
                fig = go.Figure()
                for i, ev in enumerate(selected_for_compare):
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

            # breakdown table per benchmark
            st.markdown('<div class="section-title">Per Benchmark Detail</div>', unsafe_allow_html=True)

            # collect all benchmark names from selected evaluations
            all_benchmarks: set[str] = set()
            for ev in selected_for_compare:
                all_benchmarks.update(ev.get("per_benchmark", {}).keys())

            if all_benchmarks:
                table_rows = []
                for bm in sorted(all_benchmarks):
                    row: dict[str, object] = {"Benchmark": _short_source(bm)}
                    for ev in selected_for_compare:
                        bm_data = ev.get("per_benchmark", {}).get(bm, {})
                        row[ev["name"]] = bm_data.get("answer_correctness", "—")
                    table_rows.append(row)
                compare_df = pd.DataFrame(table_rows)

                # format numeric columns
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
                st.info("No per-benchmark data available.")

            # summary: best evaluation among the selected ones
            best_ev = max(selected_for_compare, key=lambda e: e.get("aggregate", {}).get("answer_correctness", 0))
            best_score = best_ev.get("aggregate", {}).get("answer_correctness", 0)
            best_ts = datetime.fromisoformat(best_ev["timestamp"]).strftime("%d/%m/%Y %H:%M")

            summary_html = (
                '<div class="eval-card" style="border-left:4px solid #00E676;">'
                '  <div class="eval-card-header">'
                '    <span class="eval-card-name">🏆 Best Evaluation</span>'
                f'    <span class="eval-best-badge">{best_score:.2f}</span>'
                '  </div>'
                '  <div class="eval-card-body">'
                f'    <span class="eval-card-detail"><strong>{best_ev["name"]}</strong></span>'
                f'    <span class="eval-card-detail">Model: {best_ev["model"]}</span>'
                f'    <span class="eval-card-detail">{best_ts}</span>'
                '  </div>'
                '</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)
