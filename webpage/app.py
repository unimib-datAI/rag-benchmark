import json
import random
import requests
import pandas as pd
import streamlit as st

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama

from evaluator import run_evaluation, parse_llm_endpoint_or_key
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
        base_url, client_kwargs = parse_llm_endpoint_or_key(ollama_url)
        llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0,
            **({"client_kwargs": client_kwargs} if client_kwargs else {})
        )
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
if "current_eval_name" not in st.session_state:
    st.session_state.current_eval_name = ""

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
    st.markdown('<p class="sidebar-label">URL o chiave API LLM AS A JUDGE</p>', unsafe_allow_html=True)
    ollama_url = st.text_input(
        "ollama_url",
        "http://localhost:11434",
        placeholder="es. http://localhost:11434 oppure la tua chiave API",
        label_visibility="collapsed",
    )

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
    sample_size_param = st.number_input("sample_size_param", min_value=1, value=3, step=1, label_visibility="collapsed")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Selection Mode</p>', unsafe_allow_html=True)
    query_selection_mode = st.radio("query_selection_mode", options=["Random", "Fixed", "Repeat Previous"], horizontal=True, label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Evaluation Name")
    st.markdown('<p class="sidebar-label">Evaluation Name <span style="color:#FF5252">*</span></p>', unsafe_allow_html=True)
    eval_name_input = st.text_input(
        "eval_name_input",
        placeholder="e.g. test_bm25_v1",
        label_visibility="collapsed",
    )
    eval_name = eval_name_input.strip() if eval_name_input else ""

    if not eval_name:
        st.caption("⚠️ Evaluation name is required to start")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # start/stop button
    if not st.session_state.running:
        can_start = len(selected_files) > 0 and bool(eval_name)
        if st.button("Start Evaluation", type="primary", disabled=not can_start, width='stretch'):
            st.session_state.running = True
            st.session_state.results = []
            st.session_state.selected_files = selected_files
            st.session_state.sample_size_param = sample_size_param
            st.session_state.query_selection_mode = query_selection_mode
            st.session_state.eval_name = eval_name
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
        st.session_state.current_eval_name = complete_evals.get("name", "")

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

        previous_eval_ids = set()
        if st.session_state.get("query_selection_mode") == "Repeat Previous":
            saved_evals = list_evaluations()
            if saved_evals:
                last_eval = load_evaluation(saved_evals[0]["filepath"])
                for res in last_eval.get("results", []):
                    previous_eval_ids.add(res.get("query_id"))
            else:
                st.warning("No previous evaluation found. Falling back to Fixed selection.")
                st.session_state.query_selection_mode = "Fixed"

        # datasets are loaded from the selected files in the sidebar
        dataset_items = []
        files_to_run = st.session_state.get("selected_files", list(BENCHMARK_MAPPING.values()))
        for file_name in files_to_run:
            path = RUN_DIR / file_name
            if path.exists():
                with open(path, 'r', encoding='utf-8-sig') as f:
                    elements = json.load(f)
                    
                    if st.session_state.get("query_selection_mode") == "Repeat Previous":
                        if previous_eval_ids:
                            filtered_elements = [e for e in elements if e["id"] in previous_eval_ids]
                            dataset_items.extend(filtered_elements)
                        continue

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
            name=st.session_state.get('eval_name', ''),
        )
        st.session_state.last_saved_id = saved['id']
        st.session_state.current_eval_name = saved['name']

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
        curr_name = st.session_state.get("current_eval_name", "")
        if curr_name:
            st.markdown(f'<div class="section-title">Overview &mdash; <span style="color: #a29bfe; font-size: 1.1rem; text-transform: none; margin-left: 8px;">{curr_name}</span></div>', unsafe_allow_html=True)
        else:
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

        # diagnostics and expert advice
        st.markdown('<div class="section-title">Diagnostics & Expert Report</div>', unsafe_allow_html=True)

        html_issues = dash.get_diagnostics_html(avg_r, avg_f, avg_c)
        st.markdown(html_issues, unsafe_allow_html=True)

        worst_query = min(reports, key=lambda x: x['correctness_voto'] + x['retrieval_voto'], default=None)
        best_query = max(reports, key=lambda x: x['correctness_voto'] + x['retrieval_voto'], default=None)

        if worst_query and "llm_advice" not in st.session_state:
            with st.spinner("🧠 AI RAG Expert is writing the analysis report..."):
                # build per-benchmark summary
                benchmark_summary = ""
                for src in sources:
                    src_df = df[df['source'] == src]
                    recall_str = f"{src_df['recall_voto'].mean():.2f}" if 'recall_voto' in src_df.columns else "N/A"
                    benchmark_summary += (
                        f"  - {src}: Correctness={src_df['correctness_voto'].mean():.2f}, "
                        f"Faithfulness={src_df['faithfulness_voto'].mean():.2f}, "
                        f"Precision={src_df['retrieval_voto'].mean():.2f}, "
                        f"Recall={recall_str} "
                        f"({len(src_df)} queries)\n"
                    )

                PROMPT_EXPERT = f"""You are a Senior RAG System Architect. Provide a clear, clean and concise evaluation report.

EVALUATION METRICS (averaged over {len(reports)} queries):
- Answer Correctness: {avg_c:.2f}
- Faithfulness: {avg_f:.2f}
- Context Precision: {avg_r:.2f}
- Context Recall: {avg_recall:.2f}

PER-BENCHMARK BREAKDOWN:
{benchmark_summary}

WORST QUERY:
  Question: {worst_query['domanda']}
  RAG Answer: {worst_query['risposta_rag'][:300]}
  Gold Answer: {worst_query['gold']}
  Scores: Correctness={worst_query['correctness_voto']:.2f}, Faithfulness={worst_query['faithfulness_voto']:.2f}, Precision={worst_query['retrieval_voto']:.2f}

BEST QUERY:
  Question: {best_query['domanda']}
  RAG Answer: {best_query['risposta_rag'][:300]}
  Gold Answer: {best_query['gold']}
  Scores: Correctness={best_query['correctness_voto']:.2f}, Faithfulness={best_query['faithfulness_voto']:.2f}, Precision={best_query['retrieval_voto']:.2f}

Write a clean, easy-to-read Markdown report with these sections:

### 📊 Summary
2-3 concise sentences summarizing what the numbers show.

### ✅ Strengths
- 2 bullet points on what works well.

### ⚠️ Main Bottlenecks
- 2 bullet points explaining where and why it fails (cite the metrics or benchmark).

### 🔧 Recommended Fixes
- 3 clear, actionable engineering steps to solve the issue.

### 🔮 Next Steps
- 2 future architecture enhancements.

Keep the language clear, concise and directly actionable. Total length under 300 words."""

                st.session_state.llm_advice = invoke_ollama(PROMPT_EXPERT, ollama_url, ollama_model)

        llm_advice_text = st.session_state.get("llm_advice", "")

        if llm_advice_text:
            st.markdown(f'<div class="expert-report-card">{llm_advice_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Expert analysis not available. Make sure the LLM Judge is accessible.")

        # footer with evaluation info
        st.markdown(dash.get_footer_html(len(reports), ollama_model, datetime.now().strftime('%d/%m/%Y %H:%M'), st.session_state.get("current_eval_name", "")), unsafe_allow_html=True)

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
                    st.session_state.current_eval_name = full_eval.get("name", "")
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
                row = {"Evaluation": ev["name"], "RAG Engine": ev.get("rag_engine", "—"), "LLM Judge": ev.get("model", "—")}
                for k, label in zip(metric_keys, metric_labels):
                    row[label] = round(agg.get(k, 0.0), 2)
                table_agg_rows.append(row)
            
            st.dataframe(pd.DataFrame(table_agg_rows), use_container_width=True)

            # load full results for all selected evaluations
            loaded_evals = {}
            for ev in selected_for_compare:
                loaded_evals[ev["name"]] = load_evaluation(ev["filepath"])

            # build a map: query_id -> {eval_name: full_result_dict}
            query_results_map: dict[str, dict] = {}
            query_meta: dict[str, dict] = {}
            for ev in selected_for_compare:
                for res in loaded_evals[ev["name"]].get("results", []):
                    qid = res.get("query_id", "")
                    if qid not in query_results_map:
                        query_results_map[qid] = {}
                        query_meta[qid] = {
                            "source": res.get("source", ""),
                            "domanda": res.get("domanda", ""),
                            "gold": res.get("gold", ""),
                        }
                    query_results_map[qid][ev["name"]] = res

            # Compared Single Queries summary table
            st.markdown('<div class="section-title">Compared Single Queries Detail</div>', unsafe_allow_html=True)
            cmp_table_rows = []
            for qid in sorted(query_results_map.keys()):
                meta = query_meta[qid]
                row = {"Query ID": qid, "Benchmark": _short_source(meta["source"])}
                for ev in selected_for_compare:
                    res = query_results_map[qid].get(ev["name"])
                    corr = res.get("correctness_voto") if res else None
                    faith = res.get("faithfulness_voto") if res else None
                    try:
                        row[f"Corr ({ev['name']})"] = round(float(corr), 2) if corr is not None else None
                    except (ValueError, TypeError):
                        row[f"Corr ({ev['name']})"] = None
                    try:
                        row[f"Faith ({ev['name']})"] = round(float(faith), 2) if faith is not None else None
                    except (ValueError, TypeError):
                        row[f"Faith ({ev['name']})"] = None
                cmp_table_rows.append(row)
            if cmp_table_rows:
                cmp_df = pd.DataFrame(cmp_table_rows)
                numeric_cols = [c for c in cmp_df.columns if c.startswith("Corr") or c.startswith("Faith")]
                for c in numeric_cols:
                    cmp_df[c] = pd.to_numeric(cmp_df[c], errors='coerce')
                cmp_fmt = {c: '{:.2f}' for c in numeric_cols}
                st.dataframe(
                    cmp_df.style.format(cmp_fmt, na_rep="—")
                    .background_gradient(subset=numeric_cols, cmap='RdYlGn', vmin=0, vmax=1),
                    width='stretch', hide_index=True,
                )

            # Interactive single-query side-by-side comparison
            if len(selected_for_compare) >= 2:
                st.markdown('<div class="section-title">🔍 Single Query Deep Comparison</div>', unsafe_allow_html=True)
                st.caption("Select a query to see side-by-side answers, contexts and metrics from each evaluation.")

                # find queries present in at least 2 evaluations
                shared_qids = [qid for qid, evals in query_results_map.items() if len(evals) >= 2]
                if shared_qids:
                    query_options = []
                    for qid in sorted(shared_qids):
                        meta = query_meta[qid]
                        q_short = meta["domanda"][:80] + "…" if len(meta["domanda"]) > 80 else meta["domanda"]
                        query_options.append(f"{qid}  |  {_short_source(meta['source'])}  |  {q_short}")

                    selected_query_label = st.selectbox(
                        "Select query to compare",
                        options=query_options,
                        key="explore_deep_compare_query",
                        label_visibility="collapsed",
                    )

                    if selected_query_label:
                        selected_qid = selected_query_label.split("  |  ")[0].strip()
                        meta = query_meta[selected_qid]

                        st.markdown("**Question**")
                        st.code(meta["domanda"], language=None)
                        st.markdown("**Gold Answer (Ground Truth)**")
                        st.success(meta["gold"])

                        # side-by-side columns for each evaluation
                        eval_names_with_data = [ev["name"] for ev in selected_for_compare if ev["name"] in query_results_map.get(selected_qid, {})]
                        cols = st.columns(len(eval_names_with_data))

                        for col_idx, eval_name in enumerate(eval_names_with_data):
                            res = query_results_map[selected_qid][eval_name]
                            ev_info = next((e for e in selected_for_compare if e["name"] == eval_name), {})

                            with cols[col_idx]:
                                st.markdown(f"#### 📋 {eval_name}")
                                st.caption(f"RAG: **{ev_info.get('rag_engine', '?')}** · Judge: **{ev_info.get('model', '?')}**")

                                # metrics mini-cards
                                metrics_data = [
                                    ("C.Prec", res.get("retrieval_voto", 0)),
                                    ("C.Rec", res.get("recall_voto", 0)),
                                    ("Faith", res.get("faithfulness_voto", 0)),
                                    ("Corr", res.get("correctness_voto", 0)),
                                    ("Relev", res.get("relevancy_voto", 0)),
                                ]
                                metrics_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.8rem;">'
                                for m_label, m_val in metrics_data:
                                    m_val_f = float(m_val) if pd.notna(m_val) else 0.0
                                    m_color = {"green": "#00E676", "yellow": "#FFC400", "red": "#FF1744"}.get(_value_color(m_val_f), "#a2a2c0")
                                    metrics_html += f'<span style="background:rgba(255,255,255,0.05);border-radius:8px;padding:4px 8px;font-size:0.75rem;">{m_label}: <strong style="color:{m_color}">{m_val_f:.2f}</strong></span>'
                                metrics_html += '</div>'
                                st.markdown(metrics_html, unsafe_allow_html=True)

                                st.markdown("**RAG Answer**")
                                st.info(res.get("risposta_rag", "N/A"))

                                with st.expander("📄 Retrieved Documents"):
                                    st.text(res.get("docs", "No documents available"))

                        # delta analysis between evaluations
                        if len(eval_names_with_data) == 2:
                            st.markdown("---")
                            st.markdown("##### 📊 Delta Analysis")
                            res_a = query_results_map[selected_qid][eval_names_with_data[0]]
                            res_b = query_results_map[selected_qid][eval_names_with_data[1]]
                            delta_cols = st.columns(5)
                            for d_col, m_key, m_label in zip(delta_cols,
                                ["retrieval_voto", "recall_voto", "faithfulness_voto", "correctness_voto", "relevancy_voto"],
                                ["C. Precision", "C. Recall", "Faithfulness", "Correctness", "Relevancy"]):
                                val_a = float(res_a.get(m_key, 0) or 0)
                                val_b = float(res_b.get(m_key, 0) or 0)
                                delta = val_b - val_a
                                with d_col:
                                    st.metric(m_label, f"{val_b:.2f}", f"{delta:+.2f}")
                else:
                    st.info("No shared queries found between the selected evaluations. Make sure the evaluations were run on overlapping datasets.")

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
                for i, ev in enumerate(selected_for_compare):
                    agg = ev.get("aggregate", {})
                    values = [agg.get(k, 0) for k in metric_keys]
                    values_closed = values + [values[0]] 
                    labels_closed = metric_labels + [metric_labels[0]]
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

            all_benchmarks: set[str] = set()
            for ev in selected_for_compare:
                all_benchmarks.update(ev.get("per_benchmark", {}).keys())

            if all_benchmarks:
                table_rows = []
                for bm in sorted(all_benchmarks):
                    row: dict[str, object] = {"Benchmark": _short_source(bm)}
                    for ev in selected_for_compare:
                        bm_data = ev.get("per_benchmark", {}).get(bm, {})
                        val = bm_data.get("answer_correctness")
                        try:
                            row[ev["name"]] = round(float(val), 2) if val is not None and val != "—" else None
                        except (ValueError, TypeError):
                            row[ev["name"]] = None
                    table_rows.append(row)
                compare_df = pd.DataFrame(table_rows)

                numeric_cols = [c for c in compare_df.columns if c != "Benchmark"]
                for c in numeric_cols:
                    compare_df[c] = pd.to_numeric(compare_df[c], errors='coerce')

                fmt = {c: '{:.2f}' for c in numeric_cols}
                st.dataframe(
                    compare_df.style.format(fmt, na_rep="—")
                    .background_gradient(
                        subset=numeric_cols,
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
                f'    <span class="eval-card-detail">RAG: {best_ev.get("rag_engine", "?")}</span>'
                f'    <span class="eval-card-detail">Model: {best_ev["model"]}</span>'
                f'    <span class="eval-card-detail">{best_ts}</span>'
                '  </div>'
                '</div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)

            # Comparative Expert RAG Report (LLM-generated)
            if len(selected_for_compare) >= 2:
                st.markdown('<div class="section-title">🤖 Comparative RAG Expert Report</div>', unsafe_allow_html=True)

                compare_report_key = "compare_report_" + "_".join(sorted(ev["id"] for ev in selected_for_compare))

                if compare_report_key not in st.session_state:
                    if st.button("🧠 Generate Comparative Expert Analysis", key="btn_compare_report", type="primary"):
                        with st.spinner("🧠 AI RAG Expert is analyzing the evaluations..."):
                            # build comparison summary for LLM
                            eval_summaries = []
                            for ev in selected_for_compare:
                                agg = ev.get("aggregate", {})
                                summary_text = (
                                    f"- **{ev['name']}** (RAG: {ev.get('rag_engine', '?')}, Judge: {ev.get('model', '?')}, "
                                    f"Queries: {ev.get('num_queries', '?')})\n"
                                    f"  Correctness={agg.get('answer_correctness', 0):.2f}, "
                                    f"Faithfulness={agg.get('faithfulness', 0):.2f}, "
                                    f"Context Precision={agg.get('context_precision', 0):.2f}, "
                                    f"Context Recall={agg.get('context_recall', 0):.2f}"
                                )
                                summary_text += "\n  Per-benchmark: "
                                for bm_name, bm_data in ev.get("per_benchmark", {}).items():
                                    summary_text += f"{_short_source(bm_name)}(corr={bm_data.get('answer_correctness', 0):.2f}) "
                                eval_summaries.append(summary_text)

                            # find a shared query where the two evals differ the most
                            example_query_text = ""
                            if shared_qids:
                                max_delta_qid = None
                                max_delta = -1
                                for qid in shared_qids:
                                    scores = []
                                    for ev in selected_for_compare:
                                        r = query_results_map[qid].get(ev["name"])
                                        if r:
                                            scores.append(float(r.get("correctness_voto", 0) or 0))
                                    if len(scores) >= 2:
                                        delta = abs(max(scores) - min(scores))
                                        if delta > max_delta:
                                            max_delta = delta
                                            max_delta_qid = qid
                                if max_delta_qid:
                                    meta_q = query_meta[max_delta_qid]
                                    example_query_text = f"\nMost divergent query (ID: {max_delta_qid}):\n  Question: {meta_q['domanda']}\n  Gold Answer: {meta_q['gold']}\n"
                                    for ev in selected_for_compare:
                                        r = query_results_map[max_delta_qid].get(ev["name"])
                                        if r:
                                            example_query_text += f"  [{ev['name']}] Answer: {r.get('risposta_rag', 'N/A')[:300]} | Correctness: {r.get('correctness_voto', 0):.2f}\n"

                            PROMPT_COMPARE = f"""You are a Senior RAG System Architect conducting a comparative analysis of multiple RAG evaluation runs.

Here are the evaluations being compared:
{chr(10).join(eval_summaries)}

{example_query_text}

Write a structured Markdown comparison report with these sections:

## 📊 Comparative Overview
Brief summary comparing the evaluations (which is better overall and by how much).

## ⚔️ Head-to-Head Analysis
For each key metric (Correctness, Faithfulness, Context Precision, Context Recall), explain which evaluation wins and what that implies about the RAG pipeline or LLM Judge.

## ✅ What Works Best
Identify the strongest configuration and explain WHY it works better (2-3 bullet points).

## ❌ Key Weaknesses Found
Highlight the weakest areas across both evaluations (2-3 bullet points with specific metric evidence).

## 🔧 Recommended Actions
Give 3-4 concrete, actionable technical recommendations to improve the weaker system (e.g., "Switch from BM25 to hybrid search", "Increase chunk overlap", "Add a re-ranking step with cross-encoder").

## 🔮 Strategic Outlook
2 forward-looking suggestions for the overall RAG architecture evolution.

Write in English, be specific with numbers, keep it under 500 words total."""

                            st.session_state[compare_report_key] = invoke_ollama(PROMPT_COMPARE, ollama_url, ollama_model)
                            st.rerun()
                else:
                    st.markdown(st.session_state[compare_report_key])
                    if st.button("🔄 Regenerate Report", key="btn_regen_compare_report"):
                        del st.session_state[compare_report_key]
                        st.rerun()
