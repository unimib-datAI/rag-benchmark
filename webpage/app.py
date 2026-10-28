import json
import random
import pandas as pd
import streamlit as st

from collections import defaultdict
from datetime import datetime
from pathlib import Path


from evaluator import run_evaluation, build_judge_llm, classify_query_verdict
from taxonomy import metric_mean, metric_value, needs_reclassification
from persistence import save_evaluation, list_evaluations, load_evaluation, delete_evaluation
from benchmark_download import build_benchmarks_zip

import dashboard as dash
from dashboard import get_pattern_legend_html

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
    "LegalBench-RAG (Contract Precision)": "Tutto_LEGALBENCH.jsonl",
    "MultiHop-RAG (Multi-Document Synthesis)": "Tutto_MULTIHOP.jsonl",
    "CRAG (HTML Parsing & Traps)": "Tutto_CRAG_DAVE.jsonl",
    "MTRAG (Multi-Turn Conversations)": "Tutto_MTRAG_DAVE.jsonl",
    "OHRBench (OCR Noise / Scans)": "Tutto_OHRBench_DAVE.jsonl",
    "SQuAD-it (Italian Q&A on Wikipedia)": "Tutto_SQUAD.jsonl"
}

BENCHMARK_SHORT = {
    "LegalBench-RAG": "LegalBench",
    "MultiHop-RAG": "MultiHop",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "OHRBench": "OHRBench",
    "SQuAD-it": "SQuAD",
}

def _load_benchmark_jsonl(path, skip_context=True):
    """Load a JSONL benchmark file line-by-line, optionally skipping the context field to save memory."""
    elements = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if skip_context:
                record.pop('context', None)
            elements.append(record)
    return elements

def invoke_ollama(prompt: str, ollama_url: str, model: str, llm_api_key: str = "") -> str:
    """Call Ollama via langchain_ollama and return the response as a string."""
    try:
        llm, _ = build_judge_llm(ollama_url, model, llm_api_key)
        return llm.invoke(prompt).content
    except Exception as exc:
        st.error(f"LLM request failed: {exc}")
        return ""

# color based on score (green/yellow/red)
def _value_color(v: float) -> str:
    if v is None or pd.isna(v): return "grey"
    if v >= 0.7: return "green"
    if v >= 0.4: return "yellow"
    return "red"

def _fmt_score(v) -> str:
    # metrics not computed by RAGAS are None and shown as N/A
    return "N/A" if v is None else f"{v:.2f}"

def _fmt_metric(rep: dict, key: str) -> str:
    # single-query score, N/A when RAGAS did not compute it
    return _fmt_score(metric_value(rep, key))

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
    st.markdown('<p class="sidebar-label">Indirizzo LLM (Endpoint URL)</p>', unsafe_allow_html=True)
    ollama_url = st.text_input(
        "ollama_url",
        "http://localhost:11434",
        placeholder="es. http://localhost:11434 o URL dell'università",
        label_visibility="collapsed",
    )

    st.markdown('<p class="sidebar-label">API Key (Opzionale)</p>', unsafe_allow_html=True)
    llm_api_key = st.text_input(
        "llm_api_key",
        "",
        placeholder="es. sk-12345...",
        type="password",
        label_visibility="collapsed",
    )

    st.markdown('<p class="sidebar-label">Model</p>', unsafe_allow_html=True)
    # Permette all'utente di inserire un modello personalizzato
    ollama_model = st.text_input("model_select", value="gemma4:31b-cloud", placeholder="es. gemma4:31b-cloud o il modello dell'università", label_visibility="collapsed")

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

    # the raw benchmarks are not in the repository: download them from the official sources
    if st.button("Download Benchmark", width='stretch', disabled=st.session_state.running,
                 help="Downloads the raw files of the 6 benchmarks (~1 GB) into a .zip in the project folder"):
        # clicking Cancel makes Streamlit interrupt this run at the next progress update;
        # build_benchmarks_zip then removes the partial zip and the temporary files
        cancel_slot = st.empty()
        cancel_slot.button("Cancel Download", width='stretch',
                           on_click=lambda: st.session_state.update(download_cancelled=True))
        bar = st.progress(0.0, text="Starting download...")

        def _show_progress(step, n_steps, label, done, total):
            fraction = (step + (done / total if total else 0)) / n_steps
            size = f"{done / 2**20:.0f}/{total / 2**20:.0f} MB" if total else f"{done / 2**20:.0f} MB"
            bar.progress(min(fraction, 1.0), text=f"[{step + 1}/{n_steps}] {label}: {size}")

        try:
            zip_path = build_benchmarks_zip(progress=_show_progress)
            cancel_slot.empty()
            bar.empty()
            st.success(f"Saved `{zip_path}`. Extract it in the project folder, then run the `Run/extract_*.py` scripts.")
        except Exception as e:
            cancel_slot.empty()
            bar.empty()
            st.error(f"Benchmark download failed: {e}")
    elif st.session_state.pop("download_cancelled", False):
        st.info("Download cancelled.")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Queries per Benchmark</p>', unsafe_allow_html=True)
    sample_size_param = st.number_input("sample_size_param", min_value=1, value=3, step=1, label_visibility="collapsed")

    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Metriche RAGAS da Calcolare</p>', unsafe_allow_html=True)
    all_metrics = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness", "context_recall"]
    selected_metrics = st.multiselect("metrics_select", options=all_metrics, default=all_metrics, label_visibility="collapsed")
    
    st.markdown('<p class="sidebar-label" style="margin-top: 1rem;">Richieste in Parallelo (Velocità)</p>', unsafe_allow_html=True)
    max_workers_param = st.slider("max_workers_param", min_value=1, max_value=10, value=2, step=1, label_visibility="collapsed")

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
            st.session_state.pop("eval_error", None)
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
        loaded_results = complete_evals.get("results", [])
        # retroactive classification for old evaluations (missing verdict or older taxonomy)
        for rep in loaded_results:
            if needs_reclassification(rep):
                rep.update(classify_query_verdict(rep))
        st.session_state.results = loaded_results
        st.session_state.current_eval_name = complete_evals.get("name", "")

# main tabs
tab_evaluation, tab_explore = st.tabs(["🔬 Evaluation", "📊 Explore"])

with tab_evaluation:
    if not st.session_state.running and st.session_state.get("eval_error"):
        st.error(f"❌ Evaluation failed, nothing was saved: {st.session_state.eval_error}"
                 + ("\n\nShowing the latest saved evaluation." if st.session_state.results else ""))
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
                # the gold "context" is never sent to the RAG: every benchmark retrieves over its corpus
                elements = _load_benchmark_jsonl(path, skip_context=True)
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

        # placeholder per le statistiche dei token
        tps_placeholder = st.empty()
        def _tps_callback(total, input_tok, output_tok, max_input, tps):
            html = f'''
            <div style="text-align: center; padding: 10px; background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; margin-bottom: 10px; font-size: 0.9em;">
                ⚡ <strong>Totale:</strong> {total} (In: {input_tok} | Out: {output_tok}) | 
                <strong>Max Input/Req:</strong> {max_input} | 
                <strong>Velocità:</strong> {tps:.1f} tok/s
            </div>
            '''
            tps_placeholder.markdown(html, unsafe_allow_html=True)

        with st.spinner("LLM Ragas evaluation in progress... waiting for model response"):
            try:
                ui_reports = run_evaluation(
                    dataset=dataset_items,
                    rag_base_url=rag_base_url,
                    threshold=threshold,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    llm_api_key=llm_api_key,
                    progress_callback=_streamlit_progress,
                    tps_callback=_tps_callback,
                    motore_rag=motore_rag,
                    benchmark_files=files_to_run,
                    selected_metrics=selected_metrics,
                    max_workers=max_workers_param
                )
            # if there is an error, show the message and reset results
            except RuntimeError as e:
                # kept in session state: st.rerun() below would wipe a plain st.error
                st.session_state.eval_error = str(e)
                ui_reports = []

        progress_bar.progress(1.0)
        st.session_state.results = ui_reports

        # save evaluation automatically, only if it produced results: an empty
        # evaluation would otherwise become "the latest" shown at startup
        if ui_reports:
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

        # averages of main metrics, only over the queries where RAGAS computed them
        avg_c = metric_mean(reports, 'correctness_voto')
        avg_f = metric_mean(reports, 'faithfulness_voto')
        avg_r = metric_mean(reports, 'retrieval_voto')
        avg_recall = metric_mean(reports, 'recall_voto')

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
        rag_error_queries = sum(1 for r in reports if r.get("rag_error"))
        if failed_queries > 0 or truncated_contexts > 0 or rag_error_queries > 0:
            msg_parts = []
            if rag_error_queries > 0:
                msg_parts.append(f"**{rag_error_queries}/{len(reports)}** queries not answered by the RAG (RAG Error: not evaluated, excluded from the averages)")
            if truncated_contexts > 0:
                msg_parts.append(f"**{truncated_contexts}** truncated contexts (exceeded model limit)")
            if failed_queries > 0:
                msg_parts.append(f"**{failed_queries}/{len(reports)}** queries with failed RAGAS metrics (excluded from the averages)")
            st.markdown('<div style="margin-top: 1.5rem; margin-bottom: 1rem;">', unsafe_allow_html=True)
            st.warning("⚠️ " + " · ".join(msg_parts) + ". The averages might not reflect all queries.")
            st.markdown('</div>', unsafe_allow_html=True)

        # time spent per phase (only evaluations run after the timings were added)
        timings = ragas_stats.get("tempi")
        if timings:
            def _fmt(sec):
                m, s = divmod(round(sec), 60)
                h, m = divmod(m, 60)
                return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{sec:.1f}s"
            ingest, queries = timings.get("ingestion", {}), timings.get("query", {})
            with st.expander(f"⏱️ Timings — total {_fmt(timings.get('totale', 0))}"):
                st.markdown(f"**Ingestion** {_fmt(sum(ingest.values()))} · **RAG queries** {_fmt(sum(queries.values()))}"
                            f" · **RAGAS** {_fmt(timings.get('ragas', 0))}")
                rows = []
                for src in sorted(set(ingest) | set(queries)):
                    n_q = sum(1 for r in reports if r.get("source") == src)
                    q = queries.get(src, 0)
                    rows.append({"Benchmark": _short_source(src), "Ingestion": _fmt(ingest.get(src, 0)),
                                 "RAG queries": _fmt(q), "Per query": f"{q / n_q if n_q else 0:.1f}s"})
                st.table(rows)

        # if there are multiple benchmarks, show breakdown
        sources = df['source'].unique()
        if len(sources) > 1:
            st.markdown('<div class="section-title">Per Benchmark</div>', unsafe_allow_html=True)
            cols = st.columns(len(sources))
            for i, src in enumerate(sources):
                src_df = df[df['source'] == src]
                sc = metric_mean([r for r in reports if r['source'] == src], 'correctness_voto')
                with cols[i]:
                    st.markdown(dash.get_breakdown_card_html(_short_source(src), sc, len(src_df), _value_color(sc)), unsafe_allow_html=True)

        # verdict distribution summary
        pattern_counts = defaultdict(int)
        for rep in reports:
            pattern_counts[rep.get('error_pattern', 'UNKNOWN')] += 1
        st.markdown(dash.get_verdict_summary_html(reports), unsafe_allow_html=True)
        
        with st.expander("📖 Mostra Legenda & Significato dei Pattern di Errore"):
            st.markdown(get_pattern_legend_html(), unsafe_allow_html=True)

        # results table
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        cols_needed = ['query_id', 'source', 'verdict_label', 'error_pattern_label', 'retrieval_voto']
        if 'recall_voto' in df.columns:
            cols_needed.append('recall_voto')
        cols_needed.extend(['faithfulness_voto', 'correctness_voto', 'relevancy_voto'])
        
        # Ensure fallback for legacy records
        for col in ['verdict_label', 'error_pattern_label']:
            if col not in df.columns:
                df[col] = ''

        display_df = df[[c for c in cols_needed if c in df.columns]].copy()
        for col in display_df.columns:
            if col.endswith('_voto'):
                display_df[col] = [metric_value(r, col) for r in reports]
        
        column_rename = {
            'query_id': 'Query',
            'source': 'Benchmark',
            'verdict_label': 'Verdict',
            'error_pattern_label': 'Pattern',
            'retrieval_voto': 'C. Precision',
            'recall_voto': 'C. Recall',
            'faithfulness_voto': 'Faithfulness',
            'correctness_voto': 'Correctness',
            'relevancy_voto': 'Relevancy'
        }
        display_df = display_df.rename(columns=column_rename)
        
        numeric_cols = [c for c in ['C. Precision', 'C. Recall', 'Faithfulness', 'Correctness', 'Relevancy'] if c in display_df.columns]
        st.dataframe(
            display_df.style.format({c: '{:.2f}' for c in numeric_cols}, na_rep="—")
            .background_gradient(subset=numeric_cols, cmap='RdYlGn', vmin=0, vmax=1),
            width='stretch', hide_index=True,
        )

        # single query details with expander
        st.markdown('<div class="section-title">Query Details</div>', unsafe_allow_html=True)
        
        # Filter controls for Query Details
        all_verdicts = sorted(list({r.get('verdict_label', '') for r in reports if r.get('verdict_label')}))
        all_patterns = sorted(list({r.get('error_pattern_label', '') for r in reports if r.get('error_pattern_label')}))
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_verdict = st.multiselect("Filtra per Verdetto", options=all_verdicts, default=[], placeholder="Tutti i verdetti")
        with col_f2:
            filter_pattern = st.multiselect("Filtra per Pattern di Errore", options=all_patterns, default=[], placeholder="Tutti i pattern")
            
        filtered_reports = reports
        if filter_verdict:
            filtered_reports = [r for r in filtered_reports if r.get('verdict_label') in filter_verdict]
        if filter_pattern:
            filtered_reports = [r for r in filtered_reports if r.get('error_pattern_label') in filter_pattern]

        if not filtered_reports:
            st.info("Nessuna query corrisponde ai filtri selezionati.")
            
        for rep in filtered_reports:
            corr = metric_value(rep, 'correctness_voto')
            v_label = rep.get('verdict_label', '')
            pattern = rep.get('error_pattern_label', '')
            pattern_suffix = f"  |  {pattern}" if rep.get('verdict') == 'wrong' else ""
            if rep.get('verdict') == 'partial' and rep.get('error_cause_label'):
                pattern_suffix = f"  |  cause: {rep['error_cause_label']}"
            label = f"{rep['query_id']}  |  {rep['source']}  |  {v_label}{pattern_suffix}  |  Correctness: {_fmt_score(corr)}"
            with st.expander(label):
                st.markdown("**Question**")
                st.code(rep['domanda'], language=None)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**RAG Answer**")
                    if rep.get('rag_error'):
                        # the RAG failed: there is no answer, show the infrastructure error
                        st.error(f"🔌 RAG Error: {rep['rag_error']}")
                    else:
                        st.info(rep['risposta_rag'])
                with c2:
                    st.markdown("**Gold Answer**")
                    st.success(rep['gold'])

                st.markdown("---")
                st.markdown("**Ragas Metrics (LLM)**")
                m1, m2, m3, m4, m5 = st.columns(5)
                for col, metric_key, metric_label in [(m1, 'retrieval', 'Context Precis.'), (m2, 'recall', 'Context Recall'), (m3, 'faithfulness', 'Faithfulness'), (m4, 'correctness', 'Correctness'), (m5, 'relevancy', 'Relevancy')]:
                    raw_v = metric_value(rep, f'{metric_key}_voto')
                    v = round(raw_v, 3) if raw_v is not None else "N/A"
                    reason = ""  # ragas returns only scores, not reasons unfortunately
                    with col:
                        st.markdown(dash.get_judge_box_html(metric_label, v, reason, _value_color(raw_v)), unsafe_allow_html=True)

                # verdict card with error pattern classification
                st.markdown(dash.get_verdict_html(
                    rep.get('verdict', 'wrong'),
                    rep.get('verdict_label', '❌ Wrong'),
                    rep.get('error_pattern_label', 'Unknown'),
                    rep.get('error_reason', ''),
                ), unsafe_allow_html=True)

                with st.expander("Retrieved Documents"):
                    st.text(rep['docs'])

        # diagnostics and expert advice
        st.markdown('<div class="section-title">Diagnostics & Expert Report</div>', unsafe_allow_html=True)

        html_issues = dash.get_diagnostics_html(reports)
        st.markdown(html_issues, unsafe_allow_html=True)

        # queries not evaluated by RAGAS have fake 0.0 scores: keep them out of best/worst
        evaluated = [r for r in reports if r.get('verdict') != 'not_evaluated']
        # ranked by Correctness, then Precision (metrics not computed count as 0)
        def _rank(x):
            return ((metric_value(x, 'correctness_voto') or 0), (metric_value(x, 'retrieval_voto') or 0))
        worst_query = min(evaluated, key=_rank, default=None)
        best_query = max(evaluated, key=_rank, default=None)

        if worst_query and "llm_advice" not in st.session_state:
            with st.spinner("🧠 AI RAG Expert is writing the analysis report..."):
                # build per-benchmark summary
                benchmark_summary = ""
                for src in sources:
                    src_reports = [r for r in reports if r['source'] == src]
                    benchmark_summary += (
                        f"  - {src}: Correctness={_fmt_score(metric_mean(src_reports, 'correctness_voto'))}, "
                        f"Faithfulness={_fmt_score(metric_mean(src_reports, 'faithfulness_voto'))}, "
                        f"Precision={_fmt_score(metric_mean(src_reports, 'retrieval_voto'))}, "
                        f"Recall={_fmt_score(metric_mean(src_reports, 'recall_voto'))} "
                        f"({len(src_reports)} queries)\n"
                    )

                # build error patterns summary
                err_pattern_summary = ""
                for p_id, p_count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
                    err_pattern_summary += f"  - {p_id}: {p_count} queries\n"

                PROMPT_EXPERT = f"""You are a Senior RAG System Architect. Provide a clear, clean and concise evaluation report.

EVALUATION METRICS (averaged over {len(reports)} queries):
- Answer Correctness: {_fmt_score(avg_c)}
- Faithfulness: {_fmt_score(avg_f)}
- Context Precision: {_fmt_score(avg_r)}
- Context Recall: {_fmt_score(avg_recall)}

ERROR PATTERNS DETECTED:
{err_pattern_summary}
PER-BENCHMARK BREAKDOWN:
{benchmark_summary}

WORST QUERY:
  Question: {worst_query['domanda']}
  RAG Answer: {worst_query['risposta_rag'][:300]}
  Gold Answer: {worst_query['gold']}
  Scores: Correctness={_fmt_metric(worst_query, 'correctness_voto')}, Faithfulness={_fmt_metric(worst_query, 'faithfulness_voto')}, Precision={_fmt_metric(worst_query, 'retrieval_voto')}

BEST QUERY:
  Question: {best_query['domanda']}
  RAG Answer: {best_query['risposta_rag'][:300]}
  Gold Answer: {best_query['gold']}
  Scores: Correctness={_fmt_metric(best_query, 'correctness_voto')}, Faithfulness={_fmt_metric(best_query, 'faithfulness_voto')}, Precision={_fmt_metric(best_query, 'retrieval_voto')}

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

                st.session_state.llm_advice = invoke_ollama(PROMPT_EXPERT, ollama_url, ollama_model, llm_api_key)

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
            avg_corr = ev.get("aggregate", {}).get("answer_correctness")
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
                f'    <span class="eval-card-detail">Correctness: <strong style="color:{color_hex};font-family:JetBrains Mono,monospace">{_fmt_score(avg_corr)}</strong></span>'
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
                    loaded_results = full_eval.get("results", [])
                    for rep in loaded_results:
                        if needs_reclassification(rep):
                            rep.update(classify_query_verdict(rep))
                    st.session_state.results = loaded_results
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
                    row[label] = round(agg[k], 2) if agg.get(k) is not None else None
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
                    if needs_reclassification(res):
                        res.update(classify_query_verdict(res))
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
                    # metric_value: metrics not computed (RAGAS failure, RAG Error) stay empty, not 0.00
                    corr = metric_value(res, "correctness_voto") if res else None
                    faith = metric_value(res, "faithfulness_voto") if res else None
                    pattern = res.get("error_pattern_label") if res else None
                    try:
                        row[f"Corr ({ev['name']})"] = round(float(corr), 2) if corr is not None else None
                    except (ValueError, TypeError):
                        row[f"Corr ({ev['name']})"] = None
                    try:
                        row[f"Faith ({ev['name']})"] = round(float(faith), 2) if faith is not None else None
                    except (ValueError, TypeError):
                        row[f"Faith ({ev['name']})"] = None
                    row[f"Pattern ({ev['name']})"] = pattern or "—"
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
                                    ("C.Prec", "retrieval_voto"),
                                    ("C.Rec", "recall_voto"),
                                    ("Faith", "faithfulness_voto"),
                                    ("Corr", "correctness_voto"),
                                    ("Relev", "relevancy_voto"),
                                ]
                                metrics_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:0.8rem;">'
                                for m_label, m_key in metrics_data:
                                    m_val = metric_value(res, m_key)
                                    m_text = _fmt_score(m_val)
                                    m_color = {"green": "#00E676", "yellow": "#FFC400", "red": "#FF1744"}.get(_value_color(m_val), "#a2a2c0")
                                    metrics_html += f'<span style="background:rgba(255,255,255,0.05);border-radius:8px;padding:4px 8px;font-size:0.75rem;">{m_label}: <strong style="color:{m_color}">{m_text}</strong></span>'
                                metrics_html += '</div>'
                                st.markdown(metrics_html, unsafe_allow_html=True)

                                # Verdict card in deep compare
                                st.markdown(dash.get_verdict_html(
                                    res.get('verdict', 'wrong'),
                                    res.get('verdict_label', '❌ Wrong'),
                                    res.get('error_pattern_label', 'Unknown'),
                                    res.get('error_reason', ''),
                                ), unsafe_allow_html=True)

                                st.markdown("**RAG Answer**")
                                if res.get("rag_error"):
                                    st.error(f"🔌 RAG Error: {res['rag_error']}")
                                else:
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
                                val_a = metric_value(res_a, m_key)
                                val_b = metric_value(res_b, m_key)
                                with d_col:
                                    if val_a is None or val_b is None:
                                        st.metric(m_label, _fmt_score(val_b))
                                    else:
                                        st.metric(m_label, f"{val_b:.2f}", f"{val_b - val_a:+.2f}")
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
                    values = [agg.get(k) for k in metric_keys]
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
                    values = [agg.get(k) for k in metric_keys]
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
            best_ev = max(selected_for_compare, key=lambda e: e.get("aggregate", {}).get("answer_correctness") or 0)
            best_score = best_ev.get("aggregate", {}).get("answer_correctness")
            best_ts = datetime.fromisoformat(best_ev["timestamp"]).strftime("%d/%m/%Y %H:%M")

            summary_html = (
                '<div class="eval-card" style="border-left:4px solid #00E676;">'
                '  <div class="eval-card-header">'
                '    <span class="eval-card-name">🏆 Best Evaluation</span>'
                f'    <span class="eval-best-badge">{_fmt_score(best_score)}</span>'
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
                                    f"  Correctness={_fmt_score(agg.get('answer_correctness'))}, "
                                    f"Faithfulness={_fmt_score(agg.get('faithfulness'))}, "
                                    f"Context Precision={_fmt_score(agg.get('context_precision'))}, "
                                    f"Context Recall={_fmt_score(agg.get('context_recall'))}"
                                )
                                summary_text += "\n  Per-benchmark: "
                                for bm_name, bm_data in ev.get("per_benchmark", {}).items():
                                    summary_text += f"{_short_source(bm_name)}(corr={_fmt_score(bm_data.get('answer_correctness'))}) "
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
