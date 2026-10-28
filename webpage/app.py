import json
import os
import random
from html import escape
import re
import sys
import time
import requests
import pandas as pd
import streamlit as st

from collections import defaultdict
from datetime import datetime
from pathlib import Path


from evaluator import run_evaluation, build_judge_llm, classify_query_verdict, FETCH_K
from taxonomy import metric_mean, metric_value, needs_reclassification
from persistence import save_evaluation, list_evaluations, load_evaluation, delete_evaluation, set_expert_report
import api_guide
import answerable
import ingest_http
from benchmark_download import build_benchmarks_zip, extract_benchmarks_zip
import jobs
from settings import load_config, save_config, dave_settings, DAVE_RETRIEVAL_METHODS

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

# benchmark file -> Run/ script that builds it
EXTRACT_SCRIPTS = {
    "Tutto_LEGALBENCH.jsonl": "extract_legalbenchmark.py",
    "Tutto_MULTIHOP.jsonl": "extract_multihop.py",
    "Tutto_CRAG_DAVE.jsonl": "extract_crag.py",
    "Tutto_MTRAG_DAVE.jsonl": "extract_mtrag.py",
    "Tutto_OHRBench_DAVE.jsonl": "extract_ohrbench.py",
    "Tutto_SQUAD.jsonl": "extract_squad.py",
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

# --- settings helpers -------------------------------------------------------

if "cfg" not in st.session_state:
    st.session_state.cfg = load_config()


def _init(key, value):
    """Seed st.session_state once (widget state is dropped when a widget is hidden)."""
    if key not in st.session_state:
        st.session_state[key] = value


def _persist(values: dict):
    """Write the changed keys to config.json right away."""
    cfg = st.session_state.cfg
    changed = {k: v for k, v in values.items() if cfg.get(k) != v}
    if changed:
        cfg.update(changed)
        save_config(cfg)


def _sync_rag_url(widget_key):
    st.session_state.rag_url = st.session_state[widget_key]


def _clamp(value, lo, hi, default):
    try:
        return min(max(type(default)(value), lo), hi)
    except (TypeError, ValueError):
        return default


def _open_evaluation(full_eval: dict):
    """Make a saved evaluation the current one, with the expert report stored in it (if any)."""
    results = full_eval.get("results", [])
    # retroactive classification for old evaluations (missing verdict or older taxonomy)
    for rep in results:
        if needs_reclassification(rep):
            rep.update(classify_query_verdict(rep))
    st.session_state.results = results
    st.session_state.current_eval_name = full_eval.get("name", "")
    st.session_state.current_eval_filepath = full_eval.get("filepath")
    st.session_state.pop("llm_advice", None)
    if full_eval.get("expert_report"):
        st.session_state.llm_advice = full_eval["expert_report"]


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+")


def _style_report(text: str) -> str:
    """Report of the LLM expert: no emojis, section titles in colour."""
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"^(#{1,4})[ \t]+(.+?)[ \t]*$", r"\1 :violet[\2]", text, flags=re.MULTILINE)
    return text


def _indexed_benchmarks():
    """Benchmarks ready to evaluate with DAVE: collection created and documents uploaded.

    OHRBench has no collection by design (its documents travel with each question)."""
    import ingest_dave
    collections, cache = ingest_dave.load_collections(), ingest_dave.load_doc_cache()
    ready = []
    for name in BENCHMARK_MAPPING:
        short = next((v for k, v in BENCHMARK_SHORT.items() if name.startswith(k)), name)
        if short == "OHRBench" or (short in collections and cache.get(short)):
            ready.append(name)
    return ready


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
    # sidebar settings are kept in config.json ("ui") so a page reload does not lose them
    ui_saved = st.session_state.cfg.get("ui", {})
    ENGINES = ["DAVE (Elasticsearch)", "Custom RAG (HTTP)"]
    _init("engine_select", ui_saved.get("engine") if ui_saved.get("engine") in ENGINES else ENGINES[0])
    motore_rag = st.radio(
        "RAG Engine", ENGINES, key="engine_select",
        help="DAVE: the DAVE server configured in the Setup tab. Custom RAG: your own RAG, reached over HTTP "
             "with the POST /ask and /ingest routes described in the API Guide tab.")

    if "DAVE" in motore_rag:
        _init("w_dave_retrieval_method", dave_settings(st.session_state.cfg)["retrieval_method"])
        if st.session_state.w_dave_retrieval_method not in DAVE_RETRIEVAL_METHODS:
            st.session_state.w_dave_retrieval_method = "full"
        dave_retrieval = st.radio(
            "DAVE Retrieval Type", list(DAVE_RETRIEVAL_METHODS), format_func=DAVE_RETRIEVAL_METHODS.get,
            key="w_dave_retrieval_method",
            help="How DAVE finds the chunks. Hybrid: vector search + full-text search (and entities) merged with "
                 "Reciprocal Rank Fusion. Dense: vector (embedding) search only. "
                 "Sparse: keyword full-text search only. Used by evaluations and by the retrieval test.")
        # saved right away: the evaluation below reads it from config.json
        _persist({"dave": {**dave_settings(st.session_state.cfg), "retrieval_method": dave_retrieval}})

    # one value for the custom RAG address, edited here and in Setup > Generic RAG (HTTP)
    _init("rag_url", ui_saved.get("rag_base_url", "http://127.0.0.1:8000"))
    if "DAVE" not in motore_rag:
        st.session_state.rag_url_sidebar = st.session_state.rag_url
        st.text_input(
            "Base URL RAG", key="rag_url_sidebar", on_change=_sync_rag_url, args=("rag_url_sidebar",),
            help="Address of your RAG: the tool calls <Base URL>/ask and <Base URL>/ingest. "
                 "From the Docker container, a RAG running on the host is http://host.docker.internal:PORT, not localhost.")
        st.caption("Upload the benchmark corpus to your RAG in Setup > Generic RAG (HTTP).")
        if st.button("Test RAG endpoint", width='stretch', key="test_rag_endpoint",
                     help="Sends one sample question to <Base URL>/ask and checks the reply."):
            with st.spinner("Waiting for your RAG..."):
                checks = api_guide.check_endpoint(st.session_state.rag_url, "SQuAD-it")
            for level, message in checks:
                {"ok": st.success, "info": st.info, "warning": st.warning, "error": st.error}[level](message[:300])
    rag_base_url = st.session_state.rag_url

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### LLM Judge", help="The judge LLM scores the answers with RAGAS metrics and classifies errors.")
    # judge settings are kept in config.json (same keys as the CLI) so a page reload does not lose them
    _init("ollama_url", st.session_state.cfg.get("ollama_url", "http://localhost:11434"))
    _init("llm_api_key", st.session_state.cfg.get("llm_api_key", ""))
    _init("model_select", st.session_state.cfg.get("model", "gemma4:31b-cloud"))

    ollama_url = st.text_input(
        "LLM Address (Endpoint URL)", key="ollama_url",
        placeholder="e.g. http://localhost:11434 or the university URL",
        help="Ollama or OpenAI-compatible server hosting the judge model.")

    llm_api_key = st.text_input(
        "API Key (Optional)", key="llm_api_key", placeholder="e.g. sk-12345...", type="password",
        help="Only needed for cloud/university endpoints that require authentication.")

    # lets the user enter a custom model
    ollama_model = st.text_input(
        "Model", key="model_select",
        placeholder="e.g. gemma4:31b-cloud or the university model",
        help="Name of the judge model, as known by the endpoint.")
    _persist({"ollama_url": ollama_url.strip(), "model": ollama_model.strip(), "llm_api_key": llm_api_key})

    if st.button("Test judge", width='stretch', key="test_judge",
                 help="Sends a one-word prompt to the judge with the endpoint, key and model above."):
        started = time.time()
        try:
            with st.spinner("Contacting the judge..."):
                llm, _ = build_judge_llm(ollama_url, ollama_model, llm_api_key, timeout=30)
                reply = str(llm.invoke("Reply with the single word: OK").content).strip()
            st.success(f"✅ Judge reachable ({time.time() - started:.1f}s). Reply: {reply[:80] or '(empty)'}")
        except Exception as e:
            st.error(f"❌ Judge not reachable: {str(e)[:400]}")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Benchmark", help="Which benchmark query sets to evaluate.")
    # DAVE: only the benchmarks already indexed; custom RAG: all (the corpus is uploaded to /ingest)
    default_benchmarks = _indexed_benchmarks() if "DAVE" in motore_rag else list(BENCHMARK_MAPPING)
    _init("bench_select", default_benchmarks if ui_saved.get("benchmarks") is None else ui_saved["benchmarks"])
    if st.session_state.get("engine_prev", motore_rag) != motore_rag:
        st.session_state.bench_select = default_benchmarks   # switching engine resets the selection
    st.session_state.engine_prev = motore_rag
    st.session_state.bench_select = [n for n in st.session_state.bench_select if n in BENCHMARK_MAPPING]
    selected_names = st.multiselect(
        "Benchmarks",
        options=list(BENCHMARK_MAPPING.keys()),
        key="bench_select",
        label_visibility="collapsed",
        help="Which benchmark query sets to evaluate. With DAVE the default is the ones already indexed "
             "(plus OHRBench, whose documents travel with each question); with a custom RAG, all of them.",
    )

    selected_files = [BENCHMARK_MAPPING[name] for name in selected_names]
    st.caption(f"{len(selected_names)} of {len(BENCHMARK_MAPPING)} selected")

    _init("sample_size_widget", _clamp(ui_saved.get("sample_size"), 1, 10**6, 3))
    sample_size_param = st.number_input(
        "Queries per Benchmark", min_value=1, step=1, key="sample_size_widget",
        help="How many queries to evaluate for each selected benchmark.")

    all_metrics = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness", "context_recall"]
    metric_help = {
        "faithfulness": "Is the answer grounded in the retrieved context?",
        "answer_relevancy": "Does the answer address the question?",
        "context_precision": "Are the retrieved docs relevant (ranked first)?",
        "answer_correctness": "Does the answer match the gold answer?",
        "context_recall": "Does the retrieved context cover the gold answer?",
    }
    st.markdown("RAGAS Metrics to Compute", help="Fewer metrics = faster and cheaper evaluation.")
    saved_metrics = ui_saved.get("metrics", all_metrics)
    for m in all_metrics:
        _init(f"metric_{m}", m in saved_metrics)
    selected_metrics = [m for m in all_metrics if st.checkbox(m, key=f"metric_{m}", help=metric_help[m])]

    _init("max_workers_param", _clamp(ui_saved.get("max_workers"), 1, 10, 2))
    max_workers_param = st.slider(
        "Parallel Requests (Speed)", min_value=1, max_value=10, step=1, key="max_workers_param",
        help="Queries evaluated at the same time. Higher is faster but may overload DAVE or the judge.")

    modes = ["Random", "Fixed", "Repeat Previous"]
    _init("query_selection_mode_radio", ui_saved.get("selection_mode") if ui_saved.get("selection_mode") in modes else "Random")
    query_selection_mode = st.radio(
        "Selection Mode", options=modes, horizontal=True, key="query_selection_mode_radio",
        help="Random: new queries each run. Fixed: always the same first queries. "
             "Repeat Previous: the queries of the last saved evaluation, for a like-for-like comparison.")

    # the benchmark selection is saved only when it differs from the default (the indexed ones),
    # so the default keeps following what has been indexed
    _persist({"ui": {
        "rag_base_url": rag_base_url.strip(),
        "engine": motore_rag,
        "benchmarks": None if set(selected_names) == set(default_benchmarks) else selected_names,
        "sample_size": int(sample_size_param),
        "metrics": selected_metrics,
        "max_workers": int(max_workers_param),
        "selection_mode": query_selection_mode,
    }})

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("### Evaluation Name")
    eval_name_input = st.text_input(
        "Evaluation Name *", placeholder="e.g. test_bm25_v1", key="eval_name_input",
        help="Label used to save the run; it appears in the Explore tab.")
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
        _open_evaluation(load_evaluation(last_eval["filepath"]))

# main tabs
tab_evaluation, tab_explore, tab_setup, tab_guide = st.tabs(["🔬 Evaluation", "📊 Explore", "🛠️ Setup", "📖 API Guide"])

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

        # seeded sampling: the seed is saved with the evaluation, so the Random selection can be reproduced
        sampling_seed = random.randrange(2**32)
        rng = random.Random(sampling_seed)

        # datasets are loaded from the selected files in the sidebar
        dataset_items = []
        sampling_notes = []
        files_to_run = st.session_state.get("selected_files", list(BENCHMARK_MAPPING.values()))
        for file_name in files_to_run:
            path = RUN_DIR / file_name
            if path.exists():
                # the gold "context" is never sent to the RAG: every benchmark retrieves over its corpus
                elements = _load_benchmark_jsonl(path, skip_context=True)
                # DAVE answers from the uploaded documents: ask only questions whose documents are there
                if st.session_state.get("query_selection_mode") != "Repeat Previous":
                    elements, info = answerable.filter_answerable(
                        elements, None if "DAVE" in motore_rag else ingest_http.uploaded_names(rag_base_url))
                    if info["indexed"] and info["tracked"]:
                        note = f"{info['source']}: {info['answerable']} of {info['total']} queries are answerable with the uploaded documents"
                        if info["unknown"]:
                            note += f" ({info['unknown']} without document information, kept)"
                        sampling_notes.append(note)
                        if not elements:
                            st.warning(f"{info['source']}: no query can be answered, upload its documents first (Setup tab).")
                            continue
                    elif info["tracked"] and info["source"] in answerable.COLLECTION:
                        sampling_notes.append(f"{info['source']}: documents index missing, no filtering "
                                              "(Setup tab > Build benchmark files)")
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
                        rng.shuffle(conv_keys)
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
                        dataset_items.extend(rng.sample(elements, size))
                    else:
                        dataset_items.extend(elements[:size])

        for note in sampling_notes:
            st.caption(note)
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

        # placeholder for the token statistics
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
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    llm_api_key=llm_api_key,
                    progress_callback=_streamlit_progress,
                    tps_callback=_tps_callback,
                    motore_rag=motore_rag,
                    benchmark_files=None,  # a custom RAG gets its corpus from Setup > Generic RAG (HTTP)
                    selected_metrics=selected_metrics,
                    max_workers=max_workers_param,
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
                sample_size=st.session_state.get('sample_size_param', 3),
                benchmarks_used=files_to_run,
                rag_engine=(f"{motore_rag} · {DAVE_RETRIEVAL_METHODS.get(dave_settings(st.session_state.cfg)['retrieval_method'], '')}"
                            if "DAVE" in motore_rag else motore_rag),
                name=st.session_state.get('eval_name', ''),
                config={
                    "selection_mode": st.session_state.get("query_selection_mode", "Random"),
                    "seed": sampling_seed,
                    "metrics": selected_metrics,
                    "fetch_k": FETCH_K,
                    "max_workers": max_workers_param,
                },
            )
            st.session_state.last_saved_id = saved['id']
            st.session_state.current_eval_filepath = saved['filepath']
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
            st.markdown(f'<div class="section-title">Overview &mdash; <span style="color: #a29bfe; font-size: 1.1rem; text-transform: none; margin-left: 8px;">{escape(curr_name)}</span></div>', unsafe_allow_html=True)
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
        
        with st.expander("📖 Show Legend & Meaning of the Error Patterns"):
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
            filter_verdict = st.multiselect("Filter by Verdict", options=all_verdicts, default=[], placeholder="All verdicts")
        with col_f2:
            filter_pattern = st.multiselect("Filter by Error Pattern", options=all_patterns, default=[], placeholder="All patterns")
            
        filtered_reports = reports
        if filter_verdict:
            filtered_reports = [r for r in filtered_reports if r.get('verdict_label') in filter_verdict]
        if filter_pattern:
            filtered_reports = [r for r in filtered_reports if r.get('error_pattern_label') in filter_pattern]

        if not filtered_reports:
            st.info("No query matches the selected filters.")
            
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

### Summary
2-3 concise sentences summarizing what the numbers show.

### Strengths
- 2 bullet points on what works well.

### Main Bottlenecks
- 2 bullet points explaining where and why it fails (cite the metrics or benchmark).

### Recommended Fixes
- 3 clear, actionable engineering steps to solve the issue.

### Next Steps
- 2 future architecture enhancements.

Keep the language clear, concise and directly actionable. Do not use emojis. Total length under 300 words."""

                report = invoke_ollama(PROMPT_EXPERT, ollama_url, ollama_model, llm_api_key)
                st.session_state.llm_advice = report
                # stored in the evaluation file: reloading the page does not generate it again
                eval_path = st.session_state.get("current_eval_filepath")
                if report and eval_path and Path(eval_path).exists():
                    set_expert_report(eval_path, report)

        llm_advice_text = st.session_state.get("llm_advice", "")

        if llm_advice_text:
            # LLM output is untrusted (it echoes benchmark documents): escape any HTML, the Markdown still renders
            with st.container(key="expert_report"):
                st.markdown(_style_report(escape(llm_advice_text, quote=False)))
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
                f'    <span class="eval-card-name">{escape(ev["name"])}</span>'
                f'    <span class="eval-card-ts">{ts}</span>'
                f'  </div>'
                f'  <div class="eval-card-body">'
                f'    <span class="eval-card-detail">RAG: <strong>{escape(str(ev.get("rag_engine", "Unknown")))}</strong></span>'
                f'    <span class="eval-card-detail">Model: <strong>{escape(str(ev.get("model", "Unknown")))}</strong></span>'
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
                    _open_evaluation(load_evaluation(ev["filepath"]))
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
                            st.markdown("##### Delta Analysis")
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
                f'    <span class="eval-card-detail"><strong>{escape(best_ev["name"])}</strong></span>'
                f'    <span class="eval-card-detail">RAG: {escape(str(best_ev.get("rag_engine", "?")))}</span>'
                f'    <span class="eval-card-detail">Model: {escape(str(best_ev["model"]))}</span>'
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

## Comparative Overview
Brief summary comparing the evaluations (which is better overall and by how much).

## Head-to-Head Analysis
For each key metric (Correctness, Faithfulness, Context Precision, Context Recall), explain which evaluation wins and what that implies about the RAG pipeline or LLM Judge.

## What Works Best
Identify the strongest configuration and explain WHY it works better (2-3 bullet points).

## Key Weaknesses Found
Highlight the weakest areas across both evaluations (2-3 bullet points with specific metric evidence).

## Recommended Actions
Give 3-4 concrete, actionable technical recommendations to improve the weaker system (e.g., "Switch from BM25 to hybrid search", "Increase chunk overlap", "Add a re-ranking step with cross-encoder").

## Strategic Outlook
2 forward-looking suggestions for the overall RAG architecture evolution.

Write in English, be specific with numbers, do not use emojis, keep it under 500 words total."""

                            st.session_state[compare_report_key] = invoke_ollama(PROMPT_COMPARE, ollama_url, ollama_model)
                            st.rerun()
                else:
                    st.markdown(_style_report(st.session_state[compare_report_key]))
                    if st.button("🔄 Regenerate Report", key="btn_regen_compare_report"):
                        del st.session_state[compare_report_key]
                        st.rerun()


# setup tab - DAVE connection, corpus ingestion, benchmark files
def _job_panel(name):
    """Status, progress and log of a background job; refreshes alone while it runs."""
    seen_key = f"job_seen_running_{name}"

    def render():
        snap = jobs.snapshot(name)
        if snap is None:
            return
        if snap["running"]:
            st.session_state[seen_key] = True
            st.info(f"⏳ {snap['label']} — running for {snap['elapsed']}s")
            if snap["progress"]:
                st.caption(snap["progress"])
            if st.button("Stop", key=f"w_stop_{name}"):
                jobs.stop(name)
        elif st.session_state.pop(seen_key, False):
            st.rerun()  # just finished: refresh tables and buttons once
        elif snap["returncode"] == 0:
            st.success(f"✅ {snap['label']} finished in {snap['elapsed']}s")
        else:
            st.error(f"❌ {snap['label']} failed (exit code {snap['returncode']})")
        if snap["lines"]:
            st.code("\n".join(snap["lines"][-30:]), language=None)

    st.fragment(render, run_every=2 if jobs.is_running(name) else None)()


with tab_setup:
    import ingest_dave
    from dave_client import DaveClient

    cfg = st.session_state.cfg
    saved = dave_settings(cfg)

    setup_dave, setup_rag = st.tabs(["DAVE", "Generic RAG (HTTP)"])

    with setup_dave:
        # 1. DAVE connection
        st.subheader("DAVE Connection")
        for k, v in saved.items():
            _init(f"w_dave_{k}", v)
        c1, c2 = st.columns(2)
        with c1:
            base_url = st.text_input("Base URL", key="w_dave_base_url", help="Address of the DAVE server, e.g. http://host:8001. Trailing slashes are removed.")
            username = st.text_input("Username", key="w_dave_username", help="DAVE account used for login, ingestion and retrieval.")
        with c2:
            base_path = st.text_input("Base path", key="w_dave_base_path", help="Path prefix where DAVE is served, e.g. /holmes24. Leave empty if DAVE runs at the root of the URL.")
            password = st.text_input("Password", key="w_dave_password", type="password",
                                     help="Stored in config.json (chmod 600, git-ignored). Passed to jobs via environment variables, never on the command line.")

        # 2. DAVE generation
        st.subheader("DAVE Generation")
        g1, g2, g3, g4, g5 = st.columns(5)
        temperature = g1.number_input("temperature", 0.0, 2.0, step=0.1, key="w_dave_temperature",
                                    help="Randomness of the answers. 0 = deterministic (best for comparable runs); higher = more varied.")
        top_p = g2.number_input("top_p", 0.0, 1.0, step=0.05, key="w_dave_top_p",
                                help="Nucleus sampling: only the most likely tokens whose probabilities sum to top_p are considered. 1.0 = off.")
        max_tokens = g3.number_input("max_tokens", 16, 32768, step=64, key="w_dave_max_tokens",
                                    help="Maximum length of each generated answer, in tokens.")
        top_k = g4.number_input("top_k", 1, 200, key="w_dave_top_k",
                                help="Sampling only from the k most likely tokens at each step.")
        rep_pen = g5.number_input("repetition penalty", 1.0, 2.0, step=0.05, key="w_dave_repetition_penalty",
                                  help="Penalizes tokens already generated, to reduce repetitions. 1.0 = off.")
        st.caption("Defaults are deterministic so runs are comparable; DAVE's chat UI uses different values.")

        dave_now = {
            "base_url": base_url.strip().rstrip("/"),
            "base_path": base_path.strip(),
            "username": username.strip(),
            "password": password,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "top_k": int(top_k),
            "repetition_penalty": float(rep_pen),
            "retrieval_method": saved["retrieval_method"] if "w_dave_retrieval_method" not in st.session_state
                                else st.session_state.w_dave_retrieval_method,
        }
        if dave_now != cfg.get("dave"):
            _persist({"dave": dave_now})
            st.rerun()  # the next evaluation picks up the new settings
        dave_ready = bool(dave_now["base_url"] and dave_now["username"] and dave_now["password"])

        if st.button("Test DAVE login", key="w_dave_test"):
            try:
                DaveClient(**dave_now)
                st.success("DAVE login OK")
            except Exception as e:
                st.error(f"DAVE login failed: {e}")

        st.divider()

        # 3. DAVE ingestion
        st.subheader("DAVE Ingestion")
        collections = ingest_dave.load_collections()
        doc_cache = ingest_dave.load_doc_cache()
        sources = {name: ingest_dave.find_source(name) for name in ingest_dave.INGESTERS}

        def _answerable_label(name):
            source = next((k for k, v in answerable.COLLECTION.items() if v == name), None)
            found = answerable.counts(source) if source else None
            return f"{found[0]} / {found[1]}" if found else ("all" if name == "OHRBench" else "-")
        st.dataframe(pd.DataFrame([
            {
                "Benchmark": name,
                "Corpus source": str(src.relative_to(ROOT_DIR)) if src else "missing",
                "Collection ID": collections.get(name, ""),
                "Docs indexed (local cache)": len(doc_cache.get(name, [])),
                "Answerable queries": _answerable_label(name),
            }
            for name, src in sources.items()
        ]), hide_index=True, width="stretch")

        available = [n for n, src in sources.items() if src]
        cfg_ingest = cfg.get("ingest", {})
        default_sel = [n for n in available if n != "OHRBench" and n not in ingest_dave.DEFAULT_BENCHMARKS_EXCLUDED]
        _init("w_ingest_benchmarks", [n for n in cfg_ingest.get("benchmarks", default_sel) if n in available])
        st.session_state.w_ingest_benchmarks = [n for n in st.session_state.w_ingest_benchmarks if n in available]
        sel_bench = st.multiselect(
            "Benchmarks to upload", available, key="w_ingest_benchmarks",
            help="Corpora to send to DAVE (only those found on disk). OHRBench is off by default because its documents travel with each question; MTRAG because its corpus is very large.",
        )

        _init("w_ingest_batch_docs", _clamp(cfg_ingest.get("batch_docs"), 1, 200, ingest_dave.MAX_BATCH_DOCS))
        _init("w_ingest_batch_mb", _clamp(cfg_ingest.get("batch_mb"), 0.1, 500.0, ingest_dave.MAX_BATCH_BYTES / 2**20))
        _init("w_ingest_sleep", _clamp(cfg_ingest.get("sleep"), 0.0, 120.0, float(ingest_dave.SLEEP_BETWEEN)))
        _init("w_ingest_max_docs", _clamp(cfg_ingest.get("max_docs"), 0, 10**7, 0))

        with st.expander("Batch settings"):
            b1, b2, b3 = st.columns(3)
            batch_docs = b1.number_input("Max docs per batch", 1, 200, key="w_ingest_batch_docs",
                                      help="A batch is closed when it reaches this many documents (or the size limit).")
            batch_mb = b2.number_input("Max MB per batch", 0.1, 500.0, step=0.1, key="w_ingest_batch_mb",
                                    help="A batch is closed when its payload reaches this size (or the document limit). Lower it if the server rejects uploads.")
            sleep_s = b3.number_input("Pause between batches (s)", 0.0, 120.0, step=1.0, key="w_ingest_sleep",
                                    help="Wait after each batch so the DAVE server is not overloaded.")
        max_docs = st.number_input(
            "Max documents per benchmark (0 = all)", min_value=0, step=50, key="w_ingest_max_docs",
            help="A random subset is uploaded. Documents already uploaded count towards the limit and are kept. "
                 "Questions whose source document is missing can't be answered.",
        )
        _persist({"ingest": {
            "benchmarks": sel_bench, "batch_docs": int(batch_docs), "batch_mb": float(batch_mb),
            "sleep": float(sleep_s), "max_docs": int(max_docs),
        }})

        ingest_running = jobs.is_running("ingest")
        blocked = ingest_running or not sel_bench or not dave_ready
        _init("recreate_round", 0)

        def _launch(label, extra, bench_args):
            env = {**os.environ,
                   "DAVE_BASE_URL": dave_now["base_url"], "DAVE_BASE_PATH": dave_now["base_path"],
                   "DAVE_USERNAME": dave_now["username"], "DAVE_PASSWORD": dave_now["password"]}
            args = [sys.executable, "-u", str(ROOT_DIR / "ingest_dave.py"), "--benchmarks", *sel_bench, *bench_args, *extra]
            return jobs.start("ingest", args, label, env=env, cwd=str(ROOT_DIR))

        upload_args = ["--batch-docs", str(int(batch_docs)), "--batch-mb", str(float(batch_mb)),
                       "--sleep", str(float(sleep_s)), "--max-docs", str(int(max_docs))]

        bt1, bt2, _ = st.columns([1, 1, 3])
        if bt1.button("Start DAVE ingestion", type="primary", disabled=blocked, key="w_ingest_start"):
            _launch("DAVE ingestion", [], upload_args)
            st.rerun()
        if bt2.button("Verify index", disabled=blocked, key="w_ingest_verify"):
            _launch("DAVE index verification", ["--verify"], [])
            st.rerun()

        with st.expander("Recreate collections"):
            st.caption("Creates new collections and re-uploads everything. The old ones stay on DAVE, unused: delete them there by hand.")
            if sel_bench:
                st.write("Selected: " + ", ".join(
                    f"{n} (has collection)" if n in collections else n for n in sel_bench))
            confirm_key = f"w_recreate_confirm_{st.session_state.recreate_round}"
            confirmed = st.checkbox("I understand, create new collections and re-upload", key=confirm_key,
                                  help="Required to enable the button. Resets after each launch.")
            if st.button("Create new and re-upload", disabled=blocked or not confirmed, key="w_ingest_recreate"):
                _launch("DAVE re-creation", ["--recreate"], upload_args)
                st.session_state.recreate_round += 1  # the checkbox comes back unticked
                st.rerun()

        if not dave_ready:
            st.caption("⚠️ Fill in URL, username and password to enable ingestion.")
        _job_panel("ingest")

        st.markdown("**Retrieval test**")
        if collections:
            r1, r2 = st.columns([1, 3])
            test_col = r1.selectbox("Collection", list(collections), key="w_retr_collection",
                                    help="DAVE collection to query (from .dave_collections.json).")
            test_query = r2.text_input("Query", key="w_retr_query",
                                       help="Test question: shows what DAVE retrieves, without running a full evaluation.")
            if st.button("Retrieve", key="w_retr_go", disabled=not (dave_ready and test_query.strip())):
                try:
                    found = DaveClient(**dave_now).retrieve_chunks(test_query, collections[test_col])
                    if found:
                        st.write(f"{len(found)} documents")
                        text = "\n".join(c.get("text", "") for c in found[0].get("chunks", []))
                        st.code(text[:600], language=None)
                    else:
                        st.warning("No documents: collection empty or DAVE still indexing.")
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
        else:
            st.caption("No collections yet: run an ingestion first.")

    with setup_rag:
        st.caption("For any RAG that exposes the POST /ask and /ingest routes described in the API Guide tab. "
                   "To evaluate it, choose **Custom RAG (HTTP)** in the sidebar.")

        # 1. connection
        st.subheader("Connection")
        st.session_state.rag_url_setup = st.session_state.rag_url
        st.text_input(
            "Base URL", key="rag_url_setup", on_change=_sync_rag_url, args=("rag_url_setup",),
            help="The tool calls <Base URL>/ask and <Base URL>/ingest. From the Docker container, a RAG running on "
                 "the host is http://host.docker.internal:PORT, not localhost.")
        rag_url = ingest_http.normalize_url(st.session_state.rag_url)
        c1, c2 = st.columns([1, 3])
        check_bench = c2.selectbox("Benchmark used for the test question", ingest_http.UPLOADABLE + ["OHRBench"],
                                   index=2, key="w_rag_check_bench", label_visibility="collapsed")
        if c1.button("Test endpoint", key="w_rag_check", disabled=not rag_url,
                     help="Sends one sample question to <Base URL>/ask and checks the reply."):
            with st.spinner("Waiting for your RAG..."):
                checks = api_guide.check_endpoint(rag_url, answerable.COLLECTION_SOURCE.get(check_bench, check_bench))
            for level, message in checks:
                {"ok": st.success, "info": st.info, "warning": st.warning, "error": st.error}[level](message[:400])

        st.divider()

        # 2. corpus upload
        st.subheader("Corpus Upload")
        rag_uploaded = ingest_http.uploaded_names(rag_url)
        rag_sources = {name: ingest_dave.find_source(name) for name in ingest_http.UPLOADABLE}

        def _rag_answerable(name):
            found = answerable.counts(ingest_http.SOURCE_OF[name], rag_uploaded) if name in rag_uploaded else None
            return f"{found[0]} / {found[1]}" if found else "-"

        st.dataframe(pd.DataFrame([
            {
                "Benchmark": name,
                "Corpus source": str(src.relative_to(ROOT_DIR)) if src else "missing",
                "Docs uploaded to this URL": len(rag_uploaded.get(name, [])),
                "Answerable queries": _rag_answerable(name),
            }
            for name, src in rag_sources.items()
        ]), hide_index=True, width="stretch")
        st.caption("OHRBench is not listed: its documents travel with each question. A benchmark that was never "
                   "uploaded from here is not filtered (the evaluation assumes your RAG is already indexed).")

        rag_available = [n for n, src in rag_sources.items() if src]
        cfg_rag = cfg.get("ingest_rag", {})
        rag_default = [n for n in rag_available if n not in ingest_dave.DEFAULT_BENCHMARKS_EXCLUDED]
        _init("w_rag_benchmarks", [n for n in cfg_rag.get("benchmarks", rag_default) if n in rag_available])
        st.session_state.w_rag_benchmarks = [n for n in st.session_state.w_rag_benchmarks if n in rag_available]
        rag_sel = st.multiselect(
            "Benchmarks to upload", rag_available, key="w_rag_benchmarks",
            help="Corpora to send to POST <Base URL>/ingest (only those found on disk). MTRAG is off by default "
                 "because its corpus is very large.")

        _init("w_rag_batch_docs", _clamp(cfg_rag.get("batch_docs"), 1, 5000, 100))
        _init("w_rag_batch_mb", _clamp(cfg_rag.get("batch_mb"), 0.1, 500.0, 5.0))
        _init("w_rag_sleep", _clamp(cfg_rag.get("sleep"), 0.0, 120.0, 0.0))
        _init("w_rag_max_docs", _clamp(cfg_rag.get("max_docs"), 0, 10**7, 0))
        with st.expander("Batch settings"):
            g1, g2, g3 = st.columns(3)
            rag_batch_docs = g1.number_input("Max docs per request", 1, 5000, key="w_rag_batch_docs",
                                             help="A request is closed when it reaches this many documents (or the size limit).")
            rag_batch_mb = g2.number_input("Max MB per request", 0.1, 500.0, step=0.1, key="w_rag_batch_mb",
                                           help="A request is closed when its payload reaches this size (or the document limit).")
            rag_sleep = g3.number_input("Pause between requests (s)", 0.0, 120.0, step=1.0, key="w_rag_sleep",
                                        help="Wait after each request so your RAG is not overloaded.")
        rag_max_docs = st.number_input(
            "Max documents per benchmark (0 = all)", min_value=0, step=50, key="w_rag_max_docs",
            help="A random subset is uploaded. Questions whose source document is not in the subset can't be "
                 "answered: the evaluation skips them (see the Answerable queries column).")
        _persist({"ingest_rag": {
            "benchmarks": rag_sel, "batch_docs": int(rag_batch_docs), "batch_mb": float(rag_batch_mb),
            "sleep": float(rag_sleep), "max_docs": int(rag_max_docs),
        }})

        rag_running = jobs.is_running("ingest_rag")
        u1, u2, _ = st.columns([1, 1, 3])
        if u1.button("Start corpus upload", type="primary", key="w_rag_upload",
                     disabled=rag_running or not rag_sel or not rag_url,
                     help="Replaces what your RAG holds for the selected benchmarks (the first request of each "
                          "benchmark has batch.index 0)."):
            jobs.start("ingest_rag", [
                sys.executable, "-u", str(ROOT_DIR / "ingest_http.py"), "--url", rag_url, "--benchmarks", *rag_sel,
                "--batch-docs", str(int(rag_batch_docs)), "--batch-mb", str(float(rag_batch_mb)),
                "--sleep", str(float(rag_sleep)), "--max-docs", str(int(rag_max_docs)),
            ], "Corpus upload to the generic RAG", cwd=str(ROOT_DIR))
            st.rerun()
        if u2.button("Forget uploads", key="w_rag_forget", disabled=rag_running or not rag_uploaded,
                     help="Deletes only the local record of what was uploaded to this URL (your RAG is not touched). "
                          "The evaluation then stops filtering questions for it."):
            ingest_http.forget(rag_url)
            st.rerun()
        _job_panel("ingest_rag")

    st.divider()

    # shared: raw benchmarks and query files, used by both engines
    st.subheader("Benchmark Files")

    def _start_extract(names):
        scripts = [str(RUN_DIR / EXTRACT_SCRIPTS[BENCHMARK_MAPPING[n]]) for n in names]
        scripts.append(str(RUN_DIR / "build_gold_docs.py"))  # which documents each query needs
        code = ("import runpy, sys\n"
                "for p in sys.argv[1:]:\n"
                "    print('== ' + p, flush=True)\n"
                "    runpy.run_path(p, run_name='__main__')\n")
        return jobs.start("extract", [sys.executable, "-u", "-c", code, *scripts],
                          "Benchmark file extraction", cwd=str(ROOT_DIR))

    # the raw benchmarks are not in the repository: download them from the official sources
    if st.button("Download Benchmark", width='stretch', disabled=st.session_state.running,
                 help="Downloads the raw files of the 6 benchmarks (~1 GB), extracts them into Benchmarks/ and builds the query files in Run/"):
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
            bar.progress(1.0, text="Extracting files...")
            out_dir = extract_benchmarks_zip(
                zip_path, progress=lambda i, n: bar.progress(i / n, text=f"Extracting files: {i}/{n}"))
            zip_path.unlink()  # extracted: the archive is no longer needed
            bar.empty()
            _start_extract(list(BENCHMARK_MAPPING))
            st.session_state.download_msg = f"Downloaded and extracted into `{out_dir}`. Building the benchmark files..."
            st.rerun()
        except Exception as e:
            cancel_slot.empty()
            bar.empty()
            st.error(f"Benchmark download failed: {e}")
    elif st.session_state.pop("download_cancelled", False):
        st.info("Download cancelled.")
    if "download_msg" in st.session_state:
        st.success(st.session_state.pop("download_msg"))

    rows, missing_names = [], []
    for name, fname in BENCHMARK_MAPPING.items():
        path = RUN_DIR / fname
        if not path.exists():
            missing_names.append(name)
        rows.append({
            "Benchmark": name,
            "File in Run/": fname,
            "Size": f"{path.stat().st_size / 2**20:.1f} MB" if path.exists() else "missing",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    _init("w_extract_benchmarks", missing_names)
    st.session_state.w_extract_benchmarks = [
        n for n in st.session_state.w_extract_benchmarks if n in BENCHMARK_MAPPING]
    to_build = st.multiselect("Benchmarks to (re)build", list(BENCHMARK_MAPPING), key="w_extract_benchmarks",
                          help="Runs the Run/extract_*.py scripts that turn the raw downloaded benchmarks into the query files. Missing ones are preselected.")
    if st.button("Build benchmark files", disabled=jobs.is_running("extract") or not to_build,
                 key="w_extract_start"):
        _start_extract(to_build)
        st.rerun()
    _job_panel("extract")


# api guide tab - how a third-party RAG must be structured
with tab_guide:
    api_guide.render(default_url=rag_base_url)
