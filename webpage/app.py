import json
import math
import os
import random
from html import escape
import re
import sys
import time
import requests
import pandas as pd
import streamlit as st

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


from evaluator import run_evaluation, build_judge_llm, classify_query_verdict, FETCH_K
from taxonomy import metric_mean, metric_value, needs_reclassification
from persistence import save_evaluation, list_evaluations, load_evaluation, delete_evaluation, set_expert_report, EVAL_DIR
import api_guide
import corpus_sources
import answerable
import ingest_http
import cleanup
from benchmark_download import build_benchmarks_zip, extract_benchmarks_zip, is_downloaded
import jobs
from settings import load_config, save_config, dave_settings, DAVE_RETRIEVAL_METHODS

import dashboard as dash
import devmode  # DEV MODE
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

# Query Details shows this many queries at a time: every click rebuilds the whole page
QUERY_DETAILS_PAGE_SIZE = 20

# benchmark names mapping -> json files in Run folder
BENCHMARK_MAPPING = {
    "LegalBench-RAG (Contract Precision)": "Tutto_LEGALBENCH.jsonl",
    "MultiHop-RAG (Multi-Document Synthesis)": "Tutto_MULTIHOP.jsonl",
    "CRAG (HTML Parsing & Traps)": "Tutto_CRAG_DAVE.jsonl",
    "MTRAG (Multi-Turn Conversations)": "Tutto_MTRAG_DAVE.jsonl",
    "OHRBench (OCR Noise / Scans)": "Tutto_OHRBench_DAVE.jsonl",
    "SQuAD-it (Italian Q&A on Wikipedia)": "Tutto_SQUAD.jsonl"
}

# built-in benchmarks plus the datasets found in Custom/ (the latter are not part of the Catalogue/download flow)
EVAL_BENCHMARKS = {**BENCHMARK_MAPPING,
                   **{name: str(path) for name, path in corpus_sources.custom_benchmarks().items()}}

# benchmark file -> Run/ script that builds it
EXTRACT_SCRIPTS = {
    "Tutto_LEGALBENCH.jsonl": "extract_legalbenchmark.py",
    "Tutto_MULTIHOP.jsonl": "extract_multihop.py",
    "Tutto_CRAG_DAVE.jsonl": "extract_crag.py",
    "Tutto_MTRAG_DAVE.jsonl": "extract_mtrag.py",
    "Tutto_OHRBench_DAVE.jsonl": "extract_ohrbench.py",
    "Tutto_SQUAD.jsonl": "extract_squad.py",
}

BENCHMARK_DESCRIPTIONS = {
    "LegalBench-RAG (Contract Precision)": "Real legal contracts (691). Tests extracting precise contract clauses without hallucinating or paraphrasing.",
    "MultiHop-RAG (Multi-Document Synthesis)": "Multi-document synthesis over 609 news articles. Needs reasoning and aggregating information spread across 2 to 4 documents.",
    "CRAG (HTML Parsing & Traps)": "Web HTML and noise (Meta CRAG). Measures resilience to unstructured web pages, noisy fragments and questions with false premises.",
    "MTRAG (Multi-Turn Conversations)": "Multi-turn dialogues. Checks handling of the conversation memory and of context-dependent references.",
    "OHRBench (OCR Noise / Scans)": "Scans and OCR noise. Business and academic documents with corrupted tables, degraded characters and math formulas.",
    "SQuAD-it (Italian Q&A on Wikipedia)": "Italian reading comprehension (2,007 Wikipedia paragraphs). Tests extraction on Italian texts and queries.",
}


def _benchmark_ready(collection):
    """What the Catalogue and Setup call "downloaded": the raw files are in Benchmarks/, the query file
    is built in Run/ and the corpus is found. `collection` is the name used by ingest_dave (e.g. "SQuAD")."""
    import ingest_dave
    if collection in corpus_sources.custom_benchmarks():  # Custom/ datasets: nothing to download, just the folder
        return ingest_dave.find_source(collection) is not None
    download_name = next(k for k, v in BENCHMARK_SHORT.items() if v == collection)
    query_file = next(f for n, f in BENCHMARK_MAPPING.items() if n.startswith(download_name))
    return (is_downloaded(download_name) and (RUN_DIR / query_file).exists()
            and ingest_dave.find_source(collection) is not None)


_COUNTS_FILE = Path(os.environ.get("STATE_DIR") or Path(__file__).resolve().parents[1]) / ".catalogue_counts.json"


# Catalogue tags: benchmark -> (domain, tasks)
BENCHMARK_TAGS = {
    "LegalBench-RAG (Contract Precision)": ("Legal", ["Retrieval Evaluation"]),
    "MultiHop-RAG (Multi-Document Synthesis)": ("News", ["Multi-hop QA"]),
    "CRAG (HTML Parsing & Traps)": ("Multi-domain", ["Time-sensitive QA", "Multi-hop QA", "False Premise Detection"]),
    "MTRAG (Multi-Turn Conversations)": ("Multi-domain", ["Multi-turn QA", "Negative Rejection"]),
    "OHRBench (OCR Noise / Scans)": ("Multi-domain", ["OCR Robustness", "Table QA"]),
    "SQuAD-it (Italian Q&A on Wikipedia)": ("Open-domain", ["Single-hop QA"]),
}


def _count_queries(path):
    """Number of queries (lines) of a Run/Tutto_*.jsonl file. The files are huge (CRAG is several GB),
    so the count is read once and kept in a small file until the size or date of the file changes."""
    stat = path.stat()
    signature = [stat.st_size, int(stat.st_mtime)]
    try:
        counts = json.loads(_COUNTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        counts = {}
    entry = counts.get(path.name)
    if entry and entry[:2] == signature:
        return entry[2]
    lines, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    lines += last != b"\n"  # last line without a trailing newline
    counts[path.name] = [*signature, lines]
    try:
        _COUNTS_FILE.write_text(json.dumps(counts), encoding="utf-8")
    except OSError:
        pass  # read-only folder: the count is just recomputed next time
    return lines


BENCHMARK_SHORT = {
    "LegalBench-RAG": "LegalBench",
    "MultiHop-RAG": "MultiHop",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "OHRBench": "OHRBench",
    "SQuAD-it": "SQuAD",
}

@st.cache_data(max_entries=1, show_spinner=False)
def _cached_evaluations(signature):
    return list_evaluations()


def _first_query_page():
    """A new filter on Query Details starts again from its first page."""
    st.session_state.query_details_page = 1


def _benchmark_filter(df, key):
    """Rows of a results table whose Benchmark is picked in a multiselect: all of them when none is picked."""
    if df["Benchmark"].nunique() < 2:
        return df
    col, _ = st.columns([1, 2])
    with col:
        picked = st.multiselect("Filter by Benchmark", sorted(df["Benchmark"].unique()), placeholder="All benchmarks", key=key)
    return df[df["Benchmark"].isin(picked)] if picked else df


def saved_evaluations():
    """Saved evaluations, read from disk again only when a file in valutazioni/ is added, removed or changed."""
    signature = tuple((f.name, f.stat().st_mtime_ns) for f in sorted(EVAL_DIR.glob("*.json")))
    return _cached_evaluations(signature)


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

# scores shown in the verdict card of a query: (label, report key)
VERDICT_METRICS = [("Precision", "retrieval_voto"), ("Recall", "recall_voto"), ("Faithfulness", "faithfulness_voto"),
                   ("Correctness", "correctness_voto"), ("Relevancy", "relevancy_voto")]
_COLOR_HEX = {"green": "#00E676", "yellow": "#FFC400", "red": "#FF1744", "grey": "#a2a2c0"}

def _verdict_card(rep: dict, best: dict = None) -> str:
    """Verdict card of a query with its scores. best: top score of each metric among compared evaluations."""
    metrics = []
    for label, key in VERDICT_METRICS:
        v = metric_value(rep, key)
        is_best = bool(best) and v is not None and best.get(key) == round(v, 2)
        metrics.append((label, _fmt_score(v), _COLOR_HEX[_value_color(v)], is_best))
    return dash.get_verdict_html(rep.get('verdict', 'wrong'), rep.get('verdict_label', '❌ Wrong'),
                                 rep.get('error_cause_label') or rep.get('error_pattern_label', 'Unknown'),
                                 rep.get('error_reason', ''), metrics)

def _answer_card(rep: dict) -> str:
    if rep.get('rag_error'):
        # the RAG failed: there is no answer, show the infrastructure error
        return dash.get_text_card_html("RAG Error", rep['rag_error'], "error")
    return dash.get_text_card_html("RAG Answer", rep.get('risposta_rag') or "N/A", "answer")

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
    """Benchmarks ready to evaluate with DAVE: collection created and documents uploaded."""
    import ingest_dave
    collections, cache = ingest_dave.load_collections(), ingest_dave.load_doc_cache()
    ready = []
    for name in BENCHMARK_MAPPING:
        short = next((v for k, v in BENCHMARK_SHORT.items() if name.startswith(k)), name)
        if short in collections and cache.get(short):
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
    devmode.sidebar_switch()  # DEV MODE
    st.markdown("### ⚙️ Run Settings")
    eval_name_input = st.text_input(
        "Evaluation Name *", placeholder="e.g. test_bm25_v1", key="eval_name_input",
        help="Label used to save the run; it appears in the Explore tab.")
    eval_name = eval_name_input.strip() if eval_name_input else ""
    if not eval_name:
        st.caption("⚠️ Evaluation name is required to start")

    # sidebar settings are kept in config.json ("ui") so a page reload does not lose them
    ui_saved = st.session_state.cfg.get("ui", {})
    ENGINES = ["DAVE (Elasticsearch)", "Custom RAG (HTTP)"]
    _init("engine_select", ui_saved.get("engine") if ui_saved.get("engine") in ENGINES else ENGINES[0])
    devmode.force_engine(ENGINES[0])  # DEV MODE: behind the scenes the engine is DAVE
    motore_rag = st.radio(
        "RAG Engine", ENGINES, key="engine_select",
        help="DAVE: the DAVE server configured in the Setup tab. Custom RAG: your own RAG, reached over HTTP "
             "with the POST /ask and /ingest routes described in the API Guide tab.")

    if "DAVE" in motore_rag:
        _init("w_dave_retrieval_method", dave_settings(st.session_state.cfg)["retrieval_method"])
        if st.session_state.w_dave_retrieval_method not in DAVE_RETRIEVAL_METHODS:
            st.session_state.w_dave_retrieval_method = "full"
        if devmode.active() and devmode.retrieval_from_url():  # DEV MODE: the URL picks the retrieval type
            st.session_state.w_dave_retrieval_method = devmode.retrieval_from_url()
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
                api_guide.show_checks(api_guide.check_endpoint(st.session_state.rag_url))
    rag_base_url = st.session_state.rag_url

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

    st.markdown("### Benchmark", help="Which benchmark query sets to evaluate.")
    # DAVE: only the benchmarks already indexed; custom RAG: all (the corpus is uploaded to /ingest)
    default_benchmarks = _indexed_benchmarks() if "DAVE" in motore_rag else list(EVAL_BENCHMARKS)
    _init("bench_select", default_benchmarks if ui_saved.get("benchmarks") is None else ui_saved["benchmarks"])
    if st.session_state.get("engine_prev", motore_rag) != motore_rag:
        st.session_state.bench_select = default_benchmarks   # switching engine resets the selection
    st.session_state.engine_prev = motore_rag
    st.session_state.bench_select = [n for n in st.session_state.bench_select if n in EVAL_BENCHMARKS]
    selected_names = st.multiselect(
        "Benchmarks",
        options=list(EVAL_BENCHMARKS.keys()),
        key="bench_select",
        label_visibility="collapsed",
        help="Which benchmark query sets to evaluate. With DAVE the default is the ones already indexed; "
             "with a custom RAG, all of them.",
    )

    selected_files = [EVAL_BENCHMARKS[name] for name in selected_names]
    st.caption(f"{len(selected_names)} of {len(EVAL_BENCHMARKS)} selected")

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
    saved_evals = saved_evaluations()
    if saved_evals:
        last_eval = saved_evals[0]
        _open_evaluation(load_evaluation(last_eval["filepath"]))

# main tabs
# the Catalogue comes first, but the page opens on the Evaluation tab
tab_catalogue, tab_setup, tab_evaluation, tab_explore, tab_guide = st.tabs(
    ["📚 Catalogue", "🛠️ Setup", "🔬 Evaluation", "📊 Explore", "📖 API Guide"], default="🔬 Evaluation")

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
            saved_evals = saved_evaluations()
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
        files_to_run = st.session_state.get("selected_files", list(EVAL_BENCHMARKS.values()))
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
        st.markdown(dash.get_metric_grid_html([
            (label, val, _value_color(val))
            for label, val in [("Answer Correctness", avg_c), ("Faithfulness", avg_f),
                               ("Context Precision", avg_r), ("Context Recall", avg_recall)]
        ]), unsafe_allow_html=True)

        # Warning on failed queries and truncated contexts
        ragas_stats = reports[0].get("_ragas_stats", {}) if reports else {}
        # not evaluated queries (RAG errors, no Correctness) are already counted in the verdict summary:
        # here only judged queries that miss some other metric, which leave that metric's average
        failed_queries = sum(1 for r in reports if r.get("ragas_fallita", False) and r.get("verdict") != "not_evaluated")
        truncated_contexts = ragas_stats.get("contesti_troncati", 0)
        if failed_queries > 0 or truncated_contexts > 0:
            msg_parts = []
            if truncated_contexts > 0:
                msg_parts.append(f"**{truncated_contexts}** truncated contexts (exceeded model limit)")
            if failed_queries > 0:
                msg_parts.append(f"**{failed_queries}/{len(reports)}** judged queries with some RAGAS metric not computed "
                                 "(excluded from the average of that metric)")
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
            queries = timings.get("query", {})
            total = timings.get("totale", 0)
            with st.container(key="timings"):  # keyed only to give it a top margin (style.css)
                with st.expander(f"⏱️ Timings — total {_fmt(total)}"):
                    rows = []
                    for src in sorted(queries):
                        n_q = sum(1 for r in reports if r.get("source") == src)
                        q = queries.get(src, 0)
                        rows.append({"Phase": f"RAG queries — {_short_source(src)}",
                                     "Time": _fmt(q), "Per query": f"{q / n_q if n_q else 0:.1f}s"})
                    # the RAG answered these queries, so RAGAS judged them (5 metrics each)
                    judged = sum(1 for r in reports if not r.get("rag_error"))
                    ragas = timings.get("ragas", 0)
                    rows.append({"Phase": "RAGAS (judge)", "Time": _fmt(ragas),
                                 "Per query": f"{ragas / judged if judged else 0:.1f}s"})
                    # the generic HTTP RAG receives the corpus at the start of the run; DAVE is loaded from Setup
                    ingestion = sum(timings.get("ingestion", {}).values())
                    if ingestion:
                        rows.append({"Phase": "Ingestion", "Time": _fmt(ingestion), "Per query": "—"})
                    other = total - sum(queries.values()) - ragas - ingestion
                    rows.append({"Phase": "Other (setup)", "Time": _fmt(max(other, 0)), "Per query": "—"})
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
        display_df = _benchmark_filter(display_df.rename(columns=column_rename), key="results_benchmark_filter")

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
        
        col_f1, col_f2, col_page = st.columns([2, 2, 1])
        with col_f1:
            filter_verdict = st.multiselect("Filter by Verdict", options=all_verdicts, default=[], placeholder="All verdicts",
                                            on_change=_first_query_page)
        with col_f2:
            filter_pattern = st.multiselect("Filter by Error Pattern", options=all_patterns, default=[], placeholder="All patterns",
                                            on_change=_first_query_page)

        filtered_reports = reports
        if filter_verdict:
            filtered_reports = [r for r in filtered_reports if r.get('verdict_label') in filter_verdict]
        if filter_pattern:
            filtered_reports = [r for r in filtered_reports if r.get('error_pattern_label') in filter_pattern]

        pages = max(1, math.ceil(len(filtered_reports) / QUERY_DETAILS_PAGE_SIZE))
        if st.session_state.get("query_details_page", 1) > pages:
            st.session_state.query_details_page = pages  # fewer queries than before (another evaluation opened)
        with col_page:
            page = st.number_input("Page", min_value=1, max_value=pages, key="query_details_page")
        first = (page - 1) * QUERY_DETAILS_PAGE_SIZE
        page_reports = filtered_reports[first:first + QUERY_DETAILS_PAGE_SIZE]

        if not filtered_reports:
            st.info("No query matches the selected filters.")
        else:
            st.caption(f"Queries {first + 1}–{first + len(page_reports)} of {len(filtered_reports)} · page {page} of {pages}")

        for rep in page_reports:
            corr = metric_value(rep, 'correctness_voto')
            v_label = rep.get('verdict_label', '')
            pattern = rep.get('error_pattern_label', '')
            cause = rep.get('error_cause_label') or pattern
            pattern_suffix = f"  |  cause: {cause}" if rep.get('verdict') != 'correct' and cause else ""
            label = f"{rep['query_id']}  |  {rep['source']}  |  {v_label}{pattern_suffix}  |  Correctness: {_fmt_score(corr)}"
            with st.expander(label):
                st.markdown(dash.get_text_card_html("Question", rep['domanda'], "question"), unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(_answer_card(rep), unsafe_allow_html=True)
                with c2:
                    st.markdown(dash.get_text_card_html("Gold Answer", rep['gold'], "gold"), unsafe_allow_html=True)

                # verdict, error pattern and RAGAS scores
                st.markdown(_verdict_card(rep), unsafe_allow_html=True)

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

# explore tab - evaluation history and comparison
with tab_explore:
    import plotly.graph_objects as go

    if "explore_msg" in st.session_state:
        st.success(st.session_state.explore_msg)
        del st.session_state.explore_msg

    saved_evals = saved_evaluations()

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

            # benchmarks that actually have results in the run; older files fall back to the query files used
            run_benchmarks = list(ev.get("per_benchmark") or {}) or [
                next((n for n, f in EVAL_BENCHMARKS.items() if f == used), used).split(" (")[0]
                for used in ev.get("benchmarks_used", [])]
            benchmarks_text = ", ".join(run_benchmarks) or "—"

            # html of the card for each evaluation
            card_html = (
                f'<div class="eval-card">'
                f'  <div class="eval-card-header">'
                f'    <span class="eval-card-name">{escape(ev["name"])}</span>'
                f'    <span class="eval-card-ts">{ts}</span>'
                f'  </div>'
                f'  <div class="eval-card-body">'
                f'    <span class="eval-card-detail">RAG: <strong>{escape(str(ev.get("rag_engine", "Unknown")))}</strong></span>'
                f'    <span class="eval-card-detail">Benchmarks: <strong>{escape(benchmarks_text)}</strong></span>'
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
       # st.markdown('<div class="section-title">Evaluation Comparison</div>', unsafe_allow_html=True)

        if len(selected_for_compare) == 0:
            st.info("Select one or more evaluations to compare them.")
        else:
            metric_keys = ["answer_correctness", "faithfulness", "context_precision", "context_recall"]
            metric_labels = ["Answer Correctness", "Faithfulness", "Context Precision", "Context Recall"]

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

            st.markdown('<div class="section-title">Metrics Comparison</div>', unsafe_allow_html=True)

            # Overall or a single benchmark: the same metrics, from the averages saved with each evaluation
            benchmarks = sorted({bm for ev in selected_for_compare for bm in ev.get("per_benchmark", {})})
            views = {"Overall": None} | {_short_source(bm): bm for bm in benchmarks}
            view = "Overall"
            if len(benchmarks) > 1:
                view = st.segmented_control("View", list(views), default="Overall", required=True,
                                            key="explore_metrics_view", label_visibility="collapsed")

            # one horizontal bar per evaluation for each metric, with its value at the end of the bar:
            # the techniques often differ by a few hundredths, too little to see from the bar length alone
            palette = ["#6c5ce7", "#00E676", "#FFC400", "#FF1744", "#00B0FF", "#FF6D00", "#D500F9", "#76FF03"]
            fig = go.Figure()
            for i, ev in enumerate(selected_for_compare):
                agg = ev.get("aggregate", {}) if views[view] is None else ev.get("per_benchmark", {}).get(views[view], {})
                # an evaluation without this benchmark, or a metric not computed, has no bar
                values = [v if isinstance(v, (int, float)) else None for v in (agg.get(k) for k in metric_keys)]
                fig.add_trace(go.Bar(
                    y=metric_labels,
                    x=values,
                    orientation='h',
                    name=ev["name"],
                    marker_color=palette[i % len(palette)],
                    opacity=0.85,
                    text=["N/A" if v is None else f"{v:.2f}" for v in values],
                    textposition='outside',
                    cliponaxis=False,
                ))
            fig.update_layout(
                barmode='group',
                bargap=0.25,  # space between the metrics, so the thinner bars still read as groups
                height=100 + 18 * len(metric_keys) * len(selected_for_compare),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,30,0.6)",
                font=dict(family="Inter", color="#e0e0e0"),
                xaxis=dict(range=[0, 1.08], gridcolor="rgba(255,255,255,0.05)"),  # room for the value labels
                yaxis=dict(autorange="reversed"),  # metrics top to bottom in the order of metric_labels
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, width='stretch')

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
                cmp_df = _benchmark_filter(pd.DataFrame(cmp_table_rows), key="compare_benchmark_filter")
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

                        st.markdown(dash.get_text_card_html("Question", meta["domanda"], "question"), unsafe_allow_html=True)
                        st.markdown(dash.get_text_card_html("Gold Answer", meta["gold"], "gold"), unsafe_allow_html=True)

                        # side-by-side columns for each evaluation
                        eval_names_with_data = [ev["name"] for ev in selected_for_compare if ev["name"] in query_results_map.get(selected_qid, {})]
                        compared = [query_results_map[selected_qid][name] for name in eval_names_with_data]
                        # top score of each metric, marked in the verdict cards; equal scores mark nothing
                        best = {}
                        for _, key in VERDICT_METRICS:
                            scores = {round(v, 2) for v in (metric_value(r, key) for r in compared) if v is not None}
                            best[key] = max(scores) if len(scores) > 1 else None
                        cols = st.columns(len(eval_names_with_data))

                        for col_idx, eval_name in enumerate(eval_names_with_data):
                            res = compared[col_idx]

                            with cols[col_idx]:
                                st.markdown(f"#### 📋 {eval_name}")
                                # the answer first, so the answers of all evaluations start at the same height
                                st.markdown(_answer_card(res), unsafe_allow_html=True)
                                st.markdown(_verdict_card(res, best), unsafe_allow_html=True)

                                with st.expander("📄 Retrieved Documents"):
                                    st.text(res.get("docs", "No documents available"))

                else:
                    st.info("No shared queries found between the selected evaluations. Make sure the evaluations were run on overlapping datasets.")

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

                            compare_report = invoke_ollama(PROMPT_COMPARE, ollama_url, ollama_model, llm_api_key)
                        if compare_report:  # on failure invoke_ollama shows the error, a rerun would wipe it
                            st.session_state[compare_report_key] = compare_report
                            st.rerun()
                else:
                    st.markdown(_style_report(st.session_state[compare_report_key]))
                    if st.button("🔄 Regenerate Report", key="btn_regen_compare_report"):
                        del st.session_state[compare_report_key]
                        st.rerun()


# setup tab - DAVE connection, corpus ingestion, benchmark files
def _dave_env(dave_now):
    """Environment of the ingest_dave.py subprocess: the DAVE connection typed in Setup."""
    return {**os.environ,
            "DAVE_BASE_URL": dave_now["base_url"], "DAVE_BASE_PATH": dave_now["base_path"],
            "DAVE_USERNAME": dave_now["username"], "DAVE_PASSWORD": dave_now["password"]}


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
            log = "\n".join(snap["lines"][-30:])
            st.code(log.replace("DAVE", "RAG") if devmode.active() else log, language=None)  # DEV MODE


    st.fragment(render, run_every=2 if jobs.is_running(name) else None)()


with tab_catalogue:
    # raw benchmarks and query files, used by both engines
    st.caption("Download the benchmarks here. Only the downloaded ones can be indexed into the RAG from the Setup tab.")

    def _start_extract(names):
        scripts = [str(RUN_DIR / EXTRACT_SCRIPTS[BENCHMARK_MAPPING[n]]) for n in names]
        scripts.append(str(RUN_DIR / "build_gold_docs.py"))  # which documents each query needs
        code = ("import runpy, sys\n"
                "for p in sys.argv[1:]:\n"
                "    print('== ' + p, flush=True)\n"
                "    runpy.run_path(p, run_name='__main__')\n")
        return jobs.start("extract", [sys.executable, "-u", "-c", code, *scripts],
                          "Benchmark file extraction", cwd=str(ROOT_DIR))

    import ingest_dave

    # a fragment: ticking a box or typing in the search reruns only this part, not the whole app
    @st.fragment
    def _catalogue():
        # every entry shown in the table (the placeholders only exist in DEV MODE): name, description, domain, tasks
        placeholders = devmode.catalogue_placeholders(BENCHMARK_SHORT)
        all_tags = [BENCHMARK_TAGS.get(n, ("", [])) for n in BENCHMARK_MAPPING] + \
                   [devmode.placeholder_tags(n) for n, _, _ in placeholders]
        f1, f2, f3 = st.columns([3, 2, 2])
        catalogue_search = f1.text_input(
            "Search", key="w_catalogue_search", placeholder="Search a benchmark by name or description",
            label_visibility="collapsed")
        terms = catalogue_search.lower().split()
        domain_counts = Counter(d for d, _ in all_tags if d)
        task_counts = Counter(t for _, tasks in all_tags for t in tasks)
        domain_filter = f2.multiselect("Domain", sorted(domain_counts), key="w_catalogue_domain",
                                       format_func=lambda d: f"{d} ({domain_counts[d]})",
                                       placeholder="Domain", label_visibility="collapsed")
        task_filter = f3.multiselect("Task", sorted(task_counts), key="w_catalogue_task",
                                     format_func=lambda t: f"{t} ({task_counts[t]})",
                                     placeholder="Task", label_visibility="collapsed")

        def _matches(name, description, domain, tasks):
            """The search looks at name and description only (never at the content); the selects at the tags."""
            return (all(term in f"{name} {description}".lower() for term in terms)
                    and (not domain_filter or domain in domain_filter)
                    and (not task_filter or any(t in task_filter for t in tasks)))

        def _tag_cells(cells, domain, tasks):
            cells[3].markdown(f":violet-badge[{domain}]" if domain else "-")
            cells[4].markdown(" ".join(f":gray-badge[{t}]" for t in tasks) if tasks else "-")

        def _download_name(name):
            """Name of the benchmark in benchmark_download.BENCHMARKS."""
            return next(k for k in BENCHMARK_SHORT if name.startswith(k))

        def _short(name):
            return BENCHMARK_SHORT[_download_name(name)]

        def _downloaded(name):
            """Raw data, query file and corpus all there: what the Setup tab needs to index it."""
            return _benchmark_ready(_short(name))

        # a plain table of page elements (not a canvas grid): every row always shows its own checkbox
        widths = [0.5, 2.0, 3.6, 1.4, 2.8, 1.5, 1.0]
        header = st.columns(widths)
        for col, label in zip(header, ["", "Benchmark", "Description", "Domain", "Tasks", "Status", "Queries"]):
            if label:  # the checkbox column has no title
                col.markdown(f"**{label}**")
        selected, shown = [], 0
        for name, fname in BENCHMARK_MAPPING.items():
            path = RUN_DIR / fname
            description = BENCHMARK_DESCRIPTIONS.get(name, "")
            domain, tasks = BENCHMARK_TAGS.get(name, ("", []))
            if not _matches(name, description, domain, tasks):
                continue
            shown += 1
            cells = st.columns(widths, vertical_alignment="center")
            if cells[0].checkbox(f"Select {name}", key=f"w_catalogue_sel_{_short(name)}", label_visibility="collapsed"):
                selected.append(name)
            cells[1].markdown(f"**{name}**")
            cells[2].write(description)
            _tag_cells(cells, domain, tasks)
            cells[5].write("✅ Downloaded" if _downloaded(name) else "Not downloaded")
            cells[6].write(f"{_count_queries(path):,}" if path.exists() else "-")
        # DEV MODE: placeholder datasets, they only look downloaded (see devmode.py)
        for index, (name, description, count) in enumerate(placeholders):
            domain, tasks = devmode.placeholder_tags(name)
            if not _matches(name, description, domain, tasks):
                continue
            shown += 1
            cells = st.columns(widths, vertical_alignment="center")
            if cells[0].checkbox(f"Select {name}", key=f"w_catalogue_sel_placeholder_{index}", label_visibility="collapsed"):
                selected.append(name)
            cells[1].markdown(f"**{name}**")
            cells[2].write(description)
            _tag_cells(cells, domain, tasks)
            cells[5].write("✅ Downloaded" if devmode.is_marked_downloaded(name) else "Not downloaded")
            cells[6].write(count or "-")
        if not shown:
            st.info("No benchmark matches the search.")

        # the raw benchmarks are not in the repository: download the selected ones from the official sources
        def _mark_placeholders():  # DEV MODE: fake download, a short delay and then they look downloaded
            names = [name for i, (name, _, _) in enumerate(devmode.catalogue_placeholders(BENCHMARK_SHORT))
                     if st.session_state.get(f"w_catalogue_sel_placeholder_{i}")]
            if names:
                with st.spinner("Downloading " + ", ".join(names) + "..."):
                    time.sleep(random.uniform(2.5, 4.5))
                devmode.mark_downloaded(names)

        d1, _ = st.columns([1, 4])
        if d1.button("Download selected", type="primary", disabled=st.session_state.running or not selected,
                     key="w_catalogue_download",
                     help="Downloads the raw files of the selected benchmarks, extracts them into Benchmarks/ and "
                          "builds their query files in Run/ automatically. Tick the benchmarks in the table first."):
            real = [n for n in selected if n in BENCHMARK_MAPPING]
            _mark_placeholders()
            if not real:  # only DEV MODE placeholders: nothing to fetch
                st.session_state.download_msg = "Downloaded: " + ", ".join(selected)
                st.rerun()  # the whole page, not just this fragment: the Setup tab lists them too
            if real:
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
                    zip_path = build_benchmarks_zip(progress=_show_progress, benchmarks=[_download_name(n) for n in real])
                    cancel_slot.empty()
                    bar.progress(1.0, text="Extracting files...")
                    out_dir = extract_benchmarks_zip(
                        zip_path, progress=lambda i, n: bar.progress(i / n, text=f"Extracting files: {i}/{n}"))
                    zip_path.unlink()  # extracted: the archive is no longer needed
                    bar.empty()
                    _start_extract(real)
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

        # start over: delete every local file of the datasets (re-download with "Download selected")
        busy = any(jobs.is_running(job) for job in ("extract", "ingest", "ingest_rag")) or st.session_state.running
        _init("catalogue_clear_round", 0)
        with st.expander("Clear downloaded files"):
            st.caption(
                "⚠️ Deletes all the local files of the datasets: the downloaded benchmarks (Benchmarks/) and the query "
                "files and corpora prepared from them (Run/). Download them again with **Download selected**. "
                "What was indexed is not touched: clear it from the Setup tab.")
            confirmed = st.checkbox("I understand, delete the files", key=f"catalogue_clear_confirm_{st.session_state.catalogue_clear_round}",
                                    help="Required to enable the button. Resets after each use.")
            if st.button("Clear downloaded files", disabled=busy or not confirmed, key="catalogue_clear"):
                deleted, freed, errors = cleanup.clear_downloads()
                for key in [k for k in st.session_state if k.startswith("w_catalogue_sel_")] + ["dev_fake_downloaded"]:
                    st.session_state.pop(key, None)  # selections, and the DEV MODE "downloaded" placeholders
                st.session_state.catalogue_clear_round += 1  # the checkbox comes back unticked
                st.session_state.download_msg = (
                    f"Deleted {deleted} download/prepared items ({freed / 2**30:.2f} GB)."
                    + (f" {len(errors)} could not be removed: {errors[0]}" if errors else ""))
                st.rerun()
            if busy:
                st.caption("Wait for the running download, extraction or upload to finish.")

    _catalogue()
    _job_panel("extract")


def _clear_index(key, systems, text, extra_keys=()):
    """Expander that forgets what was indexed in `systems` ("generic", "dave"). Only the records are removed:
    no downloaded file is touched and nothing is sent to the RAG or to DAVE."""
    busy = any(jobs.is_running(job) for job in ("ingest", "ingest_rag")) or st.session_state.running
    _init(f"{key}_round", 0)
    if f"{key}_msg" in st.session_state:
        st.success(st.session_state.pop(f"{key}_msg"))
    with st.expander("Clear index"):
        st.caption(text)
        confirmed = st.checkbox("I understand, clear the index", key=f"{key}_confirm_{st.session_state[f'{key}_round']}",
                                help="Required to enable the button. Resets after each use.")
        if st.button("Clear index", disabled=busy or not confirmed, key=key):
            removed, errors = cleanup.clear_index_records(systems)
            for extra in extra_keys:
                st.session_state.pop(extra, None)
            st.session_state[f"{key}_round"] += 1  # the checkbox comes back unticked
            st.session_state[f"{key}_msg"] = ("Index cleared." if removed else "Nothing was indexed.") + \
                (f" {len(errors)} could not be removed: {errors[0]}" if errors else "")
            st.rerun()
        if busy:
            st.caption("Wait for the running upload to finish.")


with tab_setup:
    import ingest_dave
    from dave_client import DaveClient

    cfg = st.session_state.cfg
    saved = dave_settings(cfg)

    if devmode.active():  # DEV MODE: no tabs, the DAVE part still runs inside a container that is only hidden
        setup_rag, setup_dave = st.container(), st.container(key="dave_setup_hidden")
    else:
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
        sources = {name: ingest_dave.find_source(name) if _benchmark_ready(name) else None
                   for name in ingest_dave.INGESTERS}

        def _answerable_label(name):
            source = next((k for k, v in answerable.COLLECTION.items() if v == name), None)
            found = answerable.counts(source) if source else None
            return f"{found[0]} / {found[1]}" if found else "-"
        st.dataframe(pd.DataFrame([
            {
                "Benchmark": name,
                "Corpus source": str(src.relative_to(ROOT_DIR)) if src else "missing",
                "Collection ID": collections.get(name, ""),
                "Docs indexed (local cache)": len(doc_cache.get(name, [])),
                "Answerable queries": _answerable_label(name),
            }
            for name, src in sources.items() if src  # only the downloaded ones (Catalogue tab)
        ]), hide_index=True, width="stretch")
        if not any(sources.values()):
            st.info("No benchmark downloaded yet: download them in the Catalogue tab.")

        available = [n for n, src in sources.items() if src]
        cfg_ingest = cfg.get("ingest", {})
        default_sel = list(available)
        _init("w_ingest_benchmarks", [n for n in cfg_ingest.get("benchmarks", default_sel) if n in available])
        st.session_state.w_ingest_benchmarks = [n for n in st.session_state.w_ingest_benchmarks if n in available]
        sel_bench = st.multiselect(
            "Benchmarks to upload", available, key="w_ingest_benchmarks",
            help="Corpora to send to DAVE (only those downloaded).",
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
            env = _dave_env(dave_now)
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

        _clear_index("w_clear_index_dave", ["dave"],
                     "⚠️ Forgets the collections, uploaded files and jobs recorded for DAVE. Nothing is deleted on DAVE: "
                     "the collections already created stay there, unused, and the next ingestion creates new ones. "
                     "The downloaded files are not touched.")

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
        if st.button("Test endpoint", key="w_rag_check", disabled=not rag_url,
                     help="Sends one probe question to <Base URL>/ask and checks the shape of the reply."):
            with st.spinner("Waiting for your RAG..."):
                # DEV MODE: the reply is always a healthy one
                checks = devmode.fake_check_endpoint(rag_url) if devmode.active() else api_guide.check_endpoint(rag_url)
            api_guide.show_checks(checks)

        st.divider()

        # 2. corpus upload
        st.subheader("Index corpus" if devmode.active() else "Corpus Upload")
        rag_uploaded = ingest_http.uploaded_names(rag_url)
        if devmode.active():  # DEV MODE: the documents really are in DAVE
            rag_uploaded = ingest_dave.load_doc_cache()
        rag_sources = {name: ingest_dave.find_source(name) if _benchmark_ready(name) else None
                       for name in ingest_http.UPLOADABLE}
        # DEV MODE: placeholders "downloaded" in the Catalogue are listed too (they have nothing on disk)
        fake_names = [n for n, _, _ in devmode.catalogue_placeholders(BENCHMARK_SHORT) if devmode.is_marked_downloaded(n)]

        def _indexed_label(n_docs):
            return "✅ Indexed" if n_docs else "Not indexed"

        def _rag_answerable(name):
            found = answerable.counts(ingest_http.SOURCE_OF[name], rag_uploaded) if name in rag_uploaded else None
            return f"{found[0]} / {found[1]}" if found else "-"

        st.dataframe(pd.DataFrame([
            {
                "Benchmark": name,
                "Corpus source": str(src.relative_to(ROOT_DIR)) if src else "missing",
                "Indexed": _indexed_label(len(rag_uploaded.get(name, []))),
                "Docs uploaded to this URL": len(rag_uploaded.get(name, [])),
                "Answerable queries": _rag_answerable(name),
            }
            for name, src in rag_sources.items() if src  # only the downloaded ones (Catalogue tab)
        ] + [{"Benchmark": name, "Corpus source": devmode.fake_corpus_path(name),
              "Indexed": _indexed_label(devmode.fake_upload_stats(name)[0]),
              "Docs uploaded to this URL": devmode.fake_upload_stats(name)[0],
              "Answerable queries": devmode.fake_upload_stats(name)[1]}
             for name in fake_names]), hide_index=True, width="stretch")
        if not any(rag_sources.values()) and not fake_names:
            st.info("No benchmark downloaded yet: download them in the Catalogue tab.")

        rag_available = [n for n, src in rag_sources.items() if src]
        rag_options = rag_available + fake_names
        cfg_rag = cfg.get("ingest_rag", {})
        rag_default = list(rag_available)
        _init("w_rag_benchmarks", [n for n in cfg_rag.get("benchmarks", rag_default) if n in rag_options])
        st.session_state.w_rag_benchmarks = [n for n in st.session_state.w_rag_benchmarks if n in rag_options]
        rag_sel = st.multiselect(
            "Benchmarks to upload", rag_options, key="w_rag_benchmarks",
            help="Corpora to send to POST <Base URL>/ingest (only those downloaded).")

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
            "benchmarks": [n for n in rag_sel if n in rag_available], "batch_docs": int(rag_batch_docs), "batch_mb": float(rag_batch_mb),
            "sleep": float(rag_sleep), "max_docs": int(rag_max_docs),
        }})

        rag_running = jobs.is_running("ingest_rag")
        u1, u2, _ = st.columns([1, 1, 3])
        if u1.button("Start corpus upload", type="primary", key="w_rag_upload",
                     disabled=rag_running or not rag_sel or not (
                         (dave_ready or not any(n in rag_available for n in rag_sel)) if devmode.active() else rag_url),
                     help="Replaces what your RAG holds for the selected benchmarks (the first request of each "
                          "benchmark has batch.index 0)."):
            batch_args = ["--batch-docs", str(int(rag_batch_docs)), "--batch-mb", str(float(rag_batch_mb)),
                          "--sleep", str(float(rag_sleep)), "--max-docs", str(int(rag_max_docs))]
            real_sel = [n for n in rag_sel if n in rag_available]
            fake_sel = [n for n in rag_sel if n in fake_names]
            if fake_sel:  # DEV MODE: pretend to index the placeholders
                devmode.simulate_upload(fake_sel)
            if not real_sel:
                st.rerun()  # shows the new numbers in the table
            elif devmode.active():  # DEV MODE: same settings and look, but the corpus goes to DAVE
                jobs.start("ingest_rag", [sys.executable, "-u", str(ROOT_DIR / "ingest_dave.py"),
                                          "--benchmarks", *real_sel, *batch_args],
                           "Corpus upload to the generic RAG", env=_dave_env(dave_now), cwd=str(ROOT_DIR))
                st.rerun()
            else:
                jobs.start("ingest_rag", [
                    sys.executable, "-u", str(ROOT_DIR / "ingest_http.py"), "--url", rag_url, "--benchmarks", *real_sel,
                    *batch_args,
                ], "Corpus upload to the generic RAG", cwd=str(ROOT_DIR))
                st.rerun()
        if u2.button("Forget uploads", key="w_rag_forget", disabled=rag_running or not rag_uploaded or devmode.active(),
                     help="Deletes only the local record of what was uploaded to this URL (your RAG is not touched). "
                          "The evaluation then stops filtering questions for it."):
            ingest_http.forget(rag_url)
            st.rerun()
        if "dev_upload_msg" in st.session_state:  # DEV MODE: result of the fake indexing
            st.success(st.session_state.pop("dev_upload_msg"))
        # DEV MODE: this page fronts DAVE, so clearing it clears DAVE's records too (and the fake indexing)
        _clear_index("w_clear_index_rag", ["generic", "dave"] if devmode.active() else ["generic"],
                     "⚠️ Forgets which documents were uploaded to the RAG"
                     + ("" if devmode.active() else " (all URLs)") + ". The RAG itself is not touched. "
                     "The downloaded files are not touched.",
                     extra_keys=("dev_fake_uploaded", "dev_upload_msg"))
        _job_panel("ingest_rag")


# api guide tab - how a third-party RAG must be structured
with tab_guide:
    api_guide.render(default_url=rag_base_url)
