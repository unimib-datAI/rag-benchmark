"""Secret dev mode, temporary: to remove it delete this file and the lines marked `# DEV MODE` in app.py.

Pressing R+A+G together reveals a "Dev mode" switch at the top of the sidebar (pressing it again only hides the switch: the mode keeps being on or off). While the switch is on
the DAVE settings and the Setup > DAVE tab are only hidden (they keep running and saving as usual),
and the retrieval type is read from the URL (the sidebar radio is hidden):
    http://<host>/dense, /fulltext, /hybrid        or        http://<host>/?retrieval=dense
"""
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

# URL value -> DAVE retrieval method (keys of settings.DAVE_RETRIEVAL_METHODS)
URL_RETRIEVAL = {"dense": "dense", "fulltext": "full-text", "full-text": "full-text",
                 "sparse": "full-text", "hybrid": "full"}

_ALWAYS_CSS = ".st-key-dev_unlock, .st-key-dev_js {display: none;}"
_DEV_CSS = """
.st-key-engine_select, .st-key-w_dave_retrieval_method, .st-key-dave_setup_hidden {display: none;}
"""

_KEY_LISTENER = """
<script>
const parent = window.parent;
if (!parent.__ragDevKeys) {
  parent.__ragDevKeys = true;
  const down = new Set();
  const doc = parent.document;
  doc.addEventListener('keydown', e => {
    const t = e.target;
    // typing in a text field must not count, but a focused switch or checkbox (the Dev mode toggle) must
    const typing = t.tagName === 'TEXTAREA' || t.isContentEditable ||
      (t.tagName === 'INPUT' && !/^(checkbox|radio|button|submit|range)$/.test(t.type));
    if (typing) return;
    down.add(e.key.toLowerCase());
    if (down.has('r') && down.has('a') && down.has('g')) {
      down.clear();
      const button = doc.querySelector('.st-key-dev_unlock button');
      if (button) button.click();
    }
  });
  doc.addEventListener('keyup', e => down.delete(e.key.toLowerCase()));
  parent.addEventListener('blur', () => down.clear());
}
</script>
"""


def sidebar_switch():
    """Key listener plus, once R+A+G was pressed, the switch. Call at the top of the sidebar."""
    st.markdown(f"<style>{_ALWAYS_CSS}</style>", unsafe_allow_html=True)
    if st.button("dev-unlock", key="dev_unlock"):  # R+A+G again hides the switch, the mode keeps its value
        st.session_state.dev_unlocked = not st.session_state.get("dev_unlocked", False)
    if st.session_state.get("dev_unlocked"):
        # the value lives in dev_enabled: a widget's own state is dropped while the widget is not drawn
        st.session_state.dev_enabled = st.toggle("Dev mode", value=st.session_state.get("dev_enabled", False),
                                                 key="dev_mode")
    with st.container(key="dev_js"):
        components.html(_KEY_LISTENER, height=0)
    if active():
        st.markdown(f"<style>{_DEV_CSS}</style>", unsafe_allow_html=True)


def active():
    return bool(st.session_state.get("dev_enabled"))


def retrieval_from_url():
    """DAVE retrieval method requested by the URL (last path segment or ?retrieval=), else None."""
    value = st.query_params.get("retrieval")
    if not value:
        try:
            value = urlparse(st.context.url).path.rstrip("/").rsplit("/", 1)[-1]
        except Exception:
            value = ""
    return URL_RETRIEVAL.get(str(value).lower())


def force_engine(engine, key="engine_select"):
    """While dev mode is on the evaluation always runs on `engine` (DAVE) even though the UI only shows
    the generic RAG; the engine chosen before is restored when the mode is switched off."""
    if active():
        if "dev_prev_engine" not in st.session_state:
            st.session_state.dev_prev_engine = st.session_state.get(key)
        st.session_state[key] = engine
    elif "dev_prev_engine" in st.session_state:
        previous = st.session_state.pop("dev_prev_engine")
        if previous:
            st.session_state[key] = previous


# Catalogue placeholders: (name, description, count). They only look downloaded, nothing is fetched.
# A placeholder whose name starts with the name of a real benchmark (BENCHMARK_SHORT) is skipped.
PLACEHOLDER_BENCHMARKS = [
    ("RGB (Noise & Rejection)", "Recent news in English and Chinese, with documents retrieved from search engines. Four testbeds: noise robustness, negative rejection, information integration and counterfactual robustness.", ""),
    ("MultiHop-RAG (Multi-Document Reasoning)", "English news articles (609). Each query requires 2-4 pieces of evidence (inference, comparison, temporal, null).", "2,556"),
    ("FollowRAG (Instruction Constraints)", "Checks whether the system respects instruction constraints (22 categories, e.g. format and length) on 4 knowledge-intensive QA datasets. The associated training dataset (VIF-RAG-QA) exceeds 100k samples.", "~3,000 (test)"),
    ("LegalBench-RAG (Contract Precision)", "Real contracts and legal documents (714 documents from CUAD, MAUD, ContractNLI, PrivacyQA). Tests the extraction of precise clauses.", "6,858"),
    ("MIRAGE (Medical QA)", "Multiple-choice medical questions from 5 datasets (MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, BioASQ-Y/N). MedRAG is the associated toolkit with corpora and retrievers.", "7,663"),
    ("CRAG (Dynamic Facts & Noise)", "Factual Q&A across 5 domains and 8 question types, with noisy HTML pages and mock APIs. Tests facts that change over time, false premises and negative rejection.", "4,409"),
    ("OmniEval (Financial Matrix)", "Financial domain in Chinese, organized in a matrix of 5 task classes × 16 topics. It has a GPT-4-generated part (11.4k examples) and a hand-annotated one (1.7k). The retrieval corpus is about 362k documents, but this figure comes from a single secondary source.", "~13,100 (11.4k + 1.7k)"),
    ("DomainRAG (University Enrollment)", "Enrollment system of a Chinese university (1,686 web pages, 14,406 passages). Six abilities: conversational, structural, extractive, time-sensitive, multi-document and faithful QA.", "395"),
    ("RAG Fairness (Demographic Bias)", "Scenario-based multiple-choice questions that check whether retrieval and generation favour one demographic group: gender and geographic location (TREC Fair Ranking 2022, 48 queries over a Wikipedia corpus) and socio-economic status (a BBQ subset with ambiguous and disambiguated contexts).", "4,800 per scenario + 1,104 (BBQ)"),
    ("CRUD-RAG (Create/Read/Update/Delete)", "Recent Chinese news (over 80,000 articles) with tasks of text continuation, single- and multi-document QA, hallucination correction and multi-document summarization.", ""),
    ("OHRBench (OCR Noise)", "PDF pages (8,561) from 7 domains. Tests the impact of OCR noise on text, tables, formulas, charts and reading order.", "8,498"),
    ("MTRAG (Multi-Turn Conversations)", "110 human-written multi-turn conversations (7.7 turns on average) over 4 corpora (ClapNQ, FiQA, Govt, Cloud). Tests coreference and non-standalone or unanswerable questions.", "842"),
]


def catalogue_placeholders(real_prefixes):
    """Placeholder entries (name, description, count) to add to the Catalogue while dev mode is on,
    without the ones that already exist as real benchmarks."""
    if not active():
        return []
    return [entry for entry in PLACEHOLDER_BENCHMARKS if not entry[0].startswith(tuple(real_prefixes))]


def mark_downloaded(names):
    """The "download" of a placeholder only remembers it for this session: nothing is fetched."""
    st.session_state.setdefault("dev_fake_downloaded", set()).update(names)


def is_marked_downloaded(name):
    return name in st.session_state.get("dev_fake_downloaded", ())


# ── fake indexing of the placeholders ──────────────────────────────

def _fake_queries(name):
    """Query count written in the placeholder list ("2,556", "~13,100 (11.4k + 1.7k)"), else a stable made-up one."""
    count = next((c for n, _, c in PLACEHOLDER_BENCHMARKS if n == name), "")
    digits = "".join(ch for ch in count.split("(")[0] if ch.isdigit())
    return int(digits) if digits else 300 + sum(map(ord, name)) % 2000


def _fake_docs(name):
    return max(1, _fake_queries(name) // 4)


def fake_upload_stats(name):
    """(documents uploaded, "answerable / total" text) of a placeholder, or (0, "-") if not "indexed" yet."""
    if name not in st.session_state.get("dev_fake_uploaded", ()):
        return 0, "-"
    queries = _fake_queries(name)
    return _fake_docs(name), f"{queries} / {queries}"


def simulate_upload(names):
    """Pretend to index the placeholders: a progress bar and a log, then they count as uploaded. Nothing is sent."""
    import time
    bar = st.progress(0.0, text="Starting upload...")
    log = st.empty()
    lines = []
    steps = 6
    for name in names:
        for step in range(1, steps + 1):
            time.sleep(0.35)
            bar.progress(step / steps, text=f"[{name}] uploading batch {step}/{steps}")
        lines.append(f"[{name}] done: {_fake_docs(name)} documents uploaded and indexed, 0 failed requests.")
        log.code("\n".join(lines), language=None)
        st.session_state.setdefault("dev_fake_uploaded", set()).add(name)
    bar.empty()
    st.session_state.dev_upload_msg = "Corpus upload completed: " + ", ".join(names)



def fake_corpus_path(name):
    """A believable but made-up corpus path for a placeholder, stable for the same name."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", name.split(" (")[0].lower()).strip("-")
    variants = [f"Benchmarks/{slug}/data/corpus", f"Benchmarks/{slug}/dataset/corpus.json",
                f"Benchmarks/{slug}/corpus/documents.jsonl", f"Run/{slug}_corpus.json"]
    return variants[sum(map(ord, name)) % len(variants)]


def fake_check_endpoint(base_url):
    """What api_guide.check_endpoint reports for a healthy RAG, without contacting anything."""
    import random
    import time
    time.sleep(0.8)
    url = base_url.strip().rstrip("/") + "/ask"
    return [("info", f"POST {url} answered HTTP 200 in {random.uniform(0.9, 2.4):.1f}s"),
            ("ok", "answer: 'Yes, the endpoint is working.'"),
            ("ok", "contexts: 6 documents")]


# Catalogue tags of the placeholders: name -> (domain, tasks).
PLACEHOLDER_TAGS = {
    "RGB (Noise & Rejection)": ("News", ["Noise Robustness", "Negative Rejection", "Counterfactual Robustness"]),
    "FollowRAG (Instruction Constraints)": ("Open-domain", ["Instruction Following"]),
    "MIRAGE (Medical QA)": ("Medical", ["Multiple-choice QA"]),
    "OmniEval (Financial Matrix)": ("Finance", ["Multi-hop QA", "Multi-turn QA", "Long-form QA"]),
    "DomainRAG (University Enrollment)": ("Education", ["Multi-turn QA", "Time-sensitive QA", "Noise Robustness"]),
    "RAG Fairness (Demographic Bias)": ("Open-domain", ["Fairness Evaluation", "Multiple-choice QA"]),
    "CRUD-RAG (Create/Read/Update/Delete)": ("News", ["Text Generation", "Multi-hop QA", "Summarization"]),
}


def placeholder_tags(name):
    return PLACEHOLDER_TAGS.get(name, ("", []))
