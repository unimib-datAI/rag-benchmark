"""Corpus of each benchmark, shared by the local RAG ingestion (webpage/evaluator.py)
and the DAVE ingestion (ingest_dave.py), so that both engines index the same documents.

The "context" field of Run/Tutto_*.jsonl holds only the gold documents, so it is used
as corpus only where the benchmark itself provides the documents per question (CRAG
search results). CRAG pages are sent as raw HTML: finding the information inside the
web code is part of what the RAG under test is evaluated on. OHRBench pages are the
OCR-parsed text of its PDFs with simulated OCR errors (OHR_VERSION).
Documents longer than MAX_DOC_CHARS are split into parts, because the university DAVE
server cannot ingest files that large.
"""
import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = ROOT_DIR / "Benchmarks"
RUN_DIR = ROOT_DIR / "Run"

MAX_DOC_CHARS = 500_000

# version of the OHRBench corpus (folder of retrieval.zip): the same 8500+ PDF pages
# parsed with realistic OCR errors (MinerU, moderate intensity); "gt" is the clean text
OHR_VERSION = "semantic_noise_MinerU_moderate"

# source name (field "source" of Run/Tutto_*.jsonl) -> benchmark file
BENCHMARK_FILES = {
    "MultiHop-RAG": "Tutto_MULTIHOP.jsonl",
    "MTRAG": "Tutto_MTRAG_DAVE.jsonl",
    "LegalBench-RAG": "Tutto_LEGALBENCH.jsonl",
    "SQuAD-it": "Tutto_SQUAD.jsonl",
    "CRAG": "Tutto_CRAG_DAVE.jsonl",
    "OHRBench": "Tutto_OHRBench_DAVE.jsonl",
}

# DAVE collection name -> benchmark: ingest_dave.py loads each corpus into its collection,
# webpage/evaluator.py searches the collection of each question's benchmark
DAVE_COLLECTIONS = {
    "LegalBench": "LegalBench-RAG",
    "MultiHop": "MultiHop-RAG",
    "CRAG": "CRAG",
    "MTRAG": "MTRAG",
    "SQuAD": "SQuAD-it",
    "OHRBench": "OHRBench",
}


def split_context_raw(context):
    """Splits a context into documents without cleaning the HTML."""
    if isinstance(context, list):
        return [str(c) for c in context if str(c).strip()]
    text = str(context or "").strip()
    if not text: return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines: return [text]

    docs, current = [], []
    for line in lines:
        if line.startswith("[Doc ") or line.startswith("[Fonte:") or line.startswith("[Documento "):
            if current:
                docs.append(" ".join(current).strip())
                current = []
        current.append(line)
    if current: docs.append(" ".join(current).strip())
    return docs


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def benchmark_source(path):
    for item in iter_jsonl(path):
        return item.get("source", "default")
    return "default"


def _multihop():
    path = BENCHMARKS_DIR / "MultiHop-RAG" / "dataset" / "corpus.json"
    def docs():
        with open(path, "r", encoding="utf-8-sig") as f:
            articles = json.load(f)
        for i, art in enumerate(articles):
            yield f"doc_{i}", f"{art.get('title', '')}\n{art.get('body', '')}".strip()
    return [path], docs


def _mtrag():
    corpus_dir = BENCHMARKS_DIR / "mt-rag-benchmark" / "corpora" / "document_level"
    paths = sorted(p for p in corpus_dir.glob("**/*.jsonl") if p.is_file()) if corpus_dir.is_dir() else []
    def docs():
        for path in paths:
            for i, rec in enumerate(iter_jsonl(path)):
                doc_id = rec.get("_id") or rec.get("document_id") or rec.get("id") or f"{path.stem}_{i}"
                yield str(doc_id), rec.get("text", "")
    return paths or [corpus_dir], docs


def _legalbench():
    corpus_dir = BENCHMARKS_DIR / "legalbenchrag" / "data" / "corpus"
    def docs():
        for path in sorted(corpus_dir.rglob("*.txt")):
            yield path.name, path.read_text(encoding="utf-8-sig")
    return [corpus_dir], docs


def _squad():
    path = RUN_DIR / "squad_corpus.json"
    def docs():
        with open(path, "r", encoding="utf-8-sig") as f:
            for rec in json.load(f):
                yield str(rec["id"]), rec.get("text", "")
    return [path], docs


def _ohrbench():
    """One document per PDF page: <domain>_<document>_p<page>, e.g. finance_JPMORGAN_2021Q1_10Q_p24."""
    corpus_dir = BENCHMARKS_DIR / "OHR-Bench" / "data" / "retrieval_base" / OHR_VERSION
    def docs():
        for path in sorted(corpus_dir.rglob("*.json")):
            doc_name = path.relative_to(corpus_dir).with_suffix("").as_posix().replace("/", "_")
            with open(path, "r", encoding="utf-8-sig") as f:
                for page in json.load(f):
                    yield f"{doc_name}_p{page.get('page_idx')}", str(page.get("text", ""))
    return [corpus_dir], docs


def _per_question(source):
    """Documents attached to each question by the benchmark itself (CRAG), as is."""
    path = RUN_DIR / BENCHMARK_FILES[source]
    def docs():
        for item in iter_jsonl(path):
            for raw in split_context_raw(item.get("context", "")):
                yield f"{source}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}", raw
    return [path], docs


_CORPORA = {
    "MultiHop-RAG": _multihop,
    "MTRAG": _mtrag,
    "LegalBench-RAG": _legalbench,
    "SQuAD-it": _squad,
    "CRAG": lambda: _per_question("CRAG"),
    "OHRBench": _ohrbench,
}


def _split_long_text(text, max_chars):
    """Splits text into parts of at most max_chars, preferring whitespace boundaries."""
    parts = []
    while len(text) > max_chars:
        cut = text.rfind(" ", int(max_chars * 0.9), max_chars)
        cut = cut if cut > 0 else max_chars
        parts.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def _part_id(doc_id, n):
    """Id of the n-th part of a split document; the first part keeps the original id."""
    if n == 1:
        return doc_id
    stem, dot, ext = doc_id.rpartition(".")
    return f"{stem}_part{n}.{ext}" if dot and stem else f"{doc_id}_part{n}"


class Corpus:
    """Unique, non-empty documents of a benchmark as (id, text) pairs."""

    def __init__(self, source):
        if source not in _CORPORA:
            raise RuntimeError(f"Benchmark '{source}' has no corpus registered in corpus_sources.py (_CORPORA): "
                               "see aggiunta_dataset.md, Step 2.")
        self.source = source
        self.paths, self._iter = _CORPORA[source]()

    def missing_paths(self):
        return [p for p in self.paths if not p.exists()]

    def ids(self):
        return [doc_id for doc_id, _ in self.docs(only_ids=True)]

    def docs(self, only_ids=False, include=None):
        """Yields (id, text); with include, only the documents whose id is in it."""
        seen = set()
        for doc_id, text in self._iter():
            if doc_id in seen or not text.strip():
                continue
            seen.add(doc_id)
            for n, part in enumerate(_split_long_text(text, MAX_DOC_CHARS), 1):
                part_id = _part_id(doc_id, n)
                if include is None or part_id in include:
                    yield part_id, (None if only_ids else part)
