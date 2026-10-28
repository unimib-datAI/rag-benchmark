"""Documents of a benchmark, read from Run/Tutto_*.jsonl as the tool sends them to /ingest, and their chunks."""
import json
from pathlib import Path

from bs4 import BeautifulSoup

RUN_DIR = Path(__file__).resolve().parents[1] / "Run"

BENCHMARK_FILES = {
    "SQuAD-it": "Tutto_SQUAD.jsonl",
    "MultiHop-RAG": "Tutto_MULTIHOP.jsonl",
    "MTRAG": "Tutto_MTRAG_DAVE.jsonl",
    "LegalBench-RAG": "Tutto_LEGALBENCH.jsonl",
    "CRAG": "Tutto_CRAG_DAVE.jsonl",
    "OHRBench": "Tutto_OHRBench_DAVE.jsonl",
}

DOC_MARKERS = ("[Doc ", "[Fonte:", "[Documento ")
CHUNK_WORDS = 200
OVERLAP_WORDS = 40


def split_context(context):
    """A context holds one or more documents: a new one starts at each line beginning with a DOC_MARKER."""
    if isinstance(context, list):
        return [str(doc) for doc in context]
    docs = []
    for line in str(context).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(DOC_MARKERS) or not docs:
            docs.append(line)
        else:
            docs[-1] += " " + line
    return docs


def clean_html(text):
    if "<" not in text and "&" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def load_documents(benchmark):
    """Unique documents of a benchmark, in order of first appearance."""
    seen = set()
    documents = {}
    with open(RUN_DIR / BENCHMARK_FILES[benchmark], encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            for doc in split_context(json.loads(line).get("context") or ""):
                # many records share the same document: clean it only the first time
                if hash(doc) in seen:
                    continue
                seen.add(hash(doc))
                documents[clean_html(doc)] = None
    return [doc for doc in documents if doc]


def chunk(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Windows of `size` words; consecutive windows share `overlap` words."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), size - overlap)]
