"""Which benchmark queries can be answered with the documents uploaded to DAVE.

"Max documents per benchmark" uploads a random subset of the corpus: a question whose source
document was not uploaded cannot be answered, whatever the RAG does. Run/gold_docs.json (built by
Run/build_gold_docs.py) lists the documents each query needs; the uploaded ones are in the local
cache .dave_docs_cache.json (written by ingest_dave.py after DAVE confirms each file).
"""
import json
from pathlib import Path

from settings import ROOT_DIR  # also puts the project root on sys.path

import ingest_dave
from corpus_docs import mtrag_base_id

GOLD_FILE = ROOT_DIR / "Run" / "gold_docs.json"

# benchmark as written in the "source" field of the queries -> collection / cache name
COLLECTION = {"LegalBench-RAG": "LegalBench", "MultiHop-RAG": "MultiHop", "SQuAD-it": "SQuAD",
              "CRAG": "CRAG", "MTRAG": "MTRAG"}

# collection / corpus name -> benchmark name in the questions (+ OHRBench, which has no collection)
COLLECTION_SOURCE = {v: k for k, v in COLLECTION.items()} | {"OHRBench": "OHRBench"}

_loaded = {"mtime": None, "index": {}}


def gold_index() -> dict:
    """Run/gold_docs.json ({} if it was not built yet); reloaded only when the file changes."""
    try:
        mtime = GOLD_FILE.stat().st_mtime_ns
    except OSError:
        return {}
    if _loaded["mtime"] != mtime:
        _loaded["index"] = json.loads(GOLD_FILE.read_text(encoding="utf-8"))
        _loaded["mtime"] = mtime
    return _loaded["index"]


def _uploaded(source: str, uploaded: dict = None) -> set:
    """Files uploaded for a benchmark: DAVE's local cache, or `uploaded` ({corpus name: [files]})."""
    cache = ingest_dave.load_doc_cache() if uploaded is None else uploaded
    names = set(cache.get(COLLECTION[source], []))
    if source == "MTRAG":  # passages of one document are separate files
        return {mtrag_base_id(n[:-4] if n.endswith(".txt") else n) for n in names}
    return names


def _answerable(needed: list, mode: str, uploaded: set) -> bool:
    if not needed:
        return True  # the query needs no document (e.g. MultiHop null queries)
    return bool(uploaded & set(needed)) if mode == "any" else set(needed) <= uploaded


def filter_answerable(elements: list, uploaded: dict = None):
    """Keep the queries of one benchmark whose documents are uploaded.

    `uploaded` is the record of a generic RAG ({corpus name: [files]}); None means DAVE. A benchmark
    missing from that record was not uploaded through the tool: nothing is filtered (tracked False).
    Returns (kept, info); info has total, answerable, unknown (queries without gold information,
    kept) and indexed (False when the benchmark has no gold information at all: nothing filtered)."""
    source = elements[0].get("source") if elements else None
    entry = gold_index().get(source) if source in COLLECTION else None
    info = {"source": source, "total": len(elements), "answerable": len(elements), "unknown": 0,
            "indexed": entry is not None,
            "tracked": uploaded is None or COLLECTION.get(source) in uploaded}
    if entry is None or not info["tracked"]:  # OHRBench, index not built, or not uploaded via the tool
        return elements, info

    uploaded = _uploaded(source, uploaded)
    kept, unknown = [], 0
    for e in elements:
        needed = entry["queries"].get(e["id"])
        if needed is None:
            unknown += 1
            kept.append(e)
        elif _answerable(needed, entry["mode"], uploaded):
            kept.append(e)
    info.update(answerable=len(kept), unknown=unknown)
    return kept, info


def counts(source: str, uploaded: dict = None):
    """(answerable, total) queries of a benchmark with the current uploads, or None if not indexed."""
    entry = gold_index().get(source)
    if entry is None:
        return None
    uploaded = _uploaded(source, uploaded)
    ok = sum(1 for needed in entry["queries"].values() if _answerable(needed, entry["mode"], uploaded))
    return ok, len(entry["queries"])
