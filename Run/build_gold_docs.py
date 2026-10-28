"""Which documents does each benchmark query need? Writes Run/gold_docs.json.

The evaluation uses it to ask only questions whose source documents were uploaded to DAVE
(a random subset of the corpus is uploaded when "Max documents per benchmark" is set).
The names are the ones ingest_dave.py gives to the uploaded files (see corpus_docs.py).

  mode "all": every listed document is needed (LegalBench, MultiHop, SQuAD, MTRAG)
  mode "any": at least one is needed (CRAG: the question comes with up to 10 web pages and
              the dataset does not say which ones hold the answer)
OHRBench is not listed: its documents travel with each question.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "Run"
BENCH = ROOT / "Benchmarks"
sys.path.insert(0, str(ROOT))
from corpus_docs import crag_pages, mtrag_base_id  # noqa: E402

OUTPUT = RUN / "gold_docs.json"


def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def legalbench():
    """The context of each query starts every document with a '[folder/file.txt]' line."""
    path = RUN / "Tutto_LEGALBENCH.jsonl"
    queries = {}
    for r in _iter_jsonl(path):
        names = re.findall(r"(?m)^\[([^\[\]\n]+?\.txt)\]$", r.get("context", ""))
        queries[r["id"]] = sorted({os.path.basename(n) for n in names})
    return "all", queries


def multihop():
    """Evidence titles -> position in corpus.json (ingest_dave names the files doc_<position>.txt)."""
    corpus = json.load(open(BENCH / "MultiHop-RAG" / "dataset" / "corpus.json", encoding="utf-8-sig"))
    position = {d.get("title"): i for i, d in enumerate(corpus)}
    items = json.load(open(BENCH / "MultiHop-RAG" / "dataset" / "MultiHopRAG.json", encoding="utf-8-sig"))
    queries = {}
    for i, item in enumerate(items):
        titles = [e.get("title") for e in item.get("evidence_list", [])]
        found = sorted({f"doc_{position[t]}.txt" for t in titles if t in position})
        if len(found) or not titles:
            queries[f"mh_{i + 1:04d}"] = found
    return "all", queries


def squad():
    """The context of each query is the text of one document of squad_corpus.json."""
    corpus = json.load(open(RUN / "squad_corpus.json", encoding="utf-8-sig"))
    doc_of_text = {d["text"]: f"{d['id']}.txt" for d in corpus}
    queries = {}
    for r in _iter_jsonl(RUN / "Tutto_SQUAD.jsonl"):
        name = doc_of_text.get(r.get("context", ""))
        if name:
            queries[r["id"]] = [name]
    return "all", queries


def mtrag():
    """Same walk as extract_mtrag.py: the relevant contexts of each agent turn."""
    path = BENCH / "mt-rag-benchmark" / "mtrag-human" / "conversations" / "conversations.json"
    conversations = json.load(open(path, encoding="utf-8"))
    queries = {}
    for c_idx, conv in enumerate(conversations):
        turn = 1
        for msg in conv.get("turns", conv.get("messages", [])):
            if msg.get("speaker", "").lower() != "agent":
                continue
            names = set()
            for ctx in msg.get("contexts", []):
                feedback = ctx.get("feedback", {}).get("relevant", {})
                if any(f.get("value") == "yes" for f in feedback.values()) or not feedback:
                    base = mtrag_base_id(ctx.get("document_id", ""))
                    if base:
                        names.add(base)
            queries[f"mtrag_c{c_idx + 1:03d}_t{turn:02d}"] = sorted(names)
            turn += 1
    return "all", queries


def crag():
    queries = {}
    for r in _iter_jsonl(RUN / "Tutto_CRAG_DAVE.jsonl"):
        queries[r["id"]] = sorted({name for name, _ in crag_pages(r.get("context", ""))})
    return "any", queries


# benchmark (as in the "source" field) -> (function, files it needs)
BUILDERS = {
    "LegalBench-RAG": (legalbench, [RUN / "Tutto_LEGALBENCH.jsonl"]),
    "MultiHop-RAG": (multihop, [BENCH / "MultiHop-RAG" / "dataset" / "corpus.json"]),
    "SQuAD-it": (squad, [RUN / "Tutto_SQUAD.jsonl", RUN / "squad_corpus.json"]),
    "MTRAG": (mtrag, [BENCH / "mt-rag-benchmark" / "mtrag-human" / "conversations" / "conversations.json"]),
    "CRAG": (crag, [RUN / "Tutto_CRAG_DAVE.jsonl"]),
}


def main():
    index = {}
    for source, (build, needed) in BUILDERS.items():
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            print(f"[{source}] skipped, missing: {', '.join(missing)}", flush=True)
            continue
        mode, queries = build()
        index[source] = {"mode": mode, "queries": queries}
        print(f"[{source}] {len(queries)} queries, mode '{mode}'", flush=True)

    tmp = OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUTPUT)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
