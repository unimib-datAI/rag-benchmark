"""Uploads the benchmark corpora to a generic RAG over HTTP (POST <url>/ingest).

The RAG contract is described in the API Guide tab. One request per batch of documents:

  {"benchmark_name": "SQuAD-it",
   "documents": [{"id": "squad_doc_1", "text": "..."}, ...],
   "batch": {"index": 0, "count": 12}}

batch.index 0 means "replace everything you hold for this benchmark", the next ones append.
What has been uploaded is recorded in .rag_docs_cache.json (per URL), so the evaluation can ask
only the questions whose documents the RAG holds.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

import ingest_dave
from corpus_sources import custom_benchmarks

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("STATE_DIR") or BASE_DIR)  # a mounted folder in Docker
STATE_FILE = STATE_DIR / ".rag_docs_cache.json"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300

# benchmark name used in the questions ("source" field) <-> corpus name of ingest_dave
SOURCE_OF = {"LegalBench": "LegalBench-RAG", "MultiHop": "MultiHop-RAG", "SQuAD": "SQuAD-it",
             "CRAG": "CRAG", "MTRAG": "MTRAG"}
# OHRBench documents travel with each question: nothing to upload
SOURCE_OF.update({name: ingest_dave.COLLECTIONS[name] for name in custom_benchmarks()})  # datasets in Custom/
UPLOADABLE = list(SOURCE_OF)


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def uploaded_names(url: str) -> dict:
    """{corpus name: [file names]} uploaded to this URL."""
    return load_state().get(normalize_url(url), {})


def _save(url: str, name: str, names: list):
    state = load_state()
    state.setdefault(normalize_url(url), {})[name] = names
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def forget(url: str):
    """Drop the local record of the uploads to this URL (the RAG itself is not touched)."""
    state = load_state()
    if state.pop(normalize_url(url), None) is not None:
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def _post(endpoint: str, body: dict):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(endpoint, json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return None
            error = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            error = str(e)
        if attempt < MAX_RETRIES:
            wait = 5 * 2 ** (attempt - 1)
            tqdm.write(f"  attempt {attempt}/{MAX_RETRIES} failed ({error}). Retrying in {wait}s...")
            time.sleep(wait)
    return error


def upload(url: str, name: str, batch_docs: int, batch_bytes: int, sleep: float, max_docs: int) -> int:
    """Upload one corpus; returns the number of batches that failed."""
    src = ingest_dave.find_source(name)
    if src is None:
        print(f"[{name}] corpus not found: skipped.")
        return 0
    print(f"[{name}] reading {src.relative_to(BASE_DIR)}...")
    docs = [d for d in ingest_dave.INGESTERS[name](src) if d["content"].strip()]
    if max_docs and len(docs) > max_docs:
        docs = random.sample(docs, max_docs)
    if not docs:
        print(f"[{name}] no documents found.")
        return 0

    batches = ingest_dave._make_batches(docs, batch_docs, batch_bytes)
    endpoint = normalize_url(url) + "/ingest"
    print(f"[{name}] sending {len(docs)} documents in {len(batches)} requests to {endpoint}")
    sent, failed = [], 0
    for index, batch in enumerate(tqdm(batches, desc=f"[{name}] uploading", unit="batch")):
        body = {
            "benchmark_name": SOURCE_OF[name],
            "documents": [{"id": d["fileName"][:-4] if d["fileName"].endswith(".txt") else d["fileName"],
                           "text": d["content"]} for d in batch],
            "batch": {"index": index, "count": len(batches)},
        }
        error = _post(endpoint, body)
        if error:
            failed += 1
            tqdm.write(f"[{name}] batch {index + 1}/{len(batches)} failed: {error}")
            if index == 0:  # nothing was replaced: the previous upload is still there
                print(f"[{name}] the first batch failed, upload abandoned.")
                return failed
            continue
        sent += [d["fileName"] for d in batch]
        _save(url, name, sent)
        time.sleep(sleep)
    print(f"[{name}] done: {len(sent)} of {len(docs)} documents uploaded, {failed} failed requests.")
    return failed


def main():
    parser = argparse.ArgumentParser(description="Upload the benchmark corpora to a generic RAG (POST /ingest)")
    parser.add_argument("--url", required=True, help="base URL of the RAG")
    parser.add_argument("--benchmarks", nargs="+", choices=UPLOADABLE,
                        default=UPLOADABLE)
    parser.add_argument("--batch-docs", type=int, default=100, help="documents per request")
    parser.add_argument("--batch-mb", type=float, default=5, help="MB per request")
    parser.add_argument("--sleep", type=float, default=0, help="pause between requests (s)")
    parser.add_argument("--max-docs", type=int, default=0, help="0 = all (random subset otherwise)")
    args = parser.parse_args()

    failed = 0
    for name in args.benchmarks:
        try:
            failed += upload(args.url, name, args.batch_docs, int(args.batch_mb * 2**20), args.sleep, args.max_docs)
        except Exception as e:
            failed += 1
            print(f"[{name}] error: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
