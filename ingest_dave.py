import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm
from dave_client import DaveClient
# same corpora that webpage/evaluator.py sends to the local RAG
from corpus_sources import Corpus, DAVE_COLLECTIONS as COLLECTIONS

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("STATE_DIR") or BASE_DIR)  # a mounted folder in Docker
COLLECTION_DB = STATE_DIR / ".dave_collections.json"
DOC_CACHE_DB = STATE_DIR / ".dave_docs_cache.json"
# upload jobs started and not yet confirmed, per collection id: DAVE processes the files asynchronously and some
# can fail after the job was accepted. Normally empty: each job is waited for before the next one starts.
JOBS_DB = STATE_DIR / ".dave_upload_jobs.json"
MAX_UPLOAD_ROUNDS = 3  # passes over the documents DAVE failed to index
JOB_TIMEOUT = 3600  # seconds to wait for a single batch to be indexed
JOB_POLL_SECONDS = 3

MAX_BATCH_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_BATCH_DOCS = 20
MAX_RETRIES = 3
SLEEP_BETWEEN = 0  # seconds between batches (each batch is already waited for)


def _load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _load_doc_cache(mapping):
    """Documents already uploaded, per collection id. Old caches were keyed by benchmark name:
    they are moved to the id of the matching collection."""
    doc_cache = _load_json(DOC_CACHE_DB)
    for name, collection_id in mapping.items():
        if name in doc_cache and collection_id not in doc_cache:
            doc_cache[collection_id] = doc_cache.pop(name)
    return doc_cache


def _file_name(doc_id):
    return doc_id if doc_id.endswith(".txt") else f"{doc_id}.txt"


# ── API used by the web UI and ingest_http.py ──────────────────────

def load_collections():
    """Collection name -> DAVE collection id."""
    return _load_json(COLLECTION_DB)


def load_doc_cache():
    """Collection name -> fileNames already uploaded (the file itself is keyed by collection id)."""
    mapping = load_collections()
    doc_cache = _load_doc_cache(mapping)
    return {name: doc_cache.get(collection_id, []) for name, collection_id in mapping.items()}


def find_source(name):
    """Main path of the corpus of a collection, or None if any of its files is missing."""
    corpus = Corpus(COLLECTIONS[name])
    return corpus.paths[0] if corpus.paths and not corpus.missing_paths() else None


def _load_corpus_docs(name):
    return [{"fileName": _file_name(doc_id), "content": text} for doc_id, text in Corpus(COLLECTIONS[name]).docs()]


# collection name -> loader(source path) -> [{"fileName", "content"}]
INGESTERS = {name: (lambda src, name=name: _load_corpus_docs(name)) for name in COLLECTIONS}


def _make_batches(docs, max_docs, max_bytes):
    batches, current, size = [], [], 0
    for doc in docs:
        doc_size = len(doc["content"].encode("utf-8"))
        if current and (size + doc_size > max_bytes or len(current) >= max_docs):
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += doc_size
    if current:
        batches.append(current)
    return batches


def settle_jobs(client: DaveClient, collection_id: str, name: str, doc_cache: dict, jobs: dict):
    """Wait for the upload jobs left unconfirmed by an interrupted run and take their result: the files DAVE
    failed to index are removed from the cache, so the next upload sends them again."""
    for job_id in list(jobs.get(collection_id, [])):
        print(f"[{name}] Waiting for the job left over from a previous run ({job_id})...")
        try:
            job = client.wait_for_upload_job(job_id, timeout=JOB_TIMEOUT, poll=JOB_POLL_SECONDS)
        except Exception as e:
            print(f"[{name}] Job {job_id} not available: {e}")
            jobs[collection_id].remove(job_id)
            continue
        failed = {f["fileName"] for f in job.get("files", []) if f.get("status") != "completed"}
        if failed:
            print(f"[{name}] {len(failed)} files of the previous job were not indexed: they will be uploaded again.")
            doc_cache[collection_id] = [n for n in doc_cache.get(collection_id, []) if n not in failed]
            _save_json(DOC_CACHE_DB, doc_cache)
        jobs[collection_id].remove(job_id)
    _save_json(JOBS_DB, jobs)


def _create_job(client: DaveClient, collection_id: str, batch: list, name: str):
    """Start the upload job of a batch, retrying with exponential backoff. Returns the job id or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.create_txt_upload_job(collection_id, batch)["jobId"]
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = 10 * (2 ** (attempt - 1))
                tqdm.write(f"[{name}] Attempt {attempt}/{MAX_RETRIES} to start the batch failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                tqdm.write(f"[{name}] Batch not started after {MAX_RETRIES} attempts: {e}. Skipped.")
    return None


def upload_corpus(client: DaveClient, collection_id: str, corpus: Corpus, name: str, doc_cache: dict, jobs: dict,
                  batch_docs=MAX_BATCH_DOCS, batch_bytes=MAX_BATCH_BYTES, sleep=SLEEP_BETWEEN, max_docs=0):
    """Upload the documents of the corpus that are not in the collection yet, one batch at a time: a batch is
    started only when DAVE finished indexing the previous one, so the progress bar counts documents that are
    really indexed. Returns the number of documents that were not indexed."""
    already_uploaded = set(doc_cache.setdefault(collection_id, []))
    ids = corpus.ids()
    missing_ids = {doc_id for doc_id in ids if _file_name(doc_id) not in already_uploaded}

    if max_docs:
        # documents already uploaded count towards the limit and stay
        allowed = max(0, max_docs - (len(ids) - len(missing_ids)))
        if len(missing_ids) > allowed:
            missing_ids = set(random.Random(0).sample(sorted(missing_ids), allowed))
    if not missing_ids:
        print(f"[{name}] All {len(ids)} documents are already uploaded. Nothing to do.")
        return 0
    print(f"[{name}] {len(ids) - len(missing_ids)} documents already uploaded. "
          f"Uploading {len(missing_ids)} new documents, one batch at a time...")

    def batches():
        batch, size = [], 0
        for doc_id, text in corpus.docs(include=missing_ids):
            doc_size = len(text.encode("utf-8"))
            if batch and (size + doc_size > batch_bytes or len(batch) >= batch_docs):
                yield batch
                batch, size = [], 0
            batch.append({"fileName": _file_name(doc_id), "content": text})
            size += doc_size
        if batch:
            yield batch

    failed_total = 0
    bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} docs [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
    with tqdm(total=len(missing_ids), desc=f"[{name}] Indexed", unit="doc", bar_format=bar_format) as bar:
        for number, batch in enumerate(batches(), 1):
            job_id = _create_job(client, collection_id, batch, name)
            if not job_id:
                failed_total += len(batch)
                continue
            jobs.setdefault(collection_id, []).append(job_id)
            _save_json(JOBS_DB, jobs)
            counted = 0  # documents of this batch already shown in the bar

            def on_progress(job, batch=batch, number=number):
                nonlocal counted
                files = job.get("files", [])
                finished = sum(1 for f in files if f.get("status") in ("completed", "failed"))
                if finished > counted:
                    bar.update(finished - counted)
                    counted = finished
                bar.set_postfix_str(f"batch {number}: {job.get('status', '?')}")

            try:
                job = client.wait_for_upload_job(job_id, timeout=JOB_TIMEOUT, poll=JOB_POLL_SECONDS,
                                                 on_progress=on_progress)
            except Exception as e:
                tqdm.write(f"[{name}] Batch {number}: could not follow job {job_id}: {e}. "
                           "It stays in the list to check next time.")
                bar.update(len(batch) - counted)
                failed_total += len(batch)
                continue

            files = job.get("files", [])
            done = [f["fileName"] for f in files if f.get("status") == "completed"]
            bad = [f for f in files if f.get("status") != "completed"]
            if job.get("error"):
                tqdm.write(f"[{name}] Batch {number}: job {job.get('status')}: {job['error']}")
            for f in bad[:5]:
                tqdm.write(f"[{name}]   not indexed: {f.get('fileName')} ({f.get('status')}) {f.get('error', '')}")
            if len(bad) > 5:
                tqdm.write(f"[{name}]   ... and {len(bad) - 5} more files not indexed")
            bar.update(len(batch) - counted)  # whatever the job did not report
            failed_total += len(batch) - len(done)
            if done:
                doc_cache[collection_id].extend(done)
                _save_json(DOC_CACHE_DB, doc_cache)
            jobs[collection_id].remove(job_id)
            _save_json(JOBS_DB, jobs)
            if sleep:
                time.sleep(sleep)

    print(f"[{name}] Done: {len(missing_ids) - failed_total} documents indexed, {failed_total} not indexed.")
    return failed_total


def verify(client: DaveClient, collection_id: str, name: str, doc_cache: dict):
    """Compare the local cache with the documents in DAVE; uploads nothing.

    The name of a document in DAVE is the fileName without .txt. Files in the cache that DAVE does not
    really have are removed from the cache, so the next ingestion uploads them again."""
    strip = lambda n: n[:-4] if n.lower().endswith(".txt") else n
    local = {strip(n) for n in doc_cache.get(collection_id, [])}
    remote = set(client.list_documents(collection_id))
    missing, extra = local - remote, remote - local
    print(f"[{name}] local cache: {len(local)} | in the RAG: {len(remote)} | "
          f"missing in the RAG: {len(missing)} | in the RAG but not in the cache: {len(extra)}")
    for n in sorted(missing)[:10]:
        print(f"    missing: {n}")
    if missing:
        doc_cache[collection_id] = [n for n in doc_cache[collection_id] if strip(n) not in missing]
        _save_json(DOC_CACHE_DB, doc_cache)
        print(f"[{name}] {len(missing)} entries removed from the cache: the next ingestion uploads them again.")
    return not missing


def recreate(client: DaveClient, mapping: dict, doc_cache: dict, jobs: dict, name: str):
    """Create a new collection in place of the existing one and reset the local state.
    The old one stays in DAVE (it is not deleted) and has to be removed by hand."""
    old_id = mapping.get(name)
    collection_id = client.create_collection(name)
    mapping[name] = collection_id
    if old_id:
        doc_cache.pop(old_id, None)
        jobs.pop(old_id, None)
    _save_json(COLLECTION_DB, mapping)
    _save_json(DOC_CACHE_DB, doc_cache)
    _save_json(JOBS_DB, jobs)
    print(f"[{name}] Created a new collection (ID: {collection_id})"
          + (f"; the previous one ({old_id}) stays there and is no longer used" if old_id else ""))


def upload_with_retries(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts):
    """Upload the corpus; documents DAVE failed to index are sent again (up to MAX_UPLOAD_ROUNDS passes).
    Returns True if everything is indexed."""
    for round_ in range(1, MAX_UPLOAD_ROUNDS + 1):
        failed = upload_corpus(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts)
        if not failed:
            return True
        if round_ < MAX_UPLOAD_ROUNDS:
            print(f"[{name}] Uploading the {failed} documents that failed again (pass {round_ + 1}/{MAX_UPLOAD_ROUNDS})...")
    print(f"[{name}] {failed} documents are still not indexed after {MAX_UPLOAD_ROUNDS} passes: run the script again to retry.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Upload the benchmark corpora and wait for each batch to be indexed")
    parser.add_argument("--benchmarks", nargs="*", default=[], metavar="BENCHMARK",
                        help=f"collections to upload (default: all). Values: {', '.join(COLLECTIONS)}")
    parser.add_argument("--batch-docs", type=int, default=MAX_BATCH_DOCS)
    parser.add_argument("--batch-mb", type=float, default=MAX_BATCH_BYTES / 2**20)
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--max-docs", type=int, default=0, help="0 = all (random subset otherwise)")
    parser.add_argument("--recreate", action="store_true",
                        help="create new collections in place of the existing ones and upload everything again "
                             "(the old ones stay in DAVE and have to be deleted by hand)")
    parser.add_argument("--verify", action="store_true",
                        help="compare the cache with DAVE, uploads nothing")
    args = parser.parse_args()
    unknown = [b for b in args.benchmarks if b not in COLLECTIONS]
    if unknown:
        parser.error(f"unknown benchmarks: {', '.join(unknown)}. Values: {', '.join(COLLECTIONS)}")

    try:
        client = DaveClient()
    except Exception as e:
        print(f"Could not connect to DAVE: {e}")
        return 1

    print("Connected to DAVE.")

    mapping = _load_json(COLLECTION_DB)
    doc_cache = _load_doc_cache(mapping)
    jobs = _load_json(JOBS_DB)

    ok = True
    for name in args.benchmarks or list(COLLECTIONS):
        try:
            if args.verify:
                if mapping.get(name):
                    ok &= verify(client, mapping[name], name, doc_cache)
                else:
                    print(f"[{name}] No collection.")
                continue

            corpus = Corpus(COLLECTIONS[name])
            missing = corpus.missing_paths()
            if missing:
                print(f"[{name}] Corpus not found: {', '.join(str(p) for p in missing)}")
                continue

            if args.recreate:
                recreate(client, mapping, doc_cache, jobs, name)
            collection_id = mapping.get(name)
            if not collection_id:
                collection_id = client.create_collection(name)
                mapping[name] = collection_id
                _save_json(COLLECTION_DB, mapping)
                print(f"[{name}] Created a new collection (ID: {collection_id})")

            # jobs left unfinished by a previous run: their failed files go back among those to upload
            settle_jobs(client, collection_id, name, doc_cache, jobs)
            upload_opts = dict(batch_docs=args.batch_docs, batch_bytes=int(args.batch_mb * 2**20),
                               sleep=args.sleep, max_docs=args.max_docs)
            ok &= upload_with_retries(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts)
        except Exception as e:
            ok = False
            print(f"[{name}] Error: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
