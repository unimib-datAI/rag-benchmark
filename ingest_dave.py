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
# job di upload non ancora verificati, per collection id: DAVE elabora i file in modo
# asincrono e alcuni possono fallire anche dopo che il job è stato accettato
JOBS_DB = STATE_DIR / ".dave_upload_jobs.json"
JOB_POLL_SECONDS = 30
MAX_UPLOAD_ROUNDS = 3

# Batch conservativo: max 1MB payload o 20 docs
MAX_BATCH_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_BATCH_DOCS = 20
MAX_RETRIES = 3
SLEEP_BETWEEN = 5  # secondi tra batch

# corpora not selected by default in the web UI (MTRAG: very large corpus)
DEFAULT_BENCHMARKS_EXCLUDED = ["MTRAG"]


def _load_json(path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def _load_doc_cache(mapping):
    """Documenti già caricati, per collection id. Le cache vecchie erano indicizzate per
    nome del benchmark: vengono spostate sull'id della collection corrispondente."""
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


def check_jobs(client: DaveClient, collection_id: str, name: str, doc_cache: dict, jobs: dict):
    """Verifica i job di upload della collection: i file falliti vengono tolti dalla cache,
    così la prossima chiamata a upload_corpus li ricarica. Ritorna (job in corso, file falliti)."""
    pending, failed = [], []
    for job_id in jobs.get(collection_id, []):
        try:
            files = client.get_upload_job_status(job_id).get("files", [])
        except Exception as e:
            print(f"[{name}] Stato del job {job_id} non disponibile: {e}")
            pending.append(job_id)
            continue
        if any(f.get("status") not in ("completed", "failed") for f in files):
            pending.append(job_id)
            continue
        for f in files:
            if f.get("status") == "failed":
                failed.append(f["fileName"])
                print(f"[{name}] {f['fileName']} non indicizzato da DAVE: {f.get('error', 'errore sconosciuto')}")

    if failed:
        failed_set = set(failed)
        doc_cache[collection_id] = [n for n in doc_cache.get(collection_id, []) if n not in failed_set]
        _save_json(DOC_CACHE_DB, doc_cache)
    jobs[collection_id] = pending
    _save_json(JOBS_DB, jobs)
    return pending, failed


def upload_corpus(client: DaveClient, collection_id: str, corpus: Corpus, name: str, doc_cache: dict, jobs: dict,
                  batch_docs=MAX_BATCH_DOCS, batch_bytes=MAX_BATCH_BYTES, sleep=SLEEP_BETWEEN, max_docs=0):
    """Carica su DAVE solo i documenti del corpus non ancora presenti nella collection."""
    already_uploaded = set(doc_cache.setdefault(collection_id, []))
    ids = corpus.ids()
    missing_ids = {doc_id for doc_id in ids if _file_name(doc_id) not in already_uploaded}

    if max_docs:
        # i documenti già caricati contano nel limite e restano
        allowed = max(0, max_docs - (len(ids) - len(missing_ids)))
        if len(missing_ids) > allowed:
            missing_ids = set(random.Random(0).sample(sorted(missing_ids), allowed))
    if not missing_ids:
        print(f"[{name}] Tutti i documenti ({len(ids)}) sono già presenti in cache. Salto l'upload.")
        return 0
    print(f"[{name}] {len(ids) - len(missing_ids)} doc già caricati. Inizio upload dei {len(missing_ids)} nuovi doc...")

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

    last_job_id = None
    with tqdm(total=len(missing_ids), desc=f"[{name}] Uploading", unit="doc") as bar:
        for batch in batches():
            job_id = None

            # Retry con backoff esponenziale
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    job_id = client.create_txt_upload_job(collection_id, batch)["jobId"]
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        wait = 10 * (2 ** (attempt - 1))
                        tqdm.write(f"[{name}] Batch tentativo {attempt}/{MAX_RETRIES} fallito: {e}. Riprovo tra {wait}s...")
                        time.sleep(wait)
                    else:
                        tqdm.write(f"[{name}] Batch fallito dopo {MAX_RETRIES} tentativi: {e}. Salto.")

            bar.update(len(batch))
            if not job_id:
                continue
            last_job_id = job_id

            # Il job è stato accettato: i file vanno in cache subito, e quelli che DAVE
            # non riesce a indicizzare vengono tolti da check_jobs
            doc_cache[collection_id].extend(d["fileName"] for d in batch)
            _save_json(DOC_CACHE_DB, doc_cache)
            jobs.setdefault(collection_id, []).append(job_id)
            _save_json(JOBS_DB, jobs)

            time.sleep(sleep)

    if last_job_id:
        print(f"[{name}] Upload completato. Ultimo job: {last_job_id}")
    return len(missing_ids)


def verify(client: DaveClient, collection_id: str, name: str, doc_cache: dict):
    """Confronta la cache locale con i documenti presenti su DAVE; non carica nulla.

    Il nome del documento su DAVE è il fileName senza .txt. I file in cache che DAVE non ha
    davvero vengono tolti dalla cache, così la prossima ingestione li ricarica."""
    strip = lambda n: n[:-4] if n.lower().endswith(".txt") else n
    local = {strip(n) for n in doc_cache.get(collection_id, [])}
    remote = set(client.list_documents(collection_id))
    missing, extra = local - remote, remote - local
    print(f"[{name}] cache locale: {len(local)} | su DAVE: {len(remote)} | "
          f"mancanti su DAVE: {len(missing)} | su DAVE ma non in cache: {len(extra)}")
    for n in sorted(missing)[:10]:
        print(f"    mancante: {n}")
    if missing:
        doc_cache[collection_id] = [n for n in doc_cache[collection_id] if strip(n) not in missing]
        _save_json(DOC_CACHE_DB, doc_cache)
        print(f"[{name}] {len(missing)} voci tolte dalla cache: la prossima ingestione le ricarica.")
    return not missing


def recreate(client: DaveClient, mapping: dict, doc_cache: dict, jobs: dict, name: str):
    """Crea una nuova collection al posto di quella esistente e azzera lo stato locale.
    La vecchia resta su DAVE (non viene eliminata) e va rimossa a mano."""
    old_id = mapping.get(name)
    collection_id = client.create_collection(name)
    mapping[name] = collection_id
    if old_id:
        doc_cache.pop(old_id, None)
        jobs.pop(old_id, None)
    _save_json(COLLECTION_DB, mapping)
    _save_json(DOC_CACHE_DB, doc_cache)
    _save_json(JOBS_DB, jobs)
    print(f"[{name}] Creata nuova collection (ID: {collection_id})"
          + (f"; la precedente ({old_id}) resta su DAVE e non è più usata" if old_id else ""))


def wait_for_indexing(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts):
    for round_ in range(1, MAX_UPLOAD_ROUNDS + 1):
        pending, failed = check_jobs(client, collection_id, name, doc_cache, jobs)
        while pending:
            print(f"[{name}] DAVE sta ancora indicizzando ({len(pending)} job in corso)...")
            time.sleep(JOB_POLL_SECONDS)
            pending, new_failed = check_jobs(client, collection_id, name, doc_cache, jobs)
            failed += new_failed
        if not failed:
            print(f"[{name}] Tutti i documenti sono indicizzati su DAVE.")
            return True
        if round_ == MAX_UPLOAD_ROUNDS:
            print(f"[{name}] {len(failed)} documenti ancora non indicizzati dopo {MAX_UPLOAD_ROUNDS} tentativi: "
                  "rilancia lo script per riprovare.")
            return False
        print(f"[{name}] Ricarico {len(failed)} documenti falliti (tentativo {round_ + 1}/{MAX_UPLOAD_ROUNDS})...")
        upload_corpus(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts)
    return False


def main():
    parser = argparse.ArgumentParser(description="Carica i corpus dei benchmark su DAVE")
    parser.add_argument("--benchmarks", nargs="*", default=[], metavar="BENCHMARK",
                        help=f"collection da caricare (default: tutte). Valori: {', '.join(COLLECTIONS)}")
    parser.add_argument("--batch-docs", type=int, default=MAX_BATCH_DOCS)
    parser.add_argument("--batch-mb", type=float, default=MAX_BATCH_BYTES / 2**20)
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    parser.add_argument("--max-docs", type=int, default=0, help="0 = tutti (sottoinsieme casuale)")
    parser.add_argument("--recreate", action="store_true",
                        help="crea nuove collection al posto di quelle esistenti e ricarica tutto "
                             "(le vecchie restano su DAVE e vanno eliminate a mano)")
    parser.add_argument("--no-wait", action="store_true",
                        help="non attende l'indicizzazione di DAVE (default: attende e ricarica i file falliti)")
    parser.add_argument("--verify", action="store_true",
                        help="confronta cache e DAVE, non carica nulla")
    args = parser.parse_args()
    unknown = [b for b in args.benchmarks if b not in COLLECTIONS]
    if unknown:
        parser.error(f"benchmark sconosciuti: {', '.join(unknown)}. Valori: {', '.join(COLLECTIONS)}")

    try:
        client = DaveClient()
    except Exception as e:
        print(f"Errore di connessione a DAVE: {e}")
        return 1

    print("Connesso a DAVE con successo!")

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
                    print(f"[{name}] Nessuna collection.")
                continue

            corpus = Corpus(COLLECTIONS[name])
            missing = corpus.missing_paths()
            if missing:
                print(f"[{name}] Corpus non trovato: {', '.join(str(p) for p in missing)}")
                continue

            if args.recreate:
                recreate(client, mapping, doc_cache, jobs, name)
            collection_id = mapping.get(name)
            if not collection_id:
                collection_id = client.create_collection(name)
                mapping[name] = collection_id
                _save_json(COLLECTION_DB, mapping)
                print(f"[{name}] Creata nuova collection (ID: {collection_id})")

            # i file falliti nelle esecuzioni precedenti tornano tra quelli da caricare
            check_jobs(client, collection_id, name, doc_cache, jobs)
            upload_opts = dict(batch_docs=args.batch_docs, batch_bytes=int(args.batch_mb * 2**20),
                               sleep=args.sleep, max_docs=args.max_docs)
            upload_corpus(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts)
            if not args.no_wait:
                ok &= wait_for_indexing(client, collection_id, corpus, name, doc_cache, jobs, **upload_opts)
        except Exception as e:
            ok = False
            print(f"[{name}] Errore: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
