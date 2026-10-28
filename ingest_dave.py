import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from dave_client import DaveClient
# same corpora that webpage/evaluator.py sends to the local RAG
from corpus_sources import Corpus, DAVE_COLLECTIONS as COLLECTIONS

BASE_DIR = Path(__file__).resolve().parent
COLLECTION_DB = BASE_DIR / ".dave_collections.json"
DOC_CACHE_DB = BASE_DIR / ".dave_docs_cache.json"
# job di upload non ancora verificati, per collection id: DAVE elabora i file in modo
# asincrono e alcuni possono fallire anche dopo che il job è stato accettato
JOBS_DB = BASE_DIR / ".dave_upload_jobs.json"
JOB_POLL_SECONDS = 30
MAX_UPLOAD_ROUNDS = 3

# Batch conservativo: max 1MB payload o 20 docs
MAX_BATCH_BYTES = 1 * 1024 * 1024  # 1 MB
MAX_BATCH_DOCS = 20
MAX_RETRIES = 3
SLEEP_BETWEEN = 5  # secondi tra batch


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


def upload_corpus(client: DaveClient, collection_id: str, corpus: Corpus, name: str, doc_cache: dict, jobs: dict):
    """Carica su DAVE solo i documenti del corpus non ancora presenti nella collection."""
    already_uploaded = set(doc_cache.setdefault(collection_id, []))
    ids = corpus.ids()
    missing_ids = {doc_id for doc_id in ids if _file_name(doc_id) not in already_uploaded}

    if not missing_ids:
        print(f"[{name}] Tutti i documenti ({len(ids)}) sono già presenti in cache. Salto l'upload.")
        return 0
    print(f"[{name}] {len(ids) - len(missing_ids)} doc già caricati. Inizio upload dei {len(missing_ids)} nuovi doc...")

    def batches():
        batch, size = [], 0
        for doc_id, text in corpus.docs(include=missing_ids):
            doc_size = len(text.encode("utf-8"))
            if batch and (size + doc_size > MAX_BATCH_BYTES or len(batch) >= MAX_BATCH_DOCS):
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

            time.sleep(SLEEP_BETWEEN)

    if last_job_id:
        print(f"[{name}] Upload completato. Ultimo job: {last_job_id}")
    return len(missing_ids)


def main():
    parser = argparse.ArgumentParser(description="Carica i corpus dei benchmark su DAVE")
    parser.add_argument("benchmarks", nargs="*", metavar="BENCHMARK",
                        help=f"collection da caricare (default: tutte). Valori: {', '.join(COLLECTIONS)}")
    parser.add_argument("--recreate", nargs="+", default=[], metavar="BENCHMARK",
                        help="crea una nuova collection invece di aggiornare quella esistente "
                             "(la vecchia resta su DAVE e va eliminata a mano)")
    parser.add_argument("--wait", action="store_true",
                        help="attende che DAVE abbia indicizzato tutti i file e ricarica quelli falliti")
    args = parser.parse_args()
    unknown = [b for b in args.benchmarks + args.recreate if b not in COLLECTIONS]
    if unknown:
        parser.error(f"benchmark sconosciuti: {', '.join(unknown)}. Valori: {', '.join(COLLECTIONS)}")

    try:
        client = DaveClient()
    except Exception as e:
        print(f"Errore di connessione a DAVE: {e}")
        return

    print("Connesso a DAVE con successo!")

    mapping = _load_json(COLLECTION_DB)
    doc_cache = _load_doc_cache(mapping)
    jobs = _load_json(JOBS_DB)

    for name in args.benchmarks or list(COLLECTIONS):
        corpus = Corpus(COLLECTIONS[name])
        missing = corpus.missing_paths()
        if missing:
            print(f"[{name}] Corpus non trovato: {', '.join(str(p) for p in missing)}")
            continue

        collection_id = mapping.get(name)
        if not collection_id or name in args.recreate:
            old_id = collection_id
            collection_id = client.create_collection(name)
            mapping[name] = collection_id
            _save_json(COLLECTION_DB, mapping)
            print(f"[{name}] Creata nuova collection (ID: {collection_id})"
                  + (f"; la precedente ({old_id}) non è più usata" if old_id else ""))

        # i file falliti nelle esecuzioni precedenti tornano tra quelli da caricare
        check_jobs(client, collection_id, name, doc_cache, jobs)
        upload_corpus(client, collection_id, corpus, name, doc_cache, jobs)
        if not args.wait:
            continue

        for round_ in range(1, MAX_UPLOAD_ROUNDS + 1):
            pending, failed = check_jobs(client, collection_id, name, doc_cache, jobs)
            while pending:
                print(f"[{name}] DAVE sta ancora indicizzando ({len(pending)} job in corso)...")
                time.sleep(JOB_POLL_SECONDS)
                pending, new_failed = check_jobs(client, collection_id, name, doc_cache, jobs)
                failed += new_failed
            if not failed:
                print(f"[{name}] Tutti i documenti sono indicizzati su DAVE.")
                break
            if round_ == MAX_UPLOAD_ROUNDS:
                print(f"[{name}] {len(failed)} documenti ancora non indicizzati dopo {MAX_UPLOAD_ROUNDS} tentativi: "
                      "rilancia lo script per riprovare.")
                break
            print(f"[{name}] Ricarico {len(failed)} documenti falliti (tentativo {round_ + 1}/{MAX_UPLOAD_ROUNDS})...")
            upload_corpus(client, collection_id, corpus, name, doc_cache, jobs)


if __name__ == "__main__":
    main()
