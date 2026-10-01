import json
import os
import time
import glob
from pathlib import Path
from tqdm import tqdm
from dave_client import DaveClient

BASE_DIR = Path(__file__).resolve().parent

def upload_docs_in_batches(client: DaveClient, collection_id: str, docs: list, benchmark_name: str):
    DOC_CACHE_DB = BASE_DIR / ".dave_docs_cache.json"
    doc_cache = {}
    if DOC_CACHE_DB.exists():
        with open(DOC_CACHE_DB, "r") as f:
            doc_cache = json.load(f)
            
    if benchmark_name not in doc_cache:
        doc_cache[benchmark_name] = []
        
    already_uploaded = set(doc_cache[benchmark_name])
    new_docs = [d for d in docs if d["fileName"] not in already_uploaded and d.get("content", "").strip()]
    
    if not new_docs:
        print(f"[{benchmark_name}] Tutti i documenti ({len(docs)}) sono già presenti in cache. Salto l'upload.")
        return
        
    print(f"[{benchmark_name}] {len(docs) - len(new_docs)} doc già caricati. Inizio upload dei {len(new_docs)} nuovi doc...")

    # Tronca documenti troppo grandi (>500KB) - il server non li regge
    MAX_DOC_CHARS = 500_000
    truncated = 0
    for doc in new_docs:
        if len(doc.get("content", "")) > MAX_DOC_CHARS:
            doc["content"] = doc["content"][:MAX_DOC_CHARS] + "\n[... troncato ...]"
            truncated += 1
    if truncated:
        print(f"[{benchmark_name}] Troncati {truncated} documenti troppo grandi (>{MAX_DOC_CHARS//1000}KB).")

    # Batch conservativo: max 1MB payload o 20 docs
    MAX_BATCH_BYTES = 1 * 1024 * 1024  # 1 MB
    MAX_BATCH_DOCS = 20
    MAX_RETRIES = 3
    SLEEP_BETWEEN = 5  # secondi tra batch
    
    batches = []
    current_batch = []
    current_size = 0
    for doc in new_docs:
        doc_size = len(doc.get("content", "").encode("utf-8"))
        if current_batch and (current_size + doc_size > MAX_BATCH_BYTES or len(current_batch) >= MAX_BATCH_DOCS):
            batches.append(current_batch)
            current_batch = []
            current_size = 0
        current_batch.append(doc)
        current_size += doc_size
    if current_batch:
        batches.append(current_batch)

    for batch_idx in tqdm(range(len(batches)), desc=f"[{benchmark_name}] Uploading", unit="batch"):
        batch = batches[batch_idx]
        job_id = None
        
        # Retry con backoff esponenziale
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                job = client.create_txt_upload_job(collection_id, batch)
                job_id = job["jobId"]
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = 10 * (2 ** (attempt - 1))
                    tqdm.write(f"[{benchmark_name}] Batch {batch_idx+1} tentativo {attempt}/{MAX_RETRIES} fallito: {e}. Riprovo tra {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"[{benchmark_name}] Batch {batch_idx+1} fallito dopo {MAX_RETRIES} tentativi: {e}. Salto.")
                    job_id = None
        
        if not job_id:
            continue
        
        # Fire-and-forget: assumiamo che se il job è stato creato (job_id presente),
        # il server lo processerà in modo asincrono. Aggiorniamo subito la cache.
        doc_cache[benchmark_name].extend([d["fileName"] for d in batch])
        with open(DOC_CACHE_DB, "w") as f:
            json.dump(doc_cache, f)
        
        time.sleep(SLEEP_BETWEEN)

def ingest_file(client: DaveClient, filepath: str, benchmark_name: str):
    print(f"[{benchmark_name}] Ingesting {filepath}...")
    
    # Leggiamo il mapping salvato in precedenza
    COLLECTION_DB = BASE_DIR / ".dave_collections.json"
    mapping = {}
    if COLLECTION_DB.exists():
        with open(COLLECTION_DB, "r") as f:
            mapping = json.load(f)
            
    collection_id = mapping.get(benchmark_name)
    if not collection_id:
        collection_id = client.create_collection(benchmark_name)
        mapping[benchmark_name] = collection_id
        with open(COLLECTION_DB, "w") as f:
            json.dump(mapping, f)
        print(f"Creata nuova collection per {benchmark_name} (ID: {collection_id})")

    # Leggec i documenti
    docs = []
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            # Gestione array di dict o dict singolo/dizionario di dizionari
            if isinstance(data, list):
                for item in data:
                    text = item.get("text", str(item))
                    doc_id = item.get("id", f"doc_{len(docs)}")
                    docs.append({"fileName": f"{doc_id}.txt", "content": text})
            elif isinstance(data, dict):
                # Potrebbe essere il formato di MultiHop: { "id1": {"text": ...}, "id2": {"text": ...} }
                # Oppure un record singolo
                if len(data) > 0 and isinstance(next(iter(data.values())), dict) and "text" in next(iter(data.values())):
                    for k, v in data.items():
                        text = v.get("text", str(v))
                        docs.append({"fileName": f"{k}.txt", "content": text})
                else:
                    text = data.get("text", str(data))
                    docs.append({"fileName": "doc_0.txt", "content": text})
    elif filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                doc_id = data.get("_id") or data.get("document_id") or data.get("id") or f"doc_{len(docs)}"
                if text:
                    docs.append({"fileName": f"{doc_id}.txt", "content": text})

    if not docs:
        print(f"Nessun documento trovato in {filepath}")
        return

    print(f"Trovati {len(docs)} documenti in {filepath}. Inizio upload a lotti (max 100 alla volta)...")
    upload_docs_in_batches(client, collection_id, docs, benchmark_name)


def main():
    try:
        client = DaveClient()
    except Exception as e:
        print(f"Errore di connessione a DAVE: {e}")
        return

    print("Connesso a DAVE con successo!")

    COLLECTION_DB = BASE_DIR / ".dave_collections.json"
    mapping = {}
    if COLLECTION_DB.exists():
        with open(COLLECTION_DB, "r") as f:
            mapping = json.load(f)

    # Ingest LegalBench (dalla cartella corpus)
    legalbench_dir = BASE_DIR / "Benchmarks" / "legalbenchrag" / "data" / "corpus"
    if legalbench_dir.exists():
        for root, _, files in os.walk(legalbench_dir):
            for file in files:
                if file.endswith(".txt"):
                    # Pass the single text files or process them differently
                    pass
        # Rielaborazione per LegalBench: raggruppare i .txt in un payload
        docs = []
        for root, _, files in os.walk(legalbench_dir):
            for file in files:
                if file.endswith(".txt"):
                    try:
                        with open(os.path.join(root, file), "r", encoding="utf-8-sig") as f:
                            docs.append({"fileName": file, "content": f.read()})
                    except Exception:
                        pass
        if docs:
            print(f"[LegalBench] Trovati {len(docs)} documenti.")
            collection_id = mapping.get("LegalBench")
            if not collection_id:
                collection_id = client.create_collection("LegalBench")
                mapping["LegalBench"] = collection_id
                with open(COLLECTION_DB, "w") as f: json.dump(mapping, f)
            print("Inizio upload a lotti per LegalBench...")
            upload_docs_in_batches(client, collection_id, docs, "LegalBench")
            print("Upload completato per LegalBench.")
    else:
        print(f"Cartella non trovata: {legalbench_dir}")

    # Ingest MultiHop
    multihop_path = BASE_DIR / "Benchmarks" / "MultiHop-RAG" / "dataset" / "corpus.json"
    if multihop_path.exists():
        ingest_file(client, str(multihop_path), "MultiHop")
    else:
        print(f"File non trovato: {multihop_path}")

    # Ingest CRAG
    crag_path = BASE_DIR / "Run" / "Tutto_CRAG_DAVE.jsonl"
    if not crag_path.exists():
        crag_path = BASE_DIR / "Run" / "Tutto_CRAG_DAVE.json"
    if crag_path.exists():
        try:
            def _iter_crag(p):
                with open(p, "r", encoding="utf-8-sig") as f:
                    if str(p).endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                yield json.loads(line)
                    else:
                        for item in json.load(f):
                            yield item

            docs = []
            doc_ids_visti = set()
            for item in _iter_crag(crag_path):
                contesto_grezzo = item.get("context", "")
                lines = [line.strip() for line in contesto_grezzo.splitlines() if line.strip()]
                
                docs_temporanei, current = [], []
                for line in lines:
                    is_new_doc = line.startswith("[") and "]" in line
                    if is_new_doc and current:
                        docs_temporanei.append(" ".join(current).strip())
                        current = [line]
                    else:
                        current.append(line)
                if current:
                    docs_temporanei.append(" ".join(current).strip())
                    
                for doc in docs_temporanei:
                    id_fittizio = doc[:50].replace(" ", "_")
                    id_fittizio = "".join(c for c in id_fittizio if c.isalnum() or c == "_")
                    if id_fittizio not in doc_ids_visti and doc.strip():
                        docs.append({"fileName": f"{id_fittizio}.txt", "content": doc})
                        doc_ids_visti.add(id_fittizio)
                        
            if docs:
                print(f"[CRAG] Trovati {len(docs)} documenti.")
                
                # Ricarica il mapping aggiornato per evitare di sovrascrivere MultiHop
                if COLLECTION_DB.exists():
                    with open(COLLECTION_DB, "r") as f:
                        mapping = json.load(f)
                
                collection_id = mapping.get("CRAG")
                if not collection_id:
                    collection_id = client.create_collection("CRAG")
                    mapping["CRAG"] = collection_id
                    with open(COLLECTION_DB, "w") as f: json.dump(mapping, f)
                print("Inizio upload a lotti per CRAG...")
                upload_docs_in_batches(client, collection_id, docs, "CRAG")
                print("Upload completato per CRAG.")
        except Exception as e:
            print(f"Errore caricamento CRAG: {e}")
    else:
        print(f"File non trovato: {crag_path}")

    # MTRAG ha tanti file .jsonl nella cartella corpora/document_level
    mtrag_dir = BASE_DIR / "Benchmarks" / "mt-rag-benchmark" / "corpora" / "document_level"
    if mtrag_dir.exists():
        for jsonl_file in glob.glob(str(mtrag_dir / "**" / "*.jsonl"), recursive=True):
           if os.path.isfile(jsonl_file):
             ingest_file(client, jsonl_file, "MTRAG")
    else:
        print(f"Cartella non trovata: {mtrag_dir}")

    # Ingest SQuAD-it
    squad_path = BASE_DIR / "Run" / "squad_corpus.json"
    if squad_path.exists():
        ingest_file(client, str(squad_path), "SQuAD")
    else:
        print(f"File non trovato: {squad_path}")

    # Ingest OHRBench
    ohrbench_path = BASE_DIR / "Run" / "Tutto_OHRBench_DAVE.jsonl"
    if not ohrbench_path.exists():
        ohrbench_path = BASE_DIR / "Run" / "Tutto_OHRBench_DAVE.json"
    if ohrbench_path.exists():
        try:
            def _iter_ohr(p):
                with open(p, "r", encoding="utf-8-sig") as f:
                    if str(p).endswith(".jsonl"):
                        for line in f:
                            if line.strip():
                                yield json.loads(line)
                    else:
                        for item in json.load(f):
                            yield item

            docs = []
            for item in _iter_ohr(ohrbench_path):
                contesto = item.get("context", "")
                doc_id = item.get("id", "doc_0")
                if contesto.strip():
                    docs.append({"fileName": f"{doc_id}.txt", "content": contesto})
            
            if docs:
                print(f"[OHRBench] Trovati {len(docs)} documenti.")
                
                if COLLECTION_DB.exists():
                    with open(COLLECTION_DB, "r") as f:
                        mapping = json.load(f)
                        
                collection_id = mapping.get("OHRBench")
                if not collection_id:
                    collection_id = client.create_collection("OHRBench")
                    mapping["OHRBench"] = collection_id
                    with open(COLLECTION_DB, "w") as f: json.dump(mapping, f)
                print("Inizio upload a lotti per OHRBench...")
                upload_docs_in_batches(client, collection_id, docs, "OHRBench")
                print("Upload completato per OHRBench.")
        except Exception as e:
            print(f"Errore caricamento OHRBench: {e}")
    else:
        print(f"File non trovato: {ohrbench_path}")

if __name__ == "__main__":
    main()
