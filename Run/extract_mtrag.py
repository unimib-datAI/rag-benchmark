import json
import os
import re


def _get_corpus_key(domain: str) -> str | None:
    d = (domain or "").lower()
    if "clapnq" in d:
        return "clapnq"
    if "fiqa" in d:
        return "fiqa"
    if "govt" in d:
        return "govt"
    if "cloud" in d:
        return "cloud"
    return None


def _parse_base_id(doc_id: str) -> str:
    if not doc_id:
        return ""
    if "_" not in doc_id:
        return doc_id
    base, tail = doc_id.split("_", 1)
    # Solo i formati clapnq come "827849752_115-357" vanno splittati;
    # gli ID cloud come "ibmcld_00001" restano interi.
    if re.match(r'^\d+-\d+$', tail):
        return base
    return doc_id


def _parse_segment_start(doc_id: str) -> int:
    if not doc_id or "_" not in doc_id:
        return 0
    tail = doc_id.split("_", 1)[1]
    start_str = tail.split("-", 1)[0]
    try:
        return int(start_str)
    except ValueError:
        return 0


def _load_document_corpus(base_dir: str, corpus_key: str) -> dict[str, str]:
    corpus_path = os.path.normpath(
        os.path.join(
            base_dir,
            "..",
            "Benchmarks",
            "mt-rag-benchmark",
            "corpora",
            "document_level",
            f"{corpus_key}.jsonl",
            f"{corpus_key}.jsonl",
        )
    )

    doc_segments: dict[str, list[tuple[int, str]]] = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get("_id") or data.get("document_id") or data.get("id") or ""
            base_id = _parse_base_id(doc_id)
            if not base_id:
                continue
            start = _parse_segment_start(doc_id)
            doc_segments.setdefault(base_id, []).append((start, data.get("text", "")))

    docs: dict[str, str] = {}
    for base_id, segments in doc_segments.items():
        segments.sort(key=lambda x: x[0])
        docs[base_id] = " ".join(seg_text for _, seg_text in segments).strip()
    return docs

def converti_mtrag(input_path, output_path):
    print(f"Lettura dei dati MTRAG da {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        dati_grezzi = json.load(f)
        
    dati_unificati = []
    corpora_cache: dict[str, dict[str, str]] = {}
    
    for c_idx, conversazione in enumerate(dati_grezzi):
        messaggi = conversazione.get("turns", conversazione.get("messages", []))
        corpus_key = _get_corpus_key(conversazione.get("domain", ""))
        if corpus_key and corpus_key not in corpora_cache:
            corpora_cache[corpus_key] = _load_document_corpus(BASE_DIR, corpus_key)
        corpus_docs = corpora_cache.get(corpus_key or "", {})
        turn_count = 1
        domanda_corrente = ""
        
        for msg in messaggi:
            speaker = msg.get("speaker", "").lower()
            testo = msg.get("text", "").strip()
            
            if speaker == "user":
                domanda_corrente = testo
                
            elif speaker == "agent":
                risposta_gold = testo
                contesto_doc = ""
                for ctx in msg.get("contexts", []):
                    feedbacks = ctx.get("feedback", {}).get("relevant", {})
                    is_relevant = any(f.get("value") == "yes" for f in feedbacks.values())
                    if is_relevant or not feedbacks:
                        base_id = _parse_base_id(ctx.get("document_id", ""))
                        full_text = corpus_docs.get(base_id)
                        if not full_text:
                            full_text = ctx.get("text", "")
                        contesto_doc += f"[{ctx.get('title', 'Doc')}] {full_text}\n"
                
                record = {
                    "id": f"mtrag_c{c_idx+1:03d}_t{turn_count:02d}", 
                    "source": "MTRAG",
                    "query_type": "Multi_Turn_Conversation",
                    "question": domanda_corrente, 
                    "context": contesto_doc.strip(),
                    "gold_answer": risposta_gold
                }
                dati_unificati.append(record)
                turn_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dati_unificati:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"MTRAG completato! Salvato in {output_path}.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MTRAG_INPUT = os.path.normpath(os.path.join(BASE_DIR, "..", "Benchmarks", "mt-rag-benchmark", "mtrag-human","conversations", "conversations.json")) 
MTRAG_OUTPUT = os.path.join(BASE_DIR, "Tutto_MTRAG_DAVE.jsonl")

if os.path.exists(MTRAG_INPUT): 
    converti_mtrag(MTRAG_INPUT, MTRAG_OUTPUT)
else: 
    print(f"❌ ATTENZIONE: File non trovato!")
    print(f"Sto cercando esattamente qui: {MTRAG_INPUT}")
    print("Controlla che la cartella si chiami così e che il file .json sia al suo interno.")