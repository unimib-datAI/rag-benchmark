import json
import os

def converti_ohrbench(input_path, output_path):
    print(f"Lettura dei dati OHRBench da {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        dati_grezzi = json.load(f)
        
    dati_unificati = []
    
    for i, item in enumerate(dati_grezzi):
        
        domande = item.get("questions", [])
        risposte = item.get("answers", [])
        
        # Estrazione sicura per domande e risposte
        domanda = domande[0] if isinstance(domande, list) and len(domande)>0 else str(domande)
        risposta_gold = risposte[0] if isinstance(risposte, list) and len(risposte)>0 else str(risposte)
        
        tipo_evidenza = str(item.get("evidence_source", "text")).lower()
        
        categoria_dave = "Estrazione_Semplice"
        if tipo_evidenza in ["table", "formula", "chart"]:
            categoria_dave = "Confronto_Aggregazione"
            
        # --- FIX: GESTIONE DELL'EVIDENCE CONTEXT COME LISTA ---
        contesto_raw = item.get("evidence_context", "")
        if isinstance(contesto_raw, list):
            # Trasforma tutti gli elementi della lista in stringhe e li unisce con "a capo"
            contesto_str = "\n".join([str(c) for c in contesto_raw])
        else:
            # Se per caso in alcuni è già una stringa, la teniamo com'è
            contesto_str = str(contesto_raw)
            
        record = {
            "id": f"ohr_{i+1:04d}",
            "source": "OHRBench",
            "query_type": categoria_dave,
            "question": domanda.strip(),
            "context": contesto_str.strip(),
            "gold_answer": risposta_gold.strip()
        }
        
        if record["question"] and record["context"]:
            dati_unificati.append(record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dati_unificati:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
    print(f"OHRBench completato! Estratte {len(dati_unificati)} query e salvate in {output_path}.")

# --- ESECUZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OHR_INPUT = os.path.normpath(os.path.join(BASE_DIR, "..", "Benchmarks", "OHR-Bench", "data", "qas.json")) 
OHR_OUTPUT = os.path.join(BASE_DIR, "Tutto_OHRBench_DAVE.jsonl")

if os.path.exists(OHR_INPUT): 
    converti_ohrbench(OHR_INPUT, OHR_OUTPUT)
else: 
    print(f"❌ File non trovato: {OHR_INPUT}")