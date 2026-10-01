import json
import os
import bz2

def converti_crag(input_path, output_path):
    print(f"Lettura del file CRAG da {input_path}...")
    
    mapping = {
        "false_premise": "Verifica_Controfattuale",
        "comparison": "Comparison",
        "aggregation": "Aggregation",
        "multi-hop": "Multi_Hop",
        "set": "Set",
    }
    
    def _iter_items():
        if input_path.endswith(".bz2"):
            with bz2.open(input_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            with open(input_path, 'r', encoding='utf-8-sig') as f:
                dati_grezzi = json.load(f)
                if not isinstance(dati_grezzi, list):
                    dati_grezzi = [dati_grezzi]
                for item in dati_grezzi:
                    yield item

    count = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for i, item in enumerate(_iter_items()):
            contesto_unito = ""
            risultati = item.get('search_results', [])
            for j, res in enumerate(risultati[:10]):
                fonte = res.get('page_name', 'Web')
                page_result = res.get('page_result', '')
                contesto_unito += f"[Fonte: {fonte}] {page_result}\n"
            
            tipo_originale = item.get('question_type', 'sconosciuto')
            tipo_originale_norm = str(tipo_originale).lower().strip()
            categoria_dave = mapping.get(tipo_originale_norm, "Estrazione_Esatta")

            record = {
                "id": f"crag_{i+1:04d}",
                "source": "CRAG",
                "query_type": categoria_dave, 
                "question": item.get("query", ""),
                "context": contesto_unito.strip(),
                "gold_answer": str(item.get("answer", ""))
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
            if count % 500 == 0:
                print(f"  Elaborate {count} query...")

    print(f"CRAG completato! Estratte {count} query e salvate in {output_path}")

# --- ESECUZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Usa il dataset completo (bz2) se disponibile, altrimenti fallback su example_data
CRAG_FULL = os.path.normpath(os.path.join(BASE_DIR, "..", "Benchmarks", "CRAG", "data", "crag_task_1_and_2_dev_v4.jsonl.bz2"))
CRAG_EXAMPLE = os.path.normpath(os.path.join(BASE_DIR, "..", "Benchmarks", "CRAG", "example_data", "dev_data.json"))
CRAG_OUTPUT = os.path.join(BASE_DIR, "Tutto_CRAG_DAVE.jsonl")

if os.path.exists(CRAG_FULL):
    converti_crag(CRAG_FULL, CRAG_OUTPUT)
elif os.path.exists(CRAG_EXAMPLE):
    print(f"Dataset completo non trovato, uso example_data...")
    converti_crag(CRAG_EXAMPLE, CRAG_OUTPUT)
else:
    print(f"Errore: nessun file CRAG trovato.")