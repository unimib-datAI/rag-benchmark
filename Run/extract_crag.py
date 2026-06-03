import json
import os

def converti_crag(input_path, output_path):
    print(f"Lettura del file CRAG da {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        dati_grezzi = json.load(f) 
        
    dati_unificati = []
    
    for i, item in enumerate(dati_grezzi):
        
        # Estraiamo il contenuto HTML completo (page_result) dai risultati di ricerca
        contesto_unito = ""
        risultati = item.get('search_results', [])
        
        # Mettiamo un limite ai documenti (massimo i primi 10 risultati) per non saturare la memoria dell'LLM
        for j, res in enumerate(risultati[:10]): 
            fonte = res.get('page_name', 'Web')
            page_result = res.get('page_result', '')
            contesto_unito += f"[Fonte: {fonte}] {page_result}\n"
        
        tipo_originale = item.get('question_type', 'sconosciuto')
        tipo_originale_norm = str(tipo_originale).lower().strip()

        # Mapping più dettagliato delle categorie (fallback: Estrazione_Esatta)
        mapping = {
                "false_premise": "Verifica_Controfattuale",
                "comparison": "Comparison",
                "aggregation": "Aggregation",
                "multi-hop": "Multi_Hop",
                "set": "Set",
          }

        categoria_dave = mapping.get(tipo_originale_norm, "Estrazione_Esatta")
        record = {
            "id": f"crag_{i+1:03d}", # Usiamo 3 cifre (es. crag_001) visti i grandi numeri
            "source": "CRAG",
            "query_type": categoria_dave, 
            "question": item.get("query", ""),
            "context": contesto_unito.strip(),
            "gold_answer": str(item.get("answer", "")) # Assicuriamoci sia sempre una stringa
        }
        
        dati_unificati.append(record)

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        json.dump(dati_unificati, f, indent=4, ensure_ascii=False)
        
    print(f"✅ OPERAZIONE COMPLETATA! Estratte TUTTE le {len(dati_unificati)} query da CRAG e salvate in {output_path}!")

# --- ESECUZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRAG_INPUT = os.path.normpath(os.path.join(BASE_DIR, "..", "Benchmarks", "CRAG", "example_data", "dev_data.json")) 
CRAG_OUTPUT = os.path.join(BASE_DIR, "Tutto_CRAG_DAVE.json")

if os.path.exists(CRAG_INPUT):
    converti_crag(CRAG_INPUT, CRAG_OUTPUT)
else:
    print(f"Errore: File non trovato in {CRAG_INPUT}")