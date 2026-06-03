import json
import os
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELLO = "gemma4:31b-cloud"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_LEGAL = os.path.join(BASE_DIR, "50_Esempi_LegalBench.json")
INPUT_MULTIHOP = os.path.join(BASE_DIR, "Tutto_MULTIHOP.json")
INPUT_CRAG = os.path.join(BASE_DIR, "Tutto_CRAG_DAVE.json")
OUTPUT_REPORT = os.path.join(BASE_DIR, "Report_Classificazione.md")
OUTPUT_IMG = os.path.join(BASE_DIR, "Grafico_Query_DAVE.png")

# Prompt riformulato per non bloccare Gemma
PROMPT_ROUTER = """Sei un assistente AI esperto in classificazione dei dati.
Il tuo compito è classificare la domanda dell'utente in UNA SOLA di queste 5 categorie:
- Estrazione_Semplice
- Multi_Hop
- Confronto_Aggregazione
- Verifica_Fatti
- Temporale

Devi rispondere ESCLUSIVAMENTE fornendo un oggetto JSON, senza alcun testo introduttivo o conclusivo. Usa questa struttura esatta:
{{"categoria": "inserisci_qui_la_categoria", "motivo": "breve spiegazione del perché"}}

Domanda da classificare: "{domanda}"
"""

def pulisci_json(testo_grezzo: str) -> str:
    match = re.search(r'\{.*\}', testo_grezzo, re.DOTALL)
    if match:
        return match.group(0)
    return testo_grezzo

def get_classificazione(domanda: str) -> tuple[str, str]:
    testo_grezzo = ""
    try:
        payload = {
            "model": MODELLO,
            "prompt": PROMPT_ROUTER.format(domanda=domanda),
            "stream": False,
            "temperature": 0.1,
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        
        testo_grezzo = r.json().get('response', '').strip()
        if not testo_grezzo:
            print("Il modello Ollama ha restituito una risposta vuota.")
            return "Errore_Vuoto", "N/A"
            
        testo_pulito = pulisci_json(testo_grezzo)
        res = json.loads(testo_pulito)
        if isinstance(res, list) and len(res) > 0:
            res = res[0]
            
        res_safe = {str(k).lower(): str(v) for k, v in res.items()}
        cat = res_safe.get('categoria', res_safe.get('category', 'Sconosciuta'))
        mot = res_safe.get('motivo', res_safe.get('reason', 'N/A'))
        
        return cat, mot
        
    except json.JSONDecodeError:
        print(f"  -> ⚠️ Errore JSON. Il modello ha generato questo testo illeggibile:\n{testo_grezzo}")
        return "Errore_Parsing", "N/A"
    except Exception as e:
        print(f"  -> ⚠️ Errore di connessione/esecuzione: {e}")
        return "Errore_Tecnico", "N/A"

def _load_samples(file_paths: list[str], sample_size: int) -> list[dict]:
    data = []
    for f_path in file_paths:
        if os.path.exists(f_path):
            with open(f_path, 'r', encoding='utf-8-sig') as f:
                items = json.load(f)
                take = min(sample_size, len(items))
                data.extend(random.sample(items, take))
    return data


def genera_report():
    print("🚀 Classificazione domande in corso...")
    data = _load_samples([INPUT_LEGAL, INPUT_MULTIHOP, INPUT_CRAG], sample_size=10)

    risultati = []
    for item in data:
        print(f"Analizzo: {item['id']}...")
        cat, mot = get_classificazione(item['question'])
        risultati.append({"ID": item['id'], "Domanda": item['question'], "Categoria": cat, "Motivo": mot})
    
    df = pd.DataFrame(risultati)
    
    plt.figure(figsize=(10, 8))
    df['Categoria'].value_counts().plot(kind='pie', autopct='%1.1f%%', title="Distribuzione Tipologia Query")
    plt.ylabel("")
    plt.savefig(OUTPUT_IMG)
    plt.close()

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# 📊 Report Classificazione Query DAVE\n\n")
        f.write(f"Questo report analizza la natura delle domande poste al sistema.\n\n")
        f.write(df.to_markdown(index=False))
    
    print(f"\n✅ Report creato con successo: {OUTPUT_REPORT}")
    print(f"✅ Grafico salvato come: {OUTPUT_IMG}")

if __name__ == "__main__":
    genera_report()