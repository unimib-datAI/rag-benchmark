import json
from pathlib import Path

# Percorsi dei file
BASE_DIR = Path(__file__).resolve().parent.parent
SQUAD_JSON_PATH = BASE_DIR / "Benchmarks" / "squad-it" / "SQuAD_it-test.json" / "SQuAD_it-test.json"
OUTPUT_BENCHMARK_PATH = BASE_DIR / "Run" / "Tutto_SQUAD.jsonl"
OUTPUT_CORPUS_PATH = BASE_DIR / "Run" / "squad_corpus.json"


def main():
    if not SQUAD_JSON_PATH.exists():
        # Fallback nel caso in cui il file json sia direttamente in squad-it e non nella sottocartella
        fallback_path = BASE_DIR / "Benchmarks" / "squad-it" / "SQuAD_it-test.json"
        if fallback_path.is_file():
            squad_path = fallback_path
        else:
            print(f"Errore: File SQuAD non trovato in {SQUAD_JSON_PATH}")
            return
    else:
        squad_path = SQUAD_JSON_PATH

    print(f"Leggo i dati da: {squad_path}")
    with open(squad_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    benchmark_data = []
    unique_contexts = set()
    corpus_data = []
    
    # Navighiamo l'albero JSON di SQuAD
    # Struttura: {"data": [ {"title": "...", "paragraphs": [ {"context": "...", "qas": [...] } ] } ] }
    
    for article in data.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            
            # Aggiungiamo al corpus se non l'abbiamo già visto
            if context and context not in unique_contexts:
                unique_contexts.add(context)
                corpus_data.append({
                    "id": f"squad_doc_{len(unique_contexts)}",
                    "text": context
                })
            
            for qa in paragraph.get("qas", []):
                question = qa.get("question", "")
                qa_id = qa.get("id", "")
                
                # SQuAD ha un array di risposte, prendiamo la prima
                answers = qa.get("answers", [])
                if not answers:
                    continue
                gold_answer = answers[0].get("text", "")
                
                # Aggiungiamo al nostro formato SuperBenchmark
                benchmark_data.append({
                    "id": f"squad_{qa_id}",
                    "source": "SQuAD-it",
                    "question": question,
                    "gold_answer": gold_answer,
                    "context": context
                })

    # Salvataggio Benchmark
    with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        for record in benchmark_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Salvataggio Corpus
    with open(OUTPUT_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    print(f"Creato con successo {OUTPUT_BENCHMARK_PATH}")
    print(f"Totale query estratte: {len(benchmark_data)}")
    print(f"Creato con successo {OUTPUT_CORPUS_PATH}")
    print(f"Totale documenti unici estratti per ingestion: {len(corpus_data)}")

if __name__ == "__main__":
    main()