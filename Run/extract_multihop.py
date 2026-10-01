import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.normpath(
    os.path.join(BASE_DIR, "..", "Benchmarks", "MultiHop-RAG", "dataset", "MultiHopRAG.json")
)
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Tutto_MULTIHOP.jsonl")


def converti_multihop(input_file, output_file):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        dati_grezzi = json.load(f)

    dati_unificati = []

    for i, item in enumerate(dati_grezzi):
        contesto = ""
        if "evidence_list" in item:
            for ev in item["evidence_list"]:
                contesto += f"[{ev['source']}] {ev['fact']}\n"

        record = {
            "id": f"mh_{i+1:04d}",
            "source": "MultiHop-RAG",
            "query_type": item.get("question_type", "unknown"),
            "question": item.get("query", ""),
            "context": contesto.strip(),
            "gold_answer": item.get("answer", ""),
        }
        dati_unificati.append(record)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in dati_unificati:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Estratte {len(dati_unificati)} query da MultiHop-RAG e salvate in {output_file}!")


if __name__ == "__main__":
    converti_multihop(DEFAULT_INPUT, DEFAULT_OUTPUT)