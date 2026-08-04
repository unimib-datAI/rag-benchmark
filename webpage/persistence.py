import json
import uuid
from datetime import datetime
from pathlib import Path

# cartella dove salvo i json delle valutazioni
EVAL_DIR = Path(__file__).resolve().parents[1] / "valutazioni"
EVAL_DIR.mkdir(exist_ok=True)


def save_evaluation(results: list[dict], model: str, endpoint: str, threshold: float,
                    sample_size: int, benchmarks_used: list[str]) -> dict:
    """Salva i risultati di una valutazione su file JSON."""
    eval_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now()
    name = f"eval_{timestamp.strftime('%Y-%m-%d_%H-%M')}_{model.split(':')[0]}"

    # calcolo medie aggregate con pandas
    import pandas as pd
    df = pd.DataFrame(results)

    # se manca una colonna metto 0.0 come default
    aggregate = {
        "answer_correctness": float(df['correctness_voto'].mean()) if 'correctness_voto' in df else 0.0,
        "faithfulness": float(df['faithfulness_voto'].mean()) if 'faithfulness_voto' in df else 0.0,
        "context_precision": float(df['retrieval_voto'].mean()) if 'retrieval_voto' in df else 0.0,
        "answer_relevancy": float(df['relevancy_voto'].mean()) if 'relevancy_voto' in df else 0.0,
        "context_recall": float(df['recall_voto'].mean()) if 'recall_voto' in df else 0.0,
        "semantic_sim": float(df['semantic_sim'].mean()) if 'semantic_sim' in df else 0.0,
    }

    # breakdown per singolo benchmark
    per_benchmark = {}
    if 'source' in df.columns:
        for src in df['source'].unique():
            src_df = df[df['source'] == src]
            per_benchmark[src] = {
                "answer_correctness": float(src_df['correctness_voto'].mean()),
                "faithfulness": float(src_df['faithfulness_voto'].mean()),
                "context_precision": float(src_df['retrieval_voto'].mean()),
                "answer_relevancy": float(src_df['relevancy_voto'].mean()) if 'relevancy_voto' in src_df else 0.0,
                "context_recall": float(src_df['recall_voto'].mean()) if 'recall_voto' in src_df else 0.0,
                "semantic_sim": float(src_df['semantic_sim'].mean()) if 'semantic_sim' in src_df else 0.0,
                "count": len(src_df),
            }

    evaluation = {
        "id": eval_id,
        "name": name,
        "timestamp": timestamp.isoformat(),
        "model": model,
        "endpoint": endpoint,
        "threshold": threshold,
        "sample_size": sample_size,
        "benchmarks_used": benchmarks_used,
        "aggregate": aggregate,
        "per_benchmark": per_benchmark,
        "results": results,
    }

    filepath = EVAL_DIR / f"{name}_{eval_id}.json"
    filepath.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding='utf-8')
    return evaluation


def list_evaluations() -> list[dict]:
    """Ritorna la lista delle valutazioni salvate (senza i risultati completi, troppo pesanti)."""
    evals = []
    for f in sorted(EVAL_DIR.glob("eval_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            # prendo solo i metadati, i results li carico solo quando servono
            evals.append({
                "id": data["id"],
                "name": data["name"],
                "timestamp": data["timestamp"],
                "model": data["model"],
                "endpoint": data.get("endpoint", ""),
                "threshold": data.get("threshold", 0.0),
                "sample_size": data.get("sample_size", 0),
                "benchmarks_used": data.get("benchmarks_used", []),
                "aggregate": data.get("aggregate", {}),
                "per_benchmark": data.get("per_benchmark", {}),
                "num_queries": len(data.get("results", [])),
                "filepath": str(f),
            })
        except (json.JSONDecodeError, KeyError):
            # file corrotto o incompleto, lo skippo
            continue
    return evals


def load_evaluation(filepath: str) -> dict:
    # carica una valutazione completa dal json
    return json.loads(Path(filepath).read_text(encoding='utf-8'))


def delete_evaluation(filepath: str) -> bool:
    # cancella il file, ritorna True se ok
    try:
        Path(filepath).unlink()
        return True
    except OSError:
        return False
