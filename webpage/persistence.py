import json
import uuid
from datetime import datetime
from pathlib import Path

# folder where evaluation JSON files are saved
EVAL_DIR = Path(__file__).resolve().parents[1] / "valutazioni"
EVAL_DIR.mkdir(exist_ok=True)


def save_evaluation(results: list[dict], model: str, endpoint: str, threshold: float,
                    sample_size: int, benchmarks_used: list[str], rag_engine: str = "Unknown",
                    name: str = "") -> dict:
    """Saves the evaluation results to a JSON file."""
    import re
    eval_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now()

    clean_name = name.strip() if name else ""
    if not clean_name:
        clean_name = f"eval_{timestamp.strftime('%Y-%m-%d_%H-%M')}_{model.split(':')[0]}"

    safe_filename = re.sub(r'[\\/*?:"<>|\s]+', '_', clean_name).strip('._')
    if not safe_filename:
        safe_filename = f"eval_{eval_id}"

    # calculate aggregate averages with pandas
    import pandas as pd
    df = pd.DataFrame(results)

    # if a column is missing, set 0.0 as default
    aggregate = {
        "answer_correctness": float(df['correctness_voto'].mean()) if 'correctness_voto' in df else 0.0,
        "faithfulness": float(df['faithfulness_voto'].mean()) if 'faithfulness_voto' in df else 0.0,
        "context_precision": float(df['retrieval_voto'].mean()) if 'retrieval_voto' in df else 0.0,
        "answer_relevancy": float(df['relevancy_voto'].mean()) if 'relevancy_voto' in df else 0.0,
        "context_recall": float(df['recall_voto'].mean()) if 'recall_voto' in df else 0.0,
    }

    # breakdown per benchmark
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
                "count": len(src_df),
            }

    # aggregate error pattern counts
    error_patterns = {}
    if 'error_pattern' in df.columns:
        for pattern, count in df['error_pattern'].value_counts().items():
            error_patterns[str(pattern)] = int(count)

    filepath = EVAL_DIR / f"{safe_filename}_{eval_id}.json"

    evaluation = {
        "id": eval_id,
        "name": clean_name,
        "filepath": str(filepath),
        "timestamp": timestamp.isoformat(),
        "model": model,
        "endpoint": endpoint,
        "threshold": threshold,
        "sample_size": sample_size,
        "benchmarks_used": benchmarks_used,
        "rag_engine": rag_engine,
        "aggregate": aggregate,
        "per_benchmark": per_benchmark,
        "error_patterns": error_patterns,
        "results": results,
    }

    filepath.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding='utf-8')
    return evaluation


def list_evaluations() -> list[dict]:
    """Returns the list of saved evaluations (without full results, which are too heavy)."""
    evals = []
    for f in EVAL_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if "id" not in data or "name" not in data:
                continue
            # take only metadata; load results only when needed
            evals.append({
                "id": data["id"],
                "name": data["name"],
                "timestamp": data.get("timestamp", ""),
                "model": data.get("model", "Unknown"),
                "endpoint": data.get("endpoint", ""),
                "threshold": data.get("threshold", 0.0),
                "sample_size": data.get("sample_size", 0),
                "benchmarks_used": data.get("benchmarks_used", []),
                "rag_engine": data.get("rag_engine", "Unknown"),
                "aggregate": data.get("aggregate", {}),
                "per_benchmark": data.get("per_benchmark", {}),
                "error_patterns": data.get("error_patterns", {}),
                "num_queries": len(data.get("results", [])),
                "filepath": str(f),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            # corrupted or incomplete file, skip it
            continue
    evals.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return evals


def load_evaluation(filepath: str) -> dict:
    # loads a complete evaluation from JSON
    return json.loads(Path(filepath).read_text(encoding='utf-8'))


def delete_evaluation(filepath: str) -> bool:
    # deletes the file, returns True if successful
    try:
        Path(filepath).unlink()
        return True
    except OSError:
        return False
