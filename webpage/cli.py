import sys
import json
import random
from collections import defaultdict
from pathlib import Path

# used to import evaluator and persistence from the same folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from evaluator import run_evaluation
    from persistence import save_evaluation, EVAL_DIR
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# paths and config
RUN_DIR = Path(__file__).resolve().parents[1] / "Run"
CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"

# map benchmark -> corresponding json file
BENCHMARK_MAP = {
    "LEGALBENCH": "Tutto_LEGALBENCH.json",
    "MULTIHOP": "Tutto_MULTIHOP.json",
    "CRAG": "Tutto_CRAG_DAVE.json",
    "MTRAG": "Tutto_MTRAG_DAVE.json",
    "OHRBENCH": "Tutto_OHRBench_DAVE.json",
    "SQUAD": "Tutto_SQUAD.json",
}
ALL_BENCHMARKS = list(BENCHMARK_MAP.keys())

HEADER = r"""
╔══════════════════════════════════════════════════════════╗
║         SuperBenchmark  —  CLI RAG Evaluation            ║
╚══════════════════════════════════════════════════════════╝
"""


# config management (save/load from json)
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


# input and interactive menu

def prompt_input(message, default):
    value = input(f"{message} [{default}]: ").strip()
    return value if value else default


def interactive_menu():
    config = load_config()

    def_endpoint = config.get("endpoint", "http://127.0.0.1:8000")
    def_threshold = config.get("threshold", "0.0")
    def_rag_engine = config.get("motore_rag", "Local RAG (BM25)")
    def_ollama_url = config.get("ollama_url", "http://localhost:11434")
    def_model = config.get("model", "gemma4:31b-cloud")
    def_benchmarks = config.get("benchmarks", "ALL")
    def_sample_size = config.get("sample_size", "3")
    def_selection_mode = config.get("selection_mode", "R")

    while True:
        print("\n--- Evaluation Configuration ---")
        endpoint = prompt_input("Enter RAG Base URL", def_endpoint)
        rag_engine = prompt_input("Select RAG Engine (DAVE or Local RAG)", def_rag_engine)

        while True:
            thresh_str = prompt_input("Enter Similarity Threshold (0.0 - 1.0)", def_threshold)
            try:
                threshold = float(thresh_str)
                break
            except ValueError:
                print("Please enter a valid decimal number.")

        ollama_url = prompt_input("Enter Ollama URL", def_ollama_url)
        model = prompt_input("Enter LLM Judge model", def_model)

        print(f"\nAvailable benchmarks: {', '.join(ALL_BENCHMARKS)}")
        benchmarks_str = prompt_input(
            "Select benchmarks for evaluation (comma-separated, or ALL)",
            def_benchmarks
        )

        if benchmarks_str.upper() == "ALL":
            benchmarks = ALL_BENCHMARKS
        else:
            benchmarks = [b.strip().upper() for b in benchmarks_str.split(",")
                          if b.strip().upper() in ALL_BENCHMARKS]
            if not benchmarks:
                print("No valid benchmarks entered. All will be used.")
                benchmarks = ALL_BENCHMARKS
                benchmarks_str = "ALL"

        while True:
            sample_size_str = prompt_input("Enter number of queries per benchmark", def_sample_size)
            try:
                sample_size = int(sample_size_str)
                break
            except ValueError:
                print("Please enter a valid number.")

        while True:
            mode_str = prompt_input("Select query mode: [R]andom or [F]ixed", def_selection_mode).strip().upper()
            if mode_str in ["R", "F"]:
                selection_mode = "Random" if mode_str == "R" else "Fixed"
                break
            print("Please enter 'R' for Random or 'F' for Fixed.")

        print("\n--- Configuration Summary ---")
        print(f"RAG Base URL:    {endpoint}")
        print(f"RAG Engine:      {rag_engine}")
        print(f"Threshold:       {threshold}")
        print(f"Ollama URL:      {ollama_url}")
        print(f"LLM Model:       {model}")
        print(f"Benchmarks:      {', '.join(benchmarks)}")
        print(f"Queries/Bench:   {sample_size}")
        print(f"Query mode:      {selection_mode}")

        choice = input("\nDo you want to [S]tart the evaluation or [M]odify parameters? [S/M]: ").strip().upper()

        if choice == "S" or choice == "":
            config["endpoint"] = endpoint
            config["motore_rag"] = rag_engine
            config["threshold"] = str(threshold)
            config["ollama_url"] = ollama_url
            config["model"] = model
            config["benchmarks"] = benchmarks_str
            config["sample_size"] = str(sample_size)
            config["selection_mode"] = mode_str
            save_config(config)

            return endpoint, threshold, rag_engine, ollama_url, model, benchmarks, sample_size, selection_mode


# data loading and evaluation

def _load_dataset(benchmarks, sample_size, selection_mode):
    """Load queries from the json files of the selected benchmarks."""
    dataset = []

    for name in benchmarks:
        filename = BENCHMARK_MAP[name]
        path = RUN_DIR / filename
        if not path.exists():
            print(f"  ⚠  File not found: {path}  — benchmark {name} skipped.")
            continue

        with open(path, 'r', encoding='utf-8-sig') as f:
            items = json.load(f)
        if name == "MTRAG":
            conversations = defaultdict(list)
            for item in items:
                conv_id = item["id"].split("_t")[0]
                conversations[conv_id].append(item)

            conv_keys = list(conversations.keys())
            if selection_mode == "Random":
                random.shuffle(conv_keys)
            else:
                conv_keys.sort()

            turns_added = 0
            for c_key in conv_keys:
                if turns_added >= sample_size:
                    break
                for turn in sorted(conversations[c_key], key=lambda x: x["id"]):
                    if turns_added < sample_size:
                        dataset.append(turn)
                        turns_added += 1
        else:
            sz = min(sample_size, len(items))
            if selection_mode == "Random":
                dataset.extend(random.sample(items, sz))
            else:
                dataset.extend(items[:sz])

    return dataset


def _cli_progress(phase, current, total, message):
    # callback to show progress in terminal
    if phase == "fetch":
        print(f"  [{current}/{total}] {message}", end="\r", flush=True)
    elif phase == "ragas":
        print(f"\n  {message}")
    elif phase == "done":
        print(f"\n  {message}")


def _print_results(results):
    n = len(results)
    if n == 0:
        print("\n  No results to show.")
        return

    # calculate averages
    avg_corr = sum(r.get("correctness_voto", 0) for r in results) / n
    avg_faith = sum(r.get("faithfulness_voto", 0) for r in results) / n
    avg_ctx = sum(r.get("retrieval_voto", 0) for r in results) / n
    avg_rel = sum(r.get("relevancy_voto", 0) for r in results) / n

    print("\n📊 Aggregate Metrics:")
    print(f"  Answer Correctness:  {avg_corr:.4f}")
    print(f"  Faithfulness:        {avg_faith:.4f}")
    print(f"  Context Precision:   {avg_ctx:.4f}")
    print(f"  Answer Relevancy:    {avg_rel:.4f}")

    # per-query table
    print("\n📋 Per Query:")
    col_num = 12
    fmt = f"  {{:<15}}  {{:<12}}  {{:>{col_num}}}  {{:>{col_num}}}  {{:>{col_num}}}  {{:>{col_num}}}"
    print(fmt.format("ID", "Benchmark", "Correctness", "Faithfulness", "Ctx Prec.", "Relevancy"))
    print(f"  {'─'*15}  {'─'*12}  {'─'*col_num}  {'─'*col_num}  {'─'*col_num}  {'─'*col_num}")

    for r in results:
        bench = r.get("source", "?").split('-')[0][:12] if '-' in r.get("source", "?") else r.get("source", "?")[:12]
        print(fmt.format(
            r.get("query_id", "?")[:15],
            bench,
            f"{r.get('correctness_voto', 0):.4f}",
            f"{r.get('faithfulness_voto', 0):.4f}",
            f"{r.get('retrieval_voto', 0):.4f}",
            f"{r.get('relevancy_voto', 0):.4f}",
        ))
    print()


def main():
    print(HEADER)

    # 1) interactive configuration
    endpoint, threshold, rag_engine, ollama_url, model, benchmarks, sample_size, selection_mode = interactive_menu()

    # 2) load the dataset
    print("\n📂 Loading dataset...")
    dataset = _load_dataset(benchmarks, sample_size, selection_mode)

    if not dataset:
        print("\n❌ Error: no data loaded. Check that benchmark files exist in the Run/ folder.")
        sys.exit(1)

    print(f"  ✔ {len(dataset)} queries loaded.\n")

    # 3) start the evaluation
    print("🚀 Starting evaluation...\n")
    try:
        results = run_evaluation(
            dataset,
            endpoint,
            threshold,
            ollama_url,
            model,
            progress_callback=_cli_progress,
            motore_rag=rag_engine
        )
    except ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        print("  Check that the RAG endpoint and Ollama are running.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Error during evaluation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")
        sys.exit(1)

    # 4) print the results
    _print_results(results)

    # 5) save to disk
    try:
        eval_data = save_evaluation(
            results=results,
            model=model,
            endpoint=endpoint,
            threshold=threshold,
            sample_size=sample_size,
            benchmarks_used=benchmarks,
        )
        saved_files = sorted(EVAL_DIR.glob(f"{eval_data['name']}*.json"))
        if saved_files:
            print(f"💾 Results saved successfully in: {saved_files[-1]}")
        else:
            print(f"💾 Results saved in folder: {EVAL_DIR}")
    except Exception as e:
        print(f"⚠  Unable to save results: {e}")

    print("\n✅ Operation completed.")

if __name__ == "__main__":
    main()
