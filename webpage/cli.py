import sys
import json
import random
from collections import defaultdict
from pathlib import Path

# serve per importare evaluator e persistence dalla stessa cartella
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from evaluator import run_evaluation
    from persistence import save_evaluation, EVAL_DIR
except ImportError as e:
    print(f"Errore di importazione: {e}")
    sys.exit(1)

# path e config
RUN_DIR = Path(__file__).resolve().parents[1] / "Run"
CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.json"

# mappa benchmark -> file json corrispondente
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
║         SuperBenchmark  —  CLI Valutazione RAG           ║
╚══════════════════════════════════════════════════════════╝
"""


# gestione config (salva/carica da json)
def carica_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def salva_config(config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


# input e menu interattivo

def richiedi_input(messaggio, default):
    valore = input(f"{messaggio} [{default}]: ").strip()
    return valore if valore else default


def menu_interattivo():
    config = carica_config()

    def_endpoint = config.get("endpoint", "http://127.0.0.1:8000")
    def_threshold = config.get("threshold", "0.0")
    def_motore = config.get("motore_rag", "Local RAG (BM25)")
    def_ollama_url = config.get("ollama_url", "http://localhost:11434")
    def_model = config.get("model", "gemma4:31b-cloud")
    def_benchmarks = config.get("benchmarks", "TUTTI")
    def_sample_size = config.get("sample_size", "3")
    def_selection_mode = config.get("selection_mode", "C")

    while True:
        print("\n--- Configurazione Valutazione ---")
        endpoint = richiedi_input("Inserire Base URL RAG", def_endpoint)
        motore_rag = richiedi_input("Selezionare Motore RAG (DAVE o Local RAG)", def_motore)

        while True:
            thresh_str = richiedi_input("Inserire Soglia Similarità (Threshold 0.0 - 1.0)", def_threshold)
            try:
                threshold = float(thresh_str)
                break
            except ValueError:
                print("Per favore, inserisci un numero decimale valido.")

        ollama_url = richiedi_input("Inserire URL di Ollama", def_ollama_url)
        model = richiedi_input("Inserire modello di LLM as a Judge", def_model)

        print(f"\nBenchmark disponibili: {', '.join(ALL_BENCHMARKS)}")
        benchmarks_str = richiedi_input(
            "Selezionare i benchmark per la valutazione (separati da virgola, o TUTTI)",
            def_benchmarks
        )

        if benchmarks_str.upper() == "TUTTI":
            benchmarks = ALL_BENCHMARKS
        else:
            benchmarks = [b.strip().upper() for b in benchmarks_str.split(",")
                          if b.strip().upper() in ALL_BENCHMARKS]
            if not benchmarks:
                print("Nessun benchmark valido inserito. Verranno usati tutti.")
                benchmarks = ALL_BENCHMARKS
                benchmarks_str = "TUTTI"

        while True:
            sample_size_str = richiedi_input("Inserire il numero di query per benchmark", def_sample_size)
            try:
                sample_size = int(sample_size_str)
                break
            except ValueError:
                print("Per favore, inserisci un numero valido.")

        while True:
            mode_str = richiedi_input("Selezionare modalità query: [C]asuali o [F]isse", def_selection_mode).strip().upper()
            if mode_str in ["C", "F"]:
                selection_mode = "Casuali" if mode_str == "C" else "Fisse"
                break
            print("Per favore, inserisci 'C' per Casuali o 'F' per Fisse.")

        print("\n--- Riepilogo Configurazione ---")
        print(f"Base URL RAG:    {endpoint}")
        print(f"Motore RAG:      {motore_rag}")
        print(f"Threshold:       {threshold}")
        print(f"Ollama URL:      {ollama_url}")
        print(f"Modello LLM:     {model}")
        print(f"Benchmark:       {', '.join(benchmarks)}")
        print(f"Query x Bench:   {sample_size}")
        print(f"Modalità query:  {selection_mode}")

        scelta = input("\nVuoi [I]niziare la valutazione o [M]odificare qualche parametro? [I/M]: ").strip().upper()

        if scelta == "I" or scelta == "":
            config["endpoint"] = endpoint
            config["motore_rag"] = motore_rag
            config["threshold"] = str(threshold)
            config["ollama_url"] = ollama_url
            config["model"] = model
            config["benchmarks"] = benchmarks_str
            config["sample_size"] = str(sample_size)
            config["selection_mode"] = mode_str
            salva_config(config)

            return endpoint, threshold, motore_rag, ollama_url, model, benchmarks, sample_size, selection_mode


# caricamento dati e valutazione

def _carica_dataset(benchmarks, sample_size, selection_mode):
    """Carica le query dai file json dei benchmark selezionati."""
    dataset = []

    for name in benchmarks:
        filename = BENCHMARK_MAP[name]
        path = RUN_DIR / filename
        if not path.exists():
            print(f"  ⚠  File non trovato: {path}  — benchmark {name} ignorato.")
            continue

        with open(path, 'r', encoding='utf-8-sig') as f:
            items = json.load(f)
        if name == "MTRAG":
            conversations = defaultdict(list)
            for item in items:
                conv_id = item["id"].split("_t")[0]
                conversations[conv_id].append(item)

            conv_keys = list(conversations.keys())
            if selection_mode == "Casuali":
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
            if selection_mode == "Casuali":
                dataset.extend(random.sample(items, sz))
            else:
                dataset.extend(items[:sz])

    return dataset


def _cli_progress(phase, current, total, message):
    # callback per mostrare il progresso a terminale
    if phase == "fetch":
        print(f"  [{current}/{total}] {message}", end="\r", flush=True)
    elif phase == "ragas":
        print(f"\n  {message}")
    elif phase == "done":
        print(f"\n  {message}")


def _stampa_risultati(results):
    n = len(results)
    if n == 0:
        print("\n  Nessun risultato da mostrare.")
        return

    # calcolo medie
    avg_corr = sum(r.get("correctness_voto", 0) for r in results) / n
    avg_faith = sum(r.get("faithfulness_voto", 0) for r in results) / n
    avg_ctx = sum(r.get("retrieval_voto", 0) for r in results) / n
    avg_rel = sum(r.get("relevancy_voto", 0) for r in results) / n
    avg_sem = sum(r.get("semantic_sim", 0) for r in results) / n

    print("\n📊 Metriche Aggregate:")
    print(f"  Answer Correctness:  {avg_corr:.4f}")
    print(f"  Faithfulness:        {avg_faith:.4f}")
    print(f"  Context Precision:   {avg_ctx:.4f}")
    print(f"  Answer Relevancy:    {avg_rel:.4f}")
    print(f"  Semantic Similarity: {avg_sem:.4f}")

    # tabella per-query
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

    # 1) configurazione interattiva
    endpoint, threshold, motore_rag, ollama_url, model, benchmarks, sample_size, selection_mode = menu_interattivo()

    # 2) carico il dataset
    print("\n📂 Caricamento dataset...")
    dataset = _carica_dataset(benchmarks, sample_size, selection_mode)

    if not dataset:
        print("\n❌ Errore: nessun dato caricato. Controlla che i file benchmark esistano nella cartella Run/.")
        sys.exit(1)

    print(f"  ✔ {len(dataset)} query caricate.\n")

    # 3) lancio la valutazione
    print("🚀 Avvio valutazione...\n")
    try:
        results = run_evaluation(
            dataset,
            endpoint,
            threshold,
            ollama_url,
            model,
            progress_callback=_cli_progress,
            motore_rag=motore_rag
        )
    except ConnectionError as e:
        print(f"\n❌ Errore di connessione: {e}")
        print("  Verifica che l'endpoint RAG e Ollama siano in esecuzione.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Errore durante la valutazione: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {type(e).__name__}: {e}")
        sys.exit(1)

    # 4) stampo i risultati
    _stampa_risultati(results)

    # 5) salvo su disco
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
            print(f"💾 Risultati salvati con successo in: {saved_files[-1]}")
        else:
            print(f"💾 Risultati salvati nella cartella: {EVAL_DIR}")
    except Exception as e:
        print(f"⚠  Impossibile salvare i risultati: {e}")

    print("\n✅ Operazione completata.")

if __name__ == "__main__":
    main()
