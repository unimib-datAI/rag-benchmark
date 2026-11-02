import os
import sys
import json
import random
import argparse
from collections import Counter, defaultdict
from pathlib import Path

# used to import evaluator and persistence from the same folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from corpus_sources import custom_benchmarks

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


try:
    from evaluator import run_evaluation, FETCH_K
    from persistence import save_evaluation, list_evaluations, load_evaluation, EVAL_DIR
    from taxonomy import diagnosis_messages, metric_mean, metric_value
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# paths and config
RUN_DIR = Path(__file__).resolve().parents[1] / "Run"
CONFIG_FILE = Path(os.environ.get("STATE_DIR") or Path(__file__).resolve().parents[1]) / "config.json"

# map benchmark -> corresponding jsonl file (aligned with app.py BENCHMARK_MAPPING)
BENCHMARK_MAP = {
    "LEGALBENCH": "Tutto_LEGALBENCH.jsonl",
    "MULTIHOP": "Tutto_MULTIHOP.jsonl",
    "CRAG": "Tutto_CRAG_DAVE.jsonl",
    "MTRAG": "Tutto_MTRAG_DAVE.jsonl",
    "OHRBENCH": "Tutto_OHRBench_DAVE.jsonl",
    "SQUAD": "Tutto_SQUAD.jsonl",
}
BENCHMARK_MAP.update({name.upper(): str(path) for name, path in custom_benchmarks().items()})  # datasets in Custom/
ALL_BENCHMARKS = list(BENCHMARK_MAP.keys())

HEADER = r"""
╔══════════════════════════════════════════════════════════╗
║       SuperRagBench  —  CLI RAG Evaluation               ║
╚══════════════════════════════════════════════════════════╝
"""


# ── config management ──────────────────────────────────────────────

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


# ── input helpers ──────────────────────────────────────────────────

def prompt_input(message, default):
    value = input(f"{message} [{default}]: ").strip()
    return value if value else default


def normalize_rag_engine(val: str) -> str:
    """Normalize RAG engine string to UI standard names."""
    v = val.strip().lower()
    if "dave" in v:
        return "DAVE (Elasticsearch)"
    return "Local RAG (BM25)"


def normalize_selection_mode(val: str) -> str:
    """Normalize selection mode."""
    v = val.strip().upper()
    if v in ["P", "PREVIOUS", "REPEAT", "REPEAT PREVIOUS"]:
        return "Repeat Previous"
    if v in ["F", "FIXED"]:
        return "Fixed"
    return "Random"


# ── diagnostics (aligned with app.py) ──────────────────────────────

def generate_diagnostic(results):
    """Aggregated diagnosis from the per-query error patterns (same rules as the UI)."""
    return [f"{title}: {text}" for title, text in diagnosis_messages(results)]


def _score(v, width=0, digits=4):
    # metrics not computed by RAGAS are None and shown as n/a
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{digits}f}"


# ── history / explore ──────────────────────────────────────────────

def print_history():
    """List saved evaluations from disk (equivalent to UI Explore tab)."""
    evals = list_evaluations()
    if not evals:
        print("\n  Nessuna valutazione precedente trovata nella cartella valutazioni/.\n")
        return

    print(f"\n📂 Valutazioni Salvate ({len(evals)} totali):")
    cw_name, cw_ts, cw_eng, cw_q, cw_m = 26, 17, 18, 8, 10
    hdr = f"  {'Nome':<{cw_name}} {'Timestamp':<{cw_ts}} {'Motore RAG':<{cw_eng}} {'Query':>{cw_q}} {'Correct':>{cw_m}} {'Faith':>{cw_m}} {'Prec':>{cw_m}}"
    print(hdr)
    print(f"  {'─'*cw_name} {'─'*cw_ts} {'─'*cw_eng} {'─'*cw_q} {'─'*cw_m} {'─'*cw_m} {'─'*cw_m}")

    for ev in evals[:25]:
        agg = ev.get("aggregate", {})
        corr = _score(agg.get('answer_correctness'), digits=3)
        faith = _score(agg.get('faithfulness'), digits=3)
        prec = _score(agg.get('context_precision'), digits=3)
        ts = ev.get("timestamp", "")[:16].replace("T", " ")
        name = ev.get("name", "?")[:cw_name]
        engine = ev.get("rag_engine", "?")[:cw_eng]
        q_count = str(ev.get("num_queries", 0))
        print(f"  {name:<{cw_name}} {ts:<{cw_ts}} {engine:<{cw_eng}} {q_count:>{cw_q}} {corr:>{cw_m}} {faith:>{cw_m}} {prec:>{cw_m}}")
    print()


# ── interactive menu ───────────────────────────────────────────────

def interactive_menu():
    config = load_config()

    def_endpoint    = config.get("endpoint", "http://127.0.0.1:8000")
    def_rag_engine  = config.get("motore_rag", "Local RAG (BM25)")
    def_ollama_url  = config.get("ollama_url", "http://localhost:11434")
    def_model       = config.get("model", "gemma4:31b-cloud")
    def_benchmarks  = config.get("benchmarks", "ALL")
    def_sample_size = config.get("sample_size", "3")
    def_sel_mode    = config.get("selection_mode", "R")
    def_workers     = config.get("max_workers", "2")
    def_ingest      = config.get("ingest_corpus", "Y")

    while True:
        print("\n--- Configurazione Valutazione (Interattiva) ---")
        while True:
            eval_name = prompt_input("Nome della Valutazione (obbligatorio)", "").strip()
            if eval_name:
                break
            print("Il nome della valutazione è obbligatorio.")

        endpoint   = prompt_input("RAG Base URL", def_endpoint)
        engine_str = prompt_input("Motore RAG [Local / DAVE]", def_rag_engine)
        rag_engine = normalize_rag_engine(engine_str)

        ollama_url = prompt_input("URL LLM Judge (es. http://localhost:11434)", def_ollama_url)
        import getpass
        llm_api_key = getpass.getpass("API Key LLM Judge (opzionale, NON salvata su disco): ").strip()
        model = prompt_input("Modello LLM Judge", def_model)

        print(f"\nBenchmark disponibili: {', '.join(ALL_BENCHMARKS)}")
        benchmarks_str = prompt_input(
            "Seleziona benchmark (separati da virgola, o ALL)",
            def_benchmarks
        )

        if benchmarks_str.upper() == "ALL":
            benchmarks = ALL_BENCHMARKS
        else:
            benchmarks = [b.strip().upper() for b in benchmarks_str.split(",")
                          if b.strip().upper() in ALL_BENCHMARKS]
            if not benchmarks:
                print("Nessun benchmark valido specificato. Verranno utilizzati tutti.")
                benchmarks = ALL_BENCHMARKS
                benchmarks_str = "ALL"

        while True:
            sample_size_str = prompt_input("Numero di query per benchmark", def_sample_size)
            try:
                sample_size = int(sample_size_str)
                if sample_size >= 1:
                    break
                print("Il numero deve essere >= 1.")
            except ValueError:
                print("Inserisci un numero intero valido.")

        while True:
            mode_str = prompt_input(
                "Modalità selezione query: [R]andom, [F]ixed o [P]revious (ripeti run precedente)",
                def_sel_mode
            ).strip().upper()
            if mode_str in ["R", "F", "P", "RANDOM", "FIXED", "PREVIOUS"]:
                selection_mode = normalize_selection_mode(mode_str)
                break
            print("Inserisci 'R' per Random, 'F' per Fixed o 'P' per Repeat Previous.")

        while True:
            workers_str = prompt_input("Worker paralleli RAGAS (1-10)", def_workers)
            try:
                max_workers = max(1, min(10, int(workers_str)))
                break
            except ValueError:
                print("Inserisci un numero intero valido (1-10).")

        ingest_str = prompt_input("Eseguire l'ingestion del corpus nel RAG? [Y/n]", def_ingest).strip().upper()
        skip_ingest = (ingest_str == "N")

        # riassunto
        print("\n--- Riepilogo Configurazione ---")
        print(f"  Nome Valutazione: {eval_name}")
        print(f"  RAG Base URL:     {endpoint}")
        print(f"  Motore RAG:       {rag_engine}")
        print(f"  LLM Judge URL:    {ollama_url}")
        print(f"  API Key:          {'***IMPOSTATA***' if llm_api_key else 'Nessuna'}")
        print(f"  Modello LLM:      {model}")
        print(f"  Benchmark:        {', '.join(benchmarks)}")
        print(f"  Query/Benchmark:  {sample_size}")
        print(f"  Campionamento:    {selection_mode}")
        print(f"  Worker RAGAS:     {max_workers}")
        print(f"  Ingestion Corpus: {'SALTATA' if skip_ingest else 'ATTIVA'}")

        choice = input("\nVuoi [A]vviare la valutazione o [M]odificare i parametri? [A/M]: ").strip().upper()

        if choice in ["A", "S", ""]:
            config["endpoint"]       = endpoint
            config["motore_rag"]     = rag_engine
            config["ollama_url"]     = ollama_url
            config["model"]          = model
            config["benchmarks"]     = benchmarks_str
            config["sample_size"]    = str(sample_size)
            config["selection_mode"] = "R" if selection_mode == "Random" else ("F" if selection_mode == "Fixed" else "P")
            config["max_workers"]    = str(max_workers)
            config["ingest_corpus"]   = "N" if skip_ingest else "Y"
            save_config(config)

            return (eval_name, endpoint, rag_engine, ollama_url, model,
                    benchmarks, sample_size, selection_mode, llm_api_key, max_workers, skip_ingest)


# ── data loading ───────────────────────────────────────────────────

def _load_repeat_ids():
    """Load query IDs from the most recent saved evaluation (Repeat Previous mode)."""
    evals = list_evaluations()
    if not evals:
        return None
    latest = load_evaluation(evals[0]["filepath"])
    return {r["query_id"] for r in latest.get("results", []) if "query_id" in r}


def _load_dataset(benchmarks, sample_size, selection_mode, seed):
    """Load queries from the jsonl files of the selected benchmarks.
    The Random selection uses its own generator seeded with `seed`, so a run can be repeated."""
    rng = random.Random(seed)
    repeat_ids = None
    if selection_mode == "Repeat Previous":
        repeat_ids = _load_repeat_ids()
        if not repeat_ids:
            print("  ⚠  Nessuna valutazione precedente trovata — fallback automatico a Fixed.")
            selection_mode = "Fixed"

    dataset = []
    files_to_run = []

    for name in benchmarks:
        filename = BENCHMARK_MAP.get(name)
        if not filename:
            continue
        path = RUN_DIR / filename
        if not path.exists():
            print(f"  ⚠  File non trovato: {path}  — benchmark {name} saltato.")
            continue

        files_to_run.append(filename)
        items = []
        with open(path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # the gold "context" is never sent to the RAG: every benchmark retrieves over its corpus
                record.pop('context', None)
                items.append(record)

        if selection_mode == "Repeat Previous" and repeat_ids:
            selected = [it for it in items if it.get("id") in repeat_ids]
            dataset.extend(selected)
        elif name == "MTRAG":
            conversations = defaultdict(list)
            for item in items:
                conv_id = item["id"].split("_t")[0]
                conversations[conv_id].append(item)

            conv_keys = list(conversations.keys())
            if selection_mode == "Random":
                rng.shuffle(conv_keys)
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
                dataset.extend(rng.sample(items, sz))
            else:
                dataset.extend(items[:sz])

    return dataset, files_to_run


# ── progress and TPS callbacks ─────────────────────────────────────

def _cli_progress(phase, current, total, message):
    if phase == "ingest":
        print(f"  📦 Ingestion [{current}/{total}] {message}", end="\r", flush=True)
    elif phase == "fetch":
        print(f"  🔍 Querying [{current}/{total}] {message}", end="\r", flush=True)
    elif phase == "ragas":
        print(f"\n  ⚖️  {message}")
    elif phase == "done":
        print(f"\n  ✅ {message}")


def _tps_callback(total_tokens, input_tokens, output_tokens, max_input, tps):
    print(
        f"  📊 Tokens: {total_tokens:,} (in:{input_tokens:,} out:{output_tokens:,}) "
        f"| max_input: {max_input:,} | {tps:.1f} tok/s",
        end="\r", flush=True
    )


# ── results display ───────────────────────────────────────────────

def _fmt_seconds(sec):
    m, s = divmod(round(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{sec:.1f}s"


def _print_timings(results, timings):
    """Seconds per phase measured by evaluator.py (ingestion, RAG queries, RAGAS)."""
    if not timings:
        return
    ingest, queries = timings.get("ingestion", {}), timings.get("query", {})
    print(f"\n⏱  Tempi: totale {_fmt_seconds(timings.get('totale', 0))}"
          f" | ingestion {_fmt_seconds(sum(ingest.values()))}"
          f" | interrogazione RAG {_fmt_seconds(sum(queries.values()))}"
          f" | RAGAS {_fmt_seconds(timings.get('ragas', 0))}")
    for src in sorted(set(ingest) | set(queries)):
        n_q = sum(1 for r in results if r.get("source") == src)
        q = queries.get(src, 0)
        print(f"  {src:<14} ingestion {_fmt_seconds(ingest.get(src, 0)):>12}"
              f" | query {_fmt_seconds(q):>12} ({q / n_q if n_q else 0:.1f}s/domanda)")


def _print_results(results):
    n = len(results)
    if n == 0:
        print("\n  Nessun risultato da mostrare.")
        return

    # RAGAS stats (attached to first result by evaluator.py)
    stats = results[0].get("_ragas_stats", {})
    # queries the RAG did not answer (RAG_ERROR): never sent to RAGAS, excluded from the averages
    rag_errors = sum(1 for r in results if r.get("rag_error"))
    if stats:
        print(f"\n📈 Copertura RAGAS: {stats.get('riuscite', n)}/{stats.get('totali', n)} query valutate con successo"
              f" | {stats.get('fallite', 0)} fallite (NaN) | {stats.get('contesti_troncati', 0)} contesti protetti da troncamento"
              + (f" | {rag_errors} errori del RAG (non valutate)" if rag_errors else ""))
    if rag_errors:
        print(f"⚠  {rag_errors}/{n} query senza risposta dal RAG (RAG_ERROR): escluse da RAGAS e dalle medie")
    _print_timings(results, stats.get("tempi"))

    # aggregate averages, only over the queries where RAGAS computed each metric
    print("\n📊 Medie Aggregate (5 Metriche RAGAS):")
    print(f"  Answer Correctness:  {_score(metric_mean(results, 'correctness_voto'))}")
    print(f"  Faithfulness:        {_score(metric_mean(results, 'faithfulness_voto'))}")
    print(f"  Context Precision:   {_score(metric_mean(results, 'retrieval_voto'))}")
    print(f"  Context Recall:      {_score(metric_mean(results, 'recall_voto'))}")
    print(f"  Answer Relevancy:    {_score(metric_mean(results, 'relevancy_voto'))}")

    # verdict and error pattern distribution (same classification as the UI)
    verdicts = Counter("rag_error" if r.get("rag_error") else r.get("verdict", "?") for r in results)
    patterns = Counter(r.get("error_pattern", "?") for r in results if r.get("verdict") == "wrong")
    partial_causes = Counter(r.get("error_cause") or "UNKNOWN" for r in results if r.get("verdict") == "partial")
    print("\n⚖️  Verdetti: " + " | ".join(f"{v}: {c}" for v, c in verdicts.most_common()))
    if patterns:
        print("  Pattern di errore: " + ", ".join(f"{p}: {c}" for p, c in patterns.most_common()))
    if partial_causes:
        print("  Cause delle risposte parziali: " + ", ".join(f"{p}: {c}" for p, c in partial_causes.most_common()))

    # diagnostics
    print("\n🩺 Diagnostica Automatica:")
    for line in generate_diagnostic(results):
        print(f"  {line}")

    # per-benchmark breakdown
    bench_groups = defaultdict(list)
    for r in results:
        bench_groups[r.get("source", "Unknown")].append(r)

    if len(bench_groups) > 1:
        print("\n📦 Dettaglio per Benchmark:")
        for src, group in sorted(bench_groups.items()):
            bc, bf, bp, br = (_score(metric_mean(group, k), digits=3)
                              for k in ("correctness_voto", "faithfulness_voto", "retrieval_voto", "recall_voto"))
            print(f"  {src:28s}  Corr={bc}  Faith={bf}  Prec={bp}  Recall={br}  ({len(group)} query)")

    # per-query table
    print("\n📋 Dettaglio per Query:")
    cw = 9
    hdr = f"  {'ID':<16} {'Benchmark':<14} {'Correct':>{cw}} {'Faith':>{cw}} {'Prec':>{cw}} {'Recall':>{cw}} {'Relev':>{cw}} {'Stato':>7}"
    print(hdr)
    print(f"  {'─'*16} {'─'*14} {'─'*cw} {'─'*cw} {'─'*cw} {'─'*cw} {'─'*cw} {'─'*7}")

    for r in results:
        src = r.get("source", "?")
        short_src = src.split('-')[0][:14] if '-' in src else src[:14]
        status = "ERR RAG" if r.get("rag_error") else "FALLITA" if r.get("ragas_fallita") else "OK"
        print(
            f"  {r.get('query_id', '?')[:16]:<16} {short_src:<14}"
            f" {_score(metric_value(r, 'correctness_voto'), cw)}"
            f" {_score(metric_value(r, 'faithfulness_voto'), cw)}"
            f" {_score(metric_value(r, 'retrieval_voto'), cw)}"
            f" {_score(metric_value(r, 'recall_voto'), cw)}"
            f" {_score(metric_value(r, 'relevancy_voto'), cw)}"
            f" {status:>7}"
        )
    print()


# ── argument parser ────────────────────────────────────────────────

# the CLI always computes all five RAGAS metrics
CLI_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness", "context_recall"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="SuperRagBench — Interfaccia CLI per la Valutazione di Sistemi RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Esempi di utilizzo:
  # Avvio in modalità interattiva:
  python cli.py

  # Consultazione archivio storico delle valutazioni:
  python cli.py --list

  # Esecuzione automatizzata in batch (headless / CI-CD):
  python cli.py --name "Test_Automated" --sample 5 --benchmarks SQUAD,CRAG --no-ingest
"""
    )
    parser.add_argument("--name", type=str, default="", help="Nome identificativo della valutazione")
    parser.add_argument("--engine", type=str, default=None, help="Motore RAG target ('Local' o 'DAVE')")
    parser.add_argument("--endpoint", type=str, default=None, help="URL base del motore RAG (es. http://127.0.0.1:8000)")
    parser.add_argument("--judge-url", type=str, default=None, help="Endpoint dell'LLM Judge (es. http://localhost:11434)")
    parser.add_argument("--model", type=str, default=None, help="Nome modello LLM Judge (es. gemma4:31b-cloud)")
    parser.add_argument("--api-key", type=str, default="", help="API Key per modelli cloud / OpenAI")
    parser.add_argument("--benchmarks", type=str, default=None, help="Lista benchmark separati da virgola (o 'ALL')")
    parser.add_argument("--sample", type=int, default=None, help="Numero di query per benchmark")
    parser.add_argument("--mode", type=str, choices=["Random", "Fixed", "Repeat", "random", "fixed", "repeat"], default=None, help="Modalità campionamento query")
    parser.add_argument("--workers", type=int, default=None, help="Worker paralleli RAGAS (1-10, default: 2)")
    parser.add_argument("--seed", type=int, default=None, help="Seed del campionamento Random (default: casuale, salvato nella valutazione)")
    parser.add_argument("--no-ingest", action="store_true", help="Salta l'ingestion preliminare del corpus nel RAG")
    parser.add_argument("--list", action="store_true", help="Mostra l'elenco delle valutazioni salvate ed esce")
    parser.add_argument("--interactive", action="store_true", help="Forza l'apertura del menù interattivo")
    return parser


# ── main ───────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Se richiesto --list, mostra storico ed esce
    if args.list:
        print(HEADER)
        print_history()
        return

    # Determina se operare in modalità interattiva o headless
    is_interactive = args.interactive or (
        len(sys.argv) == 1 or (len(sys.argv) == 2 and args.interactive)
    )

    if is_interactive:
        print(HEADER)
        try:
            (eval_name, endpoint, rag_engine, ollama_url, model,
             benchmarks, sample_size, selection_mode, llm_api_key, max_workers, skip_ingest) = interactive_menu()
        except KeyboardInterrupt:
            print("\n\nOperazione annullata dall'utente.")
            sys.exit(0)
    else:
        # Modalità headless (CI/CD o script batch)
        config = load_config()
        eval_name = args.name.strip()
        if not eval_name:
            import datetime
            eval_name = f"cli_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        endpoint       = args.endpoint or config.get("endpoint", "http://127.0.0.1:8000")
        rag_engine     = normalize_rag_engine(args.engine or config.get("motore_rag", "Local RAG (BM25)"))
        ollama_url     = args.judge_url or config.get("ollama_url", "http://localhost:11434")
        model          = args.model or config.get("model", "gemma4:31b-cloud")
        llm_api_key    = args.api_key
        max_workers    = args.workers or int(config.get("max_workers", "2"))
        sample_size    = args.sample or int(config.get("sample_size", "3"))
        selection_mode = normalize_selection_mode(args.mode or config.get("selection_mode", "Random"))
        skip_ingest    = args.no_ingest

        bench_arg = args.benchmarks or config.get("benchmarks", "ALL")
        if bench_arg.upper() == "ALL":
            benchmarks = ALL_BENCHMARKS
        else:
            benchmarks = [b.strip().upper() for b in bench_arg.split(",") if b.strip().upper() in ALL_BENCHMARKS]
            if not benchmarks:
                benchmarks = ALL_BENCHMARKS

        print(HEADER)
        print(f"🚀 Avvio valutazione automatizzata: {eval_name}")
        print(f"  Motore: {rag_engine} ({endpoint}) | Judge: {model} ({ollama_url})")
        print(f"  Benchmark: {', '.join(benchmarks)} | Campioni/Bench: {sample_size} | Mode: {selection_mode} | Workers: {max_workers}\n")

    # Caricamento del dataset (seed salvato nella valutazione per poter ripetere il campionamento)
    seed = args.seed if args.seed is not None else random.randrange(2**32)
    print(f"📂 Caricamento del dataset di benchmark... (seed {seed})")
    dataset, files_to_run = _load_dataset(benchmarks, sample_size, selection_mode, seed)

    if not dataset:
        print("\n❌ Errore: nessun dato caricato. Verifica la presenza dei file nella cartella Run/.")
        sys.exit(1)

    print(f"  ✔ {len(dataset)} query caricate per la sessione.\n")

    # Esecuzione della valutazione
    print("🚀 Esecuzione della pipeline di valutazione...\n")
    try:
        results = run_evaluation(
            dataset=dataset,
            rag_base_url=endpoint,
            ollama_url=ollama_url,
            ollama_model=model,
            llm_api_key=llm_api_key,
            progress_callback=_cli_progress,
            tps_callback=_tps_callback,
            motore_rag=rag_engine,
            benchmark_files=None if skip_ingest else files_to_run,
            selected_metrics=CLI_METRICS,
            max_workers=max_workers,
        )
    except ConnectionError as e:
        print(f"\n❌ Errore di connessione: {e}")
        print("  Verifica che il server RAG e l'LLM Judge siano attivi e raggiungibili.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Errore durante la valutazione: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nValutazione interrotta dall'utente.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore inatteso: {type(e).__name__}: {e}")
        sys.exit(1)

    # A capo pulito dopo l'output dinamico del TPS
    print()

    # Visualizzazione dei risultati
    _print_results(results)

    # Salvataggio su disco (mai una valutazione vuota: diventerebbe "l'ultima" mostrata dalla UI)
    if not results:
        print("⚠  Nessun risultato: valutazione non salvata.")
        sys.exit(1)
    try:
        eval_data = save_evaluation(
            results=results,
            model=model,
            endpoint=endpoint,
            sample_size=sample_size,
            benchmarks_used=benchmarks,
            rag_engine=rag_engine,
            name=eval_name,
            config={
                "selection_mode": selection_mode,
                "seed": seed,
                "metrics": CLI_METRICS,
                "fetch_k": FETCH_K,
                "max_workers": max_workers,
            },
        )
        saved_file = eval_data.get("filepath")
        if saved_file and Path(saved_file).exists():
            print(f"💾 Risultati salvati con successo in: {saved_file}")
        else:
            print(f"💾 Risultati salvati nella cartella: {EVAL_DIR}")
    except Exception as e:
        print(f"⚠  Impossibile salvare i risultati: {e}")

    print("\n✅ Operazione completata.")


if __name__ == "__main__":
    main()
