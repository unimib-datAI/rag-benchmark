# Guida: Come Aggiungere un Nuovo Dataset a SuperBenchmark

Per aggiungere un nuovo benchmark scaricato da GitHub (o da un'altra fonte) servono **quattro passaggi**: il terzo solo se usi DAVE.

Ogni benchmark fornisce a SuperBenchmark due cose distinte:

| Cosa | Dove | A cosa serve |
| :--- | :--- | :--- |
| **Domande e risposte** | `Run/Tutto_NUOVO.jsonl` (Step 1) | Le domande poste al RAG e le *gold answer* con cui RAGAS confronta le risposte. Il campo `context` contiene i documenti corretti (la "soluzione") e resta come riferimento: non viene mai dato al RAG. |
| **Corpus** | funzione in `corpus_sources.py` (Step 2) | I documenti in cui il RAG cerca. Lo stesso corpus viene caricato sul RAG locale e su DAVE. |

---

## Step 1: Lo Script di Estrazione (`Run/extract_*.py`)

I dataset grezzi si trovano in formati eterogenei (CSV, JSON innestati, pagine HTML). Il primo passo è scrivere uno script Python che legga il formato grezzo e lo trasformi nel **formato standard JSONL** (JSON Lines) richiesto da SuperBenchmark. L'adozione del formato JSONL consente il caricamento riga per riga in streaming, abbattendo drasticamente l'impronta di memoria.

### Output Atteso (`Tutto_NUOVO.jsonl`)

Il tuo script deve generare un file `.jsonl` in cui ogni riga è un oggetto JSON valido con questa struttura:

```json
{"id": "prefisso_0001", "source": "NomeDelBenchmark", "query_type": "Estrazione_Semplice", "question": "Qual è la capitale della Francia?", "context": "[Doc 1] La Francia è un paese europeo. La sua capitale è Parigi...", "gold_answer": "Parigi"}
```

- `id`: identificativo univoco della query (es. `leg_0001`, `crag_001`).
- `source`: nome identificativo del benchmark. **Deve essere lo stesso nome usato negli Step 2 e 3.**
- `query_type`: tipologia della domanda (es. `Estrazione_Semplice`, `Multi_Hop`, `Confronto_Aggregazione`).
- `question`: la domanda posta al RAG.
- `context`: i documenti corretti per la domanda (la soluzione), in un'unica stringa delimitata da tag (es. `[Doc 1] Testo... [Doc 2] Testo...`). È solo un riferimento: il RAG cerca nel corpus dello Step 2.
- `gold_answer`: risposta corretta di riferimento (*ground truth*).

---

## Step 2: Il Corpus da Indicizzare (`corpus_sources.py`)

Il RAG non deve cercare tra le sole soluzioni (troverebbe sempre il documento giusto): deve cercare nel **corpus completo** del benchmark, che contiene anche documenti non pertinenti.

Apri `corpus_sources.py`, scrivi una funzione che indichi dove si trovano i documenti e come leggerli, e registrala nel dizionario `_CORPORA` con lo stesso nome del campo `source`:

```python
def _nuovo():
    corpus_dir = BENCHMARKS_DIR / "NuovoBenchmark" / "documenti"
    def docs():
        for path in sorted(corpus_dir.glob("*.txt")):
            yield path.stem, path.read_text(encoding="utf-8")   # (id, testo) di ogni documento
    return [corpus_dir], docs   # file o cartelle del corpus, usati per controllare che esistano


_CORPORA = {
    "MultiHop-RAG": _multihop,
    ...
    # ⬇️ AGGIUNGI IL TUO BENCHMARK QUI ⬇️
    "NomeDelBenchmark": _nuovo,
}
```

- La funzione restituisce coppie **(id, testo)**: un documento per coppia, con id univoci e stabili.
- I duplicati e i documenti vuoti vengono scartati automaticamente; i documenti oltre i 500.000 caratteri vengono divisi in parti (`<id>_part2`, `<id>_part3`, …), lo stesso limite usato per DAVE.
- Se i file del corpus non esistono, la valutazione si ferma con un messaggio che indica il percorso mancante; se il benchmark non è registrato in `_CORPORA`, si ferma con un messaggio che rimanda a questo Step.
- Per il **RAG locale** non serve altro: all'avvio della valutazione SuperBenchmark chiede a `POST /ingest` quali documenti mancano e carica solo quelli (vedi *Collegare il proprio RAG* nel README).

> Se il benchmark non ha un corpus separato ma fornisce i documenti per ogni domanda (come CRAG con le sue pagine web), usa `_per_question("NomeDelBenchmark")` e aggiungi il file JSONL in `BENCHMARK_FILES`: il corpus viene preso dal campo `context`.

---

## Step 3: Ingestione Documenti su DAVE (`ingest_dave.py`)

Solo se utilizzi **DAVE**: i documenti del corpus devono essere caricati e indicizzati sul server.

1. In `corpus_sources.py` aggiungi la collection nel dizionario `DAVE_COLLECTIONS` (nome della collection su DAVE -> nome del benchmark). È l'unico punto da modificare: lo usano sia `ingest_dave.py` per il caricamento sia l'evaluator per sapere in quale collection cercare.
   ```python
   DAVE_COLLECTIONS = {
       ...
       "Nuovo": "NomeDelBenchmark",
   }
   ```
2. Carica il corpus:
   ```bash
   python ingest_dave.py --benchmarks Nuovo
   ```
   Lo script crea la collection (se non esiste) e invia i documenti a lotti, saltando quelli già caricati grazie alla cache locale (`.dave_docs_cache.json`). Poi attende che DAVE li abbia indicizzati e ricarica quelli falliti (`--no-wait` per saltare l'attesa; `--recreate` crea una nuova collection (la vecchia resta su DAVE); `--verify` confronta cache e DAVE).

---

## Step 4: Integrazione nella Dashboard (`app.py`) e nella CLI (`cli.py`)

Ora che i dati sono pronti, occorre registrare il nuovo benchmark nell'applicazione:

1. **Dashboard Streamlit (`webpage/app.py`):**
   Cerca il dizionario `BENCHMARK_MAPPING` all'inizio del file e aggiungi il nuovo file JSONL:
   ```python
   BENCHMARK_MAPPING = {
       "LegalBench-RAG (Contract Precision)": "Tutto_LEGALBENCH.jsonl",
       "MultiHop-RAG (Multi-Document Synthesis)": "Tutto_MULTIHOP.jsonl",
       "CRAG (HTML Parsing & Traps)": "Tutto_CRAG_DAVE.jsonl",
       "MTRAG (Multi-Turn Conversations)": "Tutto_MTRAG_DAVE.jsonl",
       "OHRBench (OCR Noise / Scans)": "Tutto_OHRBench_DAVE.jsonl",
       "SQuAD-it (Italian Q&A on Wikipedia)": "Tutto_SQUAD.jsonl",

       # ⬇️ AGGIUNGI IL TUO BENCHMARK QUI ⬇️
       "Il Mio Nuovo Dataset (Test X)": "Tutto_NUOVO.jsonl"
   }
   ```
   Facoltativo: aggiungi un nome breve in `BENCHMARK_SHORT` (es. `"NomeDelBenchmark": "Nuovo"`), usato nei grafici.

2. **Interfaccia CLI (`webpage/cli.py`):**
   Aggiungi il riferimento nel dizionario `BENCHMARK_MAP`:
   ```python
   BENCHMARK_MAP["MIOBENCHMARK"] = "Tutto_NUOVO.jsonl"
   ```

---

### Facoltativo: Download Automatico (`webpage/benchmark_download.py`)

Se vuoi che il pulsante **Download Benchmark** scarichi anche il tuo dataset, aggiungi le sue fonti alla lista `SOURCES` (domande e corpus), con la destinazione in `Benchmarks/` attesa dagli Step 1 e 2.

---

### Conclusione

Esegui `streamlit run webpage/app.py` o lancia la CLI con `python webpage/cli.py`.
Il nuovo benchmark apparirà immediatamente tra le opzioni disponibili. La suite caricherà il corpus nel RAG, eseguirà il retrieval, il filtro semantico, la valutazione RAGAS con l'LLM Judge e la diagnosi automatica degli errori.
