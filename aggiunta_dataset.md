# Guida: Come Aggiungere un Nuovo Dataset a SuperBenchmark

Per aggiungere un nuovo benchmark scaricato da GitHub (o da un'altra fonte), bastano **tre passaggi**.

---

## Step 1: Lo Script di Estrazione (`Run/extract_*.py`)

I dataset grezzi si trovano in formati eterogenei (CSV, JSON innestati, pagine HTML). Il primo passo è scrivere uno script Python che legga il formato grezzo e lo trasformi nel **formato standard JSONL** (JSON Lines) richiesto da SuperBenchmark. L'adozione del formato JSONL consente il caricamento riga per riga in streaming, abbattendo drasticamente l'impronta di memoria.

### Output Atteso (`Tutto_NUOVO.jsonl`)

Il tuo script deve generare un file `.jsonl` in cui ogni riga è un oggetto JSON valido con questa struttura:

```json
{"id": "prefisso_0001", "source": "NomeDelBenchmark", "query_type": "Estrazione_Semplice", "question": "Qual è la capitale della Francia?", "context": "[Doc 1] La Francia è un paese europeo. La sua capitale è Parigi...", "gold_answer": "Parigi"}
```

- `id`: identificativo univoco della query (es. `leg_0001`, `crag_001`).
- `source`: nome identificativo del benchmark.
- `query_type`: tipologia della domanda (es. `Estrazione_Semplice`, `Multi_Hop`, `Confronto_Aggregazione`).
- `context`: combina i documenti necessari in un'unica stringa delimitata da tag (es. `[Doc 1] Testo... [Doc 2] Testo...`).
- `gold_answer`: risposta corretta di riferimento (*ground truth*).

---

## Step 2: Ingestione Documenti su DAVE (`ingest_dave.py`)

Se utilizzi il motore RAG locale, il file JSONL generato allo Step 1 è sufficiente.
Se invece utilizzi **DAVE**, i documenti di contesto devono essere pre-caricati e indicizzati sul server.

Apri il file `ingest_dave.py` e, nella funzione `main()`, aggiungi il caricamento del tuo nuovo corpus:

```python
# Esempio per il nuovo dataset
print("\n[NuovoBenchmark] Ingesting...")
# Crea la collezione (se non esiste)
col_id = client.create_collection("NuovoBenchmark")

# Prepara la tua lista di documenti (estratti dal corpus grezzo)
# Formato richiesto da DAVE: [{"title": "NomeDoc", "text": "Testo del documento", "fileName": "doc_id.txt"}]
nuovi_docs = [...] 

# Invia i documenti al server usando l'upload a lotti integrato
upload_docs_in_batches(client, col_id, nuovi_docs, "NuovoBenchmark")
```

Lo script salterà in automatico i documenti già caricati grazie alla cache locale (`.dave_docs_cache.json`).

---

## Step 3: Integrazione nella Dashboard (`app.py`) e nella CLI (`cli.py`)

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

2. **Interfaccia CLI (`webpage/cli.py`):**
   Aggiungi il riferimento nel dizionario `BENCHMARK_MAP`:
   ```python
   BENCHMARK_MAP["MIOBENCHMARK"] = "Tutto_NUOVO.jsonl"
   ```

---

### Conclusione

Esegui `streamlit run webpage/app.py` o lancia la CLI con `python webpage/cli.py`.
Il nuovo benchmark apparirà immediatamente tra le opzioni disponibili. La suite eseguirà automaticamente il retrieval, il filtro semantico, la valutazione RAGAS con l'LLM Judge e la diagnosi automatica degli errori.
