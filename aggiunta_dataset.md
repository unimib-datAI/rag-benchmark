 

# Guida: Come Aggiungere un Nuovo Dataset a SuperBenchmark

Per aggiungere un nuovo benchmark scaricato da GitHub (o da un'altra fonte), bastano **tre passaggi**.

---

## Step 1: Lo Script di Estrazione (`Run/extract_*.py`)

I dataset grezzi si trovano in centinaia di formati diversi (CSV, JSON innestati, pagine HTML). Il primo passo è scrivere uno script Python che legga il formato grezzo e lo trasformi nel **formato standard JSON** richiesto da SuperBenchmark.

### Output Atteso (`Tutto_NUOVO.json`)

Il tuo script deve generare un file JSON che contiene un array di oggetti, dove ogni oggetto ha rigorosamente questa struttura:

```json
[
  {
    "id": "prefisso_0001",
    "source": "NomeDelBenchmark",
    "query_type": "Estrazione_Semplice", 
    "question": "Qual è la capitale della Francia?",
    "context": "[Doc 1] La Francia è un paese europeo. La sua capitale è Parigi...",
    "gold_answer": "Parigi"
  }
]
```

- `query_type`: Cerca di mapparlo su una delle categorie esistenti (es. `Estrazione_Semplice`, `Multi_Hop`, `Confronto_Aggregazione`).
- `context`: Combina tutti i documenti necessari in un'unica stringa, delimitandoli chiaramente (es. `[Doc 1] Testo... [Doc 2] Testo...`).

---

## Step 2: Ingestione Documenti su DAVE (`ingest_dave.py`)

Se utilizzi il motore RAG locale, il file JSON generato al punto 1 è sufficiente.
Se invece utilizzi **DAVE**, i documenti di contesto devono essere pre-caricati (indicizzati) sul server.

Apri il file `ingest_dave.py` e, nella funzione `main()`, aggiungi il caricamento del tuo nuovo corpus:

```python
# Esempio per il nuovo dataset
print("\n[NuovoBenchmark] Ingesting...")
# Crea la collezione (se non esiste)
col_id = client.create_collection("NuovoBenchmark")

# Prepara la tua lista di documenti (estratti dal json grezzo)
# Formato richiesto da DAVE: [{"title": "NomeDoc", "text": "Testo del documento", "fileName": "doc_id.txt"}]
nuovi_docs = [...] 

# Invia i documenti al server usando l'upload a lotti integrato
upload_docs_in_batches(client, col_id, nuovi_docs, "NuovoBenchmark")
```

Questo script salterà in automatico i documenti già caricati grazie alla cache locale (`.dave_docs_cache.json`).

---

## Step 3: Integrazione nella Dashboard (`app.py`)

Ora che i dati sono pronti, bisogna dire all'interfaccia Streamlit che esiste un nuovo benchmark.

1. Apri `webpage/app.py`.
2. Cerca il dizionario `BENCHMARK_MAPPING` all'inizio del file.
3. Aggiungi il nuovo elemento puntando al file JSON generato allo Step 1.

```python
BENCHMARK_MAPPING = {
    "LegalBench-RAG (Contract Precision)": "Tutto_LEGALBENCH.json",
    "MultiHop-RAG (Multi-Document Synthesis)": "Tutto_MULTIHOP.json",
    "CRAG (HTML Parsing & Traps)": "Tutto_CRAG_DAVE.json",
    "MTRAG (Multi-Turn Conversations)": "Tutto_MTRAG_DAVE.json",
    "OHRBench (OCR Noise / Scans)": "Tutto_OHRBench_DAVE.json",
    "SQuAD-it (Italian Q&A on Wikipedia)": "Tutto_SQUAD.json",
  
    # ⬇️ AGGIUNGI LA TUA RIGA QUI ⬇️
    "Il Mio Nuovo Dataset (Test X)": "Tutto_NUOVO.json"
}
```

### Fine

Esegui `streamlit run webpage/app.py`.
Il tuo nuovo benchmark apparirà istantaneamente nel menù a tendina. Il motore di valutazione di SuperBenchmark applicherà automaticamente il retrieval, il filtro semantico, e invocherà l'LLM-as-a-Judge di Ragas senza bisogno di toccare la logica di valutazione.
