# 🚀 SuperBenchmark — RAG Evaluation Suite

**SuperBenchmark** è una suite completa per la valutazione sistematica e standardizzata di sistemi **RAG (Retrieval-Augmented Generation)**.
Adotta il paradigma **LLM-as-a-Judge** integrando le metriche della libreria **Ragas** per misurare oggettivamente la qualità del recupero (retrieval) e della generazione delle risposte su scenari reali e casi limite (stress-test).

Il progetto include sia una **Dashboard Web interattiva** (Streamlit) per l'analisi visiva e il confronto multi-run, sia una comoda **CLI da terminale** per esecuzioni automatizzate e benchmark batch.

---

## 📑 Indice

1. [Perché SuperBenchmark](#-perché-superbenchmark)
2. [I 6 Benchmark Inclusi](#-i-6-benchmark-inclusi)
3. [Metriche di Valutazione (RAGAS)](#-metriche-di-valutazione-ragas)
4. [Motori RAG Supportati](#-motori-rag-supportati)
5. [Requisiti di Sistema](#-requisiti-di-sistema)
6. [Installazione Rapida](#-installazione-rapida)
7. [Guida all&#39;Uso](#-guida-alluso)
   - [Opzione A: Dashboard Streamlit (Consigliata)](#opzione-a-dashboard-web-streamlit)
   - [Opzione B: Riga di Comando (CLI)](#opzione-b-interfaccia-cli-da-terminale)
8. [Configurazione (`config.json` e `.env`)](#-configurazione)
9. [Integrazione con DAVE (Elasticsearch)](#-integrazione-con-dave)
10. [Aggiunta di un Nuovo Benchmark](#-aggiunta-di-un-nuovo-benchmark)
11. [Struttura del Progetto](#-struttura-del-progetto)

---

## 🎯 Perché SuperBenchmark

Molti sistemi RAG vengono valutati solo su query semplici e testi sintetici. Nel mondo reale, tuttavia, i sistemi documentali aziendali devono affrontare:

- Testi degradati da scansioni OCR o tabelle complesse.
- Risposte distribuite su più documenti separati da aggregare.
- Contratti legali in cui una singola parola parafrasata costituisce un errore grave.
- Conversazioni multi-turno in cui l'utente usa pronomi e riferimenti al contesto precedente.
- Pagine web rumorose contenenti testo HTML non strutturato e trappole controfattuali.

SuperBenchmark aggrega dataset scientifici di riferimento pensati appositamente per mettere sotto stress ogni anello della catena RAG.

---

## 📚 I 6 Benchmark Inclusi

Ogni benchmark è già pre-elaborato e normalizzato in formato standard all'interno della cartella `Run/`:

| Benchmark                | Dataset Normalizzato         | Dominio & Debolezza Testata                                                                                                        |
| :----------------------- | :--------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **LegalBench-RAG** | `Tutto_LEGALBENCH.json`    | **Contratti legali reali** (691 contratti). Misura la capacità di estrarre clausole esatte senza allucinazioni o parafrasi. |
| **MultiHop-RAG**   | `Tutto_MULTIHOP.json`      | **Sintesi multi-documento** (609 news). Richiede di connettere informazioni presenti in 2–4 documenti diversi.              |
| **CRAG** (Meta)    | `Tutto_CRAG_DAVE.json`     | **Web HTML & Rumore**. Valuta la resilienza a contesti rumorosi, pagine web grezze e domande con false premesse.             |
| **MTRAG**          | `Tutto_MTRAG_DAVE.json`    | **Dialoghi multi-turno**. Verifica la gestione della memoria conversazionale e la risoluzione di co-referenze.               |
| **OHRBench**       | `Tutto_OHRBench_DAVE.json` | **Scansioni e Rumore OCR**. Documenti aziendali/accademici con tabelle corrotte e formule matematiche degradate.             |
| **SQuAD-it**       | `Tutto_SQUAD.json`         | **Comprensione in lingua italiana** (2.007 paragrafi Wikipedia). Valuta le performance estrattive su testi non inglesi.      |

---

## 📏 Metriche di Valutazione (RAGAS)

La valutazione sfrutta un modello linguistico locale o remoto in modalità **LLM-as-a-Judge**:

| Metrica                      | Ambito          | Descrizione                                                                                            |
| :--------------------------- | :-------------- | :----------------------------------------------------------------------------------------------------- |
| **Context Precision**  | *Retrieval*   | I documenti restituiti dal motore RAG sono pertinenti alla domanda posta?                              |
| **Context Recall**     | *Retrieval*   | I documenti recuperati contengono tutte le informazioni necessarie per la risposta?                    |
| **Faithfulness**       | *Generazione* | La risposta dell'LLM si basa rigorosamente sui soli documenti recuperati (assenza di allucinazioni)?   |
| **Answer Correctness** | *Generazione* | Quanto la risposta generata è corretta e allineata con la risposta di riferimento (*ground truth*)? |
| **Answer Relevancy**   | *Generazione* | La risposta è diretta e pertinente rispetto alla domanda dell'utente?                                 |

---

## 🔌 Motori RAG Supportati

SuperBenchmark supporta nativamente due tipologie di motori:

1. **Local RAG (BM25)**:
   - Qualsiasi server RAG HTTP locale o remoto conforme allo schema JSON standard (`POST /ask`).
   - Ideale per test su macchine locali o API proprietarie.
2. **DAVE (Elasticsearch / tRPC)**:
   - Piattaforma documentale con indicizzazione vettoriale/ibrida e generazione LLM integrata.
   - Gestito tramite `dave_client.py` con autenticazione Keycloak e streaming generation.

---

## 💻 Requisiti di Sistema

- **Python**: versione 3.10 o successiva (testato anche su 3.11 e 3.13).
- **Ollama** (per il calcolo locale delle metriche LLM-as-a-Judge):
  - Scaricabile da [ollama.com](https://ollama.com/).
  - Modello consigliato: `gemma4:31b-cloud`, `llama3.1`, o `mistral`.
  - In alternativa è possibile configurare un endpoint compatibile OpenAI/cloud tramite API Key.

---

## 📦 Installazione Rapida

### 1. Clonare il Repository

```bash
git clone https://github.com/unimib-datAI/rag-benchmark.git
cd SuperBenchmark
```

### 2. Creare e Attivare l'Ambiente Virtuale

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Installare le Dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configurare le Variabili d'Ambiente (opzionale per DAVE)

Copia il template delle variabili d'ambiente:

- Su Windows: `copy .env.example .env`
- Su Linux/macOS: `cp .env.example .env`

Modifica il file `.env` inserendo le credenziali se intendi interrogare l'istanza DAVE.

---

## 🖥️ Guida all'Uso

### Opzione A: Dashboard Web (Streamlit)

La modalità grafica è consigliata per esplorare i risultati e confrontare diversi modelli o configurazioni.

```bash
streamlit run webpage/app.py
```

La dashboard si aprirà automaticamente nel browser (tipicamente su `http://localhost:8501`) e offre due sezioni:

1. **Tab "Evaluation"**:
   - Seleziona il motore RAG (*Local RAG* o *DAVE*).
   - Imposta la soglia di similarità semantica (*Threshold*), il modello Judge Ollama e il numero di query per benchmark.
   - Premi **"Avvia Valutazione"**: la suite recupera i documenti, genera le risposte, calcola le metriche RAGAS in tempo reale e salva il report in `valutazioni/`.
2. **Tab "Explore"**:
   - Visualizza lo storico di tutti i test eseguiti.
   - Confronta più run con **Radar Chart** comparativi e metriche aggregate delta.
   - Individua immediatamente la configurazione migliore premiata con il trofeo di performance.

---

### Opzione B: Interfaccia CLI da Terminale

Per esecuzioni headless, automazioni CI o test rapidi senza interfaccia grafica:

```bash
python webpage/cli.py
```

All'avvio, la CLI mostrerà un menù interattivo guidato per confermare o modificare i parametri di test (endpoint, modello giudice, benchmark da includere, modalità random o fissa). Al termine dell'esecuzione verrà stampata a video una tabella dettagliata query per query e il file JSON completo verrà salvato automaticamente nella cartella `valutazioni/`.

---

## ⚙️ Configurazione

### File `config.json`

Contiene le impostazioni predefinite utilizzate sia dalla dashboard sia dalla CLI:

```json
{
    "endpoint": "http://127.0.0.1:8000/ask",
    "top_k": "3",
    "ollama_url": "http://localhost:11434",
    "model": "gemma4:31b-cloud",
    "benchmarks": "TUTTI",
    "sample_size": "3",
    "selection_mode": "R",
    "motore_rag": "Local RAG (BM25)",
    "threshold": "0.3"
}
```

---

## 🔍 Integrazione con DAVE

Se utilizzi il motore **DAVE**:

1. **Verificare la connessione**:

   ```bash
   python dave_client.py
   ```

   Lo script verificherà l'autenticazione Keycloak e confermerà le collezioni indicizzate disponibili nel file `.dave_collections.json`.
2. **Indicizzare nuovi documenti nel corpus**:

   ```bash
   python ingest_dave.py
   ```

   Lo script carica i documenti a lotti su DAVE, saltando automaticamente i file già elaborati grazie alla cache locale (`.dave_docs_cache.json`).

---

## ➕ Aggiunta di un Nuovo Benchmark

Aggiungere un nuovo dataset di benchmark richiede 3 semplici passaggi:

1. **Estrazione e Normalizzazione**: Crea uno script `Run/extract_nome.py` che trasformi il dataset grezzo in un file `Run/Tutto_NOME.json` con lo schema standard:
   ```json
   [
     {
       "id": "query_001",
       "source": "MioBenchmark",
       "query_type": "Estrazione_Semplice",
       "question": "Testo della domanda...",
       "context": "[Doc 1] Testo del contesto...",
       "gold_answer": "Risposta corretta di riferimento..."
     }
   ]
   ```
2. **Registrazione nella Dashboard**: Apri `webpage/app.py` e aggiungi il nuovo file al dizionario `BENCHMARK_MAPPING`.
3. **Registrazione nella CLI**: In `webpage/cli.py`, aggiungi la chiave corrispondente in `BENCHMARK_MAP`.

*(Per maggiori dettagli ed esempi completi, consulta la guida dedicata in [aggiunta_dataset.md](aggiunta_dataset.md)).*

---

## 📂 Struttura del Progetto

```text
SuperBenchmark/
├── README.md                      # Questa documentazione
├── requirements.txt               # Dipendenze Python del progetto
├── config.json                    # Configurazione di default di valutazione
├── .env.example                   # Template credenziali ambiente
├── dave_client.py                 # Client API per motore DAVE (tRPC / Next.js)
├── ingest_dave.py                 # Ingestion e indicizzazione documenti su DAVE
├── .dave_collections.json         # Mapping benchmark -> ID collezioni DAVE
├── .dave_docs_cache.json          # Cache documenti già indicizzati
│
├── webpage/                       # Core dell'applicazione (Dashboard e CLI)
│   ├── app.py                     # Interfaccia web Streamlit
│   ├── cli.py                     # Interfaccia da terminale
│   ├── evaluator.py               # Motore di valutazione RAG e pipeline RAGAS
│   ├── persistence.py             # Salvataggio e gestione storico run (JSON)
│   ├── dashboard.py               # Componenti grafici e card metriche
│   └── style.css                  # Tema dark-mode per Streamlit
│
├── Run/                           # Dati pronti per l'evaluation e script di parsing
│   ├── Tutto_LEGALBENCH.json      # Dataset LegalBench-RAG normalizzato
│   ├── Tutto_MULTIHOP.json        # Dataset MultiHop-RAG normalizzato
│   ├── Tutto_CRAG_DAVE.json       # Dataset CRAG normalizzato
│   ├── Tutto_MTRAG_DAVE.json      # Dataset MTRAG normalizzato
│   ├── Tutto_OHRBench_DAVE.json   # Dataset OHRBench normalizzato
│   ├── Tutto_SQUAD.json           # Dataset SQuAD-it normalizzato
│   └── extract_*.py               # Script di conversione formati grezzi
│
├── Benchmarks/                    # Dataset grezzi scaricati dalle fonti ufficiali
├── valutazioni/                   # Report JSON salvati dalle sessioni di test
├── Paper/                         # Pubblicazioni scientifiche e documentazione metodologica
└── aggiunta_dataset.md            # Guida dettagliata all'estensione dei benchmark
```
