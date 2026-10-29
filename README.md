# 🚀 SuperRagBench — RAG Evaluation Suite

**SuperRagBench** è una suite completa per la valutazione sistematica, standardizzata e approfondita di sistemi **RAG (Retrieval-Augmented Generation)**.
Adotta il paradigma **LLM-as-a-Judge** integrando le metriche della libreria **Ragas** per misurare oggettivamente la qualità del recupero (*retrieval*) e della generazione delle risposte su scenari reali, domini verticali e casi limite (*stress-test*).

Il progetto offre sia una **Dashboard Web interattiva** (Streamlit) per l'analisi visiva, il monitoraggio in tempo reale (TPS, token) e il confronto multi-run (Radar Chart, delta metriche), sia una potente **CLI da terminale** con supporto per modalità interattiva e modalità headless/batch per pipeline di CI/CD.

---

## 📑 Indice

1. [Perché SuperRagBench](#-perché-superragbench)
2. [I 6 Benchmark Inclusi](#-i-6-benchmark-inclusi)
3. [Metriche di Valutazione (RAGAS)](#-metriche-di-valutazione-ragas)
4. [Diagnostica Automatica & Error Pattern (Novità)](#-diagnostica-automatica--error-pattern)
5. [Motori RAG & Modelli Judge Supportati](#-motori-rag--modelli-judge-supportati)
6. [Requisiti di Sistema](#-requisiti-di-sistema)
7. [Installazione Rapida](#-installazione-rapida)
8. [Guida all'Uso](#-guida-alluso)
   - [Opzione A: Dashboard Web (Streamlit)](#opzione-a-dashboard-web-streamlit)
   - [Opzione B: Interfaccia CLI da Terminale](#opzione-b-interfaccia-cli-da-terminale)
9. [Configurazione (`config.json` e `.env`)](#-configurazione)
10. [Collegare il proprio RAG (`/ingest` e `/ask`)](#-collegare-il-proprio-rag)
11. [Integrazione con DAVE (Elasticsearch)](#-integrazione-con-dave)
12. [Aggiunta di un Nuovo Benchmark](#-aggiunta-di-un-nuovo-benchmark)
13. [Struttura del Progetto](#-struttura-del-progetto)

---

## 🎯 Perché SuperRagBench

Molti sistemi RAG vengono valutati esclusivamente su query semplici, contesti sintetici o domini generici. Nella realtà applicativa e aziendale, i sistemi documentali devono affrontare sfide complesse:

- **Contratti legali:** dove una singola parola parafrasata o una data errata costituisce un errore critico.
- **Sintesi multi-documento:** la risposta non risiede in un solo passaggio ma va ricostruita aggregando 2–4 documenti distinti.
- **Pagine web rumorose:** documenti con markup HTML grezzo, snippet fuorvianti e trappole controfattuali.
- **Conversazioni multi-turno:** follow-up con pronomi, ellissi e risoluzione di co-referenze storiche.
- **Documenti degradati (OCR):** scansioni cartacee con tabelle corrotte e caratteri alterati.
- **Supporto multilingua:** comprensione linguistica puntuale su testi in lingua italiana.

SuperRagBench aggrega dataset scientifici di riferimento pensati appositamente per sollecitare e mettere sotto stress ogni singolo anello della catena RAG.

---

## 📚 I 6 Benchmark Inclusi

Ogni benchmark è pre-elaborato e normalizzato in formato **JSONL** (`Run/Tutto_*.jsonl`) per garantire un caricamento in streaming efficiente a basso consumo di memoria (eliminando i rischi di `MemoryError` su dataset estesi):

| Benchmark | Dataset Normalizzato | Dominio & Debolezza Testata |
| :--- | :--- | :--- |
| **LegalBench-RAG** | `Tutto_LEGALBENCH.jsonl` | **Contratti legali reali** (691 contratti). Valuta la capacità di estrarre clausole contrattuali precise senza allucinazioni o parafrasi. |
| **MultiHop-RAG** | `Tutto_MULTIHOP.jsonl` | **Sintesi multi-documento** (609 news). Richiede il ragionamento logico e l'aggregazione di informazioni distribuite tra 2 e 4 documenti diversi. |
| **CRAG** (Meta) | `Tutto_CRAG_DAVE.jsonl` | **Web HTML & Rumore**. Misura la resilienza a pagine web non strutturate, frammenti rumorosi e domande con false premesse. |
| **MTRAG** | `Tutto_MTRAG_DAVE.jsonl` | **Dialoghi multi-turno**. Verifica la gestione della memoria della conversazione e la corretta interpretazione dei riferimenti contestuali. |
| **OHRBench** | `Tutto_OHRBench_DAVE.jsonl` | **Scansioni e Rumore OCR**. Documenti aziendali/accademici con tabelle corrotte, caratteri degradati e formule matematiche. |
| **SQuAD-it** | `Tutto_SQUAD.jsonl` | **Comprensione in lingua italiana** (2.007 paragrafi Wikipedia). Valuta le performance estrattive su testi e query in italiano. |

---

## 📏 Metriche di Valutazione (RAGAS)

La valutazione impiega un modello linguistico avanzato in modalità **LLM-as-a-Judge**. SuperRagBench calcola le 5 metriche cardine della libreria RAGAS, permettendo all'utente di selezionare quali calcolare per ottimizzare tempi e costi:

| Metrica | Ambito | Descrizione |
| :--- | :--- | :--- |
| **Context Precision** | *Retrieval* | Valuta se i passaggi recuperati dal motore RAG sono effettivamente pertinenti alla domanda. |
| **Context Recall** | *Retrieval* | Verifica se i documenti recuperati contengono tutte le informazioni necessarie per formulare la risposta. |
| **Faithfulness** | *Generazione* | Misura se la risposta generata è fondata strettamente sui soli documenti recuperati (assenza di allucinazioni). |
| **Answer Correctness** | *Generazione* | Misura l'accuratezza semantica e fattuale della risposta rispetto alla risposta di riferimento (*ground truth*). |
| **Answer Relevancy** | *Generazione* | Misura quanto la risposta fornita sia diretta e pertinente rispetto alla domanda posta. |

> **OHRBench:** il RAG fa retrieval sulle oltre 8.500 pagine PDF del benchmark, convertite in testo con errori OCR realistici (versione `semantic_noise_MinerU_moderate` di `retrieval.zip`, impostata in `OHR_VERSION` in `corpus_sources.py`). L'evidenza gold della domanda non viene mai passata al RAG. Circa l'1,6% delle domande di `qas.json` riguarda documenti assenti da questa versione del corpus (nomi di file non allineati nei dati ufficiali).

---

## 🔬 Diagnostica Automatica & Error Pattern

SuperRagBench include un motore euristico (`webpage/taxonomy.py`, funzione `classify_query_verdict`) che analizza le metriche calcolate query per query, **senza chiamate LLM aggiuntive**, classificando lo stato e la causa del fallimento. Le soglie sono definite una sola volta in `THRESHOLDS` (Correct `0.70`, Partial `0.40`, soglia alto/basso delle metriche diagnostiche `0.50`) e usate da UI, CLI e legenda.

### 1. Verdetto Query
- ✅ **Correct** (`Answer Correctness >= 0.70`): la risposta è corretta e allineata al riferimento.
- ⚠️ **Partial** (`0.40 <= Answer Correctness < 0.70`): risposta parzialmente corretta o incompleta; viene indicata anche la **causa probabile** (stesso codice dei pattern di errore).
- ❌ **Wrong** (`Answer Correctness < 0.40`): risposta errata o rifiuto da parte del modello.
- ⚪ **Not Evaluated**: RAGAS non ha calcolato la *Answer Correctness*; la query è esclusa dalla distribuzione degli errori.

### 2. Tassonomia dei Pattern di Errore
Per le risposte errate (e come causa delle risposte parziali) il pattern deriva da una tabella decisionale su *Context Precision*, *Context Recall*, *Faithfulness* e *Answer Relevancy* (alto = `>= 0.50`):

| Retrieval (Precision / Recall) | Faithfulness | Pattern |
| :--- | :--- | :--- |
| entrambe basse | bassa | `TOTAL_FAIL` |
| entrambe basse | alta / non calcolata | `RET_FAIL` |
| almeno una alta | bassa | `HALLUC` |
| Precision alta, Recall bassa | alta | `RET_INCOMPLETE` |
| Precision bassa, Recall alta | alta | `RET_NOISE` |
| entrambe alte | alta | `OFF_TOPIC` se *Relevancy* bassa, altrimenti `NOT_EXTRACTED` |
| metriche mancanti | — | `UNKNOWN` |

| Pattern ID | Nome Pattern | Causa & Diagnostica | Riferimento |
| :--- | :--- | :--- | :--- |
| `RET_FAIL` | **Retrieval Failure** | *Precision* e *Recall* basse: il motore RAG non ha trovato i documenti corretti. Suggerisce di rivedere embeddings, chunking o adottare ricerca ibrida. | Barnett et al. 2024 — FP1 *Missing Content*, FP2 *Missed the Top Ranked Documents* |
| `RET_INCOMPLETE` | **Incomplete Retrieval** | *Precision* alta ma *Recall* bassa: trovati passaggi pertinenti ma privi dei dettagli chiave. Suggerisce di aumentare il `top_k`. | Barnett et al. 2024 — FP2, FP3 *Not in Context* |
| `RET_NOISE` | **Retrieval Noise** | *Recall* alta ma *Precision* bassa, modello fedele: i documenti utili ci sono ma sono mescolati a molti irrilevanti. Suggerisce un re-ranker o una soglia di similarità più severa. | Chen et al. 2024 (RGB, *Noise Robustness*); Barnett et al. 2024 — FP3 |
| `HALLUC` | **Hallucination** | Contesto rilevante presente, ma *Faithfulness* bassa: il modello inventa informazioni non supportate. Suggerisce di abbassare la temperatura o rivedere il system prompt. | Es et al. 2024 (RAGAS, *Faithfulness*); Niu et al. 2024 (RAGTruth) |
| `NOT_EXTRACTED` | **Context Not Extracted** | Contesto completo e modello fedele, ma risposta errata: l'LLM non riesce a estrarre la risposta dal contesto (es. effetto *lost in the middle* su contesti lunghi). Suggerisce l'uso di un re-ranker o riduzione del contesto. | Barnett et al. 2024 — FP4 *Not Extracted*; Liu et al. 2024 (*Lost in the Middle*) |
| `OFF_TOPIC` | **Off-Topic Answer** | Contesto completo e modello fedele, ma *Answer Relevancy* bassa: la risposta non risponde alla domanda. Suggerisce di rivedere il system prompt. | Es et al. 2024 (RAGAS, *Answer Relevance*) |
| `REFUSAL` | **Refusal / Non-Answer** | Il modello rifiuta di rispondere (keyword multilingua cercate nei primi 200 caratteri della risposta). La causa probabile è ricavata dal solo retrieval: se il contesto era sufficiente diventa `NOT_EXTRACTED`. | Chen et al. 2024 (RGB, *Negative Rejection*); Yang et al. 2024 (CRAG, risposte *missing*) |
| `TOTAL_FAIL` | **System Collapse** | Retrieval fallito (*Precision* e *Recall* basse) **e** *Faithfulness* bassa. Richiede una revisione dell'intera pipeline RAG. | Categoria propria (combinazione di `RET_FAIL` e `HALLUC`) |
| `PARTIAL` | **Partial Answer** | Risposta incompleta (correttezza intermedia tra 0.40 e 0.70), con causa probabile calcolata con la stessa tabella. | Barnett et al. 2024 — FP7 *Incomplete* |
| `CORRECT` | **Correct Answer** | Risposta ampiamente corretta (correttezza >= 0.70). | Es et al. 2024 (RAGAS, *Answer Correctness*) |
| `EVAL_FAIL` | **Evaluation Failure** | RAGAS non ha calcolato la *Answer Correctness* (errore del giudice, timeout o metrica non selezionata): verdetto *Not Evaluated*, esclusa dal conteggio degli errori. | Errore del valutatore, non del RAG |
| `RAG_ERROR` | **RAG Error** | Il RAG non ha risposto (HTTP diverso da `200`, timeout, eccezione, risposta vuota, es. DAVE che chiude lo stream vuoto): verdetto *Not Evaluated*, la query non viene inviata a RAGAS ed è esclusa dalle medie e dal conteggio degli errori. Se nessuna query riceve risposta la valutazione si ferma e non viene salvata. | Errore dell'infrastruttura, non una risposta sbagliata |
| `UNKNOWN` | **Unclassified Error** | Risposta errata, ma la causa non è determinabile perché mancano metriche diagnostiche (NaN o non selezionate). | — |

### 3. Metriche mancanti, medie e diagnostica aggregata
- Le metriche non calcolate (NaN o deselezionate nella UI) vengono salvate in `metriche_non_valide`, **ignorate** dal classificatore, escluse dalle medie (UI, CLI e file salvati) e mostrate come `N/A` / `—`. Per le valutazioni salvate prima di questa modifica, in una query con `ragas_fallita` i valori esattamente `0.0` sono considerati non calcolati.
- La **diagnostica aggregata** (pannello *Advanced Heuristic Analysis* e CLI) è costruita dalla distribuzione dei pattern per query: elenca le cause che riguardano almeno il 10% delle query valutate, contando anche le cause delle risposte parziali e dei rifiuti.
- Ogni classificazione salva `taxonomy_version`: le valutazioni salvate con una versione diversa vengono riclassificate automaticamente all'apertura.
- Test del classificatore: `.venv/Scripts/python.exe test/test_taxonomy.py -v`

> **Nota metodologica:** la tassonomia è un'euristica costruita sulle metriche RAGAS e allineata alla letteratura indicata; le soglie (0.4 / 0.5 / 0.7) sono scelte empiriche non calibrate, e il rilevamento di `REFUSAL` è basato su keyword. I riferimenti bibliografici completi sono in [`riferimenti_tassonomia_errori.txt`](riferimenti_tassonomia_errori.txt).

---

## 🔌 Motori RAG & Modelli Judge Supportati

### Motori RAG Sotto Esame
1. **Local RAG (BM25 / Custom HTTP)**:
   - Qualsiasi server RAG locale o remoto che espone i due endpoint `POST /ingest` e `POST /ask` (vedi [Collegare il proprio RAG](#-collegare-il-proprio-rag)).
2. **DAVE (Elasticsearch / Next.js / tRPC)**:
   - Piattaforma documentale con indicizzazione full-text/ibrida e generazione LLM integrata via streaming, con autenticazione Keycloak.

### Modelli LLM Judge
- **Ollama (Locale o Remoto)**: es. `gemma4:31b-cloud`, `llama3.1`, `mistral`, `qwen2.5`.
- **Provider Cloud / OpenAI-Compatible**: supporto per OpenAI, Azure OpenAI, vLLM o server universitari via URL base e **API Key** (`Bearer` token) con fallback automatico su `ChatOpenAI`.

---

## 💻 Requisiti di Sistema

- **Python**: 3.10 o superiore (testato su Python 3.10, 3.11, 3.12 e 3.13).
- **Ollama** (per esecuzione locale del Judge) oppure un endpoint compatibile OpenAI.
- Connessione di rete per il download dei pesi di embedding HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`).

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
- Su Windows: `copy .env.example .env`
- Su Linux/macOS: `cp .env.example .env`

Modifica `.env` inserendo le credenziali Keycloak e l'URL di DAVE se intendi testare tale motore.

### 5. Scaricare i Benchmark
I dataset grezzi non sono inclusi nel repository. Nel tab **📚 Catalogue** della dashboard si spuntano nella tabella i benchmark desiderati e il pulsante **Download selected** scarica dalle fonti ufficiali (GitHub, Hugging Face, Dropbox) solo i file usati dagli script di estrazione dei benchmark scelti (~1 GB per tutti, compreso il corpus OCR di OHRBench) e li salva in `Benchmarks_download.zip` nella cartella del progetto. In alternativa, da terminale (senza argomenti scarica tutti; altrimenti i nomi dei benchmark, es. `SQuAD-it CRAG`):

```bash
python webpage/benchmark_download.py
```

Estrai lo zip nella cartella del progetto (crea `Benchmarks/` con la struttura attesa), poi genera i file JSONL:

```bash
python Run/extract_legalbenchmark.py
python Run/extract_multihop.py
python Run/extract_crag.py
python Run/extract_mtrag.py
python Run/extract_ohrbench.py
python Run/extract_squad.py
```

---

## 🖥️ Guida all'Uso

### Opzione A: Dashboard Web (Streamlit)

La dashboard grafica offre il massimo livello di controllo e analisi visiva:

```bash
streamlit run webpage/app.py
```

La dashboard si avvierà nel browser (di default su `http://localhost:8501`) e comprende cinque tab (Evaluation, Explore, Catalogue, Setup, API Guide):

#### 1. Tab "🔬 Evaluation"
- **Configurazione Sidebar:**
  - Scelta del motore RAG (*Local RAG* o *DAVE*).
  - Impostazione dell'URL base.
  - Configurazione dell'LLM Judge (URL Ollama/OpenAI e API Key facoltativa).
  - Selezione mirata dei benchmark e delle metriche RAGAS da calcolare.
  - **Richieste in parallelo (Workers):** slider da 1 a 10 per velocizzare sensibilmente il calcolo delle metriche RAGAS.
  - **Modalità di selezione query:**
    - `Random`: campionamento casuale.
    - `Fixed`: selezione deterministica dei primi $N$ campioni.
    - `Repeat Previous`: ripete le stesse identiche query del test precedente per confronti A/B rigorosi.
  - **Nome della valutazione:** campo obbligatorio per tracciare il run.
- **Monitoraggio Live:** barra di avanzamento e metriche live di throughput (**Tokens Per Second - TPS**, token totali, input/output).
- **Report & Risultati:**
  - Card riassuntive delle metriche medie con codifica cromatica.
  - Barra riassuntiva dei verdetti e distribuzione dei pattern di errore.
  - Dettaglio per singola query espandibile con verdetto, motivazione diagnostica, prompt/risposta RAG, gold answer e contesto recuperato.
  - Diagnostica euristica avanzata complessiva.

#### 2. Tab "📊 Explore"
- Consulta lo storico di tutte le esecuzioni salvate in `valutazioni/`.
- **Confronto Multi-Run:** seleziona due o più valutazioni per visualizzare **Radar Chart comparativi** e tabelle di variazione percentuale ($\Delta$).
- Riconoscimento immediato del run con le prestazioni migliori con badge di merito.
- Cancellazione o ricaricamento di valutazioni archiviate con un click.

#### 3. Tab "📚 Catalogue"
- Tabella di **tutti i benchmark supportati** con descrizione, stato (scaricato o no), numero di query e dimensione. La barra di ricerca filtra per **nome o descrizione** (non per contenuto).
- Si spuntano i benchmark da scaricare: **Download selected** scarica i dati grezzi solo dei benchmark scelti e ne genera i file di query; **Build query files** li rigenera dai dati già scaricati.

#### 4. Tab "🛠️ Setup"
- Connessione a DAVE e indicizzazione del corpus, oppure caricamento del corpus su un RAG generico (`POST /ingest`).
- Si possono indicizzare **solo i benchmark scaricati** dal tab Catalogue: gli altri non compaiono nelle tabelle di caricamento.
- Impostazioni di batch (documenti e MB per richiesta, pausa, numero massimo di documenti) e pannello con avanzamento e log del job.

#### 5. Tab "📖 API Guide"
- Descrive le route `/ask` e `/ingest` che un RAG di terze parti deve esporre.

---

### Opzione B: Interfaccia CLI da Terminale

SuperRagBench include una CLI completa (`webpage/cli.py`) perfettamente allineata con tutte le funzionalità della dashboard, ideale per automazioni, script batch ed esecuzioni headless.

#### 1. Modalità Interattiva (Menù Guidato)
```bash
python webpage/cli.py
```
Mostra un menù interattivo a terminale che propone i valori predefiniti da `config.json` e consente di personalizzare ogni parametro prima dell'avvio.

#### 2. Modalità Batch / Headless (CI/CD)
Tutti i parametri possono essere passati direttamente come argomenti da linea di comando:

```bash
# Esecuzione completa automatizzata con output su file
python webpage/cli.py --name "exp_llama3_run1" --engine "Local" --sample 5 --benchmarks SQUAD,LEGALBENCH --workers 4

# Esecuzione su DAVE saltando l'ingestion preliminare
python webpage/cli.py --name "test_dave_batch" --engine "DAVE" --sample 10 --no-ingest

# Utilizzo di un modello OpenAI/Cloud compatibile
python webpage/cli.py --name "gpt4o_judge" --judge-url "https://api.openai.com/v1" --model "gpt-4o-mini" --api-key "sk-..."

# Consultazione rapida dello storico delle valutazioni salvate
python webpage/cli.py --list
```

#### Argomenti CLI Supportati

| Parametro | Descrizione | Default / Valori |
| :--- | :--- | :--- |
| `--name` | Nome identificativo della valutazione | `cli_run_YYYYMMDD_HHMMSS` |
| `--engine` | Motore RAG target | `'Local'` oppure `'DAVE'` |
| `--endpoint` | URL base del server RAG | Valore da `config.json` |
| `--judge-url` | Endpoint dell'LLM Judge | `http://localhost:11434` |
| `--model` | Modello LLM Judge | `gemma4:31b-cloud` |
| `--api-key` | API Key per endpoint cloud / OpenAI | `""` |
| `--benchmarks` | Elenco benchmark separati da virgola o `ALL` | `ALL` (`LEGALBENCH,MULTIHOP,CRAG,MTRAG,OHRBENCH,SQUAD`) |
| `--sample` | Numero di query per ciascun benchmark | `3` |
| `--mode` | Modalità campionamento (`Random`, `Fixed`, `Repeat`) | `Random` |
| `--workers` | Numero di worker paralleli per RAGAS (1-10) | `2` |
| `--no-ingest` | Salta la fase di ingestion documenti preliminare | Flag booleano |
| `--list` | Stampa l'elenco delle valutazioni salvate ed esce | Flag booleano |
| `--interactive`| Forza l'apertura del menù interattivo guidato | Flag booleano |

---

## ⚙️ Configurazione

### File `config.json`
Conserva le impostazioni di default condivise da CLI e Web Dashboard:

```json
{
    "endpoint": "http://127.0.0.1:8000",
    "top_k": "3",
    "ollama_url": "http://localhost:11434",
    "model": "gemma4:31b-cloud",
    "benchmarks": "TUTTI",
    "sample_size": "3",
    "selection_mode": "R",
    "motore_rag": "Local RAG (BM25)"
}
```

### File `.env` (Credenziali DAVE)
Contiene le credenziali per l'interazione con l'istanza DAVE:
```env
DAVE_BASE_URL=https://vm.chronos.disco.unimib.it
DAVE_USERNAME=il_tuo_username
DAVE_PASSWORD=la_tua_password
DAVE_CLIENT_SECRET=il_tuo_secret
```

---

## 🔗 Collegare il proprio RAG

Per valutare il proprio sistema con il motore **Local RAG**, il server deve esporre **due endpoint HTTP** sotto lo stesso Base URL (campo *Base URL RAG* della dashboard, `--endpoint` da CLI, es. `http://127.0.0.1:8000`):

| Endpoint | A cosa serve |
| :--- | :--- |
| `POST /ingest` | Controllare quali documenti del corpus mancano e caricarli nell'indice del RAG |
| `POST /ask` | Rispondere a una domanda restituendo la risposta e i documenti recuperati |

### Flusso di una valutazione

Per ogni benchmark selezionato, prima di porre le domande:

1. SuperRagBench legge il **corpus completo** del benchmark (lo stesso caricato su DAVE da `ingest_dave.py`), non i soli documenti gold:

   | Benchmark | Corpus indicizzato |
   | :--- | :--- |
   | MultiHop-RAG | `Benchmarks/MultiHop-RAG/dataset/corpus.json` |
   | MTRAG | `Benchmarks/mt-rag-benchmark/corpora/document_level/*.jsonl` |
   | LegalBench-RAG | `Benchmarks/legalbenchrag/data/corpus/` |
   | SQuAD-it | `Run/squad_corpus.json` |
   | OHRBench | `Benchmarks/OHR-Bench/data/retrieval_base/semantic_noise_MinerU_moderate/` (una pagina PDF per documento, testo con rumore OCR) |
   | CRAG | Pagine web fornite dal benchmark per ogni domanda (campo `context` di `Run/Tutto_CRAG_DAVE.jsonl`), come **HTML grezzo**: estrarre l'informazione dal codice web è compito del RAG |

   I documenti più lunghi di 500.000 caratteri vengono divisi in parti (`<id>_part2`, `<id>_part3`, …), lo stesso limite usato per DAVE.

   Se il corpus non è stato scaricato, la valutazione si ferma con un errore (vedi [Scaricare i Benchmark](#5-scaricare-i-benchmark)).
2. Chiede a `/ingest` quali documenti mancano nell'indice (`check_only`).
3. Se non manca nulla passa subito alla valutazione; altrimenti invia **solo i documenti mancanti**, a lotti (max 500 documenti o 20 MB per richiesta).
4. Esegue una chiamata `/ask` per ogni domanda.

Da CLI l'ingestion si può saltare con `--no-ingest`.

### `POST /ingest` — controllo dei documenti presenti

Richiesta:
```json
{
  "benchmark_name": "MTRAG",
  "check_only": true,
  "expected_count": 252767,
  "ids": ["827849752_115-357", "ibmcld_00001", "..."]
}
```
Risposta attesa: gli `id` ricevuti che **non** sono ancora nell'indice di quel `benchmark_name`.
```json
{ "missing": ["ibmcld_00001"] }
```
Con `"missing": []` (oppure `{"complete": true}`) l'ingestion del benchmark viene saltata. Un server che non gestisce `check_only` riceve l'intero corpus a ogni valutazione.

### `POST /ingest` — caricamento dei documenti

Richiesta (una per lotto):
```json
{
  "benchmark_name": "MTRAG",
  "documents": [
    { "id": "827849752_115-357", "text": "Egocentrism is the inability to ..." }
  ]
}
```
Il server deve rispondere `200` e **aggiungere** i documenti all'indice esistente di quel `benchmark_name`, ignorando gli `id` già presenti. Arrivano più richieste per lo stesso benchmark: un `/ingest` che ricrea l'indice a ogni chiamata conserverebbe solo l'ultimo lotto. Una risposta diversa da `200` (o un errore di rete) durante il caricamento **ferma la valutazione** con un errore, per non valutare il RAG su un indice incompleto: alla valutazione successiva il caricamento riprende dai documenti mancanti.

### `POST /ask` — risposta a una domanda

Richiesta:
```json
{
  "question": "Who is ...?",
  "top_k": 10,
  "benchmark_name": "MultiHop-RAG",
  "search_query": "Who is ...?",
  "documents": ["testo documento 1", "testo documento 2"]
}
```
- `benchmark_name`: indice in cui cercare (lo stesso usato in `/ingest`).
- `question`: domanda da passare al generatore. Per MTRAG contiene anche la storia della conversazione (`[HISTORY] ... [NEW QUESTION] ...`).
- `search_query`: testo da usare per il retrieval (per MTRAG include i turni precedenti).
- `documents` (opzionale): se presente, il RAG **non** deve fare retrieval e deve rispondere usando solo questi testi. Viene inviato per rigenerare la risposta sui soli documenti rimasti dopo il filtro di *Similarity Threshold*.

Risposta attesa:
```json
{
  "answer": "Risposta generata",
  "contexts": ["testo del primo documento recuperato", "..."]
}
```
`contexts` può contenere stringhe oppure oggetti `{"text": "..."}`: sono i documenti che RAGAS usa per le metriche di retrieval e di faithfulness. In caso di errore il server risponde con uno status diverso da `200` e `{"error": "descrizione"}`. Una risposta diversa da `200`, non JSON o con `answer` mancante/vuoto non viene mai valutata come risposta: la query è marcata `RAG_ERROR` ed esclusa da RAGAS e dalle medie.

---

## 🔍 Integrazione con DAVE

Se utilizzi **DAVE** come motore RAG documentale:

1. **Test di Connessione e Autenticazione:**
   ```bash
   python dave_client.py
   ```
   Verifica l'autenticazione Keycloak e sincronizza il mapping delle collezioni in `.dave_collections.json`.

2. **Indicizzazione Massiva del Corpus:**
   ```bash
   python ingest_dave.py
   ```
   Carica su DAVE i corpus dei benchmark, a batch, tracciando i file già inviati nella cache locale `.dave_docs_cache.json` per evitare re-upload non necessari. Poi attende che DAVE li abbia indicizzati e ricarica quelli falliti. Opzioni: `--benchmarks` (default: tutti), `--batch-docs`, `--batch-mb`, `--sleep`, `--max-docs`, `--no-wait` (non attendere l'indicizzazione), `--verify` (confronta cache e DAVE senza caricare nulla), `--recreate` (crea nuove collezioni al posto di quelle esistenti; le vecchie restano su DAVE).

---

## ➕ Aggiunta di un Nuovo Benchmark

Aggiungere un nuovo dataset alla suite richiede 3 passaggi:

1. **Estrazione e Normalizzazione (`Run/extract_nome.py`):**
   Crea uno script che trasformi il dataset grezzo in un file JSONL (`Run/Tutto_NOME.jsonl`), dove ogni riga rappresenta un record JSON valido:
   ```json
   {"id": "query_001", "source": "MioBenchmark", "query_type": "Estrazione_Semplice", "question": "Testo della domanda...", "context": "[Doc 1] Contesto...", "gold_answer": "Risposta di riferimento..."}
   ```
2. **Registrazione nella Dashboard:**
   In `webpage/app.py`, aggiungi la voce in `BENCHMARK_MAPPING`:
   ```python
   BENCHMARK_MAPPING["MioBenchmark"] = "Tutto_NOME.jsonl"
   ```
3. **Registrazione nella CLI:**
   In `webpage/cli.py`, inserisci la chiave corrispondente in `BENCHMARK_MAP`.

*(Per la guida passo-passo dettagliata e l'indicizzazione su DAVE, consulta [aggiunta_dataset.md](aggiunta_dataset.md)).*

---

## 📂 Struttura del Progetto

```text
SuperRagBench/
├── README.md                      # Documentazione generale del progetto
├── requirements.txt               # Dipendenze Python (Streamlit, RAGAS, LangChain, ecc.)
├── config.json                    # Configurazione predefinita per UI e CLI
├── .env.example                   # Template delle variabili d'ambiente DAVE/Keycloak
├── dave_client.py                 # Client API per motore DAVE (tRPC, sessioni e streaming)
├── ingest_dave.py                 # Script per caricamento e indicizzazione corpus su DAVE
├── corpus_sources.py              # Corpus di ogni benchmark, condiviso da RAG locale e DAVE
├── .dave_collections.json         # Mapping benchmark -> UUID collezioni DAVE
├── .dave_docs_cache.json          # Cache locale per tracciamento documenti già indicizzati
├── .dave_upload_jobs.json         # Job di upload DAVE da verificare (file falliti da ricaricare)
│
├── webpage/                       # Core dell'applicazione (Dashboard e CLI)
│   ├── app.py                     # Interfaccia web Streamlit (Evaluation ed Explore)
│   ├── cli.py                     # Interfaccia a riga di comando (Interattiva e Batch)
│   ├── evaluator.py               # Motore di valutazione, query ai RAG e calcolo RAGAS
│   ├── taxonomy.py                # Tassonomia errori: verdetti, pattern, cause e diagnostica aggregata
│   ├── persistence.py             # Gestione salvataggio, caricamento ed eliminazione run
│   ├── benchmark_download.py      # Download dei benchmark grezzi dalle fonti ufficiali (zip)
│   ├── dashboard.py               # Componenti grafici HTML/CSS, card metriche e leggende errori
│   └── style.css                  # Foglio di stile personalizzato dark-mode per Streamlit
│
├── Run/                           # Dataset normalizzati e script di estrazione
│   ├── Tutto_LEGALBENCH.jsonl     # LegalBench-RAG in formato JSONL
│   ├── Tutto_MULTIHOP.jsonl       # MultiHop-RAG in formato JSONL
│   ├── Tutto_CRAG_DAVE.jsonl      # CRAG (Meta) in formato JSONL
│   ├── Tutto_MTRAG_DAVE.jsonl     # MTRAG conversazionale in formato JSONL
│   ├── Tutto_OHRBench_DAVE.jsonl  # OHRBench (OCR noise) in formato JSONL
│   ├── Tutto_SQUAD.jsonl          # SQuAD-it in formato JSONL
│   ├── squad_corpus.json          # Corpus di SQuAD-it (paragrafi unici), generato da extract_squad.py
│   └── extract_*.py               # Script di conversione dai formati grezzi a JSONL
│
├── test/                          # Test automatici (tassonomia, errori del RAG, pipeline DAVE)
│
├── Benchmarks/                    # Dataset grezzi originali (non versionati, vedi "Scaricare i Benchmark")
├── valutazioni/                   # Report JSON archiviati con i risultati di ciascuna sessione
├── Paper/                         # Pubblicazioni scientifiche di riferimento sui dataset e su RAGAS
└── aggiunta_dataset.md            # Guida dettagliata per integrare nuovi dataset
```
