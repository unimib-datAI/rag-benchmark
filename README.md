# SuperRagBench — RAG Evaluation Suite

**SuperRagBench** is a complete suite for the systematic, standardized and in-depth evaluation of **RAG (Retrieval-Augmented Generation)** systems.
It adopts the **LLM-as-a-Judge** paradigm, integrating the metrics of the **Ragas** library to objectively measure the quality of retrieval and answer generation on real-world scenarios, vertical domains and edge cases (*stress tests*).

The project offers both an interactive **Web Dashboard** (Streamlit) for visual analysis, real-time monitoring (TPS, tokens) and multi-run comparison (Radar Chart, metric deltas), and a powerful **terminal CLI** supporting an interactive mode and a headless/batch mode for CI/CD pipelines.

---

## Table of Contents

1. [Why SuperRagBench](#why-superragbench)
2. [The 6 Included Benchmarks](#the-6-included-benchmarks)
3. [Evaluation Metrics (RAGAS)](#evaluation-metrics-ragas)
4. [Automatic Diagnostics & Error Patterns](#automatic-diagnostics--error-patterns)
5. [Supported RAG Engines & Judge Models](#supported-rag-engines--judge-models)
6. [System Requirements](#system-requirements)
7. [Quick Installation](#quick-installation)
8. [Usage Guide](#usage-guide)
   - [Option A: Web Dashboard (Streamlit)](#option-a-web-dashboard-streamlit)
   - [Option B: Terminal CLI](#option-b-terminal-cli)
9. [Configuration (`config.json` and `.env`)](#configuration)
10. [Connecting Your Own RAG (`/ingest` and `/ask`)](#connecting-your-own-rag)
11. [DAVE Integration (Elasticsearch)](#dave-integration)
12. [Creating Your Own Dataset (Practical Guide)](#creating-your-own-dataset-practical-guide)
13. [Adding a New Benchmark](#adding-a-new-benchmark)
14. [Project Structure](#project-structure)

---

## Why SuperRagBench

Many RAG systems are evaluated only on simple queries, synthetic contexts or generic domains. In real business applications, document systems must face complex challenges:

- **Legal contracts:** where a single paraphrased word or a wrong date is a critical error.
- **Multi-document synthesis:** the answer does not live in a single passage but must be reconstructed by aggregating 2–4 distinct documents.
- **Noisy web pages:** documents with raw HTML markup, misleading snippets and counterfactual traps.
- **Multi-turn conversations:** follow-ups with pronouns, ellipsis and resolution of historical co-references.
- **Degraded documents (OCR):** paper scans with corrupted tables and altered characters.
- **Multilingual support:** accurate language understanding on Italian-language texts.

SuperRagBench aggregates reference scientific datasets designed specifically to stress every single link in the RAG chain.

---

## The 6 Included Benchmarks

Each benchmark is pre-processed and normalized into **JSONL** format (`Run/Tutto_*.jsonl`) to guarantee efficient, low-memory streaming loading (eliminating the risk of `MemoryError` on large datasets):

| Benchmark | Normalized Dataset | Domain & Weakness Tested |
| :--- | :--- | :--- |
| **LegalBench-RAG** | `Tutto_LEGALBENCH.jsonl` | **Real legal contracts** (691 contracts). Evaluates the ability to extract precise contract clauses without hallucinations or paraphrasing. |
| **MultiHop-RAG** | `Tutto_MULTIHOP.jsonl` | **Multi-document synthesis** (609 news articles). Requires logical reasoning and aggregation of information spread across 2 to 4 different documents. |
| **CRAG** (Meta) | `Tutto_CRAG_DAVE.jsonl` | **Web HTML & Noise**. Measures resilience to unstructured web pages, noisy fragments and questions with false premises. |
| **MTRAG** | `Tutto_MTRAG_DAVE.jsonl` | **Multi-turn dialogues**. Verifies conversation memory handling and correct interpretation of contextual references. |
| **OHRBench** | `Tutto_OHRBench_DAVE.jsonl` | **Scans & OCR Noise**. Business/academic documents with corrupted tables, degraded characters and mathematical formulas. |
| **SQuAD-it** | `Tutto_SQUAD.jsonl` | **Italian-language comprehension** (2,007 Wikipedia paragraphs). Evaluates extractive performance on Italian texts and queries. |

---

## Evaluation Metrics (RAGAS)

The evaluation uses an advanced language model in **LLM-as-a-Judge** mode. SuperRagBench computes the 5 core metrics of the RAGAS library, letting the user choose which ones to compute to optimize time and cost:

| Metric | Scope | Description |
| :--- | :--- | :--- |
| **Context Precision** | *Retrieval* | Evaluates whether the passages retrieved by the RAG engine are actually relevant to the question. |
| **Context Recall** | *Retrieval* | Checks whether the retrieved documents contain all the information needed to formulate the answer. |
| **Faithfulness** | *Generation* | Measures whether the generated answer is grounded strictly in the retrieved documents alone (no hallucinations). |
| **Answer Correctness** | *Generation* | Measures the semantic and factual accuracy of the answer against the reference answer (*ground truth*). |
| **Answer Relevancy** | *Generation* | Measures how direct and pertinent the provided answer is with respect to the question asked. |

> **OHRBench:** the RAG performs retrieval over the 8,500+ PDF pages of the benchmark, converted to text with realistic OCR errors (the `semantic_noise_MinerU_moderate` version of `retrieval.zip`, set in `OHR_VERSION` in `corpus_sources.py`). The gold evidence of the question is never passed to the RAG. About 1.6% of the questions in `qas.json` refer to documents missing from this corpus version (file names not aligned in the official data).

---

## Automatic Diagnostics & Error Patterns

SuperRagBench includes a heuristic engine (`webpage/taxonomy.py`, function `classify_query_verdict`) that analyzes the metrics computed query by query, **with no additional LLM calls**, classifying the state and cause of failure. The thresholds are defined once in `THRESHOLDS` (Correct `0.70`, Partial `0.40`, high/low threshold of the diagnostic metrics `0.50`) and used by the UI, CLI and legend.

### 1. Query Verdict
- **Correct** (`Answer Correctness >= 0.70`): the answer is correct and aligned with the reference.
- **Partial** (`0.40 <= Answer Correctness < 0.70`): partially correct or incomplete answer; the **probable cause** is also reported (same code as the error patterns).
- **Wrong** (`Answer Correctness < 0.40`): wrong answer or refusal by the model.
- **Not Evaluated**: RAGAS did not compute *Answer Correctness*; the query is excluded from the error distribution.

### 2. Error Pattern Taxonomy
For wrong answers (and as the cause of partial answers) the pattern comes from a decision table on *Context Precision*, *Context Recall*, *Faithfulness* and *Answer Relevancy* (high = `>= 0.50`):

| Retrieval (Precision / Recall) | Faithfulness | Pattern |
| :--- | :--- | :--- |
| both low | low | `TOTAL_FAIL` |
| both low | high / not computed | `RET_FAIL` |
| at least one high | low | `HALLUC` |
| Precision high, Recall low | high | `RET_INCOMPLETE` |
| Precision low, Recall high | high | `RET_NOISE` |
| both high | high | `OFF_TOPIC` if *Relevancy* is low, otherwise `NOT_EXTRACTED` |
| missing metrics | — | `UNKNOWN` |

| Pattern ID | Pattern Name | Cause & Diagnostics | Reference |
| :--- | :--- | :--- | :--- |
| `RET_FAIL` | **Retrieval Failure** | Low *Precision* and *Recall*: the RAG engine did not find the correct documents. Suggests reviewing embeddings, chunking or adopting hybrid search. | Barnett et al. 2024 — FP1 *Missing Content*, FP2 *Missed the Top Ranked Documents* |
| `RET_INCOMPLETE` | **Incomplete Retrieval** | High *Precision* but low *Recall*: relevant passages were found but lack key details. Suggests increasing `top_k`. | Barnett et al. 2024 — FP2, FP3 *Not in Context* |
| `RET_NOISE` | **Retrieval Noise** | High *Recall* but low *Precision*, faithful model: the useful documents are there but mixed with many irrelevant ones. Suggests a re-ranker or a stricter similarity threshold. | Chen et al. 2024 (RGB, *Noise Robustness*); Barnett et al. 2024 — FP3 |
| `HALLUC` | **Hallucination** | Relevant context present, but low *Faithfulness*: the model invents unsupported information. Suggests lowering the temperature or reviewing the system prompt. | Es et al. 2024 (RAGAS, *Faithfulness*); Niu et al. 2024 (RAGTruth) |
| `NOT_EXTRACTED` | **Context Not Extracted** | Complete context and faithful model, but wrong answer: the LLM fails to extract the answer from the context (e.g. *lost in the middle* effect on long contexts). Suggests using a re-ranker or reducing the context. | Barnett et al. 2024 — FP4 *Not Extracted*; Liu et al. 2024 (*Lost in the Middle*) |
| `OFF_TOPIC` | **Off-Topic Answer** | Complete context and faithful model, but low *Answer Relevancy*: the answer does not address the question. Suggests reviewing the system prompt. | Es et al. 2024 (RAGAS, *Answer Relevance*) |
| `REFUSAL` | **Refusal / Non-Answer** | The model refuses to answer (multilingual keywords searched in the first 200 characters of the answer). The probable cause is derived from retrieval alone: if the context was sufficient it becomes `NOT_EXTRACTED`. | Chen et al. 2024 (RGB, *Negative Rejection*); Yang et al. 2024 (CRAG, *missing* answers) |
| `TOTAL_FAIL` | **System Collapse** | Retrieval failed (low *Precision* and *Recall*) **and** low *Faithfulness*. Requires a review of the entire RAG pipeline. | Own category (combination of `RET_FAIL` and `HALLUC`) |
| `PARTIAL` | **Partial Answer** | Incomplete answer (correctness between 0.40 and 0.70), with the probable cause computed using the same table. | Barnett et al. 2024 — FP7 *Incomplete* |
| `CORRECT` | **Correct Answer** | Largely correct answer (correctness >= 0.70). | Es et al. 2024 (RAGAS, *Answer Correctness*) |
| `EVAL_FAIL` | **Evaluation Failure** | RAGAS did not compute *Answer Correctness* (judge error, timeout or metric not selected): *Not Evaluated* verdict, excluded from the error count. | Evaluator error, not a RAG error |
| `RAG_ERROR` | **RAG Error** | The RAG did not answer (HTTP status other than `200`, timeout, exception, empty answer, e.g. DAVE closing an empty stream): *Not Evaluated* verdict, the query is not sent to RAGAS and is excluded from averages and the error count. If no query gets an answer the evaluation stops and is not saved. | Infrastructure error, not a wrong answer |
| `UNKNOWN` | **Unclassified Error** | Wrong answer, but the cause cannot be determined because diagnostic metrics are missing (NaN or not selected). | — |

### 3. Missing Metrics, Averages and Aggregate Diagnostics
- Metrics that were not computed (NaN or deselected in the UI) are ignored by the classifier, excluded from the averages and shown as `N/A`.
- The **aggregate diagnostics** (*Advanced Heuristic Analysis* panel and CLI) are built from the per-query pattern distribution: they list the causes affecting at least 10% of the evaluated queries, also counting the causes of partial answers and refusals.
- Each classification saves `taxonomy_version`: evaluations saved with a different version are automatically reclassified when opened.
- Classifier tests: `.venv/Scripts/python.exe test/test_taxonomy.py -v`

> **Methodological note:** the taxonomy is a heuristic built on the RAGAS metrics and aligned with the cited literature; the thresholds (0.4 / 0.5 / 0.7) are uncalibrated empirical choices, and `REFUSAL` detection is keyword-based. Full bibliographic references are in [`riferimenti_tassonomia_errori.txt`](riferimenti_tassonomia_errori.txt).

---

## Supported RAG Engines & Judge Models

### RAG Engines Under Test
1. **Local RAG (BM25 / Custom HTTP)**:
   - Any local or remote RAG server exposing the two endpoints `POST /ingest` and `POST /ask` (see [Connecting Your Own RAG](#connecting-your-own-rag)).
2. **DAVE (Elasticsearch / Next.js / tRPC)**:
   - Document platform with full-text/hybrid indexing and built-in LLM generation via streaming, with Keycloak authentication.

### LLM Judge Models
- **Ollama (Local or Remote)**: e.g. `gemma4:31b-cloud`, `llama3.1`, `mistral`, `qwen2.5`.
- **Cloud / OpenAI-Compatible Providers**: support for OpenAI, Azure OpenAI, vLLM or university servers via base URL and **API Key** (`Bearer` token) with automatic fallback to `ChatOpenAI`.

---

## System Requirements

- **Python**: 3.10 or higher (tested on Python 3.10, 3.11, 3.12 and 3.13).
- **Ollama** (to run the Judge locally) or an OpenAI-compatible endpoint.
- Network connection to download the HuggingFace embedding weights (`paraphrase-multilingual-MiniLM-L12-v2`).

---

## Quick Installation

### 1. Clone the Repository
```bash
git clone https://github.com/unimib-datAI/rag-benchmark.git
cd SuperBenchmark
```

### 2. Create and Activate the Virtual Environment
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

### 3. Install the Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the Environment Variables (optional, DAVE only)
- On Windows: `copy .env.example .env`
- On Linux/macOS: `cp .env.example .env`

Edit `.env` with the Keycloak credentials and the DAVE URL if you intend to test that engine.

### 5. Download the Benchmarks
The raw datasets are not included in the repository. In the **Catalogue** tab of the dashboard, tick the desired benchmarks in the table and the **Download selected** button downloads from the official sources (GitHub, Hugging Face, Dropbox) only the files used by the extraction scripts of the chosen benchmarks (~1 GB for all of them, including the OHRBench OCR corpus) and saves them in `Benchmarks_download.zip` in the project folder. Alternatively, from the terminal (with no arguments it downloads all of them; otherwise pass the benchmark names, e.g. `SQuAD-it CRAG`):

```bash
python webpage/benchmark_download.py
```

Extract the zip in the project folder (this creates `Benchmarks/` with the expected structure), then generate the JSONL files:

```bash
python Run/extract_legalbenchmark.py
python Run/extract_multihop.py
python Run/extract_crag.py
python Run/extract_mtrag.py
python Run/extract_ohrbench.py
python Run/extract_squad.py
```

---

## Usage Guide

### Option A: Web Dashboard (Streamlit)

The graphical dashboard offers the highest level of control and visual analysis:

```bash
streamlit run webpage/app.py
```

The dashboard starts in the browser (by default at `http://localhost:8501`) and has five tabs (Catalogue, Evaluation, Explore, Setup, API Guide); when opened it lands on the Evaluation tab:

#### 1. "Catalogue" Tab
- Table of **all supported benchmarks** with description, status (downloaded or not) and number of queries. The search bar filters by **name or description** (not by content).
- Tick the benchmarks to download: **Download selected** downloads the raw data of the chosen benchmarks only and automatically generates their query files.
- **Clear downloaded files** (expander at the bottom of the tab) starts over from scratch with the local dataset files: it deletes the downloaded benchmarks (`Benchmarks/`) and the prepared query files and corpora (`Run/`). It does not touch the `Run/extract_*.py` scripts or anything that has been indexed. It requires confirmation and does not start if a download, extraction or upload is in progress.

#### 2. "Evaluation" Tab
- **Sidebar Configuration:**
  - RAG engine choice (*Local RAG* or *DAVE*).
  - Base URL setting.
  - LLM Judge configuration (Ollama/OpenAI URL and optional API Key).
  - Targeted selection of the benchmarks and RAGAS metrics to compute.
  - **Parallel requests (Workers):** slider from 1 to 10 to noticeably speed up the computation of the RAGAS metrics.
  - **Query selection mode:**
    - `Random`: random sampling.
    - `Fixed`: deterministic selection of the first $N$ samples.
    - `Repeat Previous`: repeats the exact same queries of the previous test for rigorous A/B comparisons.
  - **Evaluation name:** required field to track the run.
- **Live Monitoring:** progress bar and live throughput metrics (**Tokens Per Second - TPS**, total tokens, input/output).
- **Reports & Results:**
  - Summary cards of the average metrics with color coding.
  - Summary bar of the verdicts and distribution of the error patterns.
  - Expandable per-query detail with verdict, diagnostic reasoning, RAG prompt/answer, gold answer and retrieved context.
  - Overall advanced heuristic diagnostics.

#### 3. "Explore" Tab
- Browse the history of all runs saved in `valutazioni/`.
- **Multi-Run Comparison:** select two or more evaluations to view **comparative Radar Charts** and percentage-change tables ($\Delta$).
- Instant recognition of the best-performing run through a merit badge.
- Delete or reload archived evaluations with one click.

#### 4. "Setup" Tab
- Connect to DAVE and index the corpus, or upload the corpus to a generic RAG (`POST /ingest`).
- Only the benchmarks **downloaded** from the Catalogue tab can be indexed: the others do not appear in the upload tables.
- Batch settings (documents and MB per request, pause, maximum number of documents) and a panel with progress and job log.
- **Clear index** (an expander in each of the two sections) deletes only the records of what has been indexed: the generic RAG one in one section, the DAVE one (collections, uploaded files, jobs) in the other. It does not touch the downloaded files and sends nothing to remote systems: DAVE collections already created stay there, unused, and a new upload creates new ones.

#### 5. "API Guide" Tab
- Describes the `/ask` and `/ingest` routes that a third-party RAG must expose.

---

### Option B: Terminal CLI

SuperRagBench includes a complete CLI (`webpage/cli.py`) fully aligned with all the dashboard features, ideal for automation, batch scripts and headless runs.

#### 1. Interactive Mode (Guided Menu)
```bash
python webpage/cli.py
```
Shows an interactive terminal menu that proposes the default values from `config.json` and lets you customize every parameter before starting.

#### 2. Batch / Headless Mode (CI/CD)
All parameters can be passed directly as command-line arguments:

```bash
# Fully automated run with output to file
python webpage/cli.py --name "exp_llama3_run1" --engine "Local" --sample 5 --benchmarks SQUAD,LEGALBENCH --workers 4

# Run on DAVE skipping the preliminary ingestion
python webpage/cli.py --name "test_dave_batch" --engine "DAVE" --sample 10 --no-ingest

# Using an OpenAI/Cloud-compatible model
python webpage/cli.py --name "gpt4o_judge" --judge-url "https://api.openai.com/v1" --model "gpt-4o-mini" --api-key "sk-..."

# Quick look at the history of saved evaluations
python webpage/cli.py --list
```

#### Supported CLI Arguments

| Parameter | Description | Default / Values |
| :--- | :--- | :--- |
| `--name` | Identifying name of the evaluation | `cli_run_YYYYMMDD_HHMMSS` |
| `--engine` | Target RAG engine | `'Local'` or `'DAVE'` |
| `--endpoint` | Base URL of the RAG server | Value from `config.json` |
| `--judge-url` | LLM Judge endpoint | `http://localhost:11434` |
| `--model` | LLM Judge model | `gemma4:31b-cloud` |
| `--api-key` | API Key for cloud / OpenAI endpoints | `""` |
| `--benchmarks` | Comma-separated list of benchmarks or `ALL` | `ALL` (`LEGALBENCH,MULTIHOP,CRAG,MTRAG,OHRBENCH,SQUAD`) |
| `--sample` | Number of queries per benchmark | `3` |
| `--mode` | Sampling mode (`Random`, `Fixed`, `Repeat`) | `Random` |
| `--workers` | Number of parallel RAGAS workers (1-10) | `2` |
| `--no-ingest` | Skips the preliminary document ingestion phase | Boolean flag |
| `--list` | Prints the list of saved evaluations and exits | Boolean flag |
| `--interactive`| Forces the guided interactive menu to open | Boolean flag |

---

## Configuration

### `config.json` File
Stores the default settings shared by the CLI and the Web Dashboard:

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

### `.env` File (DAVE Credentials)
Contains the credentials to interact with the DAVE instance:
```env
DAVE_BASE_URL=https://vm.chronos.disco.unimib.it
DAVE_USERNAME=your_username
DAVE_PASSWORD=your_password
DAVE_CLIENT_SECRET=your_secret
```

---

## Connecting Your Own RAG

To evaluate your own system with the **Local RAG** engine, the server must expose **two HTTP endpoints** under the same Base URL (the *Base URL RAG* field of the dashboard, `--endpoint` from the CLI, e.g. `http://127.0.0.1:8000`):

| Endpoint | Purpose |
| :--- | :--- |
| `POST /ingest` | Check which corpus documents are missing and load them into the RAG index |
| `POST /ask` | Answer a question, returning the answer and the retrieved documents |

### Evaluation Flow

For each selected benchmark, before asking the questions:

1. SuperRagBench reads the benchmark's **full corpus** (the same one loaded on DAVE by `ingest_dave.py`), not just the gold documents:

   | Benchmark | Indexed corpus |
   | :--- | :--- |
   | MultiHop-RAG | `Benchmarks/MultiHop-RAG/dataset/corpus.json` |
   | MTRAG | `Benchmarks/mt-rag-benchmark/corpora/document_level/*.jsonl` |
   | LegalBench-RAG | `Benchmarks/legalbenchrag/data/corpus/` |
   | SQuAD-it | `Run/squad_corpus.json` |
   | OHRBench | `Benchmarks/OHR-Bench/data/retrieval_base/semantic_noise_MinerU_moderate/` (one PDF page per document, text with OCR noise) |
   | CRAG | Web pages provided by the benchmark for each question (`context` field of `Run/Tutto_CRAG_DAVE.jsonl`), as **raw HTML**: extracting the information from the web code is the RAG's job |

   Documents longer than 500,000 characters are split into parts (`<id>_part2`, `<id>_part3`, …), the same limit used for DAVE.

   If the corpus has not been downloaded, the evaluation stops with an error (see [Download the Benchmarks](#5-download-the-benchmarks)).
2. It asks `/ingest` which documents are missing from the index (`check_only`).
3. If nothing is missing it moves straight to the evaluation; otherwise it sends **only the missing documents**, in batches (max 500 documents or 20 MB per request).
4. It makes one `/ask` call for each question.

From the CLI the ingestion can be skipped with `--no-ingest`.

### `POST /ingest` — checking which documents are present

Request:
```json
{
  "benchmark_name": "MTRAG",
  "check_only": true,
  "expected_count": 252767,
  "ids": ["827849752_115-357", "ibmcld_00001", "..."]
}
```
Expected response: the received `id`s that are **not** yet in the index of that `benchmark_name`.
```json
{ "missing": ["ibmcld_00001"] }
```
With `"missing": []` (or `{"complete": true}`) the benchmark's ingestion is skipped. A server that does not handle `check_only` receives the entire corpus at every evaluation.

### `POST /ingest` — uploading documents

Request (one per batch):
```json
{
  "benchmark_name": "MTRAG",
  "documents": [
    { "id": "827849752_115-357", "text": "Egocentrism is the inability to ..." }
  ]
}
```
The server must respond `200` and **add** the documents to the existing index of that `benchmark_name`, ignoring `id`s that are already present. Several requests arrive for the same benchmark: an `/ingest` that recreates the index on every call would keep only the last batch. A response other than `200` (or a network error) during upload **stops the evaluation** with an error, to avoid evaluating the RAG on an incomplete index: at the next evaluation the upload resumes from the missing documents.

### `POST /ask` — answering a question

Request:
```json
{
  "question": "Who is ...?",
  "top_k": 10,
  "benchmark_name": "MultiHop-RAG",
  "search_query": "Who is ...?",
  "documents": ["text of document 1", "text of document 2"]
}
```
- `benchmark_name`: index to search in (the same one used in `/ingest`).
- `question`: question to pass to the generator. For MTRAG it also contains the conversation history (`[HISTORY] ... [NEW QUESTION] ...`).
- `search_query`: text to use for retrieval (for MTRAG it includes the previous turns).
- `documents` (optional): if present, the RAG must **not** do retrieval and must answer using only these texts. It is sent to regenerate the answer on only the documents remaining after the *Similarity Threshold* filter.

Expected response:
```json
{
  "answer": "Generated answer",
  "contexts": ["text of the first retrieved document", "..."]
}
```
`contexts` can contain strings or `{"text": "..."}` objects: these are the documents RAGAS uses for the retrieval and faithfulness metrics. On error the server responds with a status other than `200` and `{"error": "description"}`. A response other than `200`, non-JSON, or with a missing/empty `answer` is never evaluated as an answer: the query is marked `RAG_ERROR` and excluded from RAGAS and the averages.

---

## DAVE Integration

If you use **DAVE** as your document RAG engine:

1. **Connection and Authentication Test:**
   ```bash
   python dave_client.py
   ```
   Verifies Keycloak authentication and syncs the collection mapping in `.dave_collections.json`.

2. **Bulk Corpus Indexing:**
   ```bash
   python ingest_dave.py
   ```
   Uploads the benchmark corpora to DAVE, one batch at a time: the next batch starts only when DAVE has finished indexing the previous one, so the progress bar counts documents that are actually indexed and shows elapsed time, estimated time and speed. Files already sent are tracked in the local cache `.dave_docs_cache.json` to avoid unnecessary re-uploads; those DAVE fails to index are resent (up to 3 passes). Options: `--benchmarks` (default: all), `--batch-docs`, `--batch-mb`, `--sleep`, `--max-docs`, `--verify` (compares cache and DAVE without uploading anything), `--recreate` (creates new collections in place of the existing ones; the old ones remain on DAVE).

---

## Creating Your Own Dataset (Practical Guide)

A custom benchmark lets you evaluate a RAG on your own domain (internal manuals, regulations, technical documentation) instead of the bundled datasets. It consists of two separate artifacts:

| Artifact | Role | Format |
| :--- | :--- | :--- |
| **Corpus** | The document collection the RAG indexes and retrieves from | One `(id, text)` pair per document, read by a function in `corpus_sources.py` |
| **Question set** | Queries with reference answers (*gold answers*) and the supporting passages | A JSONL file, one query per line |

At evaluation time the RAG only sees the questions and retrieves from the corpus. The gold answers and supporting passages are used exclusively by RAGAS to score the output: they are never sent to the RAG.

> **Both parts are required.** `questions.jsonl` alone is not enough: it is never indexed. The RAG retrieves only from the files in `documents/`, while the `context` field of each question is used exclusively as a reference for scoring and is never sent to the RAG or added to the index. If `documents/` is missing or empty, the evaluation stops with a "corpus not found" error.

### Folder layout

Create one folder per dataset inside `Custom/`. Nothing needs to be registered in the code: every folder that contains a `questions.jsonl` is discovered automatically at startup and appears in the Evaluation tab of the dashboard and in the CLI (`--benchmarks`, using the folder name in upper case).

```text
Custom/
└── Company Policies/              # dataset name shown in the UI
    ├── questions.jsonl            # one query per line
    └── documents/                 # the corpus: every .txt / .md file (subfolders included) is one document
        ├── vacation.txt
        ├── remote_work.txt
        └── expenses.txt
```

The document id is the file path relative to `documents/`, so it stays stable as long as the file is not renamed.

### `questions.jsonl`

One JSON object per line:

```json
{"id": "pol_001", "source": "Company Policies", "query_type": "Simple_Extraction", "question": "How many vacation days is an employee entitled to?", "context": "[Doc 1] Every employee is entitled to 26 days of vacation per year.", "gold_answer": "26 days per year"}
{"id": "pol_002", "source": "Company Policies", "query_type": "Multi_Hop", "question": "How far in advance must I request vacation, and how many remote work days can I take?", "context": "[Doc 1] Requests must be submitted 15 days in advance. [Doc 2] Remote work is allowed up to 2 days a week.", "gold_answer": "Vacation must be requested 15 days in advance; remote work is allowed up to 2 days a week."}
{"id": "pol_003", "source": "Company Policies", "query_type": "Unanswerable", "question": "What is the daily meal voucher?", "context": "", "gold_answer": "The policies do not contain this information."}
```

Fields:

- `id`: unique identifier of the query.
- `source`: dataset name, used as the index name when the corpus is uploaded to the RAG (`benchmark_name` in `/ingest` and `/ask`). Use the same value on every line; if it is missing, the folder name is used.
- `query_type`: free-form label used to group results by question category (e.g. single-passage vs. multi-document).
- `question`: the query sent to the RAG.
- `context`: the ground-truth supporting passage(s). It is **not** indexed: the passage must also exist inside a file in `documents/`, otherwise the RAG has no way to find it. It is used by RAGAS for the retrieval metrics (Context Recall in particular). Multiple passages are written as `[Doc 1] … [Doc 2] …`. Leave it empty for unanswerable questions.
- `gold_answer`: the reference answer that Answer Correctness is computed against.

### Running it

Start the dashboard (`streamlit run webpage/app.py`) or the CLI (`python webpage/cli.py --benchmarks "COMPANY POLICIES"`). On a Local RAG engine the corpus is uploaded to `/ingest` automatically, exactly as for the bundled benchmarks, and only the missing documents are sent. Custom datasets are also listed in the Setup tab, where you can upload the corpus manually. They never appear in the Catalogue tab, which only handles downloadable benchmarks: there is nothing to download.

### Generating the `.jsonl` file from an Excel/CSV sheet

If your questions are in a CSV with the columns `question`, `answer`, `passage`, a few lines are enough:

```python
import csv, json

with open("questions.csv", encoding="utf-8") as f, open("Custom/Company Policies/questions.jsonl", "w", encoding="utf-8") as out:
    for i, r in enumerate(csv.DictReader(f), 1):
        out.write(json.dumps({
            "id": f"pol_{i:03d}",
            "source": "Company Policies",
            "query_type": "Simple_Extraction",
            "question": r["question"],
            "context": f"[Doc 1] {r['passage']}",
            "gold_answer": r["answer"],
        }, ensure_ascii=False) + "\n")
```

### Tips for a good dataset

- **Concise, self-contained gold answers**: Answer Correctness compares claims between the generated and the reference answer, so a short factual statement is scored more reliably than a long paragraph.
- **Cover different query types**: single-passage lookups, multi-document questions and unanswerable questions. The latter are the only way to measure whether the RAG hallucinates or correctly refuses.
- **Keep the corpus larger than the supporting passages**: if only the gold documents are indexed, retrieval is trivially easy and Context Precision is inflated. Include distractor documents, ideally similar in topic to the relevant ones.
- **Start with a small sample**: 10-20 questions are enough to validate the pipeline end to end before scaling up.
- **Use stable document ids**: ids are used by `/ingest` to detect which documents are already indexed, so renaming files forces a full re-upload.

*(For details on very large corpora, per-question documents and automatic download, see [aggiunta_dataset.md](aggiunta_dataset.md)).*

---

## Adding a New Benchmark

Adding a new dataset to the suite takes 3 steps:

1. **Extraction and Normalization (`Run/extract_name.py`):**
   Create a script that turns the raw dataset into a JSONL file (`Run/Tutto_NAME.jsonl`), where each line is a valid JSON record:
   ```json
   {"id": "query_001", "source": "MyBenchmark", "query_type": "Simple_Extraction", "question": "Question text...", "context": "[Doc 1] Context...", "gold_answer": "Reference answer..."}
   ```
2. **Registration in the Dashboard:**
   In `webpage/app.py`, add the entry to `BENCHMARK_MAPPING`:
   ```python
   BENCHMARK_MAPPING["MyBenchmark"] = "Tutto_NAME.jsonl"
   ```
3. **Registration in the CLI:**
   In `webpage/cli.py`, insert the corresponding key in `BENCHMARK_MAP`.

*(For the detailed step-by-step guide and indexing on DAVE, see [aggiunta_dataset.md](aggiunta_dataset.md)).*

---

## Project Structure

```text
SuperRagBench/
├── README.md                      # General project documentation
├── requirements.txt               # Python dependencies (Streamlit, RAGAS, LangChain, etc.)
├── config.json                    # Default configuration for UI and CLI
├── .env.example                   # Template of the DAVE/Keycloak environment variables
├── dave_client.py                 # API client for the DAVE engine (tRPC, sessions and streaming)
├── ingest_dave.py                 # Script to upload and index the corpus on DAVE
├── corpus_sources.py              # Corpus of each benchmark, shared by the local RAG and DAVE
├── .dave_collections.json         # Benchmark -> DAVE collection UUID mapping
├── .dave_docs_cache.json          # Local cache tracking already indexed documents
├── .dave_upload_jobs.json         # DAVE upload jobs to verify (failed files to re-upload)
│
├── webpage/                       # Application core (Dashboard and CLI)
│   ├── app.py                     # Streamlit web interface (Evaluation and Explore)
│   ├── cli.py                     # Command-line interface (Interactive and Batch)
│   ├── evaluator.py               # Evaluation engine, RAG queries and RAGAS computation
│   ├── taxonomy.py                # Error taxonomy: verdicts, patterns, causes and aggregate diagnostics
│   ├── persistence.py             # Saving, loading and deleting runs
│   ├── benchmark_download.py      # Download of raw benchmarks from the official sources (zip)
│   ├── dashboard.py               # HTML/CSS graphical components, metric cards and error legends
│   └── style.css                  # Custom dark-mode stylesheet for Streamlit
│
├── Custom/                        # Your own datasets, picked up automatically (see "Creating Your Own Dataset")
│   └── <name>/questions.jsonl + documents/
│
├── Run/                           # Normalized datasets and extraction scripts
│   ├── Tutto_LEGALBENCH.jsonl     # LegalBench-RAG in JSONL format
│   ├── Tutto_MULTIHOP.jsonl       # MultiHop-RAG in JSONL format
│   ├── Tutto_CRAG_DAVE.jsonl      # CRAG (Meta) in JSONL format
│   ├── Tutto_MTRAG_DAVE.jsonl     # Conversational MTRAG in JSONL format
│   ├── Tutto_OHRBench_DAVE.jsonl  # OHRBench (OCR noise) in JSONL format
│   ├── Tutto_SQUAD.jsonl          # SQuAD-it in JSONL format
│   ├── squad_corpus.json          # SQuAD-it corpus (unique paragraphs), generated by extract_squad.py
│   └── extract_*.py               # Scripts converting raw formats to JSONL
│
├── test/                          # Automated tests (taxonomy, RAG errors, DAVE pipeline)
│
├── Benchmarks/                    # Original raw datasets (not versioned, see "Download the Benchmarks")
├── valutazioni/                   # Archived JSON reports with the results of each session
├── Paper/                         # Reference scientific publications on the datasets and on RAGAS
└── aggiunta_dataset.md            # Detailed guide to integrate new datasets
```
