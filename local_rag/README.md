# Local RAG

Server RAG per l'opzione **Local RAG** del tool, con tre tecniche di retrieval sugli stessi indici. La tecnica si sceglie dall'URL da mettere in **Base URL RAG**:

| Base URL RAG | Tecnica |
|---|---|
| `http://127.0.0.1:8000/bm25` | sparse: BM25 con stemming e stopword |
| `http://127.0.0.1:8000/dense` | densa: similarità coseno tra embedding |
| `http://127.0.0.1:8000/hybrid` | fusione delle due classifiche con Reciprocal Rank Fusion |

Le tre tecniche usano lo stesso LLM e lo stesso prompt: le differenze nei punteggi dipendono solo dal retrieval.

## Installazione

Dalla root del progetto, con il venv attivo:

```bash
pip install -r local_rag/requirements.txt
cp local_rag/.env.example local_rag/.env
```

Nel `.env` inserisci endpoint, chiave e nome del deployment di due modelli: l'LLM che genera le risposte e il modello di embedding dell'indice denso. L'URL deve finire con `/openai/v1/`.

## 1. Indicizzazione (una tantum)

```bash
python local_rag/index.py SQuAD-it MultiHop-RAG MTRAG
```

Per ogni benchmark legge i documenti da `Run/Tutto_*.jsonl` (gli stessi che il tool invierebbe a `/ingest`), li divide in chunk da 200 parole e salva in `local_rag/indices/<benchmark>/` l'indice BM25, gli embedding e `info.json`. Va rifatta solo se cambi dataset o modello di embedding.

| Benchmark | Chunk |
|---|---|
| SQuAD-it | 2192 |
| MultiHop-RAG | 1397 |
| MTRAG | 3315 |
| LegalBench-RAG | 70390 |

OHRBench non va indicizzato: il tool manda i documenti insieme a ogni domanda.

Gli embedding sono calcolati dal deployment indicato in `EMBEDDING_MODEL`, a gruppi di 256 chunk per richiesta. Ogni indice ricorda il modello con cui è stato creato e il server usa quello anche per le domande, quindi se cambi modello devi reindicizzare.

## 2. Server

```bash
python local_rag/server.py
```

All'avvio stampa i benchmark indicizzati; `http://127.0.0.1:8000/status` mostra lo stesso elenco con i dettagli. Se reindicizzi un benchmark mentre il server è acceso, riavvialo.

## API

- `POST /<tecnica>/ask` riceve `{question, benchmark_name, top_k, search_query, documents}` e risponde `{answer, contexts}`. Il retrieval usa `search_query` se c'è (per MTRAG il tool ci mette cronologia e domanda), altrimenti `question`. Se arriva `documents`, il server non cerca e risponde solo su quei testi.
- `POST /<tecnica>/ingest` non indicizza: controlla solo che il benchmark sia pronto.
- Se la tecnica è sconosciuta o il benchmark non è indicizzato, risponde 404 con un messaggio che dice cosa fare.

## Riferimento: SQuAD-it

Su 200 domande, quante volte il paragrafo che contiene la risposta è tra i chunk restituiti (embedding `text-embedding-3-small`):

| Tecnica | Al 1° posto | Nei primi 10 |
|---|---|---|
| bm25 | 66,5% | 92,0% |
| dense | 61,5% | 91,5% |
| hybrid | 74,0% | 94,0% |

Indicizzazione con Foundry: SQuAD-it 20 s, MultiHop-RAG 9 s, MTRAG 24 s.

## File

| File | Contenuto |
|---|---|
| `corpus.py` | documenti dei benchmark e chunking |
| `retrieval.py` | costruzione degli indici e le tre ricerche |
| `generation.py` | risposta dell'LLM |
| `index.py`, `server.py` | i due comandi |
| `config.py` | impostazioni lette dal `.env` |
