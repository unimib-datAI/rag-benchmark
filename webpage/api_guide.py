"""API Guide tab: what a third-party RAG must expose to be evaluated by this tool."""
import time

import requests
import streamlit as st

# benchmark_name values sent to the RAG, with what each one contains
BENCHMARKS = [
    ("LegalBench-RAG", "Questions on legal contracts (contractnli, cuad, maud, privacy_qa).", "long legal documents"),
    ("MultiHop-RAG", "Questions that need several news articles to be answered.", "news articles"),
    ("CRAG", "Questions on web pages: exact extraction, multi-hop, comparison, aggregation, counterfactual checks.", "HTML pages"),
    ("MTRAG", "Multi-turn conversations: the question carries the previous turns.", "document passages"),
    ("OHRBench", "Questions on OCR-noisy scanned documents. Documents are sent with the question.", "OCR text"),
    ("SQuAD-it", "Italian reading-comprehension questions on Wikipedia.", "Wikipedia paragraphs"),
]

ASK_REQUEST_EXAMPLE = """{
  "question": "Who is the individual associated with the cryptocurrency exchange FTX?",
  "search_query": "Who is the individual associated with the cryptocurrency exchange FTX?",
  "benchmark_name": "MultiHop-RAG",
  "top_k": 10
}"""

ASK_RESPONSE_EXAMPLE = """{
  "answer": "Sam Bankman-Fried",
  "contexts": [
    "FTX founder Sam Bankman-Fried was arrested in the Bahamas ...",
    "The collapse of the exchange FTX ..."
  ]
}"""

INGEST_REQUEST_EXAMPLE = """{
  "benchmark_name": "MultiHop-RAG",
  "documents": [
    {"id": "doc_0", "text": "FTX founder Sam Bankman-Fried was arrested ..."},
    {"id": "doc_1", "text": "The collapse of the exchange FTX ..."}
  ],
  "batch": {"index": 0, "count": 3}
}"""

SERVER_EXAMPLE = '''"""Minimal server compatible with the benchmark tool: pip install fastapi uvicorn"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
INDEX: dict[str, list[str]] = {}          # benchmark_name -> documents


class AskRequest(BaseModel):
    question: str
    benchmark_name: str
    top_k: int = 10
    search_query: str | None = None       # use this (not question) to retrieve
    documents: list[str] | None = None    # OHRBench only: answer from these, ignore your index


class Doc(BaseModel):
    id: str
    text: str


class Batch(BaseModel):
    index: int = 0
    count: int = 1


class IngestRequest(BaseModel):
    benchmark_name: str
    documents: list[Doc]
    batch: Batch = Batch()


@app.post("/ingest")
def ingest(req: IngestRequest):
    if req.batch.index == 0:                  # first request: replace what you hold for this benchmark
        INDEX[req.benchmark_name] = []
    INDEX[req.benchmark_name] += [d.text for d in req.documents]   # build your index here
    return {"message": f"{len(req.documents)} documents indexed"}


@app.post("/ask")
def ask(req: AskRequest):
    contexts = req.documents or my_retrieve(req.search_query or req.question, req.benchmark_name, req.top_k)
    return {"answer": my_generate(req.question, contexts), "contexts": contexts}

# run: uvicorn server:app --port 8000   ->   Base URL RAG = http://127.0.0.1:8000
'''


def _context_text(item):
    """Same rule as the evaluator: a context is a string, or an object with 'text' or 'chunks'."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "text" in item:
            return str(item["text"])
        if "chunks" in item and isinstance(item["chunks"], list):
            return " ".join(str(c.get("text", "")) for c in item["chunks"] if isinstance(c, dict))
    return None


def check_endpoint(base_url: str, benchmark: str = "SQuAD-it"):
    """Send one probe question to <base_url>/ask and report whether the reply follows the contract.

    The content of the answer is not judged: a RAG with nothing indexed yet can only say it does not know.
    What matters is that the route answers 200 with a JSON object holding 'answer' and 'contexts'."""
    url = base_url.strip().rstrip("/") + "/ask"
    payload = {"question": "Is this endpoint working?", "search_query": "Is this endpoint working?",
               "benchmark_name": benchmark, "top_k": 10}
    started = time.time()
    try:
        r = requests.post(url, json=payload, timeout=120)
    except Exception as e:
        return [("error", f"Could not reach {url}: {e}")]
    checks = [("info", f"POST {url} answered HTTP {r.status_code} in {time.time() - started:.1f}s")]
    if r.status_code != 200:
        checks.append(("error", "Status must be 200. Any other status is recorded as the answer "
                                "'Error <status>: <error field>' and that text is what gets scored."))
    try:
        data = r.json()
    except ValueError:
        return checks + [("error", "The body is not JSON.")]
    if not isinstance(data, dict):
        return checks + [("error", "The JSON body must be an object with 'answer' and 'contexts'.")]
    if r.status_code != 200:
        return checks + [("info", f"error field: {data.get('error', '(missing)')}")]

    answer = data.get("answer")
    if not isinstance(answer, str):
        checks.append(("error", "'answer' must be a string."))
    elif not answer.strip():
        checks.append(("warning", "'answer' is empty. Fine for a test, but during an evaluation an empty answer "
                                  "counts as a RAG error."))
    else:
        checks.append(("ok", f"answer: {answer[:100]!r}" + ("..." if len(answer) > 100 else "")))

    contexts = data.get("contexts")
    if not isinstance(contexts, list):
        checks.append(("error", "'contexts' must be a list (the retrieved documents, empty if there are none)."))
    else:
        texts = [_context_text(c) for c in contexts]
        if None in texts:
            checks.append(("error", "Every item of 'contexts' must be a string, or an object with 'text' "
                                    "(or 'chunks': [{'text': ...}])."))
        else:
            checks.append(("ok", f"contexts: {len(contexts)} documents"))
        if len(contexts) > payload["top_k"]:
            checks.append(("warning", f"{len(contexts)} contexts returned, top_k was {payload['top_k']}."))
    return checks


def show_checks(checks):
    """One box for the whole endpoint test: green, yellow or red by the worst finding, details inside."""
    levels = [level for level, _ in checks]
    worst = "error" if "error" in levels else "warning" if "warning" in levels else "ok"
    title = {"ok": "Endpoint OK", "warning": "Endpoint reachable, with warnings", "error": "Endpoint problem"}[worst]
    icon = {"ok": "✅", "warning": "⚠️", "error": "❌", "info": "•"}
    body = "\n".join(f"- {icon[level]} {message[:400]}" for level, message in checks)
    {"ok": st.success, "warning": st.warning, "error": st.error}[worst](f"**{title}**\n\n{body}")


def render(default_url: str = "http://127.0.0.1:8000"):
    st.markdown("## How to connect your RAG to this tool")
    st.markdown(
        "The tool sends benchmark questions to your RAG, collects the **answer** and the **retrieved documents**, "
        "and scores them with an LLM judge (RAGAS): *faithfulness*, *answer relevancy*, *context precision*, "
        "*context recall* and *answer correctness*. Your system only has to expose two HTTP routes, "
        "`POST /ask` (required) and `POST /ingest` (optional). Both take and return JSON."
    )
    st.info(
        "**How to try your RAG.** Start your server, open **Setup > Generic RAG (HTTP)**, enter its Base URL, press "
        "*Test endpoint* and (if it has `/ingest`) *Start corpus upload*. Then choose **Custom RAG (HTTP)** in the "
        "sidebar, pick the benchmarks and press **Start Evaluation**. "
        "From the command line: `python webpage/cli.py --engine Local --endpoint http://host:port`."
    )
    st.warning(
        "**Running in Docker?** `localhost` inside the container is the container itself. A RAG running on the "
        "host machine is reachable at `http://host.docker.internal:PORT` (already enabled in docker-compose.yml), "
        "and it must listen on `0.0.0.0`, not only on `127.0.0.1`."
    )

    st.markdown("### 1. Base URL")
    st.markdown(
        "You give the tool a *base URL* (**Setup > Generic RAG (HTTP)** or the sidebar when the engine is *Custom RAG (HTTP)*; `--endpoint` in the CLI). "
        "The tool appends `/ask` and `/ingest` to it after removing a trailing `/`, so any path prefix works: "
        "`http://host:8000` → `http://host:8000/ask`; `https://api.example.com/rag/v2` → `https://api.example.com/rag/v2/ask`. "
        "No authentication header is sent: put the service on a trusted network or behind a key in the URL."
    )

    st.markdown("### 2. `POST /ask` (required)")
    st.markdown("Called once per benchmark question, **one request at a time, in order** (timeout: 120 s).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Request body**")
        st.table({
            "Field": ["question", "search_query", "benchmark_name", "top_k", "documents"],
            "Type": ["string", "string", "string", "integer", "list of strings (optional)"],
            "Meaning": [
                "The question to answer. In MTRAG it also carries the previous turns (see below).",
                "The text to use for retrieval. Equal to `question` except in MTRAG, where it is the "
                "conversation so far plus the new question, without the `User:`/`Assistant:` labels.",
                "Which benchmark the question belongs to (table in section 4). Use it to select the right index.",
                "How many documents to return at most. Always 10.",
                "Only sent for OHRBench: the documents the question refers to. Answer from these and ignore your index.",
            ],
        })
        st.code(ASK_REQUEST_EXAMPLE, language="json")
    with c2:
        st.markdown("**Response body (HTTP 200)**")
        st.table({
            "Field": ["answer", "contexts"],
            "Type": ["string", "list"],
            "Meaning": [
                "The final answer of your RAG, generated from `contexts`.",
                "The documents retrieved for this question, **best first**. Each item is a string, or an object "
                "with `text`, or an object with `chunks: [{\"text\": ...}]`. These are the documents the judge "
                "uses to compute context precision/recall and faithfulness.",
            ],
        })
        st.code(ASK_RESPONSE_EXAMPLE, language="json")
    st.markdown(
        "- **Errors:** any status other than 200 is recorded as the answer `Error <status>: <error field>` "
        "(put a readable message in an `error` field), with no documents, and that text is what gets scored. "
        "The run does not stop. A timeout or connection error is recorded the same way, as `Exception: ...`.\n"
        "- **Empty `contexts`:** counted as zero retrieved documents, so context precision and recall are 0.\n"
        "- **The answer must be generated from the returned `contexts`.** The judge only sees those documents: "
        "facts that come from anywhere else are scored as hallucinations.\n"
        "- **Order matters:** context precision rewards relevant documents ranked first."
    )

    st.markdown("**MTRAG (multi-turn).** The conversation is replayed turn by turn, so the order of requests matters. "
                "From the second turn on, `question` has this shape (the tool keeps the history, your RAG can be stateless):")
    st.code("[HISTORY]\nUser: where do the arizona cardinals play?\nAssistant: At State Farm Stadium.\n\n"
            "[NEW QUESTION]\nand when is the next game?", language="text")

    st.markdown("**OHRBench (documents in the request).** `documents` contains the texts to answer from. "
                "Do not query your index for this benchmark: just generate the answer from these documents and "
                "return them as `contexts`.")

    st.markdown("### 3. `POST /ingest` (optional)")
    st.markdown(
        "The corpus is uploaded from **Setup > Generic RAG (HTTP)** (button *Start corpus upload*), not during the "
        "evaluation. The documents of each selected benchmark are sent in several requests (timeout: 300 s each; "
        "size set in *Batch settings*). Use the route to build your index. If your RAG is already indexed, or has no "
        "such route, skip the upload: the evaluation only calls `/ask`."
    )
    st.code(INGEST_REQUEST_EXAMPLE, language="json")
    st.markdown(
        "- **`batch`**: `index` is the position of the request (from 0) and `count` how many there are for this "
        "benchmark. **`index` 0 means: delete what you hold for this `benchmark_name` and start again**; the next "
        "requests append. A server that ignores `batch` works when the whole corpus fits in one request "
        "(set a large *Max docs per request*).\n"
        "- Answer HTTP 200 (optionally `{\"message\": \"...\"}`). Any other status is retried 3 times, then the batch "
        "is reported as failed in the job log.\n"
        "- Document ids are the corpus file names without `.txt`; they stay the same between uploads.\n"
        "- The tool remembers what it uploaded to each URL: with *Max documents per benchmark* set, the evaluation "
        "asks only the questions whose documents were uploaded."
    )

    st.markdown("### 4. Benchmarks (`benchmark_name`)")
    st.table({
        "benchmark_name": [b[0] for b in BENCHMARKS],
        "Questions": [b[1] for b in BENCHMARKS],
        "Documents are": [b[2] for b in BENCHMARKS],
    })
    st.markdown("Questions come from the files in `Run/` (built from the official datasets in the **Setup** tab). "
                "The gold answers are never sent to your RAG.")

    st.markdown("### 5. Minimal server")
    st.code(SERVER_EXAMPLE, language="python")
    st.markdown("Test it from a terminal:")
    st.code("curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \\\n"
            "  -d '{\"question\": \"What is the capital of France?\", \"benchmark_name\": \"SQuAD-it\", \"top_k\": 10}'",
            language="bash")

    st.markdown("### 6. Check your endpoint")
    st.caption("Sends one probe question to `<Base URL>/ask` and checks the shape of the reply against the contract above. "
               "The answer itself is not judged, so it works even if nothing is indexed yet.")
    url = st.text_input("Base URL", value=default_url, key="guide_check_url")
    if st.button("Check endpoint", key="guide_check_go", disabled=not url.strip()):
        with st.spinner("Waiting for your RAG..."):
            show_checks(check_endpoint(url))
