"""HTTP API for the benchmark tool. The retrieval mode is the first part of the path, e.g. POST /hybrid/ask."""
import json
from functools import cache

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAIError
from pydantic import BaseModel

import config
from generation import answer
from retrieval import INDEX_DIR, MODES, Index

HOST, PORT = "127.0.0.1", 8000

app = FastAPI(title="Local RAG")


class AskRequest(BaseModel):
    question: str
    benchmark_name: str
    top_k: int = 10
    search_query: str | None = None
    documents: list[str] | None = None


class IngestRequest(BaseModel):
    benchmark_name: str


def indexed_benchmarks():
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(INDEX_DIR.glob("*/info.json"))}


@cache
def load_index(benchmark):
    return Index(benchmark)


def request_error(mode, benchmark):
    if mode not in MODES:
        return (f"Unknown retrieval mode '{mode}': the RAG URL must end with one of "
                f"{', '.join(MODES)}, e.g. http://{HOST}:{PORT}/hybrid")
    if benchmark not in indexed_benchmarks():
        return f"Benchmark '{benchmark}' is not indexed: run python local_rag/index.py {benchmark}"
    return None


@app.get("/status")
def status():
    return {"modes": MODES, "indexed": indexed_benchmarks()}


@app.post("/ingest")
@app.post("/{mode}/ingest")
def ingest(request: IngestRequest, mode: str = ""):
    """Indexing is done beforehand with index.py: here we only check that the benchmark is ready."""
    if error := request_error(mode, request.benchmark_name):
        return JSONResponse({"error": error}, status_code=404)
    return {"message": f"{request.benchmark_name} is already indexed"}


@app.post("/ask")
@app.post("/{mode}/ask")
def ask(request: AskRequest, mode: str = ""):
    contexts = request.documents
    if contexts is None and (error := request_error(mode, request.benchmark_name)):
        return JSONResponse({"error": error}, status_code=404)
    try:
        if contexts is None:
            query = request.search_query or request.question
            contexts = load_index(request.benchmark_name).search(mode, query, request.top_k)
        return {"answer": answer(request.question, contexts), "contexts": contexts}
    except OpenAIError as e:
        return JSONResponse({"error": f"Model endpoint call failed: {e}"}, status_code=502)


if __name__ == "__main__":
    config.require("LLM_BASE_URL", "LLM_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_API_KEY")
    indexed = indexed_benchmarks()
    print("Indexed benchmarks:", ", ".join(f"{name} ({info['chunks']} chunks, {info['embedding_model']})"
                                           for name, info in indexed.items())
          or "none, run local_rag/index.py first")
    uvicorn.run(app, host=HOST, port=PORT)
