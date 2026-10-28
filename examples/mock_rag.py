"""A tiny RAG that follows the API Guide contract, to try the "Generic RAG (HTTP)" flow end to end.

No dependencies. It keeps the uploaded documents in memory, retrieves with word overlap and answers
with the best-matching sentence: the scores are meaningless, the point is to see uploads, questions
and the answerable-queries filter working. It stops by itself after --minutes.

  python examples/mock_rag.py                 # listens on 127.0.0.1:8765
  Base URL in the tool: http://127.0.0.1:8765

With Docker nothing has to be published: start it inside the app container and use the container's
own localhost:   docker compose exec -d app python examples/mock_rag.py
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INDEX = {}  # benchmark_name -> [(doc id, text, word set)]
LOCK = threading.Lock()
LOG = None


def words(text):
    return set(re.findall(r"\w{3,}", text.lower()))


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def best_sentence(question, text):
    q = words(question)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) or [text]
    return max(sentences, key=lambda s: len(q & words(s)))[:300]


def ingest(body):
    name, docs, batch = body["benchmark_name"], body["documents"], body.get("batch", {"index": 0, "count": 1})
    with LOCK:
        if batch.get("index", 0) == 0:  # first request: replace what we hold for this benchmark
            INDEX[name] = []
        INDEX[name] += [(d["id"], d["text"], words(d["text"])) for d in docs]
        total = len(INDEX[name])
    log(f"ingest {name}: batch {batch.get('index', 0) + 1}/{batch.get('count', 1)}, +{len(docs)} docs, {total} held")
    return {"message": f"{len(docs)} documents indexed ({total} held for {name})"}


def ask(body):
    question, name, top_k = body["question"], body["benchmark_name"], int(body.get("top_k", 10))
    if body.get("documents") is not None:  # OHRBench: answer from the documents sent with the question
        contexts = body["documents"][:top_k]
    else:
        q = words(body.get("search_query") or question)
        with LOCK:
            docs = INDEX.get(name, [])
            ranked = sorted(docs, key=lambda d: len(q & d[2]), reverse=True)[:top_k]
        contexts = [text for _, text, _ in ranked]
    answer = best_sentence(question, contexts[0]) if contexts else "I don't know."
    log(f"ask {name}: {question[:60]!r} -> {len(contexts)} contexts")
    return {"answer": answer, "contexts": contexts}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with LOCK:
            self._reply(200, {name: len(docs) for name, docs in INDEX.items()})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            route = {"/ask": ask, "/ingest": ingest}.get(self.path.rstrip("/"))
            if route is None:
                return self._reply(404, {"error": f"unknown route {self.path}"})
            self._reply(200, route(body))
        except (KeyError, ValueError) as e:
            self._reply(400, {"error": f"bad request: {e}"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--minutes", type=float, default=60, help="stop by itself after this time")
    parser.add_argument("--log", default="/tmp/mock_rag.log")
    args = parser.parse_args()
    LOG = args.log
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Timer(args.minutes * 60, server.shutdown).start()
    log(f"mock RAG on http://{args.host}:{args.port} (stops in {args.minutes:g} minutes)")
    server.serve_forever()
    log("stopped")
