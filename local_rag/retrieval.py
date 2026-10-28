"""Sparse (BM25), dense and hybrid search over the chunks of a benchmark."""
import json
from datetime import datetime
from functools import cache
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from openai import OpenAI
from tqdm import tqdm

import config
from corpus import chunk

INDEX_DIR = Path(__file__).resolve().parent / "indices"
MODES = ("bm25", "dense", "hybrid")  # each one is a ranking method of Index
ITALIAN_BENCHMARKS = {"SQuAD-it"}
CANDIDATES = 50
RRF_K = 60
EMBEDDING_BATCH = 256


@cache
def embedding_client():
    # large corpora hit the deployment's rate limit: the client waits and retries
    return OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.EMBEDDING_API_KEY, max_retries=10)


def embed(texts, model, progress=False):
    """Unit-length embeddings, so that the dot product of two of them is their cosine similarity."""
    vectors = []
    for start in tqdm(range(0, len(texts), EMBEDDING_BATCH), disable=not progress):
        response = embedding_client().embeddings.create(model=model, input=texts[start:start + EMBEDDING_BATCH])
        vectors += [item.embedding for item in response.data]
    vectors = np.array(vectors, dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def tokenize(texts, language):
    return bm25s.tokenize(texts, stopwords=language, stemmer=Stemmer.Stemmer(language), show_progress=False)


def build_index(benchmark, documents, model_name):
    """Save chunks, BM25 index and embeddings. info.json is written last: its presence marks a complete index."""
    folder = INDEX_DIR / benchmark
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "info.json").unlink(missing_ok=True)
    language = "italian" if benchmark in ITALIAN_BENCHMARKS else "english"
    chunks = [piece for doc in documents for piece in chunk(doc)]
    (folder / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    bm25 = bm25s.BM25()
    bm25.index(tokenize(chunks, language), show_progress=False)
    bm25.save(folder / "bm25")

    np.save(folder / "embeddings.npy", embed(chunks, model_name, progress=True))

    info = {
        "documents": len(documents),
        "chunks": len(chunks),
        "language": language,
        "embedding_model": model_name,
        "created": datetime.now().isoformat(timespec="minutes"),
    }
    (folder / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


class Index:
    def __init__(self, benchmark):
        folder = INDEX_DIR / benchmark
        self.info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
        self.chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        self.bm25_index = bm25s.BM25.load(folder / "bm25")
        self.embeddings = np.load(folder / "embeddings.npy")

    def search(self, mode, query, k):
        return [self.chunks[i] for i in getattr(self, mode)(query, k)]

    def bm25(self, query, k):
        k = min(k, len(self.chunks))
        ids, scores = self.bm25_index.retrieve(tokenize([query], self.info["language"]), k=k, show_progress=False)
        return [int(i) for i, score in zip(ids[0], scores[0]) if score > 0]

    def dense(self, query, k):
        scores = self.embeddings @ embed([query], self.info["embedding_model"])[0]
        return np.argsort(-scores)[:k].tolist()

    def hybrid(self, query, k):
        """Reciprocal Rank Fusion: a chunk gets 1 / (RRF_K + rank) from each ranking it appears in."""
        scores = {}
        for ranking in (self.bm25(query, CANDIDATES), self.dense(query, CANDIDATES)):
            for rank, i in enumerate(ranking, start=1):
                scores[i] = scores.get(i, 0) + 1 / (RRF_K + rank)
        return sorted(scores, key=scores.get, reverse=True)[:k]
