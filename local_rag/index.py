"""Index benchmark corpora once, before starting the server: python local_rag/index.py SQuAD-it MultiHop-RAG"""
import argparse
import time

import config
from corpus import BENCHMARK_FILES, load_documents
from retrieval import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the BM25 and dense indexes of the given benchmarks.")
    parser.add_argument("benchmarks", nargs="+", choices=list(BENCHMARK_FILES))
    args = parser.parse_args()
    config.require("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY")

    for benchmark in args.benchmarks:
        start = time.time()
        documents = load_documents(benchmark)
        print(f"{benchmark}: {len(documents)} documents, embedding model {config.EMBEDDING_MODEL}")
        info = build_index(benchmark, documents, config.EMBEDDING_MODEL)
        print(f"{benchmark}: {info['chunks']} chunks indexed in {time.time() - start:.0f}s")
