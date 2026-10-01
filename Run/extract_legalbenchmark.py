import json
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGALBENCH_DIR = os.path.normpath(
	os.path.join(BASE_DIR, "..", "Benchmarks", "legalbenchrag")
)
BENCHMARKS_DIR = os.path.join(LEGALBENCH_DIR, "data", "benchmarks")
CORPUS_DIR = os.path.join(LEGALBENCH_DIR, "data", "corpus")

DEFAULT_OUTPUT = os.path.join(BASE_DIR, "Tutto_LEGALBENCH.jsonl")
MAX_EXAMPLES = None  # Nessun limite, usa tutte le domande disponibili
SAMPLE_SEED = 42


def _load_corpus_file(corpus_cache, corpus_dir, file_path):
	if file_path not in corpus_cache:
		full_path = os.path.join(corpus_dir, file_path)
		with open(full_path, "r", encoding="utf-8-sig") as f:
			corpus_cache[file_path] = f.read()
	return corpus_cache[file_path]


def _resolve_file_path(corpus_dir, file_path):
	if not file_path:
		return None
	candidates = [part.strip() for part in str(file_path).split("||") if part.strip()]
	if not candidates:
		return None
	for candidate in candidates:
		full_path = os.path.join(corpus_dir, candidate)
		if os.path.isfile(full_path):
			return candidate
	return None


def _extract_snippet_text(content, span):
	if not isinstance(span, (list, tuple)) or len(span) != 2:
		return ""
	try:
		start = int(span[0])
		end = int(span[1])
	except (TypeError, ValueError):
		return ""
	if start < 0:
		start = 0
	if end < 0:
		end = 0
	if start > len(content):
		start = len(content)
	if end > len(content):
		end = len(content)
	if end <= start:
		return ""
	return content[start:end]


def _get_snippet_text(snippet, corpus_cache, corpus_dir, resolved_file_path=None):
	answer = snippet.get("answer")
	if isinstance(answer, str):
		answer = answer.strip()
		if answer:
			return answer
	elif isinstance(answer, list):
		parts = [str(part).strip() for part in answer if str(part).strip()]
		if parts:
			return "\n".join(parts)

	file_path = resolved_file_path or snippet.get("file_path", "")
	span = snippet.get("span", None)
	if not file_path or span is None:
		return ""
	content = _load_corpus_file(corpus_cache, corpus_dir, file_path)
	return _extract_snippet_text(content, span)


def converti_legalbench(benchmarks_dir, corpus_dir, output_file, max_examples=MAX_EXAMPLES):
	if not os.path.isdir(benchmarks_dir) or not os.path.isdir(corpus_dir):
		print("ERRORE: cartelle LegalBench-RAG mancanti.")
		print(f"Controlla: {benchmarks_dir}")
		print(f"Controlla: {corpus_dir}")
		print("Scarica il dataset dal link nel README e metti le cartelle in data/.")
		return

	benchmark_files = sorted(
		f for f in os.listdir(benchmarks_dir) if f.lower().endswith(".json")
	)
	if not benchmark_files:
		print(f"ERRORE: nessun file .json trovato in {benchmarks_dir}")
		return

	corpus_cache = {}
	dati_unificati = []

	for benchmark_file in benchmark_files:
		benchmark_name = os.path.splitext(benchmark_file)[0]
		bench_path = os.path.join(benchmarks_dir, benchmark_file)
		with open(bench_path, "r", encoding="utf-8") as f:
			data = json.load(f)

		tests = data.get("tests", [])
		for test in tests:
			query = test.get("query", "")
			snippets = test.get("snippets", [])
			if not query or not snippets:
				continue

			context_parts = []
			answer_parts = []
			seen_paths = set()
			for snippet in snippets:
				file_path = snippet.get("file_path", "")
				resolved_path = _resolve_file_path(corpus_dir, file_path)
				if not resolved_path:
					continue
				try:
					if resolved_path not in seen_paths:
						content = _load_corpus_file(
							corpus_cache,
							corpus_dir,
							resolved_path,
						)
						context_parts.append(f"[{resolved_path}]\n{content}")
						seen_paths.add(resolved_path)

					snippet_text = _get_snippet_text(
						snippet,
						corpus_cache,
						corpus_dir,
						resolved_path,
					)
				except FileNotFoundError:
					continue
				if snippet_text:
					answer_parts.append(snippet_text)

			if not context_parts:
				continue

			record = {
				"id": "",
				"source": "LegalBench-RAG",
				"query_type": benchmark_name,
				"question": query,
				"context": "\n".join(context_parts).strip(),
				"gold_answer": "\n".join(answer_parts).strip(),
			}
			dati_unificati.append(record)

	if max_examples is not None and len(dati_unificati) > max_examples:
		rng = random.Random(SAMPLE_SEED)
		dati_unificati = rng.sample(dati_unificati, max_examples)

	for indice, item in enumerate(dati_unificati):
		item["id"] = f"leg_{indice+1:04d}"

	with open(output_file, "w", encoding="utf-8") as f:
		for record in dati_unificati:
			f.write(json.dumps(record, ensure_ascii=False) + '\n')

	print(
		"Estratte "
		+ str(len(dati_unificati))
		+ " query da LegalBench-RAG e salvate in "
		+ output_file
		+ "!"
	)


if __name__ == "__main__":
	converti_legalbench(BENCHMARKS_DIR, CORPUS_DIR, DEFAULT_OUTPUT)
