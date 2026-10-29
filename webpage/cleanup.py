"""Two separate clean-ups: the local files of the datasets (Catalogue) and the records of what was indexed (Setup).

Only an explicit list of paths is ever touched (never a whole folder like Benchmarks/ or Run/, which also hold
mounted data and the extract_*.py scripts), and each one is checked to be inside the project folder.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_download import SOURCES
from corpus_sources import BENCHMARK_FILES

STATE_DIR = Path(os.environ.get("STATE_DIR") or ROOT_DIR)

# built by the Run/extract_*.py scripts next to the Tutto_*.jsonl query files
_EXTRA_PREPARED = ["squad_corpus.json", "gold_docs.json"]
# what the web app records about indexing, per system
INDEX_STATE_FILES = {
    "generic": [".rag_docs_cache.json"],  # documents uploaded to each RAG URL
    "dave": [".dave_collections.json", ".dave_docs_cache.json", ".dave_upload_jobs.json"],  # collections, files, jobs
}
_DERIVED_FILES = [".catalogue_counts.json"]


def download_paths(root=ROOT_DIR):
    """Existing raw benchmark folders in Benchmarks/ (one per top-level folder of benchmark_download.SOURCES)
    and the leftovers of an interrupted download."""
    root = Path(root)
    folders = dict.fromkeys(Path(dest).parts[1] for _, _, dest, *_ in SOURCES if Path(dest).parts[0] == "Benchmarks")
    paths = [root / "Benchmarks" / name for name in folders]
    paths += [root / "Benchmarks" / "Benchmarks_download.zip", root / "Benchmarks" / "Benchmarks_download.zip.part"]
    return [p for p in paths if p.exists()]


def prepared_paths(root=ROOT_DIR):
    """Existing query files and corpora built in Run/ from the raw benchmarks."""
    root = Path(root)
    paths = [root / "Run" / name for name in [*BENCHMARK_FILES.values(), *_EXTRA_PREPARED]]
    return [p for p in paths if p.exists()]


def index_state_paths(systems, state_dir=STATE_DIR):
    names = [name for system in systems for name in INDEX_STATE_FILES[system]]
    return [p for p in (Path(state_dir) / name for name in names) if p.exists()]


def _size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _remove(path, base, errors):
    """Delete one file or folder, only if it is inside `base`. Returns (bytes freed, deleted)."""
    resolved, base = path.resolve(), Path(base).resolve()
    if base not in resolved.parents:
        errors.append(f"{path}: outside {base}, not deleted")
        return 0, False
    freed = _size(path)
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError as e:
        errors.append(f"{path}: {e}")
        return 0, False
    return freed, True


def clear_downloads(root=ROOT_DIR, state_dir=STATE_DIR):
    """Delete the raw benchmarks, the prepared query files and the cached query counts.
    Returns (items deleted, bytes freed, errors)."""
    targets = [(p, root) for p in [*download_paths(root), *prepared_paths(root)]]
    targets += [(Path(state_dir) / f, state_dir) for f in _DERIVED_FILES if (Path(state_dir) / f).exists()]
    errors, freed, deleted = [], 0, 0
    for path, base in targets:
        size, ok = _remove(path, base, errors)
        freed += size
        deleted += ok
    return deleted, freed, errors


def clear_index_records(systems, state_dir=STATE_DIR):
    """Forget what was indexed in the given systems ("generic", "dave"): collections, uploaded files, jobs.
    Nothing is sent to them: the old DAVE collections stay there, unused, and the next ingestion creates new ones.
    Downloaded files are not touched. Returns (names removed, errors)."""
    errors, removed = [], []
    for path in index_state_paths(systems, state_dir):
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as e:
            errors.append(f"{path}: {e}")
    return removed, errors
