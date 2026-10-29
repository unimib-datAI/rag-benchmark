"""Build a zip with the raw files of the 6 benchmarks, downloaded from their official sources.

The benchmarks are not in the GitHub repository (they weigh several GB with their
git history); this zip contains only the files read by Run/extract_*.py and
ingest_dave.py, already laid out as Benchmarks/<name>/... so it can be extracted
in the project root.
"""
import gzip
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
# inside Benchmarks/: in Docker the project root is read-only for the app user, Benchmarks/ is a mounted volume
DEFAULT_OUTPUT = ROOT_DIR / "Benchmarks" / "Benchmarks_download.zip"

sys.path.append(str(ROOT_DIR))
from corpus_sources import OHR_VERSION

_MTRAG_CORPUS = "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/document_level/{0}.jsonl.zip"

# (label, url, destination in the zip, kind[, only])
# kind: "file" copied as is, "gunzip" decompressed, "unzip" extracted under the destination folder
# only: for "unzip", extract just the members under this folder of the archive
SOURCES = [
    ("CRAG", "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2",
     "Benchmarks/CRAG/data/crag_task_1_and_2_dev_v4.jsonl.bz2", "file"),
    ("LegalBench-RAG", "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
     "?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1",
     "Benchmarks/legalbenchrag/data/", "unzip"),
    ("MultiHop-RAG", "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json",
     "Benchmarks/MultiHop-RAG/dataset/MultiHopRAG.json", "file"),
    ("MultiHop-RAG corpus", "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json",
     "Benchmarks/MultiHop-RAG/dataset/corpus.json", "file"),
    ("MTRAG", "https://github.com/IBM/mt-rag-benchmark/raw/main/mtrag-human/conversations/conversations.json",
     "Benchmarks/mt-rag-benchmark/mtrag-human/conversations/conversations.json", "file"),
    *[(f"MTRAG corpus {key}", _MTRAG_CORPUS.format(key),
       f"Benchmarks/mt-rag-benchmark/corpora/document_level/{key}.jsonl/", "unzip")
      for key in ("clapnq", "cloud", "fiqa", "govt")],
    ("OHRBench", "https://github.com/opendatalab/OHR-Bench/raw/main/data/qas.json",
     "Benchmarks/OHR-Bench/data/qas.json", "file"),
    ("OHRBench corpus", "https://huggingface.co/datasets/opendatalab/OHR-Bench/resolve/main/retrieval.zip",
     "Benchmarks/OHR-Bench/data/retrieval_base/", "unzip", f"{OHR_VERSION}/"),
    ("SQuAD-it", "https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz",
     "Benchmarks/squad-it/SQuAD_it-test.json", "gunzip"),
]

_CHUNK = 1024 * 1024


def source_benchmark(label):
    """Benchmark a source belongs to: "MTRAG corpus clapnq" -> "MTRAG"."""
    return label.split(" corpus")[0]


BENCHMARKS = list(dict.fromkeys(source_benchmark(label) for label, *_ in SOURCES))

# characters not allowed in Windows file names: some LegalBench files contain "|" or ":",
# and a single invalid name makes Windows Explorer refuse the whole archive
_INVALID_NAME_CHARS = str.maketrans({c: "_" for c in '<>:"\\|?*'})


def _download(url, dest, on_bytes):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                on_bytes(done, total)


def _entry(name):
    # entries written as streams would otherwise be dated 1980
    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _safe_member_name(name):
    """Member path of a downloaded zip, or "" if it could escape Benchmarks/ when the
    final archive is extracted (zip slip: ".." segments, absolute or drive paths)."""
    parts = name.replace("\\", "/").split("/")
    if ".." in parts or name.startswith(("/", "\\")) or (parts and ":" in parts[0]):
        return ""
    return "/".join(p for p in parts if p not in ("", "."))


def _add_to_zip(zf, src, dest, kind, only=None):
    if kind == "file":
        # already compressed files (.bz2) are stored, the others deflated
        compress = zipfile.ZIP_STORED if dest.endswith(".bz2") else zipfile.ZIP_DEFLATED
        zf.write(src, dest, compress_type=compress)
    elif kind == "gunzip":
        with gzip.open(src, "rb") as fin, zf.open(_entry(dest), "w", force_zip64=True) as fout:
            shutil.copyfileobj(fin, fout, _CHUNK)
    elif kind == "unzip":
        with zipfile.ZipFile(src) as inner:
            for member in inner.infolist():
                name = _safe_member_name(member.filename)
                if member.is_dir() or not name or name.startswith("__MACOSX"):
                    continue
                if only and not name.startswith(only):
                    continue
                name = name.translate(_INVALID_NAME_CHARS)
                with inner.open(member) as fin, zf.open(_entry(dest + name), "w", force_zip64=True) as fout:
                    shutil.copyfileobj(fin, fout, _CHUNK)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def build_benchmarks_zip(output=DEFAULT_OUTPUT, progress=None, benchmarks=None):
    """Download the sources of `benchmarks` (default: all, see BENCHMARKS) and write the zip;
    progress(step, n_steps, label, done, total) is optional.

    The zip is written to a .part file and renamed only at the end, so a failed
    download never leaves a truncated archive behind.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    sources = [src for src in SOURCES if benchmarks is None or source_benchmark(src[0]) in benchmarks]
    if not sources:
        raise ValueError(f"No source for {benchmarks}: valid benchmarks are {BENCHMARKS}")
    n_steps = len(sources)
    try:
        with tempfile.TemporaryDirectory() as tmp, \
                zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for step, (label, url, dest, kind, *only) in enumerate(sources):
                src = Path(tmp) / f"source_{step}"

                def on_bytes(done, total, step=step, label=label):
                    if progress:
                        progress(step, n_steps, label, done, total)

                try:
                    _download(url, src, on_bytes)
                except requests.RequestException as e:
                    raise RuntimeError(f"Download of {label} failed ({url}): {e}") from e
                _add_to_zip(zf, src, dest, kind, *only)
                src.unlink()
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def extract_benchmarks_zip(zip_path=DEFAULT_OUTPUT, dest=ROOT_DIR, progress=None):
    """Extract the zip in the project folder; progress(done, total) counts the archive members."""
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for i, member in enumerate(members):
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:  # refuse paths escaping the folder
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            zf.extract(member, dest)
            if progress:
                progress(i + 1, len(members))
    return dest / "Benchmarks"


if __name__ == "__main__":
    def _print_progress(step, n_steps, label, done, total):
        size = f"{done / _CHUNK:.0f}/{total / _CHUNK:.0f} MB" if total else f"{done / _CHUNK:.0f} MB"
        print(f"\r[{step + 1}/{n_steps}] {label}: {size}".ljust(70), end="", flush=True)

    path = build_benchmarks_zip(progress=_print_progress, benchmarks=sys.argv[1:] or None)
    print(f"\nSaved {path}, extracting...")
    print(f"Extracted in {extract_benchmarks_zip(path)}")
