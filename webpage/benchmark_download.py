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
# kind: "file" copied as is, "gunzip" decompressed, "unzip" extracted under the destination folder,
#       "hf" a Hugging Face dataset repo (url = "owner/name"): its corpus/ and benchmarks/ folders, verified with its SHA256SUMS
# only: for "unzip", extract just the members under this folder of the archive
SOURCES = [
    ("CRAG", "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2",
     "Benchmarks/CRAG/data/crag_task_1_and_2_dev_v4.jsonl.bz2", "file"),
    # the authors' Dropbox link (github.com/zeroentropy-ai/legalbenchrag) is disabled: this mirror holds the same
    # files "as distributed by the authors, without changes" (714 documents, 6,889 questions)
    ("LegalBench-RAG", "awinml/legalbench-rag", "Benchmarks/legalbenchrag/data/", "hf"),
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


def is_downloaded(benchmark):
    """True if every file of the benchmark (see SOURCES) is in Benchmarks/, extracted."""
    for label, _url, dest, kind, *only in SOURCES:
        if source_benchmark(label) != benchmark:
            continue
        path = ROOT_DIR / dest
        if kind in ("unzip", "hf"):
            folder = path / (only[0] if only else "")
            if not (folder.is_dir() and any(f.is_file() for f in folder.rglob("*"))):
                return False
        elif not path.is_file():
            return False
    return True


def _download(url, dest, on_bytes):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", "").lower():
            # sharing sites answer 200 with a web page when a link is disabled, over quota or needs a login:
            # saving it would give a corrupt "zip" or a json file full of HTML
            raise RuntimeError("the server returned a web page instead of the file (link disabled, over quota "
                               "or login required: open the URL in a browser to see why)")
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(_CHUNK):
                f.write(chunk)
                done += len(chunk)
                on_bytes(done, total)


def _add_hf_dataset(zf, repo, dest, on_bytes):
    """Write the corpus/ and benchmarks/ files of a Hugging Face dataset repo into the zip under `dest`,
    checking each one against the repo's SHA256SUMS."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote

    base = f"https://huggingface.co/datasets/{repo}"
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true", timeout=60)
    r.raise_for_status()
    files = [e for e in r.json() if e["type"] == "file" and e["path"].split("/")[0] in ("corpus", "benchmarks")]
    if not files:
        raise RuntimeError(f"no corpus/ or benchmarks/ files found in {repo}")
    sums = requests.get(f"{base}/resolve/main/SHA256SUMS", timeout=60)
    sums.raise_for_status()
    expected = {name.strip(): digest for digest, name in (line.split(None, 1) for line in sums.text.splitlines() if line.strip())}
    total, done = sum(e.get("size", 0) for e in files), 0

    def fetch(entry):
        """Content of one file (they are small); the many requests run in parallel, the latency dominates."""
        resp = session.get(f"{base}/resolve/main/{quote(entry['path'])}", timeout=60)
        resp.raise_for_status()
        return resp.content

    with requests.Session() as session, ThreadPoolExecutor(max_workers=8) as pool:
        # map() yields in order, so the zip is written sequentially while the next files are already downloading
        for entry, content in zip(files, pool.map(fetch, files)):
            path = entry["path"]
            if expected.get(path) and hashlib.sha256(content).hexdigest() != expected[path]:
                raise RuntimeError(f"checksum mismatch for {path}")
            zf.writestr(_entry(dest + path.translate(_INVALID_NAME_CHARS)), content)
            done += len(content)
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
                    if kind == "hf":
                        _add_hf_dataset(zf, url, dest, on_bytes)
                        continue
                    _download(url, src, on_bytes)
                except (requests.RequestException, RuntimeError) as e:
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
