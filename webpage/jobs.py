"""Long tasks run as subprocesses so they survive Streamlit reruns.

State lives in module globals: Streamlit imports this module once per server process.
"""
import subprocess
import threading
import time
from collections import deque

_JOBS = {}
_LOCK = threading.Lock()
MAX_LINES = 500


def _reader(job):
    """Split the child's stdout on \\n (log lines) vs \\r (progress-bar refresh)."""
    buf = b""
    stream = job["proc"].stdout
    while True:
        chunk = stream.read(4096)  # raw pipe: returns as soon as some bytes are available
        if not chunk:
            break
        buf += chunk
        while True:
            cut = [i for i in (buf.find(b"\n"), buf.find(b"\r")) if i != -1]
            if not cut:
                break
            i = min(cut)
            sep, text = buf[i:i + 1], buf[:i].decode("utf-8", errors="replace")
            buf = buf[i + 1:]
            if not text.strip():
                continue
            with _LOCK:
                if sep == b"\r":
                    job["progress"] = text
                else:
                    job["lines"].append(text)
                    job["progress"] = ""
    if buf.strip():
        with _LOCK:
            job["lines"].append(buf.decode("utf-8", errors="replace"))
    job["proc"].wait()
    job["ended"] = time.time()


def is_running(name):
    job = _JOBS.get(name)
    return bool(job) and job["proc"].poll() is None


def start(name, args, label, env=None, cwd=None):
    """Launch a job; return False if a job with the same name is still running."""
    with _LOCK:
        if is_running(name):
            return False
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=cwd, bufsize=0,
        )
        job = {
            "proc": proc, "label": label, "lines": deque(maxlen=MAX_LINES),
            "progress": "", "started": time.time(), "ended": None,
        }
        _JOBS[name] = job
    threading.Thread(target=_reader, args=(job,), daemon=True).start()
    return True


def stop(name):
    if is_running(name):
        _JOBS[name]["proc"].terminate()


def snapshot(name):
    """Return {label, running, returncode, lines, progress, elapsed} or None if never started."""
    job = _JOBS.get(name)
    if not job:
        return None
    with _LOCK:
        lines, progress = list(job["lines"]), job["progress"]
    running = job["proc"].poll() is None
    end = time.time() if running else (job["ended"] or time.time())
    return {
        "label": job["label"],
        "running": running,
        "returncode": None if running else job["proc"].returncode,
        "lines": lines,
        "progress": progress,
        "elapsed": int(end - job["started"]),
    }
