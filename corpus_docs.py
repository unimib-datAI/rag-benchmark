"""How benchmark texts are split into the documents uploaded to DAVE.

Shared by ingest_dave.py (what gets uploaded) and Run/build_gold_docs.py (which documents each
query needs), so the file names match exactly.
"""
import re


def crag_pages(context: str) -> list[tuple[str, str]]:
    """The context of a CRAG question holds several web pages, each starting with '[...'.

    Returns (fileName, text) for each page; the name comes from the first 50 characters."""
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    parts, current = [], []
    for line in lines:
        if line.startswith("[") and "]" in line and current:
            parts.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append(" ".join(current).strip())

    pages = []
    for doc in parts:
        if doc.strip():
            fake_id = "".join(c for c in doc[:50].replace(" ", "_") if c.isalnum() or c == "_")
            pages.append((f"{fake_id}.txt", doc))
    return pages


def mtrag_base_id(doc_id: str) -> str:
    """MTRAG passages of the same document ('827849752_115-357') share the part before '_'.

    Ids like 'ibmcld_00001' are already whole documents."""
    if not doc_id or "_" not in doc_id:
        return doc_id or ""
    base, tail = doc_id.split("_", 1)
    return base if re.match(r"^\d+-\d+$", tail) else doc_id
