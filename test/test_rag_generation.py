#!/usr/bin/env python3
"""
Example script: exercises the text-generation-with-RAG flow against a
collection, retrieving relevant chunks from Elasticsearch and asking the LLM
to answer a question grounded in them - end to end, the same path the chat
UI drives (see frontend/hooks/use-chat.ts and frontend/pages/api/generate.ts).

By default it targets the collection created by
scripts/create_test_collection.py (read from scripts/.last_test_collection.json).
Pass --collection-id to target a different one.

Two modes:
  - standard RAG (default): calls the `search.mostSimilarDocuments` tRPC
    mutation to retrieve chunks via vector/hybrid search scoped to the
    collection, builds the same context-injected system prompt the frontend
    uses, then streams a completion from POST /api/generate.
  - --multi-agent: retrieval happens server-side inside the multi-agent
    pipeline (frontend/lib/multiAgent.ts); this script just streams
    /api/generate with useMultiAgent=true and decodes its two sentinel types
    (step progress + retrieved-context) out of the raw text stream.

Requirements:
  pip install requests python-dotenv

Usage:
  python scripts/test_rag_generation.py
  python scripts/test_rag_generation.py --query "Who won the Wimbledon final?"
  python scripts/test_rag_generation.py --collection-id <id> --multi-agent
  DAVE_USERNAME=me@example.com DAVE_PASSWORD=hunter2 python scripts/test_rag_generation.py
  DAVE_TOKEN=eyJ... python scripts/test_rag_generation.py

Auth: same convention as create_test_collection.py - tRPC calls accept an
optional `token` field (Keycloak Bearer token) in the JSON payload. Leave
DAVE_USERNAME/DAVE_PASSWORD/DAVE_TOKEN all unset if the server runs with
USE_AUTH=false; otherwise set DAVE_USERNAME/DAVE_PASSWORD (or
--username/--password) and this script logs in for you via
POST /api/auth/keycloak-login, or set DAVE_TOKEN directly if you already
have an access token.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("test_rag_generation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dotenv import load_dotenv
load_dotenv()

NEXTJS_BASE_URL = os.environ.get("DAVE_BASE_URL", "http://vm.chronos.disco.unimib.it:8001")
# The Next.js app is served under a basePath in dev (see frontend/.env
# NEXT_PUBLIC_BASE_PATH) - every route, including /api/trpc/* and
# /api/generate, lives under it.
NEXTJS_BASE_PATH = os.environ.get("DAVE_BASE_PATH", "/holmes24")
DAVE_TOKEN = os.environ.get("DAVE_TOKEN", "") or None
DAVE_USERNAME = os.environ.get("DAVE_USERNAME")
DAVE_PASSWORD = os.environ.get("DAVE_PASSWORD")

STATE_FILE = Path(__file__).parent / ".last_test_collection.json"

# Mirrors frontend/atoms/llmSettings.ts DEFAULT_SYSTEM_PROMPT exactly, so the
# standard-RAG path here matches what the chat UI actually sends.
DEFAULT_SYSTEM_PROMPT = """You are an expert assistant that answers questions based on provided context.

<input>

<context>
{{CONTEXT}}
</context>

<question language="auto">
{{QUESTION}}
</question>

<instructions>
- Answer the question using ONLY information explicitly stated in the context.
- Integrate information from multiple documents only if they are consistent.
- Do NOT infer, speculate, generalize, or rely on external knowledge.
- The answer MUST be written in the same language as the question.
- If answering requires translating information from the context, translate faithfully
  without adding, omitting, or reinterpreting any content.
- Do NOT mention documents, context, retrieval, or sources explicitly.
- If the context is insufficient, incomplete, or ambiguous, respond EXACTLY with:
  "The information provided is not sufficient to answer with certainty." and give an explanation about why you can't answer.
Always assume that the user is asking you about information contained in the documents provided
- Use a clear, precise, and domain-appropriate technical style.
Make sure to ALWAYS answer in the same language used by the user to ask the question, don't ming the documents language
</instructions>

</input>"""

# Sentinels emitted by the multi-agent pipeline inside the raw text stream
# (see frontend/lib/multiAgent.ts / frontend/hooks/use-chat.ts).
STEP_RE = re.compile(r"\x02DAVE_STEP\x1E([\s\S]*?)\x03")
CTX_RE = re.compile(r"\x02DAVE_CTX\x1E([\s\S]*?)\x03")


class TRPCError(RuntimeError):
    pass


def _with_token(payload: Dict[str, Any]) -> Dict[str, Any]:
    if DAVE_TOKEN:
        return {**payload, "token": DAVE_TOKEN}
    return payload


def trpc_mutation(procedure: str, payload: Dict[str, Any]) -> Any:
    """Calls a tRPC v9 mutation with a plain (non-batched) HTTP POST."""
    url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/trpc/{procedure}"
    response = requests.post(url, json=_with_token(payload), timeout=120)
    body = response.json()
    if "error" in body:
        raise TRPCError(f"{procedure} failed: {body['error'].get('message')}")
    return body["result"]["data"]


def login(username: str, password: str) -> str:
    """Logs in via Keycloak's Resource Owner Password Credentials grant
    (POST /api/auth/keycloak-login, see keycloakService.ts's loginUser()) and
    returns the access token to use as the `token` field on every call below."""
    url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/auth/keycloak-login"
    response = requests.post(url, json={"username": username, "password": password}, timeout=30)
    if not response.ok:
        raise RuntimeError(f"Login failed ({response.status_code}): {response.text}")
    return response.json()["access_token"]


def default_collection_id() -> Optional[str]:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text()).get("collectionId")
    except (json.JSONDecodeError, OSError):
        return None


def retrieve_chunks(
    query: str,
    collection_id: str,
    retrieval_method: str,
    filter_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Vector/hybrid search scoped to the collection - same call the chat UI
    makes before generation (frontend/server/routers/search.ts)."""
    return trpc_mutation(
        "search.mostSimilarDocuments",
        {
            "query": query,
            "collectionId": collection_id,
            "retrievalMethod": retrieval_method,
            **({"filter_ids": filter_ids} if filter_ids else {}),
        },
    )


def build_system_prompt(docs: List[Dict[str, Any]], question: str) -> str:
    """Numbers chunks across all retrieved docs and injects them into
    DEFAULT_SYSTEM_PROMPT, mirroring frontend/hooks/use-chat.ts appendMessage()."""
    context_parts = []
    chunk_num = 0
    for i, doc in enumerate(docs):
        lines = []
        for chunk in doc.get("chunks", []):
            chunk_num += 1
            lines.append(f"[{chunk_num}] {chunk['text']}")
        doc_content = f"Nome Documento {doc['title']} - Contenuto:\n" + "\n".join(lines)
        context_parts.append(f'<document id="DOC_{i + 1}" name="{doc["title"]}">\n{doc_content}\n</document>')
    context_str = "\n".join(context_parts)

    return DEFAULT_SYSTEM_PROMPT.replace("{{CONTEXT}}", context_str).replace("{{QUESTION}}", question)


def stream_generate(payload: Dict[str, Any], multi_agent: bool) -> str:
    """POSTs to /api/generate and prints the streamed completion as it
    arrives, decoding multi-agent sentinels along the way. Returns the final
    clean (sentinel-stripped) text."""
    url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/generate"
    raw = ""

    with requests.post(url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                raw += chunk

    # Sentinels can straddle chunk boundaries, so parse once the stream is
    # fully buffered rather than incrementally.
    steps = []

    def _take_step(m: "re.Match[str]") -> str:
        try:
            steps.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
        return ""

    visible = STEP_RE.sub(_take_step, raw)

    ctx_payload = None
    ctx_match = CTX_RE.search(visible)
    if ctx_match:
        try:
            ctx_payload = json.loads(ctx_match.group(1))
        except json.JSONDecodeError:
            ctx_payload = None
        visible = CTX_RE.sub("", visible)

    if multi_agent and steps:
        print("Agent steps:")
        for s in steps:
            status = s.get("status", "?")
            print(f"  [{status}] {s.get('agent')}: {s.get('step')}")
        print()

    print("Answer:")
    print(visible.strip())

    if multi_agent and ctx_payload:
        docs = ctx_payload.get("docs", ctx_payload) if isinstance(ctx_payload, dict) else ctx_payload
        print("\nRetrieved context (from multi-agent pipeline):")
        for doc in docs or []:
            print(f"  - {doc.get('title')} (doc id={doc.get('id')}, {len(doc.get('chunks', []))} chunks)")

    return visible.strip()


def main() -> None:
    global DAVE_TOKEN

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--collection-id",
        default=default_collection_id(),
        help="Collection to search in (defaults to the one saved by create_test_collection.py)",
    )
    parser.add_argument(
        "--query",
        default="Who won the Wimbledon final and who did they beat?",
        help="Question to ask the RAG pipeline",
    )
    parser.add_argument(
        "--retrieval-method",
        default="full",
        choices=["full", "dense", "hibrid_no_ner", "full-text"],
        help="Retrieval strategy used by search.mostSimilarDocuments (standard RAG mode only)",
    )
    parser.add_argument("--filter-ids", nargs="*", default=None, help="Restrict retrieval to these document ids")
    parser.add_argument("--multi-agent", action="store_true", help="Use the multi-agent RAG pipeline instead")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--username", default=DAVE_USERNAME, help="Keycloak username/email to log in with")
    parser.add_argument("--password", default=DAVE_PASSWORD, help="Keycloak password to log in with")
    args = parser.parse_args()

    if not args.collection_id:
        parser.error(
            "No --collection-id given and no saved collection found. "
            "Run scripts/create_test_collection.py first, or pass --collection-id explicitly."
        )

    if not DAVE_TOKEN and args.username and args.password:
        logger.info("Logging in as %s...", args.username)
        DAVE_TOKEN = login(args.username, args.password)
        logger.info("Logged in successfully")

    logger.info("Collection: %s", args.collection_id)
    logger.info("Query: %s", args.query)

    if args.multi_agent:
        logger.info("Mode: multi-agent (retrieval happens server-side)")
        payload = {
            "messages": [{"role": "user", "content": args.query}],
            "useMultiAgent": True,
            "collectionId": args.collection_id,
            "filterIds": args.filter_ids,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
        stream_generate(payload, multi_agent=True)
        return

    logger.info("Mode: standard RAG (retrieval method=%s)", args.retrieval_method)
    docs = retrieve_chunks(args.query, args.collection_id, args.retrieval_method, args.filter_ids)

    total_chunks = sum(len(d.get("chunks", [])) for d in docs)
    logger.info("Retrieved %d chunk(s) across %d document(s)", total_chunks, len(docs))
    for doc in docs:
        print(f"  - {doc['title']} (doc id={doc['id']}, {len(doc.get('chunks', []))} chunks)")

    if total_chunks == 0:
        logger.warning(
            "No chunks retrieved - make sure ELASTIC_INDEX is configured on the server "
            "and the documents were indexed (see indexCreatedDocument in server/routers/document.ts)."
        )

    system_prompt = build_system_prompt(docs, args.query)
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.query},
        ],
        "collectionId": args.collection_id,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    print()
    stream_generate(payload, multi_agent=False)


if __name__ == "__main__":
    main()
