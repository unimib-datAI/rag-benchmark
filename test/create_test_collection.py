#!/usr/bin/env python3
"""
Example script: creates a collection and uploads a batch of test documents
into it as plain text, the same way the "Upload TXT" tab of the UI does -
Next.js runs the configured annotation pipeline and indexing itself, this
script just submits raw text.

This talks to the real HTTP surface of a running dev stack:
  - the Next.js app's tRPC API:
    - collection.create
    - document.createUploadJob (uploadType: "txt") - registers a background
      job that annotates each file (frontend/server/routers/document.ts's
      runAnnotateAndUpload, using the user's active pipeline configuration)
      and uploads/indexes the result, exactly like
      UploadDocumentsModal.tsx's handleUploadTXT.
    - document.getUploadJob - polled until the job finishes.

Requirements:
  pip install requests python-dotenv

Usage:
  python scripts/create_test_collection.py
  python scripts/create_test_collection.py --collection-name "My test set"
  DAVE_USERNAME=me@example.com DAVE_PASSWORD=hunter2 python scripts/create_test_collection.py
  DAVE_TOKEN=eyJ... python scripts/create_test_collection.py

Auth:
  Every tRPC call below accepts an optional `token` field (a Keycloak
  Bearer access token) instead of an Authorization header - that's how
  this app's resolvers read it. If the server you're targeting runs with
  USE_AUTH=false (typical local dev), leave all three unset and requests
  are treated as an anonymous user with full access.

  Otherwise, either:
  - set DAVE_USERNAME/DAVE_PASSWORD (or --username/--password) and this
    script logs in for you via POST /api/auth/keycloak-login (Keycloak's
    Resource Owner Password Credentials grant - the same one
    lib/documentsBackend/keycloakService.ts's loginUser() performs), or
  - set DAVE_TOKEN directly if you already have an access token.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("create_test_collection")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from dotenv import load_dotenv
load_dotenv()

NEXTJS_BASE_URL = os.environ.get("DAVE_BASE_URL", "http://vm.chronos.disco.unimib.it:8001")
# The Next.js app is served under a basePath in dev (see frontend/.env
# NEXT_PUBLIC_BASE_PATH) - every route, including /api/trpc/*, lives under it.
NEXTJS_BASE_PATH = os.environ.get("DAVE_BASE_PATH", "/holmes24")
DAVE_TOKEN = os.environ.get("DAVE_TOKEN", "") or None
DAVE_USERNAME = os.environ.get("DAVE_USERNAME")
DAVE_PASSWORD = os.environ.get("DAVE_PASSWORD")

# How long to wait for the background annotation+upload job to finish.
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 600

# Medium-sized example "documents" - plain strings, no files needed.
# They mention real, well-known entities so the NER/NEL pipeline produces
# non-trivial annotations (people, organizations, locations) to inspect.
SAMPLE_DOCUMENTS = [
    {
        "name": "Apple Q3 Announcement",
        "text": (
            "Apple Inc. announced today that Tim Cook will present the company's "
            "quarterly earnings from its headquarters in Cupertino, California. "
            "The announcement comes as Apple continues to compete with Samsung and "
            "Google in the smartphone market. Analysts at Goldman Sachs expect "
            "strong iPhone sales driven by demand in China and the European Union. "
            "Cook is expected to discuss the company's investments in artificial "
            "intelligence, following partnerships announced earlier this year with "
            "OpenAI. The event will be held shortly after a similar presentation "
            "by Microsoft CEO Satya Nadella in Redmond, Washington."
        ),
    },
    {
        "name": "European Union Climate Summit",
        "text": (
            "Leaders from across the European Union gathered in Brussels this week "
            "to discuss new climate targets ahead of the next United Nations "
            "climate conference. French President Emmanuel Macron and German "
            "Chancellor Olaf Scholz both emphasized the need for faster investment "
            "in renewable energy, citing recent droughts affecting the Rhine river. "
            "The European Commission, led by Ursula von der Leyen, proposed a new "
            "package of subsidies for solar and wind projects in Spain and Italy. "
            "Environmental groups such as Greenpeace welcomed the proposal but "
            "warned it did not go far enough to meet the goals of the Paris "
            "Agreement."
        ),
    },
    {
        "name": "Wimbledon Final Recap",
        "text": (
            "Novak Djokovic defeated Carlos Alcaraz in a five-set thriller at the "
            "Wimbledon final held at the All England Club in London. The match, "
            "widely covered by the BBC and ESPN, lasted nearly four hours and drew "
            "comparisons to previous finals against Roger Federer and Rafael Nadal. "
            "Djokovic, who trains for part of the year in Belgrade, Serbia, thanked "
            "his coaching team and the crowd at Centre Court after the win. "
            "Alcaraz, representing Spain, said he looked forward to a rematch at "
            "the US Open in New York later this year."
        ),
    },
    {
        "name": "Startup Funding Round",
        "text": (
            "A Berlin-based startup called GreenGrid raised 40 million euros in a "
            "funding round led by Sequoia Capital, with participation from "
            "existing investor Index Ventures. The company, founded by former "
            "Siemens engineers, builds software for managing electric vehicle "
            "charging networks across Germany and the Netherlands. GreenGrid's "
            "chief executive said the funding would be used to expand into France "
            "and to hire engineers from universities such as ETH Zurich and TU "
            "Munich. The deal was first reported by TechCrunch and later "
            "confirmed by Reuters."
        ),
    },
]


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


def trpc_query(procedure: str, payload: Dict[str, Any]) -> Any:
    """Calls a tRPC v9 query with a plain (non-batched) HTTP GET."""
    url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/trpc/{procedure}"
    response = requests.get(
        url, params={"input": json.dumps(_with_token(payload))}, timeout=60
    )
    body = response.json()
    if "error" in body:
        raise TRPCError(f"{procedure} failed: {body['error'].get('message')}")
    return body["result"]["data"]


def create_collection(name: str) -> str:
    logger.info("Creating collection %r", name)
    collection = trpc_mutation(
        "collection.create", {"name": name, "allowedUserIds": []}
    )
    collection_id = collection["id"]
    logger.info("Created collection %s (id=%s)", name, collection_id)
    return collection_id


def create_txt_upload_job(collection_id: str, docs: List[Dict[str, str]]) -> Dict[str, Any]:
    """Registers a background upload job exactly like the "Upload TXT" tab in
    UploadDocumentsModal.tsx: Next.js runs the user's active annotation
    pipeline configuration over each file's raw text, then uploads and
    indexes the result. No annotation call needed on our end."""
    files = [{"fileName": f"{doc['name']}.txt", "content": doc["text"]} for doc in docs]
    job = trpc_mutation(
        "document.createUploadJob",
        {
            "collectionId": collection_id,
            "uploadType": "txt",
            "files": files,
            "toAnonymize": False,
        },
    )
    logger.info("Upload job created: %s (%d files)", job["jobId"], len(files))
    return job


def poll_upload_job(job_id: str) -> Dict[str, Any]:
    """Polls document.getUploadJob until the background job reaches a
    terminal status, logging progress along the way."""
    terminal_statuses = {"completed", "completed_with_errors", "failed", "cancelled"}
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS

    job = trpc_query("document.getUploadJob", {"jobId": job_id})
    while job["status"] not in terminal_statuses:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Upload job {job_id} did not finish within {POLL_TIMEOUT_SECONDS}s"
            )
        done = sum(1 for f in job["files"] if f["status"] in ("completed", "failed"))
        logger.info(
            "Job %s: %s (%d/%d files processed)",
            job_id,
            job["status"],
            done,
            len(job["files"]),
        )
        time.sleep(POLL_INTERVAL_SECONDS)
        job = trpc_query("document.getUploadJob", {"jobId": job_id})

    return job


def main() -> None:
    global DAVE_TOKEN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-name",
        default=f"Test collection {time.strftime('%Y-%m-%d %H:%M:%S')}",
        help="Name for the newly created collection",
    )
    parser.add_argument("--username", default=DAVE_USERNAME, help="Keycloak username/email to log in with")
    parser.add_argument("--password", default=DAVE_PASSWORD, help="Keycloak password to log in with")
    args = parser.parse_args()

    if not DAVE_TOKEN and args.username and args.password:
        logger.info("Logging in as %s...", args.username)
        DAVE_TOKEN = login(args.username, args.password)
        logger.info("Logged in successfully")

    collection_id = create_collection(args.collection_name)

    job = create_txt_upload_job(collection_id, SAMPLE_DOCUMENTS)
    job = poll_upload_job(job["jobId"])

    print("\nSummary")
    print("-------")
    print(f"Collection: {args.collection_name} ({collection_id})")
    print(f"Upload job: {job['jobId']} ({job['status']})")
    for f in job["files"]:
        detail = f" (id={f['documentId']})" if f.get("documentId") else ""
        if f["status"] == "failed":
            detail = f" ({f.get('error')})"
        print(f"  - {f['fileName']}: {f['status']}{detail}")

    print(
        "\nNote: annotation happens server-side using the calling user's "
        "active pipeline configuration (Settings > Annotation configuration). "
        "If none is set, documents upload with no entities."
    )

    # Remember the collection so scripts/test_rag_generation.py can default to
    # it without the user having to copy/paste the id around.
    state_file = Path(__file__).parent / ".last_test_collection.json"
    state_file.write_text(
        json.dumps(
            {"collectionId": collection_id, "name": args.collection_name}, indent=2
        )
    )
    logger.info("Saved collection reference to %s", state_file)


if __name__ == "__main__":
    main()
