import json
import logging
import os
import re
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("dave_client")

NEXTJS_BASE_URL = os.environ.get("DAVE_BASE_URL", "http://vm.chronos.disco.unimib.it:8001")
NEXTJS_BASE_PATH = os.environ.get("DAVE_BASE_PATH", "/holmes24")

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

STEP_RE = re.compile(r"\x02DAVE_STEP\x1E([\s\S]*?)\x03")
CTX_RE = re.compile(r"\x02DAVE_CTX\x1E([\s\S]*?)\x03")

# Limite caratteri per il contesto nel system prompt.
# 500k chars Ôëê ~166k token, lascia margine per template + risposta sotto il limite 262k.
MAX_PROMPT_CONTEXT_CHARS = 500_000


def _tronca_testo(testo, max_chars):
    """Tronca il testo rispettando i confini di frase (. ! ?) o, in subordine,
    i confini di parola. Non taglia mai a met├á parola."""
    if len(testo) <= max_chars:
        return testo
    candidato = testo[:max_chars]
    # Non tornare indietro pi├╣ del 30% per cercare una fine di frase
    soglia_min = int(max_chars * 0.7)
    pos = -1
    for terminatore in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
        idx = candidato.rfind(terminatore, soglia_min)
        if idx > pos:
            pos = idx + 1  # include il segno di punteggiatura
    if pos > soglia_min:
        return candidato[:pos].rstrip() + "\n[... troncato ...]"
    # Fallback: confine di parola
    ultimo_spazio = candidato.rfind(' ', soglia_min)
    if ultimo_spazio > soglia_min:
        return candidato[:ultimo_spazio].rstrip() + "\n[... troncato ...]"
    # Ultima risorsa (testo senza spazi): tronca al limite
    return candidato.rstrip() + "\n[... troncato ...]"

class TRPCError(RuntimeError):
    pass

class DaveClient:
    def __init__(self, username=None, password=None):
        username = username or os.environ.get("DAVE_USERNAME")
        password = password or os.environ.get("DAVE_PASSWORD")
        if not username or not password:
            raise RuntimeError("Credenziali DAVE mancanti. Impostale nel file .env (DAVE_USERNAME, DAVE_PASSWORD)")
        self.username = username
        self.password = password
        self.token = None
        self._login()

    def _login(self):
        url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/auth/keycloak-login"
        resp = requests.post(url, json={"username": self.username, "password": self.password}, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"DAVE Login failed ({resp.status_code}): {resp.text}")
        self.token = resp.json()["access_token"]

    def _trpc_mutation(self, procedure: str, payload: Dict[str, Any]) -> Any:
        url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/trpc/{procedure}"
        if self.token:
            payload["token"] = self.token
        response = requests.post(url, json=payload, timeout=120)
        body = response.json()
        if "error" in body:
            raise TRPCError(f"{procedure} failed: {body['error'].get('message')}")
        return body["result"]["data"]

    def _trpc_query(self, procedure: str, payload: Dict[str, Any]) -> Any:
        url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/trpc/{procedure}"
        if self.token:
            payload["token"] = self.token
        response = requests.get(url, params={"input": json.dumps(payload)}, timeout=60)
        body = response.json()
        if "error" in body:
            raise TRPCError(f"{procedure} failed: {body['error'].get('message')}")
        return body["result"]["data"]

    # --- INGESTION APIs ---

    def create_collection(self, name: str) -> str:
        res = self._trpc_mutation("collection.create", {"name": name, "allowedUserIds": []})
        return res["id"]

    def list_collections(self) -> List[Dict]:
        return self._trpc_query("collection.list", {})

    def get_collection_by_name(self, name: str) -> Optional[str]:
        cols = self.list_collections()
        for c in cols:
            if c.get("name") == name:
                return c.get("id")
        return None

    def create_txt_upload_job(self, collection_id: str, files_data: List[Dict[str, str]]) -> Dict[str, Any]:
        job = self._trpc_mutation(
            "document.createUploadJob",
            {
                "collectionId": collection_id,
                "uploadType": "txt",
                "files": files_data,
                "toAnonymize": False,
            },
        )
        return job

    def get_upload_job_status(self, job_id: str) -> Dict[str, Any]:
        return self._trpc_query("document.getUploadJob", {"jobId": job_id})

    # --- RETRIEVAL & GENERATION APIs ---

    def retrieve_chunks(self, query: str, collection_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # DAVE usually has a hardcoded k or relies on dense/full parameters. We just call it.
        return self._trpc_mutation(
            "search.mostSimilarDocuments",
            {
                "query": query,
                "collectionId": collection_id,
                "retrievalMethod": "full",
            },
        )

    def build_system_prompt(self, docs: List[Dict[str, Any]], question: str, max_docs: int = 5) -> str:
        context_parts = []
        chunk_num = 0
        for i, doc in enumerate(docs[:max_docs]):
            lines = []
            for chunk in doc.get("chunks", []):
                chunk_num += 1
                lines.append(f"[{chunk_num}] {chunk['text']}")
            doc_content = f"Nome Documento {doc.get('title', 'Doc')} - Contenuto:\n" + "\n".join(lines)
            context_parts.append(f'<document id="DOC_{i + 1}" name="{doc.get("title", "Doc")}">\n{doc_content}\n</document>')
        context_str = "\n".join(context_parts)

        # Tronca il contesto se supera il limite per evitare "prompt too long"
        if len(context_str) > MAX_PROMPT_CONTEXT_CHARS:
            logger.warning(
                "Contesto prompt troppo lungo (%d chars, limite %d). Troncamento ai confini di frase...",
                len(context_str), MAX_PROMPT_CONTEXT_CHARS
            )
            context_str = _tronca_testo(context_str, MAX_PROMPT_CONTEXT_CHARS)

        return DEFAULT_SYSTEM_PROMPT.replace("{{CONTEXT}}", context_str).replace("{{QUESTION}}", question)

    def generate_answer(self, system_prompt: str, question: str, collection_id: str = None) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        if collection_id:
            payload["collectionId"] = collection_id

        url = f"{NEXTJS_BASE_URL}{NEXTJS_BASE_PATH}/api/generate"
        raw = ""
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    raw += chunk

        # Clean DAVE markers
        visible = STEP_RE.sub("", raw)
        visible = CTX_RE.sub("", visible)
        return visible.strip()
