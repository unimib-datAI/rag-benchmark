import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Forza UTF-8 per l'output su console Windows se necessario
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

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
Make sure to ALWAYS answer in the same language used by the user to ask the question, don't mind the documents language
</instructions>

</input>"""

STEP_RE = re.compile(r"\x02DAVE_STEP\x1E([\s\S]*?)\x03")
CTX_RE = re.compile(r"\x02DAVE_CTX\x1E([\s\S]*?)\x03")

# Limite caratteri per il contesto nel system prompt.
# 500k chars ~166k token, lascia margine per template + risposta sotto il limite 262k.
MAX_PROMPT_CONTEXT_CHARS = 500_000


def _tronca_testo(testo: str, max_chars: int) -> str:
    """Tronca il testo rispettando i confini di frase (. ! ?) o, in subordine,
    i confini di parola. Non taglia mai a metà parola."""
    if len(testo) <= max_chars:
        return testo
    candidato = testo[:max_chars]
    # Non tornare indietro più del 30% per cercare una fine di frase
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


class DaveEmptyAnswerError(RuntimeError):
    """DAVE ha chiuso lo stream di generazione senza testo (es. timeout del suo LLM con HTTP 200)."""


class DaveClient:
    """Client per interagire con l'istanza DAVE (Retrieval & Generation via tRPC / Next.js API)."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        self.username = username or os.environ.get("DAVE_USERNAME")
        self.password = password or os.environ.get("DAVE_PASSWORD")
        self.base_url = base_url or os.environ.get("DAVE_BASE_URL", NEXTJS_BASE_URL)
        self.base_path = base_path or os.environ.get("DAVE_BASE_PATH", NEXTJS_BASE_PATH)

        if not self.username or not self.password:
            raise RuntimeError(
                "Credenziali DAVE mancanti. Impostale nel file .env (DAVE_USERNAME, DAVE_PASSWORD)"
            )
        self.token = None
        self._login()

    def _login(self):
        url = f"{self.base_url}{self.base_path}/api/auth/keycloak-login"
        resp = requests.post(
            url,
            json={"username": self.username, "password": self.password},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"DAVE Login fallito ({resp.status_code}): {resp.text}")
        self.token = resp.json()["access_token"]

    def _trpc_mutation(self, procedure: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{self.base_path}/api/trpc/{procedure}"
        if self.token:
            payload["token"] = self.token
        response = requests.post(url, json=payload, timeout=120)
        body = response.json()
        if "error" in body:
            raise TRPCError(f"{procedure} fallito: {body['error'].get('message')}")
        return body["result"]["data"]

    def _trpc_query(self, procedure: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{self.base_path}/api/trpc/{procedure}"
        if self.token:
            payload["token"] = self.token
        response = requests.get(url, params={"input": json.dumps(payload)}, timeout=60)
        body = response.json()
        if "error" in body:
            raise TRPCError(f"{procedure} fallito: {body['error'].get('message')}")
        return body["result"]["data"]

    # --- INGESTION APIs ---

    def create_collection(self, name: str) -> str:
        """Crea una nuova collezione su DAVE e restituisce il suo UUID."""
        res = self._trpc_mutation("collection.create", {"name": name, "allowedUserIds": []})
        return res["id"]

    def create_txt_upload_job(
        self, collection_id: str, files_data: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Avvia un job di caricamento asincrono per una lista di file TXT."""
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
        """Verifica lo stato di avanzamento di un job di caricamento."""
        return self._trpc_query("document.getUploadJob", {"jobId": job_id})

    # --- RETRIEVAL & GENERATION APIs ---

    def retrieve_chunks(
        self, query: str, collection_id: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Esegue il retrieval su DAVE e restituisce al massimo top_k chunk, nell'ordine di DAVE.

        search.mostSimilarDocuments non accetta un parametro per il numero di risultati
        (topK, k, limit, size, top_k, numResults, maxResults, n vengono ignorati: verificato
        sul server) e restituisce i chunk raggruppati per documento. Qui i chunk vengono
        appiattiti, un elemento per chunk ({"id", "title", "chunks": [chunk]}), e tagliati a
        top_k, così l'unità di contesto e il top_k coincidono con quelli del RAG locale."""
        docs = self._trpc_mutation(
            "search.mostSimilarDocuments",
            {
                "query": query,
                "collectionId": collection_id,
                "retrievalMethod": "full",
            },
        ) or []
        chunks: List[Dict[str, Any]] = []
        for doc in docs:
            for chunk in doc.get("chunks", []):
                chunks.append({
                    "id": chunk.get("id", doc.get("id")),
                    "title": doc.get("title", "Doc"),
                    # text_emb (embedding del chunk) non serve: lo si scarta
                    "chunks": [{"id": chunk.get("id"), "text": chunk.get("text", "")}],
                })
        return chunks[:top_k]

    def build_system_prompt(
        self, docs: List[Dict[str, Any]], question: str, max_docs: Optional[int] = None
    ) -> str:
        """Costruisce il system prompt inserendo i frammenti dei documenti recuperati.
        Con max_docs=None usa tutti i documenti passati: chi chiama passa esattamente
        i documenti da usare (gli stessi che poi valuta RAGAS)."""
        context_parts = []
        chunk_num = 0
        for i, doc in enumerate(docs if max_docs is None else docs[:max_docs]):
            lines = []
            for chunk in doc.get("chunks", []):
                chunk_num += 1
                lines.append(f"[{chunk_num}] {chunk.get('text', '')}")
            doc_content = f"Nome Documento {doc.get('title', 'Doc')} - Contenuto:\n" + "\n".join(lines)
            context_parts.append(
                f'<document id="DOC_{i + 1}" name="{doc.get("title", "Doc")}">\n{doc_content}\n</document>'
            )
        context_str = "\n".join(context_parts)

        # Tronca il contesto se supera il limite per evitare errori di lunghezza prompt
        if len(context_str) > MAX_PROMPT_CONTEXT_CHARS:
            logger.warning(
                "Contesto prompt troppo lungo (%d chars, limite %d). Troncamento...",
                len(context_str),
                MAX_PROMPT_CONTEXT_CHARS,
            )
            context_str = _tronca_testo(context_str, MAX_PROMPT_CONTEXT_CHARS)

        return DEFAULT_SYSTEM_PROMPT.replace("{{CONTEXT}}", context_str).replace(
            "{{QUESTION}}", question
        )

    def generate_answer(
        self, system_prompt: str, question: str, collection_id: Optional[str] = None
    ) -> str:
        """Chiama l'endpoint di generazione di DAVE in streaming e restituisce la risposta pulita."""
        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        if collection_id:
            payload["collectionId"] = collection_id

        url = f"{self.base_url}{self.base_path}/api/generate"
        raw = ""
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    raw += chunk

        # Pulizia dei marker interni di DAVE
        visible = STEP_RE.sub("", raw)
        visible = CTX_RE.sub("", visible).strip()
        # DAVE risponde 200 e chiude lo stream vuoto quando il suo LLM va in timeout:
        # non è una risposta da valutare ma un errore dell'infrastruttura
        if not visible:
            raise DaveEmptyAnswerError(
                f"DAVE ha restituito una risposta vuota ({len(raw)} caratteri grezzi nello stream)"
            )
        return visible


def main():
    """Test rapido di connettività e autenticazione con il server DAVE."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== DAVE Client - Test di Connessione ===")
    try:
        client = DaveClient()
        print("[OK] Autenticazione con DAVE completata con successo!")
        print(f"     Server base: {client.base_url}{client.base_path}")
        print(f"     Utente: {client.username}")

        col_file = Path(__file__).resolve().parent / ".dave_collections.json"
        if col_file.exists():
            with open(col_file, "r", encoding="utf-8") as f:
                cols = json.load(f)
            print(f"     Collezioni mappate in .dave_collections.json ({len(cols)}):")
            for name, cid in cols.items():
                print(f"       - {name}: {cid}")
        else:
            print("     [INFO] File .dave_collections.json non trovato.")
    except Exception as e:
        print(f"[ERRORE] Connessione a DAVE fallita: {e}")


if __name__ == "__main__":
    main()
