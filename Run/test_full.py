import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from html import unescape

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELLO = "gemma4:31b-cloud"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IL SUPERBENCHMARK COMPLETO (5 Domini di test)
FILES_TO_TEST = [
    "50_Esempi_LegalBench.json", 
    "Tutto_MULTIHOP.json", 
    "Tutto_CRAG_DAVE.json",
    "Tutto_MTRAG_DAVE.json",
    "Tutto_OHRBench_DAVE.json"
]
OUTPUT_MD = os.path.join(BASE_DIR, "VALUTAZIONE_COMPLETA_DAVE.md")
RETRIEVAL_TOP_K = 3
CONTEXT_DIR = os.path.join(BASE_DIR, "report_artifacts", "contexts")

# --- LEGENDA METRICHE ---
LEGENDA_METRICHE = """
### 📚 Obiettivi del SuperBenchmark e Legenda Metriche
Questo report valuta l'assistente legale DAVE sottoponendolo a 5 stress-test differenti:
1. **LegalBench:** Estrazione di precisione da contratti reali (Assenza di allucinazioni).
2. **MultiHop-RAG:** Capacità logica di unire i puntini tra documenti diversi.
3. **CRAG:** Resistenza alle false premesse e domande a trabocchetto.
4. **MTRAG:** Mantenimento della memoria (Coreference) in una conversazione multi-turno.
5. **OHRBench:** Robustezza ai *Cascading Errors* generati da documenti scansionati male (Semantic & Formatting OCR Noise).

**Metriche (Scala 0-1):**
- **Retrieval Score:** I documenti scelti contenevano le informazioni utili?
- **Faithfulness (Fedeltà):** DAVE ha inventato cose o si è attenuto ai testi?
- **Correctness & F1-Score:** La risposta di DAVE coincide con la Ground Truth?
---
"""

PROMPT_JUDGE = """Agisci come un Giudice di sistemi AI. Analizza questi elementi:
1. DOMANDA (inclusa eventuale storia pregressa): {domanda}
2. DOCUMENTI RECUPERATI (N.B. Potrebbero contenere errori di formattazione OCR o rumore): {contesto}
3. RISPOSTA DI DAVE: {risposta}
4. RISPOSTA CORRETTA (GOLD): {gold}

Valuta con un voto da 0.0 a 1.0 (usa i decimali, es. 0.8) e spiega brevemente:
- retrieval_voto: i documenti forniti contengono le informazioni utili per rispondere?
- faithfulness_voto: DAVE si è basato SOLO sui documenti o ha inventato dati?
- correctness_voto: La risposta di DAVE è corretta rispetto alla GOLD?

REGOLE TASSATIVE: Restituisci ESCLUSIVAMENTE un oggetto JSON valido. Niente testo prima o dopo.
{{
  "retrieval_voto": 0.0,
  "retrieval_perche": "...",
  "faithfulness_voto": 0.0,
  "faithfulness_perche": "...",
  "correctness_voto": 0.0,
  "correctness_perche": "..."
}}
"""

def invoca_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": MODELLO, "prompt": prompt, "stream": False, "temperature": 0},
            timeout=120,
        )
    except requests.RequestException as exc:
        print(f"[OLLAMA ERROR] Richiesta fallita: {exc}")
        return ""

    try:
        payload = r.json()
    except ValueError:
        print(f"[OLLAMA ERROR] Risposta non JSON (HTTP {r.status_code}): {r.text[:200]}")
        return ""

    if r.status_code != 200:
        err_msg = payload.get("error") or payload.get("message") or str(payload)
        print(f"[OLLAMA ERROR] HTTP {r.status_code}: {err_msg}")
        return ""

    response = payload.get("response")
    if response is not None:
        return response

    err_msg = payload.get("error") or payload.get("message") or str(payload)
    print(f"[OLLAMA ERROR] Campo 'response' mancante: {err_msg}")
    return ""

# Funzione per pulire il testo e estrarre solo il JSON valido restituito dal LLM, evitando eventuali errori di formattazione o testo extra.
def pulisci_json(testo_grezzo: str) -> str:
    match = re.search(r'\{.*\}', testo_grezzo, re.DOTALL)
    if match:
        return match.group(0)
    return testo_grezzo

# Funzioni per il Retrieval e la valutazione testuale
def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+", text.lower())

# Le funzioni seguenti aiutano a pulire il testo HTML, gestire i file di contesto, e valutare la risposta di DAVE rispetto alla GOLD.
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html_text(text: str) -> str:
    if not text or ("<" not in text and "&" not in text):
        return text
    cleaned = _HTML_SCRIPT_STYLE_RE.sub(" ", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "").strip("_")
    return safe or "item"


def _save_artifact(dir_path: str, item_id: str, suffix: str, content: str) -> str:
    if content is None:
        return ""
    _ensure_dir(dir_path)
    safe_id = _sanitize_filename(item_id)
    file_name = f"{safe_id}_{suffix}"
    path = os.path.join(dir_path, file_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return os.path.relpath(path, BASE_DIR)


def _should_save_html_context(source: str) -> bool:
    return str(source or "").strip().upper() == "CRAG"


def _split_context_in_docs(context):
    if isinstance(context, list):
        docs = [str(c).strip() for c in context if str(c).strip()]
        return [_clean_html_text(doc) for doc in docs]
    text = str(context or "").strip()
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [text]

    docs = []
    current = []
    for line in lines:
        is_new_doc = line.startswith("[") and "]" in line
        if is_new_doc and current:
            docs.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        docs.append(" ".join(current).strip())
    return [_clean_html_text(doc) for doc in docs]

# Funzione per calcolare Precision, Recall e F1-score a livello di token
def _retrieve_top_k(question: str, docs: list[str], k: int) -> list[str]:
    if not docs:
        return []

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return docs[:k]

    scored = []
    for doc in docs:
        d_tokens = set(_tokenize(doc))
        score = len(q_tokens & d_tokens)
        scored.append((score, len(doc), doc))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_docs = [doc for _, _, doc in scored[:k]]
    return top_docs


def precision_recall_f1(gold, pred):
    gold_tokens = _tokenize(gold)
    pred_tokens = _tokenize(pred)
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    cg = Counter(gold_tokens)
    cp = Counter(pred_tokens)
    common = sum((cg & cp).values())
    precision = common / sum(cp.values()) if sum(cp.values()) > 0 else 0.0
    recall = common / sum(cg.values()) if sum(cg.values()) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def _build_prompt_with_history(item, memoria_conversazioni):
    domanda_pura = item["question"]
    if item["source"] != "MTRAG":
        return domanda_pura, None

    conv_id = item["id"].split("_t")[0]
    storia_passata = memoria_conversazioni.get(conv_id, "")
    if not storia_passata:
        return domanda_pura, conv_id

    prompt = (
        "[STORIA DELLA CONVERSAZIONE]\n"
        f"{storia_passata}\n"
        "[NUOVA DOMANDA DELL'UTENTE]\n"
        f"{domanda_pura}"
    )
    return prompt, conv_id


def _update_conversation_memory(memoria_conversazioni, conv_id, domanda, risposta):
    if not conv_id:
        return
    memoria_aggiornata = memoria_conversazioni.get(conv_id, "")
    memoria_aggiornata += f"Utente: {domanda}\nAssistente: {risposta}\n"
    memoria_conversazioni[conv_id] = memoria_aggiornata


def _format_retrieved_docs(docs: list[str]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, start=1):
        chunks.append(f"[Doc {i}]\n{doc}")
    return "\n\n".join(chunks)

def valuta_sistema():
    print("🔍 Avvio Valutazione Integrale RAG e caricamento SuperBenchmark...")
    dataset = []
    
    for fn in FILES_TO_TEST:
        path = os.path.join(BASE_DIR, fn)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                items = json.load(f)
                
                if "MTRAG" in fn:
                    sample_size = min(6, len(items))
                    dataset.extend(items[:sample_size])
                else:
                    sample_size = min(3, len(items))
                    dataset.extend(random.sample(items, sample_size))

    memoria_conversazioni = {}

    report_content = "# 🛡️ Valutazione Investigativa DAVE RAG\n"
    report_content += f"**Generato il:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    report_content += LEGENDA_METRICHE + "\n"

    for item in dataset:
        print(f"Analizzando {item['id']} ({item['source']})...")
        contesto = item['context']
        gold = item['gold_answer']
        prompt_domanda_finale, conv_id = _build_prompt_with_history(item, memoria_conversazioni)
        
        # Retrieval interno sul contesto pre-caricato
        docs = _split_context_in_docs(contesto)
        retrieved_docs = _retrieve_top_k(prompt_domanda_finale, docs, RETRIEVAL_TOP_K)
        contesto_retrieved = "\n".join(retrieved_docs).strip()
        if not contesto_retrieved:
            contesto_retrieved = str(contesto)

        if _should_save_html_context(item.get("source")):
            _save_artifact(CONTEXT_DIR, item["id"], "context.html", str(contesto))

        # DAVE genera la risposta
        prompt_dave = f"Usa questi documenti:\n{contesto_retrieved}\n\nRispondi alla domanda: {prompt_domanda_finale}"
        risposta_dave = invoca_ollama(prompt_dave)
        _update_conversation_memory(memoria_conversazioni, conv_id, item["question"], risposta_dave)
            
        voto_json_raw = invoca_ollama(PROMPT_JUDGE.format(domanda=prompt_domanda_finale, contesto=contesto_retrieved, risposta=risposta_dave, gold=gold))
        json_pulito = pulisci_json(voto_json_raw)
        
        try:
            voti = json.loads(json_pulito)
        except json.JSONDecodeError:
            print(f"\n[ERRORE PARSING LLM] per la query {item['id']}\n")
            voti = {
                "retrieval_voto": 0.0, "retrieval_perche": "Errore parsing LLM", 
                "faithfulness_voto": 0.0, "faithfulness_perche": "N/A", 
                "correctness_voto": 0.0, "correctness_perche": "N/A"
            }

        precision, recall, f1 = precision_recall_f1(gold, risposta_dave)

        report_content += f"## 📝 Query `{item['id']}` (Sorgente: *{item['source']}*)\n"
        
        if item['source'] == 'MTRAG' and conv_id and memoria_conversazioni.get(conv_id, "").count("Utente:") > 1:
            report_content += f"> 🗣️ **Test Conversazionale:** *Questa domanda include la storia pregressa costruita auto-regressivamente con le risposte reali di DAVE, testando il rischio di Cascading Errors.*\n\n"
        elif item['source'] == 'OHRBench':
            report_content += f"> 🖨️ **Test Robustezza OCR:** *Il documento fornito contiene 'Rumore Semantico o di Formattazione'. Valutiamo la capacità di DAVE di districarsi tra gli errori del parsing visivo (es. tabelle distrutte o parole storpiate).*\n\n"
            
        report_content += f"**Domanda dell'Utente:**\n```text\n{prompt_domanda_finale}\n```\n\n"
        
        report_content += "### 📂 Documenti Utilizzati (Retrieval)\n"
        report_content += f"_Doc totali nel contesto: {len(docs)} | usati: {len(retrieved_docs)}_\n\n"
        formatted_docs = _format_retrieved_docs(retrieved_docs)
        if formatted_docs:
            report_content += f"```text\n{formatted_docs}\n```\n\n"
        else:
            report_content += "Nessun documento recuperato.\n\n"
        report_content += f"**Voto Retrieval:** `{voti['retrieval_voto']}`\n"
        report_content += f"**Motivazione:** {voti['retrieval_perche']}\n\n"

        report_content += "### 🤖 Risposta di DAVE vs Gold Answer\n"
        report_content += f"**DAVE:** {risposta_dave}\n\n"
        report_content += f"**GOLD:** {gold}\n\n"
        report_content += "### 🔢 Metriche Testuali (Token-level)\n"
        report_content += f"- Precision: `{precision:.3f}` | Recall: `{recall:.3f}` | F1-score: `{f1:.3f}`\n\n"
        
        report_content += "| Metrica | Voto | Spiegazione del Giudice |\n"
        report_content += "| :--- | :---: | :--- |\n"
        report_content += f"| **Fedeltà (Faithfulness)** | {voti['faithfulness_voto']} | {voti['faithfulness_perche']} |\n"
        report_content += f"| **Correttezza** | {voti['correctness_voto']} | {voti['correctness_perche']} |\n"
        report_content += "\n---\n"

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n🎉 VALUTAZIONE COMPLETATA CON SUCCESSO! 🎉")
    print(f"Il referto per la tua tesi è stato salvato in: {OUTPUT_MD}")

if __name__ == "__main__":
    valuta_sistema()