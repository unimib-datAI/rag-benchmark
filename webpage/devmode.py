"""Secret dev mode, temporary: to remove it delete this file and the lines marked `# DEV MODE` in app.py.

Pressing R+A+G together reveals a "Dev mode" switch at the top of the sidebar (pressing it again only hides the switch: the mode keeps being on or off). While the switch is on
the DAVE settings and the Setup > DAVE tab are only hidden (they keep running and saving as usual),
and the retrieval type is read from the URL (the sidebar radio is hidden):
    http://<host>/dense, /fulltext, /hybrid        or        http://<host>/?retrieval=dense
"""
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components

# URL value -> DAVE retrieval method (keys of settings.DAVE_RETRIEVAL_METHODS)
URL_RETRIEVAL = {"dense": "dense", "fulltext": "full-text", "full-text": "full-text",
                 "sparse": "full-text", "hybrid": "full"}

_ALWAYS_CSS = ".st-key-dev_unlock, .st-key-dev_js {display: none;}"
_DEV_CSS = """
.st-key-engine_select, .st-key-w_dave_retrieval_method, .st-key-dave_setup_hidden {display: none;}
"""

_KEY_LISTENER = """
<script>
const parent = window.parent;
if (!parent.__ragDevKeys) {
  parent.__ragDevKeys = true;
  const down = new Set();
  const doc = parent.document;
  doc.addEventListener('keydown', e => {
    const t = e.target;
    // typing in a text field must not count, but a focused switch or checkbox (the Dev mode toggle) must
    const typing = t.tagName === 'TEXTAREA' || t.isContentEditable ||
      (t.tagName === 'INPUT' && !/^(checkbox|radio|button|submit|range)$/.test(t.type));
    if (typing) return;
    down.add(e.key.toLowerCase());
    if (down.has('r') && down.has('a') && down.has('g')) {
      down.clear();
      const button = doc.querySelector('.st-key-dev_unlock button');
      if (button) button.click();
    }
  });
  doc.addEventListener('keyup', e => down.delete(e.key.toLowerCase()));
  parent.addEventListener('blur', () => down.clear());
}
</script>
"""


def sidebar_switch():
    """Key listener plus, once R+A+G was pressed, the switch. Call at the top of the sidebar."""
    st.markdown(f"<style>{_ALWAYS_CSS}</style>", unsafe_allow_html=True)
    if st.button("dev-unlock", key="dev_unlock"):  # R+A+G again hides the switch, the mode keeps its value
        st.session_state.dev_unlocked = not st.session_state.get("dev_unlocked", False)
    if st.session_state.get("dev_unlocked"):
        # the value lives in dev_enabled: a widget's own state is dropped while the widget is not drawn
        st.session_state.dev_enabled = st.toggle("Dev mode", value=st.session_state.get("dev_enabled", False),
                                                 key="dev_mode")
    with st.container(key="dev_js"):
        components.html(_KEY_LISTENER, height=0)
    if active():
        st.markdown(f"<style>{_DEV_CSS}</style>", unsafe_allow_html=True)


def active():
    return bool(st.session_state.get("dev_enabled"))


def retrieval_from_url():
    """DAVE retrieval method requested by the URL (last path segment or ?retrieval=), else None."""
    value = st.query_params.get("retrieval")
    if not value:
        try:
            value = urlparse(st.context.url).path.rstrip("/").rsplit("/", 1)[-1]
        except Exception:
            value = ""
    return URL_RETRIEVAL.get(str(value).lower())


def force_engine(engine, key="engine_select"):
    """While dev mode is on the evaluation always runs on `engine` (DAVE) even though the UI only shows
    the generic RAG; the engine chosen before is restored when the mode is switched off."""
    if active():
        if "dev_prev_engine" not in st.session_state:
            st.session_state.dev_prev_engine = st.session_state.get(key)
        st.session_state[key] = engine
    elif "dev_prev_engine" in st.session_state:
        previous = st.session_state.pop("dev_prev_engine")
        if previous:
            st.session_state[key] = previous


# Catalogue placeholders: (name, description, count). They only look downloaded, nothing is fetched.
# A placeholder whose name starts with the name of a real benchmark (BENCHMARK_SHORT) is skipped.
PLACEHOLDER_BENCHMARKS = [
    ("RGB (Noise & Rejection)", "Notizie recenti in inglese e cinese, con documenti recuperati da motori di ricerca. Quattro testbed: noise robustness, negative rejection, information integration e counterfactual robustness.", ""),
    ("RECALL (Counterfactual Knowledge)", "Costruito da EventKG (conoscenza di senso comune) e da un dataset di conoscenza scientifica, inserendo informazioni controfattuali con ChatGPT. Due task: QA e generazione di testo.", ""),
    ("RAGTruth (Hallucination Corpus)", "Risposte generate da vari LLM con RAG, annotate a mano a livello di risposta e di parola, su QA, data-to-text e summarization. Serve a studiare dove nascono le allucinazioni.", "~18,000 risposte"),
    ("ReEval (Adversarial Test Generation)", "Non è un dataset fisso ma un framework che perturba l’evidenza di dataset QA open-domain esistenti, con prompt chaining su ChatGPT, per generare nuovi casi che inducono allucinazioni.", ""),
    ("MultiHop-RAG (Multi-Document Reasoning)", "Articoli di news in inglese (609). Ogni query richiede 2-4 evidenze (inference, comparison, temporal, null).", "2,556"),
    ("MEBench (Multi-Entity Reasoning)", "Domande cross-documento su molte entità (media 409 per query): ragionamento comparativo, statistico e relazionale, in 8 tipi di task.", "4,780"),
    ("FollowRAG / VIF-RAG (Instruction Constraints)", "Verifica se il sistema rispetta vincoli di istruzione (22 categorie, es. formato e lunghezza) su 4 dataset QA knowledge-intensive. Il dataset di training associato (VIF-RAG-QA) supera i 100k campioni.", "~3,000 (test)"),
    ("LegalBench-RAG (Contract Precision)", "Contratti e documenti legali reali (714 documenti da CUAD, MAUD, ContractNLI, PrivacyQA). Verifica l’estrazione di clausole precise.", "6,858"),
    ("MIRAGE / MedRAG (Medical QA)", "Domande mediche a scelta multipla da 5 dataset (MMLU-Med, MedQA-US, MedMCQA, PubMedQA*, BioASQ-Y/N). MedRAG è il toolkit associato con corpora e retriever.", "7,663"),
    ("CRAG (Dynamic Facts & Noise)", "Q&A fattuale su 5 domini e 8 tipi di domanda, con pagine HTML rumorose e API mock. Verifica fatti che cambiano nel tempo, premesse false e negative rejection.", "4,409"),
    ("OmniEval (Financial Matrix)", "Dominio finanziario in cinese, organizzato in una matrice di 5 classi di task × 16 topic. Ha una parte generata con GPT-4 (11.4k esempi) e una annotata a mano (1.7k). Il corpus di retrieval è di circa 362k documenti, ma questo dato viene da una sola fonte secondaria.", "~13,100 (11.4k + 1.7k)"),
    ("FeB4RAG (Federated Search)", "Richieste di informazione in stile chatbot su 16 sotto-collezioni di BEIR, con i risultati di ogni risorsa e giudizi di rilevanza prodotti da un LLM.", "790"),
    ("DomainRAG (University Enrollment)", "Sistema di iscrizioni di un’università cinese (1,686 pagine web, 14,406 passaggi). Sei capacità: QA conversazionale, strutturale, estrattiva, time-sensitive, multi-documento e faithful.", "395"),
    ("CRUD-RAG (Create/Read/Update/Delete)", "Notizie cinesi recenti (oltre 80,000 articoli) con task di continuazione testo, QA a uno o più documenti, correzione di allucinazioni e summarization multi-documento.", ""),
    ("CDQA (Chinese Dynamic QA)", "Domande con risposte che cambiano nel tempo, ricavate da news cinesi. Le categorie sono fast-changing, slow-changing e never-changing.", "1,339"),
    ("FreshLLMs / FreshQA (Fresh Knowledge)", "Domande in 4 categorie (never-, slow-, fast-changing e false-premise). Verifica la conoscenza aggiornata e la capacità di smontare premesse sbagliate.", "600"),
    ("RAGBench (Industry Aggregate)", "Aggregato di 12 dataset esistenti in 5 domini (biomedico, generale, legale, customer support, finanza), con metriche TRACe (utilization, relevance, adherence, completeness).", "100,000"),
    ("OHRBench (OCR Noise)", "Pagine PDF (8,561) da 7 domini. Verifica l’impatto del rumore OCR su testo, tabelle, formule, grafici e ordine di lettura.", "8,498"),
    ("MTRAG (Multi-Turn Conversations)", "110 conversazioni multi-turno scritte da umani (media 7.7 turni) su 4 corpora (ClapNQ, FiQA, Govt, Cloud). Verifica co-referenze e domande non standalone o senza risposta.", "842"),
    ("U-NIAH (Needle-in-a-Haystack)", "Framework che confronta LLM e RAG su contesti lunghi, con un corpus sintetico ambientato in un universo fittizio. Ha configurazioni multi-needle, long-needle e nested-needle.", ""),
]


def catalogue_placeholders(real_prefixes):
    """Placeholder entries (name, description, count) to add to the Catalogue while dev mode is on,
    without the ones that already exist as real benchmarks."""
    if not active():
        return []
    return [entry for entry in PLACEHOLDER_BENCHMARKS if not entry[0].startswith(tuple(real_prefixes))]


def mark_downloaded(names):
    """The "download" of a placeholder only remembers it for this session: nothing is fetched."""
    st.session_state.setdefault("dev_fake_downloaded", set()).update(names)


def is_marked_downloaded(name):
    return name in st.session_state.get("dev_fake_downloaded", ())
