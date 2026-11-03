"""Error taxonomy: per-query verdict, error pattern and aggregated diagnosis.

Uses only the RAGAS scores already computed (no LLM calls) and has no heavy
dependencies, so it is shared by evaluator, dashboard, CLI and tests.
Bibliographic references for each pattern: riferimenti_tassonomia_errori.txt
"""
from collections import Counter

# bump when rules, thresholds or pattern ids change: saved evaluations
# with a different version are reclassified when loaded
TAXONOMY_VERSION = 7

THRESHOLDS = {
    "correct": 0.70,  # Answer Correctness >= correct -> CORRECT
    "partial": 0.40,  # Answer Correctness >= partial -> PARTIAL, below -> REFUSAL or WRONG
    "metric": 0.50,   # high/low split for Precision, Recall, Faithfulness, Relevancy
}

# report key -> RAGAS metric column
METRIC_KEYS = {
    "retrieval_voto": "context_precision",
    "faithfulness_voto": "faithfulness",
    "correctness_voto": "answer_correctness",
    "relevancy_voto": "answer_relevancy",
    "recall_voto": "context_recall",
}

REFUSAL_PATTERNS = [
    "non posso", "non ho informazioni", "non sono in grado",
    "non ho abbastanza", "non è possibile determinare",
    "non è sufficiente", "non dispongo", "non ho trovato informazioni", "non lo so",
    "i cannot", "i don't have", "i don't know", "i'm unable", "insufficient information",
    "not enough information", "cannot determine", "unable to answer",
    "no information", "not sufficient", "information provided is not sufficient",
]
# refusals are looked for only at the start of the answer, so a long answer
# that mentions "no information" in passing is not taken as a refusal
REFUSAL_WINDOW = 200

# pattern id -> (label, explanation and suggested action)
PATTERNS = {
    "CORRECT": ("Correct Answer", "The RAG answer is substantially correct."),
    "PARTIAL": ("Partial Answer", "The answer captures some information but is incomplete."),
    "REFUSAL": ("Refusal / Non-Answer", "The model refused to answer or stated it lacks sufficient information."),
    "TOTAL_FAIL": ("System Collapse",
                   "Both retrieval and generation failed: the retriever provided irrelevant documents and the model "
                   "generated unfaithful content. Review the entire pipeline: embeddings, chunking strategy and LLM prompt."),
    "RET_FAIL": ("Retrieval Failure",
                 "The retriever did not find the relevant documents, so the answer cannot be correct. "
                 "Consider improving embeddings, chunk size, or adding hybrid search."),
    "RET_INCOMPLETE": ("Incomplete Retrieval",
                       "Retrieved documents are relevant but key information is missing. "
                       "Try increasing top_k or adjusting the similarity threshold."),
    "RET_NOISE": ("Retrieval Noise",
                  "The needed information was retrieved but mixed with many irrelevant documents that misled the model. "
                  "Consider a re-ranking step or a stricter similarity threshold."),
    "HALLUC": ("Hallucination",
               "The model had access to relevant context but generated content not grounded in the documents. "
               "Review the system prompt and consider lowering the temperature."),
    "NOT_EXTRACTED": ("Context Not Extracted",
                      "The context contains the right information and the model is faithful to it, but it failed to "
                      "extract the answer (e.g. \"lost in the middle\" with long contexts). "
                      "Consider reducing the number of documents or using a re-ranking step."),
    "OFF_TOPIC": ("Off-Topic Answer",
                  "The context is right and the model is faithful to it, but the answer does not address the question. "
                  "Review the system prompt and how the question is passed to the LLM."),
    "UNKNOWN": ("Unclassified Error",
                "The cause cannot be determined because some diagnostic metrics are unavailable."),
    "EVAL_FAIL": ("Evaluation Failure",
                  "RAGAS could not compute Answer Correctness (judge error, timeout or metric not selected). "
                  "This is a failure of the evaluator, not of the RAG."),
    "RAG_ERROR": ("RAG Error",
                  "The RAG under test did not answer (HTTP error, timeout, empty answer or exception): "
                  "the query is an infrastructure failure, not an answer, and is excluded from RAGAS and the averages. "
                  "Check that the RAG server and its LLM are running."),
}

_VERDICT_LABELS = {
    "correct": "✅ Correct",
    "partial": "⚠️ Partial",
    "refusal": "🟣 Refusal",
    "wrong": "❌ Wrong",
    "not_evaluated": "⚪ Not Evaluated",
}


# ── metric availability ───────────────────────────────────────────

def invalid_metrics(report: dict) -> list:
    """Report keys of the metrics RAGAS did not compute (NaN or not selected)."""
    invalid = report.get("metriche_non_valide")
    if invalid is None:
        # old evaluations: no per-metric info, only the global failure flag, and
        # NaN was saved as 0.0 -> in a failed query an exact 0.0 is taken as not computed
        if not report.get("ragas_fallita"):
            return []
        return [k for k in METRIC_KEYS if not report.get(k)]
    return invalid

def metric_value(report: dict, key: str):
    """Metric score, or None when it was not computed (instead of a fake 0.0)."""
    if key not in report or key in invalid_metrics(report):
        return None
    return float(report.get(key) or 0)

def metric_mean(reports: list, key: str):
    """Average of a metric over the queries where it was computed, None if none."""
    values = [v for v in (metric_value(r, key) for r in reports) if v is not None]
    return sum(values) / len(values) if values else None


# ── classification ────────────────────────────────────────────────

def _low(value):
    return value is not None and value < THRESHOLDS["metric"]

def _high(value):
    return value is not None and value >= THRESHOLDS["metric"]

def is_refusal(answer: str) -> bool:
    # LLMs often write the typographic apostrophe: "I don’t know"
    head = str(answer).strip().lower().replace("’", "'")[:REFUSAL_WINDOW]
    return any(p in head for p in REFUSAL_PATTERNS)

def _retrieval_state(prec, recall):
    """FAIL / INCOMPLETE / NOISE / OK from Precision and Recall, None if one is missing."""
    if prec is None or recall is None:
        return None
    return {
        (False, False): "FAIL",        # nothing useful retrieved
        (True, False): "INCOMPLETE",   # relevant but key passages missing
        (False, True): "NOISE",        # needed passages buried in irrelevant ones
        (True, True): "OK",
    }[(_high(prec), _high(recall))]

def diagnose(prec, recall, faith, relevancy):
    """Root cause of a non-correct answer, as a decision table on the metrics.

    retrieval | faithfulness | cause
    ----------+--------------+----------------------------------------------
    FAIL      | low          | TOTAL_FAIL
    FAIL      | high / n.a.  | RET_FAIL
    not FAIL  | low          | HALLUC
    INCOMPL.  | high         | RET_INCOMPLETE
    NOISE     | high         | RET_NOISE
    OK        | high         | OFF_TOPIC if Relevancy low, else NOT_EXTRACTED
    anything else (missing metrics)  -> UNKNOWN
    """
    ret = _retrieval_state(prec, recall)
    if ret == "FAIL":
        return "TOTAL_FAIL" if _low(faith) else "RET_FAIL"
    if _low(faith) and (_high(prec) or _high(recall)):
        return "HALLUC"
    if not _high(faith):
        return "UNKNOWN"
    if ret == "INCOMPLETE":
        return "RET_INCOMPLETE"
    if ret == "NOISE":
        return "RET_NOISE"
    if ret == "OK":
        return "OFF_TOPIC" if _low(relevancy) else "NOT_EXTRACTED"
    return "UNKNOWN"

def _refusal_cause(prec, recall):
    """Why the model refused: Faithfulness is meaningless for a refusal, so only retrieval is used."""
    ret = _retrieval_state(prec, recall)
    if ret is None:
        return "UNKNOWN"
    # context was sufficient: the model refused although it had the answer
    return {"FAIL": "RET_FAIL", "INCOMPLETE": "RET_INCOMPLETE", "NOISE": "RET_NOISE", "OK": "NOT_EXTRACTED"}[ret]

def classify_query_verdict(report: dict) -> dict:
    """Classify a single query result with a verdict, an error pattern and its root cause.

    Returns a dict with:
      - verdict: 'correct' | 'partial' | 'refusal' | 'wrong' | 'not_evaluated'
        (also for RAG_ERROR: the RAG did not answer, report field 'rag_error')
      - verdict_label: human-readable label with emoji
      - error_pattern: pattern ID (e.g. 'RET_FAIL'); PARTIAL / REFUSAL for those answers
      - error_pattern_label: human-readable pattern name
      - error_cause: root-cause pattern ID (None for correct / not evaluated answers);
        for PARTIAL and REFUSAL it tells why the answer is incomplete or refused
      - error_cause_label: human-readable cause name
      - error_reason: explanation string
      - taxonomy_version: TAXONOMY_VERSION
    """
    corr = metric_value(report, "correctness_voto")
    prec = metric_value(report, "retrieval_voto")
    recall = metric_value(report, "recall_voto")
    faith = metric_value(report, "faithfulness_voto")
    relevancy = metric_value(report, "relevancy_voto")

    if report.get("rag_error"):
        # the RAG did not answer: nothing to judge, the metrics were never computed
        return {
            "verdict": "not_evaluated",
            "verdict_label": _VERDICT_LABELS["not_evaluated"],
            "error_pattern": "RAG_ERROR",
            "error_pattern_label": PATTERNS["RAG_ERROR"][0],
            "error_cause": None,
            "error_cause_label": None,
            "error_reason": f"{PATTERNS['RAG_ERROR'][1]} Error: {report['rag_error']}",
            "taxonomy_version": TAXONOMY_VERSION,
        }
    if corr is None:
        # without Correctness no verdict can be given
        verdict, pattern, cause = "not_evaluated", "EVAL_FAIL", None
    elif corr >= THRESHOLDS["correct"]:
        verdict, pattern, cause = "correct", "CORRECT", None
    elif corr >= THRESHOLDS["partial"]:
        verdict, pattern, cause = "partial", "PARTIAL", diagnose(prec, recall, faith, relevancy)
    elif is_refusal(report.get("risposta_rag", "")):
        verdict, pattern, cause = "refusal", "REFUSAL", _refusal_cause(prec, recall)
    else:
        verdict, pattern = "wrong", diagnose(prec, recall, faith, relevancy)
        cause = pattern

    reason = PATTERNS[pattern][1]
    if cause and cause != pattern:
        reason += f" Likely cause — {PATTERNS[cause][0]}: {PATTERNS[cause][1]}"
    # the scores are not repeated here: the verdict card shows them next to the reason

    return {
        "verdict": verdict,
        "verdict_label": _VERDICT_LABELS[verdict],
        "error_pattern": pattern,
        "error_pattern_label": PATTERNS[pattern][0],
        "error_cause": cause,
        "error_cause_label": PATTERNS[cause][0] if cause else None,
        "error_reason": reason,
        "taxonomy_version": TAXONOMY_VERSION,
    }

def needs_reclassification(report: dict) -> bool:
    return "correctness_voto" in report and report.get("taxonomy_version") != TAXONOMY_VERSION


def aggregate_diagnosis(reports: list, min_share: float = 0.10) -> dict:
    """Dominant failure causes across all queries, built from the per-query classification.

    Same rules as the per-query verdicts, so the global diagnosis can never
    contradict them. Returns:
      - evaluated: queries with a verdict (EVAL_FAIL and RAG_ERROR excluded)
      - correct: number of correct answers
      - causes: [(pattern_id, count, share_of_evaluated)] for causes above min_share,
        most frequent first; partial and refused answers count under their root cause
    """
    evaluated = [r for r in reports if r.get("verdict") != "not_evaluated"]
    correct = sum(1 for r in evaluated if r.get("verdict") == "correct")
    counts = Counter(r.get("error_cause") or r.get("error_pattern", "UNKNOWN")
                     for r in evaluated if r.get("verdict") in ("partial", "refusal", "wrong"))
    n = len(evaluated)
    causes = [(p, c, c / n) for p, c in counts.most_common() if n and c / n >= min_share]
    return {"evaluated": n, "correct": correct, "causes": causes}

def diagnosis_messages(reports: list) -> list:
    """Aggregated diagnosis as (title, text) pairs, shared by dashboard and CLI."""
    diag = aggregate_diagnosis(reports)
    n = diag["evaluated"]
    if n == 0:
        return [("No Evaluated Queries", "RAGAS could not compute Answer Correctness for any query.")]
    messages = [(f"{PATTERNS[p][0]} ({p}) — {c}/{n} queries, {share:.0%}", PATTERNS[p][1])
                for p, c, share in diag["causes"]]
    if not messages:
        if diag["correct"] / n >= 0.8:
            messages.append(("Optimized System", f"{diag['correct']}/{n} answers are correct and no failure cause is dominant."))
        else:
            messages.append(("Scattered Errors", "No single failure cause covers at least 10% of the queries: "
                                                 "check the per-query patterns for details."))
    return messages
