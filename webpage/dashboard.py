from collections import Counter
from html import escape

from taxonomy import THRESHOLDS, diagnosis_messages

def get_header_html() -> str:
    # top bar with title
    return """
    <div class="header-bar">
        <h1>SuperBenchmark Dashboard</h1>
        <p>Interactive RAG system evaluation with LLM-as-a-Judge</p>
    </div>
    """

def get_welcome_html() -> str:
    # initial message when there are no results yet
    return """
    <div class="welcome-box">
        <h3>No evaluation performed</h3>
        <p>Configure parameters in the sidebar and press "Start Evaluation" to begin.</p>
    </div>
    """

def get_metric_card_html(label: str, val: float, color_class: str) -> str:
    # the CSS color class is passed from app.py (e.g. "green", "red" etc.)
    return f"""
    <div class="metric-card">
        <div class="label">{escape(label)}</div>
        <div class="value {color_class}">{"N/A" if val is None else f"{val:.2f}"}</div>
    </div>"""

def get_breakdown_card_html(src_name: str, score: float, count: int, color_class: str) -> str:
    # card for the per-source breakdown
    # the "c-" prefix on the color class avoids conflicts with the classes above
    return f"""
    <div class="breakdown-card">
        <div class="src-name">{escape(src_name)}</div>
        <div class="src-score c-{color_class}">{"N/A" if score is None else f"{score:.2f}"}</div>
        <div class="src-count">{count} queries</div>
    </div>"""

def get_judge_box_html(metric_label: str, v: float, reason: str, color_class: str) -> str:
    return f"""
    <div class="judge-box">
        <div class="jb-label">{escape(metric_label)}</div>
        <div class="jb-value c-{color_class}">{escape(str(v))}</div>
        <div class="jb-reason">{escape(reason)}</div>
    </div>"""


def get_diagnostics_html(reports: list):
    # built from the per-query patterns, so it can't contradict the verdicts
    html_issues = '<div class="diag-panel issues" style="height: 100%;"><h3>🔍 Advanced Heuristic Analysis</h3>'
    for title, text in diagnosis_messages(reports):
        html_issues += f'<p><b>{escape(title)}:</b> {escape(text)}</p>'
    html_issues += '</div>'

    return html_issues

def get_footer_html(num_reports: int, ollama_model: str, time_str: str, eval_name: str = "") -> str:
    name_prefix = f"{escape(eval_name)}  &middot;  " if eval_name else ""
    return f"""
<div class="footer-bar">
    {name_prefix}SuperBenchmark  &middot;  {num_reports} queries  &middot;  Judge: {escape(ollama_model)}  &middot;  {escape(time_str)}
</div>"""


def get_verdict_html(verdict: str, verdict_label: str, pattern_label: str, reason: str) -> str:
    """Card showing the per-query verdict and error-pattern explanation."""
    color_map = {
        'correct': ('#00E676', 'rgba(0, 230, 118, 0.06)', 'rgba(0, 230, 118, 0.20)'),
        'partial': ('#FFC400', 'rgba(255, 196, 0, 0.06)',  'rgba(255, 196, 0, 0.20)'),
        'wrong':   ('#FF1744', 'rgba(255, 23, 68, 0.06)',  'rgba(255, 23, 68, 0.20)'),
        'not_evaluated': ('#94A3B8', 'rgba(148, 163, 184, 0.06)', 'rgba(148, 163, 184, 0.20)'),
    }
    color, bg, border = color_map.get(verdict, color_map['wrong'])
    pattern_html = f' — {escape(pattern_label)}' if verdict != 'correct' else ''

    return f"""
    <div class="verdict-card" style="background:{bg}; border:1px solid {border}; border-left:4px solid {color};">
        <div class="verdict-header">
            <span class="verdict-title">⚖️ Query Verdict</span>
            <span class="verdict-badge" style="color:{color}; border-color:{color};">{escape(verdict_label)}{pattern_html}</span>
        </div>
        <div class="verdict-reason">{escape(reason)}</div>
    </div>"""


# Error pattern legend: (code, name, description, bibliographic reference)
# Full references in riferimenti_tassonomia_errori.txt
PATTERN_LEGEND = [
    ("RET_FAIL", "Retrieval Failure", "The retriever did not find the documents needed to answer the question (low Precision and Recall).",
     "Barnett et al. 2024, FP1 Missing Content / FP2 Missed the Top Ranked Documents"),
    ("RET_INCOMPLETE", "Incomplete Retrieval", "The retrieved documents are relevant (high Precision) but essential passages are missing (low Recall).",
     "Barnett et al. 2024, FP2 Missed the Top Ranked Documents / FP3 Not in Context"),
    ("RET_NOISE", "Retrieval Noise", "The useful documents are there (high Recall) but mixed with many irrelevant ones (low Precision) that confuse the model.",
     "Chen et al. 2024, RGB (Noise Robustness); Barnett et al. 2024, FP3 Not in Context"),
    ("NOT_EXTRACTED", "Context Not Extracted", "The context contains the information and the model is faithful, but it fails to extract the correct answer.",
     "Barnett et al. 2024, FP4 Not Extracted; possible cause: Liu et al. 2024, Lost in the Middle"),
    ("OFF_TOPIC", "Off-Topic Answer", "Correct context and faithful model, but the answer does not address the question (low Answer Relevancy).",
     "Es et al. 2024, RAGAS (Answer Relevance)"),
    ("HALLUC", "Hallucination", "The documents contain the answer, but the model invents information not supported by the context.",
     "Es et al. 2024, RAGAS (Faithfulness); Niu et al. 2024, RAGTruth"),
    ("REFUSAL", "Refusal / Non-Answer", "The model refuses to answer (detected at the start of the answer); the likely cause is given by the retrieval.",
     "Chen et al. 2024, RGB (Negative Rejection); Yang et al. 2024, CRAG ('missing' answers)"),
    ("TOTAL_FAIL", "System Collapse", "Retrieval and generation fail together (no useful document and an unfaithful answer).",
     "SuperBenchmark's own category: combination of RET_FAIL and HALLUC"),
    ("PARTIAL", "Partial Answer", f"The answer captures only part of the truth ({THRESHOLDS['partial']:.2f} &le; Correctness &lt; {THRESHOLDS['correct']:.2f}); the likely cause is shown with the same code as the error patterns.",
     "Barnett et al. 2024, FP7 Incomplete"),
    ("CORRECT", "Correct Answer", f"The answer generated by the RAG is correct and aligned with the gold answer (Correctness &ge; {THRESHOLDS['correct']:.2f}).",
     "Es et al. 2024, RAGAS (Answer Correctness)"),
    ("UNKNOWN", "Unclassified Error", "Wrong answer, but the cause cannot be determined because diagnostic metrics are missing.",
     "Technical category"),
    ("EVAL_FAIL", "Evaluation Failure", "RAGAS could not compute Correctness (judge error, timeout or metric not selected): the query is not judged.",
     "Evaluator error, not a RAG error: excluded from the error distribution"),
    ("RAG_ERROR", "RAG Error", "The RAG did not respond (HTTP error, timeout, empty answer or exception): there is no answer to judge, so the query is not sent to RAGAS and is excluded from the averages.",
     "Infrastructure error, not a wrong answer: excluded from the error distribution"),
]


def get_verdict_summary_html(reports: list) -> str:
    """Summary bar showing distribution of verdicts, error patterns and causes of partial answers."""
    total = len(reports)
    verdicts = Counter(r.get('verdict', 'wrong') for r in reports)
    correct, partial, not_eval = verdicts['correct'], verdicts['partial'], verdicts['not_evaluated']
    wrong = total - correct - partial - not_eval
    # not evaluated because the RAG itself failed (RAG_ERROR), not the judge
    rag_err = sum(1 for r in reports if r.get('error_pattern') == 'RAG_ERROR')
    not_eval_text = f"{not_eval} (RAG errors: {rag_err})" if rag_err else f"{not_eval}"
    pattern_counts = Counter(r.get('error_pattern', 'UNKNOWN') for r in reports if r.get('verdict', 'wrong') == 'wrong')
    partial_causes = Counter(r.get('error_cause') or 'UNKNOWN' for r in reports if r.get('verdict') == 'partial')

    # Percentages for the stacked bar
    pct_c = (correct / total * 100) if total else 0
    pct_p = (partial / total * 100) if total else 0
    pct_w = (wrong / total * 100) if total else 0
    pct_n = (not_eval / total * 100) if total else 0

    # Build error-pattern breakdown (wrong answers) and causes of partial answers
    error_patterns_html = "".join(f'<span class="vp-tag vp-tag-red">{escape(p)}: {c}</span>' for p, c in pattern_counts.most_common())
    partial_causes_html = "".join(f'<span class="vp-tag vp-tag-yellow">{escape(p)}: {c}</span>' for p, c in partial_causes.most_common())

    if not error_patterns_html and wrong == 0:
        error_patterns_html = '<span class="vp-tag vp-tag-green">No errors detected</span>'
    elif not error_patterns_html:
        error_patterns_html = '<span class="vp-tag vp-tag-red">Unclassified: {}</span>'.format(wrong)

    return f"""
    <div class="verdict-summary">
        <div class="vs-header">⚖️ Verdict Distribution</div>
        <div class="vs-bar">
            <div class="vs-seg vs-correct" style="width:{pct_c}%;" title="Correct: {correct}"></div>
            <div class="vs-seg vs-partial" style="width:{pct_p}%;" title="Partial: {partial}"></div>
            <div class="vs-seg vs-wrong"   style="width:{pct_w}%;" title="Wrong: {wrong}"></div>
            {f'<div class="vs-seg vs-noteval" style="width:{pct_n}%;" title="Not Evaluated: {not_eval_text}"></div>' if not_eval else ''}
        </div>
        <div class="vs-labels">
            <span class="vs-label"><span class="vs-dot" style="background:#00E676;"></span>Correct: {correct}</span>
            <span class="vs-label"><span class="vs-dot" style="background:#FFC400;"></span>Partial: {partial}</span>
            <span class="vs-label"><span class="vs-dot" style="background:#FF1744;"></span>Wrong: {wrong}</span>
            {f'<span class="vs-label"><span class="vs-dot" style="background:#94A3B8;"></span>Not Evaluated: {not_eval_text}</span>' if not_eval else ''}
        </div>
        <div class="vs-patterns">
            <span class="vs-patterns-title">Error Patterns:</span>
            {error_patterns_html}
        </div>
        {f'<div class="vs-patterns"><span class="vs-patterns-title">Partial Causes:</span>{partial_causes_html}</div>' if partial_causes_html else ''}
    </div>"""


def get_pattern_legend_html() -> str:
    """Returns the error pattern legend cards grid."""
    legend_rows_html = ""
    for code, title, desc, ref in PATTERN_LEGEND:
        tag_class = {"CORRECT": "vp-tag-green", "PARTIAL": "vp-tag-yellow", "EVAL_FAIL": "vp-tag-grey", "RAG_ERROR": "vp-tag-grey"}.get(code, "vp-tag-red")
        legend_rows_html += f'<div class="vs-legend-item"><span class="vp-tag {tag_class}">{code}</span><div class="vs-legend-desc"><strong>{title}</strong>: {desc}<div class="vs-legend-ref">📚 {ref}</div></div></div>'

    return f'<div class="vs-legend-grid">{legend_rows_html}</div>'
