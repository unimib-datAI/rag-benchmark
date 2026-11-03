import re
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

def get_metric_grid_html(cards: list) -> str:
    """Metric cards in one CSS grid: the cards of a row always have the same height, however the window is resized.
    cards: (label, value, color_class) tuples."""
    return '<div class="metric-grid">' + "".join(get_metric_card_html(*card) for card in cards) + '</div>'

def get_breakdown_card_html(src_name: str, score: float, count: int, color_class: str) -> str:
    # card for the per-source breakdown
    # the "c-" prefix on the color class avoids conflicts with the classes above
    return f"""
    <div class="breakdown-card">
        <div class="src-name">{escape(src_name)}</div>
        <div class="src-score c-{color_class}">{"N/A" if score is None else f"{score:.2f}"}</div>
        <div class="src-count">{count} queries</div>
    </div>"""

# kind of text card -> (label colour, background, border)
TEXT_CARD_COLORS = {
    "question": ('#a2a2c0', 'rgba(255, 255, 255, 0.03)', 'rgba(255, 255, 255, 0.08)'),
    "gold":     ('#00E676', 'rgba(0, 230, 118, 0.06)',   'rgba(0, 230, 118, 0.20)'),
    "answer":   ('#00B0FF', 'rgba(0, 176, 255, 0.06)',   'rgba(0, 176, 255, 0.20)'),
    "error":    ('#FF1744', 'rgba(255, 23, 68, 0.06)',   'rgba(255, 23, 68, 0.20)'),
}

def get_text_card_html(label: str, text: str, kind: str) -> str:
    """Question, gold answer or RAG answer, with its label inside the card like the verdict card.
    The text is shown exactly as it is (HTML escaped); its line breaks become <br>, because in Markdown
    a blank line would end the HTML block and cut the card in two."""
    color, bg, border = TEXT_CARD_COLORS[kind]
    body = escape(str(text)).replace("\n", "<br>")
    return (f'<div class="text-card" style="background:{bg}; border:1px solid {border}; border-left:4px solid {color};">'
            f'<div class="text-card-label" style="color:{color};">{escape(label)}</div>'
            f'<div class="text-card-body">{body}</div></div>')


def get_diagnostics_html(reports: list):
    # built from the per-query patterns, so it can't contradict the verdicts
    html_issues = '<div class="diag-panel issues" style="height: 100%;"><h3>🔍 Advanced Heuristic Analysis</h3>'
    for title, text in diagnosis_messages(reports):
        html_issues += f'<p><b>{escape(title)}:</b> {escape(text)}</p>'
    html_issues += '</div>'

    return html_issues

def get_verdict_html(verdict: str, verdict_label: str, pattern_label: str, reason: str, metrics=()) -> str:
    """Card showing the per-query verdict, its RAGAS scores and the error-pattern explanation.
    metrics: (label, value text, colour, best) for each score; best marks the top one among compared evaluations."""
    color_map = {
        'correct': ('#00E676', 'rgba(0, 230, 118, 0.06)', 'rgba(0, 230, 118, 0.20)'),
        'partial': ('#FFC400', 'rgba(255, 196, 0, 0.06)',  'rgba(255, 196, 0, 0.20)'),
        'refusal': ('#C084FC', 'rgba(192, 132, 252, 0.06)',  'rgba(192, 132, 252, 0.20)'),
        'wrong':   ('#FF1744', 'rgba(255, 23, 68, 0.06)',  'rgba(255, 23, 68, 0.20)'),
        'not_evaluated': ('#94A3B8', 'rgba(148, 163, 184, 0.06)', 'rgba(148, 163, 184, 0.20)'),
    }
    color, bg, border = color_map.get(verdict, color_map['wrong'])
    pattern_html = f' — {escape(pattern_label)}' if verdict != 'correct' else ''
    chips = "".join(
        f'<span class="verdict-metric{" best" if best else ""}">{escape(label)} '
        f'<strong style="color:{value_color};">{escape(value)}</strong></span>'
        for label, value, value_color, best in metrics)
    metrics_html = f'<div class="verdict-metrics">{chips}</div>' if chips else ''
    # the badge colour already tells the verdict: its label without the leading emoji ("✅ Correct" -> "Correct")
    badge_text = re.sub(r"^\W+", "", verdict_label)

    # one line, no blank lines: in Markdown a blank line ends the HTML block (e.g. a reason with line breaks)
    return (f'<div class="verdict-card" style="background:{bg}; border:1px solid {border}; border-left:4px solid {color};">'
            '<div class="verdict-header">'
            '<span class="verdict-title">Query Verdict</span>'
            f'<span class="verdict-badge" style="color:{color}; border-color:{color};">{escape(badge_text)}{pattern_html}</span>'
            '</div>'
            f'{metrics_html}'
            f'<div class="verdict-reason">{escape(reason).replace("\n", "<br>")}</div>'
            '</div>')


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
    """Summary bar showing the distribution of verdicts and, for each kind of failure, its causes."""
    total = len(reports)
    verdicts = Counter(r.get('verdict', 'wrong') for r in reports)
    correct, partial, refusal, not_eval = (verdicts['correct'], verdicts['partial'], verdicts['refusal'],
                                           verdicts['not_evaluated'])
    wrong = total - correct - partial - refusal - not_eval

    def causes_row(title, verdict, tag_color):
        # for wrong and not evaluated queries the cause is their pattern, for partial and refused answers it is stored apart
        causes = Counter(r.get('error_cause') or r.get('error_pattern', 'UNKNOWN')
                         for r in reports if r.get('verdict', 'wrong') == verdict)
        tags = "".join(f'<span class="vp-tag vp-tag-{tag_color}">{escape(p)}: {c}</span>' for p, c in causes.most_common())
        return f'<div class="vs-patterns"><span class="vs-patterns-title">{title}:</span>{tags}</div>' if tags else ""

    causes_html = (causes_row("Partial — causes", "partial", "yellow") + causes_row("Refusal — causes", "refusal", "purple")
                   + causes_row("Wrong — causes", "wrong", "red"))
    if not causes_html:
        causes_html = '<div class="vs-patterns"><span class="vp-tag vp-tag-green">No errors detected</span></div>'
    causes_html += causes_row("Not Evaluated — causes", "not_evaluated", "grey")

    # No blank lines or indentation anywhere: in Markdown a blank line ends an HTML block and the next
    # indented line becomes a code block, which cut the bar in two whenever an optional part was missing.
    # A segment's size is its count (flex-grow), so no percentage has to resolve against the bar width;
    # segments with no queries are left out, otherwise each would still show a 2px sliver of colour.
    segments = [("correct", correct, f"Correct: {correct}"), ("partial", partial, f"Partial: {partial}"),
                ("refusal", refusal, f"Refusal: {refusal}"), ("wrong", wrong, f"Wrong: {wrong}"),
                ("noteval", not_eval, f"Not Evaluated: {not_eval}")]
    bar_html = "".join(f'<div class="vs-seg vs-{kind}" style="flex:{count} 1 0;" title="{escape(title)}"></div>'
                       for kind, count, title in segments if count)
    labels = [("#00E676", f"Correct: {correct}"), ("#FFC400", f"Partial: {partial}"),
              ("#C084FC", f"Refusal: {refusal}"), ("#FF1744", f"Wrong: {wrong}")]
    if not_eval:
        labels.append(("#94A3B8", f"Not Evaluated: {not_eval}"))
    labels_html = "".join(f'<span class="vs-label"><span class="vs-dot" style="background:{color};"></span>{escape(text)}</span>'
                          for color, text in labels)
    return ('<div class="verdict-summary">'
            '<div class="vs-header">⚖️ Verdict Distribution</div>'
            f'<div class="vs-bar">{bar_html}</div>'
            f'<div class="vs-labels">{labels_html}</div>'
            f'{causes_html}'
            '</div>')


def get_pattern_legend_html() -> str:
    """Returns the error pattern legend cards grid."""
    legend_rows_html = ""
    for code, title, desc, ref in PATTERN_LEGEND:
        tag_class = {"CORRECT": "vp-tag-green", "PARTIAL": "vp-tag-yellow", "REFUSAL": "vp-tag-purple",
                     "EVAL_FAIL": "vp-tag-grey", "RAG_ERROR": "vp-tag-grey"}.get(code, "vp-tag-red")
        legend_rows_html += f'<div class="vs-legend-item"><span class="vp-tag {tag_class}">{code}</span><div class="vs-legend-desc"><strong>{title}</strong>: {desc}<div class="vs-legend-ref">📚 {ref}</div></div></div>'

    return f'<div class="vs-legend-grid">{legend_rows_html}</div>'
