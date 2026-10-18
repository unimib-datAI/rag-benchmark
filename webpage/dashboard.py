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
        <div class="label">{label}</div>
        <div class="value {color_class}">{val:.2f}</div>
    </div>"""

def get_breakdown_card_html(src_name: str, score: float, count: int, color_class: str) -> str:
    # card for the per-source breakdown
    # the "c-" prefix on the color class avoids conflicts with the classes above
    return f"""
    <div class="breakdown-card">
        <div class="src-name">{src_name}</div>
        <div class="src-score c-{color_class}">{score:.2f}</div>
        <div class="src-count">{count} queries</div>
    </div>"""

def get_judge_box_html(metric_label: str, v: float, reason: str, color_class: str) -> str:
    return f"""
    <div class="judge-box">
        <div class="jb-label">{metric_label}</div>
        <div class="jb-value c-{color_class}">{v}</div>
        <div class="jb-reason">{reason}</div>
    </div>"""


def get_diagnostics_html(avg_r: float, avg_f: float, avg_c: float):
    html_issues = '<div class="diag-panel issues" style="height: 100%;"><h3>🔍 Advanced Heuristic Analysis</h3>'
    issues = []

    if avg_r >= 0.7 and avg_c < 0.5:
        issues.append("<b>Lost in the Middle / LLM Weakness:</b> Retrieval is good (>=0.7), but correctness is low. The LLM is failing to extract information from the provided documents.")
    elif avg_r < 0.5 and avg_f >= 0.7:
        issues.append("<b>Retrieval Failure (But Reliable LLM):</b> The LLM does not hallucinate (high Faithfulness), but the retriever is providing the wrong documents. Focus on embeddings or BM25.")
    elif avg_f < 0.5 and avg_c < 0.5:
        issues.append("<b>Severe Hallucination:</b> The model ignores documents and generates wrong answers. Urgently review the System Prompt and ensure the temperature is set to 0.")
    elif avg_r < 0.5 and avg_c < 0.5:
        # this is the worst case, nothing is working
        issues.append("<b>System Collapse:</b> Neither retrieval nor generation are working well. Start by fixing the quality of document chunks.")

    # if no issues found, give generic feedback
    if not issues and avg_c >= 0.8:
        issues.append("<b>Optimized System:</b> Metrics indicate a stable and high-performing RAG!")
    elif not issues:
        issues.append("<b>Average Performance:</b> No clear critical behavior, but there is room for improvement on all fronts.")

    for iss in issues:
        html_issues += f'<p>{iss}</p>'
    html_issues += '</div>'

    return html_issues

def get_footer_html(num_reports: int, ollama_model: str, time_str: str, eval_name: str = "") -> str:
    name_prefix = f"{eval_name}  &middot;  " if eval_name else ""
    return f"""
<div class="footer-bar">
    {name_prefix}SuperBenchmark  &middot;  {num_reports} queries  &middot;  Judge: {ollama_model}  &middot;  {time_str}
</div>"""


def get_verdict_html(verdict: str, verdict_label: str, pattern_label: str, reason: str) -> str:
    """Card showing the per-query verdict and error-pattern explanation."""
    color_map = {
        'correct': ('#00E676', 'rgba(0, 230, 118, 0.06)', 'rgba(0, 230, 118, 0.20)'),
        'partial': ('#FFC400', 'rgba(255, 196, 0, 0.06)',  'rgba(255, 196, 0, 0.20)'),
        'wrong':   ('#FF1744', 'rgba(255, 23, 68, 0.06)',  'rgba(255, 23, 68, 0.20)'),
    }
    color, bg, border = color_map.get(verdict, color_map['wrong'])
    pattern_html = f' — {pattern_label}' if verdict != 'correct' else ''

    return f"""
    <div class="verdict-card" style="background:{bg}; border:1px solid {border}; border-left:4px solid {color};">
        <div class="verdict-header">
            <span class="verdict-title">⚖️ Query Verdict</span>
            <span class="verdict-badge" style="color:{color}; border-color:{color};">{verdict_label}{pattern_html}</span>
        </div>
        <div class="verdict-reason">{reason}</div>
    </div>"""


def get_verdict_summary_html(pattern_counts: dict, total: int) -> str:
    """Summary bar showing distribution of verdicts and error patterns across all queries, including a pattern legend."""
    correct = pattern_counts.get('CORRECT', 0)
    partial = pattern_counts.get('PARTIAL', 0)
    wrong = total - correct - partial

    # Percentages for the stacked bar
    pct_c = (correct / total * 100) if total else 0
    pct_p = (partial / total * 100) if total else 0
    pct_w = (wrong / total * 100) if total else 0

    # Build error-pattern breakdown (only wrong patterns, skip CORRECT/PARTIAL)
    error_patterns_html = ""
    skip = {'CORRECT', 'PARTIAL'}
    for pattern_id, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        if pattern_id in skip or count == 0:
            continue
        error_patterns_html += f'<span class="vp-tag vp-tag-red">{pattern_id}: {count}</span>'

    if not error_patterns_html and wrong == 0:
        error_patterns_html = '<span class="vp-tag vp-tag-green">No errors detected</span>'
    elif not error_patterns_html:
        error_patterns_html = '<span class="vp-tag vp-tag-red">Unclassified: {}</span>'.format(wrong)

    # Legenda dei pattern di errore
    legend_items = [
        ("RET_FAIL", "Retrieval Failure", "Il retriever non ha trovato i documenti rilevanti per rispondere alla domanda (Precision e Recall bassi)."),
        ("RET_NOISE", "Retrieval Noise", "I documenti estratti sono solo parzialmente utili; mancano passaggi informativi essenziali."),
        ("LOST_MID", "Lost in the Middle", "Il contesto contiene le informazioni e il modello è fedele, ma non riesce ad estrarle da contesti lunghi/rumorosi."),
        ("HALLUC", "Hallucination", "I documenti contengono la risposta, ma il modello inventa informazioni non supportate dal contesto."),
        ("REFUSAL", "Refusal / Non-Answer", "Il modello si rifiuta di rispondere o dichiara di non avere elementi sufficienti."),
        ("TOTAL_FAIL", "System Collapse", "Fallimento contemporaneo di retrieval e generazione (nessun documento utile e risposta non fedele)."),
        ("PARTIAL", "Partial Answer", "La risposta cattura solo una parte della verità o è incompleta rispetto alla gold answer."),
        ("CORRECT", "Correct Answer", "La risposta generata dal RAG è corretta e allineata con la gold answer (Correctness &ge; 0.70).")
    ]

    legend_rows_html = ""
    for code, title, desc in legend_items:
        tag_class = "vp-tag-green" if code == "CORRECT" else ("vp-tag-yellow" if code == "PARTIAL" else "vp-tag-red")
        legend_rows_html += f"""
        <div class="vs-legend-item">
            <span class="vp-tag {tag_class}">{code}</span>
            <div class="vs-legend-desc">
                <strong>{title}</strong>: {desc}
            </div>
        </div>"""

    return f"""
    <div class="verdict-summary">
        <div class="vs-header">⚖️ Verdict Distribution</div>
        <div class="vs-bar">
            <div class="vs-seg vs-correct" style="width:{pct_c}%;" title="Correct: {correct}"></div>
            <div class="vs-seg vs-partial" style="width:{pct_p}%;" title="Partial: {partial}"></div>
            <div class="vs-seg vs-wrong"   style="width:{pct_w}%;" title="Wrong: {wrong}"></div>
        </div>
        <div class="vs-labels">
            <span class="vs-label"><span class="vs-dot" style="background:#00E676;"></span>Correct: {correct}</span>
            <span class="vs-label"><span class="vs-dot" style="background:#FFC400;"></span>Partial: {partial}</span>
            <span class="vs-label"><span class="vs-dot" style="background:#FF1744;"></span>Wrong: {wrong}</span>
        </div>
        <div class="vs-patterns">
            <span class="vs-patterns-title">Error Patterns:</span>
            {error_patterns_html}
        </div>
    </div>"""


def get_pattern_legend_html() -> str:
    """Returns the error pattern legend cards grid."""
    legend_items = [
        ("RET_FAIL", "Retrieval Failure", "Il retriever non ha trovato i documenti rilevanti per rispondere alla domanda (Precision e Recall bassi)."),
        ("RET_NOISE", "Retrieval Noise", "I documenti estratti sono solo parzialmente utili; mancano passaggi informativi essenziali."),
        ("LOST_MID", "Lost in the Middle", "Il contesto contiene le informazioni e il modello è fedele, ma non riesce ad estrarle da contesti lunghi/rumorosi."),
        ("HALLUC", "Hallucination", "I documenti contengono la risposta, ma il modello inventa informazioni non supportate dal contesto."),
        ("REFUSAL", "Refusal / Non-Answer", "Il modello si rifiuta di rispondere o dichiara di non avere elementi sufficienti."),
        ("TOTAL_FAIL", "System Collapse", "Fallimento contemporaneo di retrieval e generazione (nessun documento utile e risposta non fedele)."),
        ("PARTIAL", "Partial Answer", "La risposta cattura solo una parte della verità o è incompleta rispetto alla gold answer."),
        ("CORRECT", "Correct Answer", "La risposta generata dal RAG è corretta e allineata con la gold answer (Correctness &ge; 0.70).")
    ]

    legend_rows_html = ""
    for code, title, desc in legend_items:
        tag_class = "vp-tag-green" if code == "CORRECT" else ("vp-tag-yellow" if code == "PARTIAL" else "vp-tag-red")
        legend_rows_html += f'<div class="vs-legend-item"><span class="vp-tag {tag_class}">{code}</span><div class="vs-legend-desc"><strong>{title}</strong>: {desc}</div></div>'

    return f'<div class="vs-legend-grid">{legend_rows_html}</div>'
