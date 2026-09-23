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
