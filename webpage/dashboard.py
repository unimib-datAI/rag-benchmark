def get_header_html() -> str:
    return """
<div class="header-bar">
    <h1>SuperBenchmark Dashboard</h1>
    <p>Valutazione interattiva di sistemi RAG con LLM-as-a-Judge</p>
</div>
"""

def get_welcome_html() -> str:
    return """
<div class="welcome-box">
    <h3>Nessuna valutazione eseguita</h3>
    <p>Configura i parametri nella barra laterale e premi "Avvia valutazione" per iniziare.</p>
</div>
"""

def get_metric_card_html(label: str, val: float, color_class: str) -> str:
    return f"""
<div class="metric-card">
    <div class="label">{label}</div>
    <div class="value {color_class}">{val:.2f}</div>
</div>"""

def get_breakdown_card_html(src_name: str, score: float, count: int, color_class: str) -> str:
    return f"""
<div class="breakdown-card">
    <div class="src-name">{src_name}</div>
    <div class="src-score c-{color_class}">{score:.2f}</div>
    <div class="src-count">{count} query</div>
</div>"""

def get_judge_box_html(metric_label: str, v: float, reason: str, color_class: str) -> str:
    return f"""
<div class="judge-box">
    <div class="jb-label">{metric_label}</div>
    <div class="jb-value c-{color_class}">{v}</div>
    <div class="jb-reason">{reason}</div>
</div>"""

def get_token_bar_html(semantic_sim: float) -> str:
    return f"""
<div class="token-bar">
    <b>Semantica Cross-lingua:</b> Semantic Similarity: <code>{semantic_sim:.2f}</code>
</div>"""

def get_diagnostics_html(avg_r: float, avg_f: float, avg_c: float, avg_f1: float):
    html_issues = '<div class="diag-panel issues" style="height: 100%;"><h3>🔍 Analisi Euristica Avanzata</h3>'
    issues = []
    
    if avg_r >= 0.7 and avg_c < 0.5:
        issues.append("<b>Lost in the Middle / Debolezza LLM:</b> Il retrieval è buono (>=0.7), ma la correttezza è bassa. L'LLM sta fallendo nell'estrarre l'informazione dai documenti forniti.")
    elif avg_r < 0.5 and avg_f >= 0.7:
        issues.append("<b>Ricerca Fallimentare (Ma LLM Sicuro):</b> L'LLM non allucina (Faithfulness alta), ma il retriever sta fornendo i documenti sbagliati. Concentrati su embeddings o BM25.")
    elif avg_f < 0.5 and avg_c < 0.5:
        issues.append("<b>Allucinazione Grave:</b> Il modello ignora i documenti e inventa risposte sbagliate. Rivedi urgentemente il System Prompt e assicurati che la temperatura sia a 0.")
    elif avg_r < 0.5 and avg_c < 0.5:
        issues.append("<b>Collasso del Sistema:</b> Né il retrieval né la generazione funzionano bene. Inizia risolvendo la qualità dei chunk dei documenti.")
    elif avg_f1 < 0.4 and avg_c >= 0.6:
        issues.append("<b>Corretto ma Verboso:</b> Le risposte contengono i fatti giusti ma divergono dalla Ground Truth (bassa Similarità Semantica). Chiedi all'LLM di essere più sintetico.")
        
    if not issues and avg_c >= 0.8:
        issues.append("<b>Sistema Ottimizzato:</b> Le metriche indicano un RAG stabile e performante!")
    elif not issues:
        issues.append("<b>Prestazioni Medie:</b> Nessun comportamento critico netto, ma c'è margine di miglioramento su tutti i fronti.")

    for iss in issues:
        html_issues += f'<p>{iss}</p>'
    html_issues += '</div>'
    
    return html_issues

def get_footer_html(num_reports: int, ollama_model: str, time_str: str) -> str:
    return f"""
<div class="footer-bar">
    SuperBenchmark  &middot;  {num_reports} query  &middot;  Giudice: {ollama_model}  &middot;  {time_str}
</div>"""
