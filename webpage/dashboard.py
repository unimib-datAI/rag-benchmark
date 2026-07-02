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
    html_issues = '<div class="diag-panel issues"><h3>Problematiche rilevate</h3>'
    issues = []
    if avg_r < 0.6:
        issues.append("<b>Retrieval debole (< 0.60):</b> I documenti recuperati non contengono le informazioni necessarie.")
    if avg_f < 0.7:
        issues.append("<b>Allucinazioni (Faithfulness < 0.70):</b> Il modello genera informazioni non presenti nel contesto fornito.")
    if avg_c < 0.5:
        issues.append("<b>Bassa correttezza (< 0.50):</b> Le risposte non corrispondono alla Ground Truth.")
    if avg_f1 < 0.3:
        issues.append("<b>Bassa Similarità Semantica (< 0.30):</b> La risposta diverge semanticamente dalla Ground Truth.")
        
    if not issues:
        html_issues += '<div class="diag-ok">Nessun problema critico rilevato.</div>'
    else:
        for iss in issues:
            html_issues += f'<p>{iss}</p>'
    html_issues += '</div>'

    html_sols = '<div class="diag-panel solutions"><h3>Possibili soluzioni</h3>'
    solutions = []
    if avg_r < 0.6:
        solutions.append("<b>Chunking:</b> Sperimenta con chunk più piccoli o sovrapposti per catturare meglio le frasi chiave.")
        solutions.append("<b>Retriever ibrido:</b> Combina BM25 con embeddings densi (FAISS + sentence-transformers).")
        solutions.append("<b>Reranking:</b> Aggiungi un modello di re-ranking dopo il primo retrieval.")
    if avg_f < 0.7:
        solutions.append("<b>System prompt:</b> Aggiungi istruzioni esplicite per limitare le risposte al solo contesto fornito.")
        solutions.append("<b>Temperature 0.0:</b> Riduci la creatività per forzare risposte deterministiche.")
    if avg_c < 0.5 or avg_f1 < 0.3:
        solutions.append("<b>Risposte concise:</b> Istruisci l'LLM a rispondere in massimo 2 frasi estraendo solo l'informazione esatta.")
        solutions.append("<b>Pulizia contesto:</b> Implementa filtri per HTML/OCR noise prima della generazione.")
        
    if not solutions:
        html_sols += '<div class="diag-ok">Prestazioni stabili.</div>'
    else:
        for sol in solutions:
            html_sols += f'<p>{sol}</p>'
    html_sols += '</div>'
    
    return html_issues, html_sols

def get_footer_html(num_reports: int, ollama_model: str, time_str: str) -> str:
    return f"""
<div class="footer-bar">
    SuperBenchmark  &middot;  {num_reports} query  &middot;  Giudice: {ollama_model}  &middot;  {time_str}
</div>"""
