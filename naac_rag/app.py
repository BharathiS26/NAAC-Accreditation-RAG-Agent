"""
FastAPI backend for the NAAC Accreditation RAG agent.
Serves a single-page web UI and a /ask JSON endpoint.
"""

import os
import sys

# Ensure naac_rag directory is on the path when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from vector_store import build_index
from rag import answer as rag_answer

app = FastAPI(title="NAAC RAG Agent")

# Build FAISS index on startup if needed
@app.on_event("startup")
def startup_event():
    pdf = os.environ.get("PDF_PATH", "")
    if not pdf:
        # Try to find the PDF one level up from naac_rag/
        parent = os.path.dirname(os.path.dirname(__file__))
        for f in os.listdir(parent):
            if f.lower().endswith(".pdf") and "naac" in f.lower():
                pdf = os.path.join(parent, f)
                os.environ["PDF_PATH"] = pdf
                break
    if pdf and os.path.exists(pdf):
        build_index(pdf)
    else:
        print(f"[WARNING] PDF not found at PDF_PATH={pdf!r}. Set PDF_PATH env var.")


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NAAC Accreditation RAG Agent</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.65;
    background: #f0f4f9;
    color: #1e2a3a;
    min-height: 100vh;
  }

  /* ── Top bar ── */
  .topbar {
    background: linear-gradient(135deg, #0f3460 0%, #1a5276 60%, #1f618d 100%);
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 62px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
  }
  .topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .topbar-logo {
    width: 36px; height: 36px;
    background: #fff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1em; color: #0f3460; letter-spacing: -0.5px;
    flex-shrink: 0;
  }
  .topbar-title {
    color: #fff;
    font-size: 1.05em;
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .topbar-title span { opacity: 0.7; font-weight: 400; font-size: 0.88em; margin-left: 8px; }
  .topbar-badge {
    background: rgba(255,255,255,0.15);
    color: #d4e8ff;
    font-size: 0.78em;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
  }

  /* ── Main layout ── */
  .page { max-width: 880px; margin: 36px auto; padding: 0 20px 60px; }

  /* ── Hero card ── */
  .hero {
    background: #fff;
    border-radius: 16px;
    padding: 32px 36px 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 24px;
  }
  .hero h1 {
    font-size: 1.55em;
    font-weight: 700;
    color: #0f3460;
    margin-bottom: 6px;
  }
  .hero p {
    color: #5a6a7e;
    font-size: 0.95em;
    max-width: 640px;
    margin-bottom: 24px;
  }

  /* ── Suggestion pills ── */
  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 22px;
  }
  .pill {
    background: #eef4ff;
    border: 1px solid #c3d9ff;
    color: #1a5276;
    font-size: 0.82em;
    padding: 5px 13px;
    border-radius: 20px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    white-space: nowrap;
  }
  .pill:hover { background: #daeaff; border-color: #90c0f8; }

  /* ── Input area ── */
  .input-wrap {
    position: relative;
  }
  textarea {
    width: 100%;
    padding: 14px 130px 14px 18px;
    font-size: 0.97em;
    font-family: inherit;
    border: 2px solid #d0dce8;
    border-radius: 12px;
    resize: none;
    outline: none;
    color: #1e2a3a;
    background: #f8fafc;
    transition: border-color 0.2s, box-shadow 0.2s;
    line-height: 1.5;
  }
  textarea:focus {
    border-color: #1a5276;
    background: #fff;
    box-shadow: 0 0 0 3px rgba(26,82,118,0.1);
  }
  textarea::placeholder { color: #9aacbc; }

  .ask-btn {
    position: absolute;
    right: 10px;
    bottom: 10px;
    background: linear-gradient(135deg, #1a5276, #0f3460);
    color: #fff;
    border: none;
    border-radius: 9px;
    padding: 10px 22px;
    font-size: 0.93em;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .ask-btn:hover { opacity: 0.9; }
  .ask-btn:active { transform: scale(0.97); }
  .ask-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .ask-btn svg { flex-shrink: 0; }

  /* ── Loading bar ── */
  .loading-bar {
    height: 3px;
    background: linear-gradient(90deg, #1a5276, #5dade2, #1a5276);
    background-size: 200% 100%;
    border-radius: 2px;
    margin-top: 10px;
    display: none;
    animation: shimmer 1.4s infinite linear;
  }
  @keyframes shimmer { 0%{background-position:100% 0} 100%{background-position:-100% 0} }
  .loading-text {
    font-size: 0.85em;
    color: #5a6a7e;
    margin-top: 8px;
    display: none;
  }

  /* ── Result card ── */
  .result-card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 20px rgba(0,0,0,0.05);
    overflow: hidden;
    display: none;
    animation: fadeUp 0.3s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Answer section */
  .answer-section {
    padding: 28px 32px 24px;
    border-bottom: 1px solid #eef1f5;
  }
  .section-label {
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8a9ab0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .section-label::before {
    content: '';
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #1a5276;
    flex-shrink: 0;
  }
  #answer {
    font-size: 0.97em;
    line-height: 1.75;
    color: #1e2a3a;
    white-space: pre-wrap;
  }
  #answer.not-found { color: #7d6608; font-style: italic; }

  /* Sources section */
  .sources-section { padding: 22px 32px 26px; background: #f8fafc; }
  .sources-grid { display: flex; flex-direction: column; gap: 10px; margin-top: 12px; }
  .source-card {
    background: #fff;
    border: 1px solid #e2eaf4;
    border-left: 4px solid #1a5276;
    border-radius: 10px;
    padding: 12px 16px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 20px;
    font-size: 0.87em;
  }
  .source-card .src-row { display: flex; flex-direction: column; }
  .src-label { font-size: 0.75em; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.06em; color: #8a9ab0; margin-bottom: 1px; }
  .src-value { color: #1e2a3a; font-weight: 500; }
  .src-badge {
    display: inline-block;
    margin-top: 6px;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
  }
  .badge-qual { background: #e8f8f5; color: #1a7a5e; }
  .badge-quant { background: #eef4ff; color: #1a5276; }
  .badge-weight { background: #fff3e0; color: #8b5e00; margin-left: 4px; }

  /* Error */
  .error-box {
    background: #fff5f5;
    border: 1px solid #fcc;
    border-left: 4px solid #e74c3c;
    border-radius: 10px;
    padding: 14px 18px;
    color: #c0392b;
    font-size: 0.93em;
  }

  /* ── Info strip ── */
  .info-strip {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 20px;
  }
  .info-chip {
    background: #fff;
    border: 1px solid #dde6f0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.82em;
    color: #4a5e74;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .info-chip strong { color: #0f3460; }

  /* ── Footer ── */
  footer {
    text-align: center;
    margin-top: 44px;
    font-size: 0.8em;
    color: #8a9ab0;
  }
  footer a { color: #1a5276; text-decoration: none; }

  @media (max-width: 600px) {
    .topbar { padding: 0 16px; }
    .hero { padding: 22px 20px; }
    .answer-section, .sources-section { padding: 20px; }
    .source-card { grid-template-columns: 1fr; }
    .topbar-badge { display: none; }
    textarea { padding-right: 18px; padding-bottom: 54px; }
    .ask-btn { right: 10px; bottom: 10px; width: calc(100% - 20px); justify-content: center; }
  }
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">N</div>
    <div class="topbar-title">
      NAAC RAG Agent
      <span>Accreditation Assistant</span>
    </div>
  </div>
  <div class="topbar-badge">IBM Granite &middot; watsonx.ai</div>
</div>

<div class="page">

  <!-- Hero / input card -->
  <div class="hero">
    <h1>Ask the NAAC Manual</h1>
    <p>
      Get grounded answers from the <strong>NAAC Revised Manual for Autonomous Colleges (2020)</strong>
      with exact Criterion &middot; Key Indicator &middot; Metric citations.
    </p>

    <!-- Quick-pick suggestions -->
    <div class="suggestions">
      <span class="pill" onclick="fillQ(this)">What is required under Key Indicator 1.1?</span>
      <span class="pill" onclick="fillQ(this)">Qualitative metrics for Curriculum Enrichment</span>
      <span class="pill" onclick="fillQ(this)">What does Academic Flexibility measure?</span>
      <span class="pill" onclick="fillQ(this)">Weightage of Metric 1.1.2</span>
      <span class="pill" onclick="fillQ(this)">Requirements for Institutional Distinctiveness</span>
    </div>

    <!-- Input -->
    <form id="qa-form">
      <div class="input-wrap">
        <textarea id="question" name="question" rows="3"
          placeholder="Type your accreditation question here…"></textarea>
        <button type="submit" class="ask-btn" id="ask-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          Ask
        </button>
      </div>
      <div class="loading-bar" id="loading-bar"></div>
      <div class="loading-text" id="loading-text">Retrieving relevant metrics and generating answer…</div>
    </form>
  </div>

  <!-- Result card -->
  <div class="result-card" id="result-card">

    <div class="answer-section">
      <div class="section-label">Answer</div>
      <div id="answer"></div>
    </div>

    <div class="sources-section">
      <div class="section-label">Source Chunks Retrieved</div>
      <div class="sources-grid" id="sources"></div>
    </div>

  </div>

  <!-- Info chips -->
  <div class="info-strip">
    <div class="info-chip">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1a5276"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
      </svg>
      <strong>120</strong> metric chunks indexed
    </div>
    <div class="info-chip">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1a5276"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
      Hybrid semantic + keyword retrieval
    </div>
    <div class="info-chip">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1a5276"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
      </svg>
      <strong>ibm/granite-4-h-small</strong> &middot; watsonx.ai
    </div>
    <div class="info-chip">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1a5276"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
      </svg>
      Grounded answers only &middot; no hallucinations
    </div>
  </div>

  <footer>
    NAAC Accreditation RAG Agent &mdash; Powered by
    <a href="https://www.ibm.com/watsonx" target="_blank">IBM Granite &amp; watsonx.ai</a>
  </footer>

</div>

<script>
function fillQ(el) {
  document.getElementById('question').value = el.textContent.trim();
  document.getElementById('question').focus();
}

document.getElementById('qa-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const q = document.getElementById('question').value.trim();
  if (!q) return;

  const btn = document.getElementById('ask-btn');
  const bar = document.getElementById('loading-bar');
  const loadTxt = document.getElementById('loading-text');
  const card = document.getElementById('result-card');
  const ansEl = document.getElementById('answer');
  const srcEl = document.getElementById('sources');

  btn.disabled = true;
  btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> Thinking…';
  bar.style.display = 'block';
  loadTxt.style.display = 'block';
  card.style.display = 'none';
  ansEl.textContent = '';
  ansEl.className = '';
  srcEl.innerHTML = '';

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await resp.json();
    const answerText = data.answer || 'No answer returned.';
    ansEl.textContent = answerText;
    if (answerText.toLowerCase().includes('not found in the manual')) {
      ansEl.className = 'not-found';
    }

    const srcs = data.sources || [];
    if (srcs.length === 0) {
      srcEl.innerHTML = '<div style="color:#8a9ab0;font-size:0.9em;font-style:italic;">No source chunks returned.</div>';
    } else {
      srcs.forEach((s, i) => {
        const typeClass = (s.type || '').toLowerCase().includes('quant') ? 'badge-quant' : 'badge-qual';
        const typeLabel = (s.type || '').toLowerCase().includes('quant') ? 'Quantitative' : 'Qualitative';
        const weightBadge = s.weight
          ? '<span class="src-badge badge-weight">Weight ' + s.weight + '</span>'
          : '';
        const d = document.createElement('div');
        d.className = 'source-card';
        d.innerHTML =
          '<div class="src-row" style="grid-column:1/-1">' +
            '<span class="src-label">Criterion</span>' +
            '<span class="src-value">' + (s.criterion || '—') + '</span>' +
          '</div>' +
          '<div class="src-row">' +
            '<span class="src-label">Key Indicator</span>' +
            '<span class="src-value">' + (s.ki || '—') + '</span>' +
          '</div>' +
          '<div class="src-row">' +
            '<span class="src-label">Metric</span>' +
            '<span class="src-value">' + (s.metric || '—') +
            '<br><span class="src-badge ' + typeClass + '">' + typeLabel + '</span>' +
            weightBadge + '</span>' +
          '</div>';
        srcEl.appendChild(d);
      });
    }

  } catch(err) {
    ansEl.innerHTML = '<div class="error-box">Request failed: ' + err.message + '</div>';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg> Ask';
    bar.style.display = 'none';
    loadTxt.style.display = 'none';
    card.style.display = 'block';
    card.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  }
});

// Enter to submit (Shift+Enter for newline)
document.getElementById('question').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('qa-form').dispatchEvent(new Event('submit'));
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_HTML)


@app.post("/ask")
async def ask_endpoint(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"answer": "Please provide a question.", "sources": []})
    try:
        result = rag_answer(question)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(
            {"answer": f"Error: {exc}", "sources": []},
            status_code=500,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
