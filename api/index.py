"""Vercel entry point: a chat page plus POST /api/ask, backed by the same graph as the CLI.

The FAISS index is built from data/docs on cold start (14 chunks, well under a second).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent import embeddings  # noqa: E402
from agent.graph import build_graph, run  # noqa: E402
from agent.index import VectorIndex  # noqa: E402
from agent.models import load_chat_model  # noqa: E402
from agent.tools import make_tools  # noqa: E402

INDEX = VectorIndex.from_directory(embeddings.from_env(), ROOT / "data" / "docs")
GRAPH = build_graph(load_chat_model(), make_tools(INDEX), max_steps=6)

app = FastAPI(title="LangGraph ReAct Agent", docs_url="/api/docs", openapi_url="/api/openapi.json")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.post("/api/ask")
def ask(body: Question) -> dict:
    started = time.perf_counter()
    try:
        result = run(GRAPH, body.question)
    except Exception as exc:  # provider outage or rate limit: say so instead of a bare 500
        raise HTTPException(
            503, f"The model provider did not answer ({type(exc).__name__}). Try again in a minute."
        ) from exc
    passages = {c.id: {"heading": c.heading, "text": c.text} for c in INDEX.chunks}
    return {
        "answer": result.answer,
        "grounded": result.grounded,
        "steps": result.steps,
        "tool_calls": [{"name": n, "args": a} for n, a in result.tool_calls],
        "citations": [{"id": cid, **passages[cid]} for cid in result.citations if cid in passages],
        "seconds": round(time.perf_counter() - started, 2),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(INDEX)}


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReAct Agent · Lumen Pay docs</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#07080f;--panel:#0f1222;--line:#232845;--ink:#e9ecf7;--muted:#9aa1bd;--faint:#5d6485;--acc:#4fe3a1;--bad:#ff5d73;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Inter Tight",system-ui,sans-serif}
main{max-width:820px;margin:0 auto;padding:40px max(16px,env(safe-area-inset-left)) 60px}
.eye{font:500 12px/1 "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--acc)}
h1{font-size:clamp(28px,4.5vw,44px);line-height:1.05;margin:10px 0 10px;letter-spacing:-.02em}
p.lede{color:var(--muted);margin:0 0 22px;max-width:62ch}
form{display:flex;gap:10px;flex-wrap:wrap}
input{flex:1 1 320px;min-width:0;background:var(--panel);border:1px solid var(--line);color:var(--ink);border-radius:12px;padding:14px 16px;font:inherit}
button{background:var(--acc);color:#04120b;border:0;border-radius:12px;padding:14px 20px;font:600 15px "Inter Tight",sans-serif;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 26px}
.chips button{background:transparent;color:var(--muted);border:1px solid var(--line);padding:8px 12px;font:400 13px "Inter Tight",sans-serif;border-radius:999px}
.chips button:hover{color:var(--ink);border-color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:16px}
.q{color:var(--muted);font-size:14px;margin-bottom:8px}.answer{font-size:17px;white-space:pre-wrap}
.cite{color:var(--acc);font:500 13px "JetBrains Mono",monospace}
.trace{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 0;padding:0;list-style:none}
.trace li{font:400 12px "JetBrains Mono",monospace;border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted)}
.badge{display:inline-block;font:600 11px "JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;padding:4px 8px;border-radius:6px;margin-right:8px}
.ok{background:#0f3a29;color:var(--acc)}.no{background:#3d1820;color:var(--bad)}
details{margin-top:12px;color:var(--muted);font-size:14px}summary{cursor:pointer}
details div{border-left:2px solid var(--line);padding-left:12px;margin-top:8px}
.err{color:var(--bad)}footer{color:var(--faint);font:12px "JetBrains Mono",monospace;margin-top:30px}
footer a{color:var(--muted)}
</style></head><body><main>
<div class="eye">LangGraph · FAISS · ReAct</div>
<h1>Ask the docs agent</h1>
<p class="lede">An agent over the documentation of Lumen Pay, a fictional payments API. It searches a FAISS index, uses a calculator when it needs one, and must cite the passages it used. An answer citing anything it never retrieved is sent back once, then flagged.</p>
<form id="f"><label for="q" hidden>Question</label><input id="q" maxlength="500" placeholder="How long after capture can a payment be refunded?" autocomplete="off"><button id="go">Ask</button></form>
<div class="chips" id="chips"></div>
<div id="out"></div>
<footer>Model: gpt-oss-120b on Groq · <a href="https://github.com/Priyanshu-bhati23/langgraph-react-rag-agent" target="_blank" rel="noopener">source on GitHub</a></footer>
</main><script>
const EX=["How many requests can a production key send in 15 minutes?","How do I verify a webhook signature?","Are refund fees returned?","What happens when I exceed the rate limit?"];
const chips=document.getElementById("chips"),out=document.getElementById("out"),q=document.getElementById("q"),go=document.getElementById("go");
EX.forEach(t=>{const b=document.createElement("button");b.type="button";b.textContent=t;b.onclick=()=>{q.value=t;ask()};chips.appendChild(b)});
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
document.getElementById("f").addEventListener("submit",e=>{e.preventDefault();ask()});
async function ask(){const text=q.value.trim()||q.placeholder;go.disabled=true;go.textContent="Thinking…";
const card=document.createElement("div");card.className="card";card.innerHTML='<div class="q">'+esc(text)+'</div><div class="answer">Searching the docs…</div>';out.prepend(card);
try{const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:text})});const d=await r.json();
if(!r.ok)throw new Error(typeof d.detail==="string"?d.detail:"Request failed ("+r.status+")");
const ans=esc(d.answer).replace(/[\\[\\u3010]([a-z0-9_-]+#\\d+)[\\]\\u3011]/g,'<span class="cite">[$1]</span>');
card.innerHTML='<div class="q">'+esc(text)+'</div><div class="answer">'+ans+'</div>'+
'<ul class="trace"><li><span class="badge '+(d.grounded?'ok">grounded':'no">not grounded')+'</span></li>'+d.tool_calls.map(c=>'<li>'+esc(c.name)+'('+esc(JSON.stringify(c.args))+')</li>').join("")+'<li>'+d.steps+' steps · '+d.seconds+'s</li></ul>'+
(d.citations.length?'<details><summary>Cited passages</summary>'+d.citations.map(c=>'<div><span class="cite">['+esc(c.id)+']</span> <b>'+esc(c.heading)+'</b><br>'+esc(c.text)+'</div>').join("")+'</details>':'');
}catch(e){card.querySelector(".answer").innerHTML='<span class="err">'+esc(e.message)+'</span>'}
go.disabled=false;go.textContent="Ask"}
</script></body></html>"""
