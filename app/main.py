from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Optional
import re
import json

from .models import Gem, GemCreate, GemUpdate, ChatRequest, ChatResponse, Message
from . import store
from .tools import list_tools, run_tool
from .llm import chat as llm_chat

app = FastAPI(title="Gems Agent API", version="0.1.0")

# Allow calling the API from a separate frontend (ngrok, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Double braces {{ }} so str.format() prints literal JSON braces
TOOLS_INSTRUCTION = (
    "You have access to the following tools: {tools}.\n"
    "When you want to call a tool, reply with ONLY this JSON (no extra text):\n"
    "{{\"tool\":\"<tool_name>\",\"input\":\"<text>\"}}\n"
    "After the tool result is provided, produce a concise final answer for the user.\n"
)

# --- simple sanitizer to avoid obviously unsafe phrases from tiny models ---
_BAN = ["child porn", "child pornography", "cp (child", "sexual content with minors"]
def _sanitize(text: str) -> str:
    low = text.lower()
    if any(b in low for b in _BAN):
        return "I can’t help with that."
    return text

# --- robust extraction of tool call JSON from a model message ---
_SMART_QUOTES = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "’": "'", "‘": "'",
}

def _normalize_quotes(s: str) -> str:
    for a, b in _SMART_QUOTES.items():
        s = s.replace(a, b)
    return s

# returns (tool_name, input) or (None, None)
def _extract_tool_call(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    t = _normalize_quotes(text.strip())

    # common case: plain JSON object somewhere in the text
    # match "tool": "...", "input": "..."
    m = re.search(r'"tool"\s*:\s*"([^"]+)"[^}]*"input"\s*:\s*"([^"]+)"', t, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # reversed order
    m = re.search(r'"input"\s*:\s*"([^"]+)"[^}]*"tool"\s*:\s*"([^"]+)"', t, re.S)
    if m:
        return m.group(2).strip(), m.group(1).strip()

    # fenced code block ```json { ... }
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t, re.I)
    if fence:
        blob = fence.group(1)
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict) and "tool" in obj and "input" in obj:
                return str(obj["tool"]).strip(), str(obj["input"]).strip()
        except Exception:
            pass

    # last resort: try to parse first {}-block as JSON and read keys
    brace = re.search(r"(\{[^{}]*\})", t, re.S)
    if brace:
        try:
            obj = json.loads(brace.group(1))
            if isinstance(obj, dict) and "tool" in obj and "input" in obj:
                return str(obj["tool"]).strip(), str(obj["input"]).strip()
        except Exception:
            pass

    return None, None


@app.get("/health")
def health():
    return {"status": "ok", "tools": list_tools()}

# ----- Gems CRUD -----
@app.get("/gems")
def list_gems():
    return [g.model_dump() for g in store.load_all()]

@app.get("/gems/{gem_id}")
def get_gem(gem_id: str):
    gem = store.get_gem(gem_id)
    if not gem:
        raise HTTPException(404, "Gem not found")
    return gem.model_dump()

@app.post("/gems")
def create_gem(body: GemCreate):
    new = Gem(
        id=store.new_id(),
        name=body.name,
        system_prompt=body.system_prompt,
        tools=body.tools or [],
        temperature=body.temperature or 0.2,
        model=body.model
    )
    store.add_gem(new)
    return new.model_dump()

@app.put("/gems/{gem_id}")
def update_gem(gem_id: str, patch: GemUpdate):
    updated = store.update_gem(gem_id, patch.model_dump())
    if not updated:
        raise HTTPException(404, "Gem not found")
    return updated.model_dump()

@app.delete("/gems/{gem_id}")
def remove_gem(gem_id: str):
    ok = store.delete_gem(gem_id)
    if not ok:
        raise HTTPException(404, "Gem not found")
    return {"deleted": True}

# ----- Chat -----
@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    gem = store.get_gem(body.gem_id)
    if not gem:
        raise HTTPException(404, "Gem not found")

    # Build system message (with optional tool instructions)
    sys = gem.system_prompt
    if body.tools_mode == "auto" and gem.tools:
        sys += "\n\n" + TOOLS_INSTRUCTION.format(tools=', '.join(gem.tools))

    convo = [{"role": "system", "content": sys}]
    for m in body.messages:
        convo.append({"role": m.role, "content": m.content})

    # First LLM turn
    first = llm_chat(convo, temperature=gem.temperature, model_override=gem.model)
    first = _sanitize(first)

    # Try to detect a tool JSON like {"tool":"...","input":"..."}
    used_tool = None
    tool_input = None
    if body.tools_mode == "auto" and gem.tools:
        tname, tinp = _extract_tool_call(first)
        if tname and tinp and tname in gem.tools:
            used_tool = tname
            tool_input = tinp
            tool_result = run_tool(tname, tinp)

            # Feed tool result back and ask for final answer
            convo.append({"role": "assistant", "content": first})
            convo.append({"role": "tool", "content": f"Tool {tname} result:\n{tool_result}"})
            final = llm_chat(convo, temperature=gem.temperature, model_override=gem.model)
            final = _sanitize(final)
            return ChatResponse(content=final, used_tool=used_tool, tool_input=tool_input)

    # No (valid) tool call; return first answer
    return ChatResponse(content=first, used_tool=used_tool, tool_input=tool_input)

# ----- Minimal Frontend (index) -----
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gems Agent UI</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- добавили Markdown и безопасную очистку -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
</head>
<body class="bg-slate-50 min-h-screen">
  <div class="max-w-3xl mx-auto p-6">
    <h1 class="text-2xl font-bold mb-4">Gems Agent — Demo</h1>

    <div class="bg-white rounded-xl shadow p-4 mb-4">
      <label class="block text-sm font-medium mb-1">Profile (Gem)</label>
      <select id="gem" class="w-full border rounded px-3 py-2"></select>
    </div>

    <div class="bg-white rounded-xl shadow p-4 mb-4">
      <label class="block text-sm font-medium mb-1">Tools mode</label>
      <select id="toolsMode" class="w-full border rounded px-3 py-2">
        <option value="auto">auto</option>
        <option value="off">off</option>
      </select>
    </div>

    <div class="bg-white rounded-xl shadow p-4 mb-4">
      <label class="block text-sm font-medium mb-1">Message</label>
      <textarea id="msg" rows="4" class="w-full border rounded px-3 py-2"
        placeholder="Plan a 2-day itinerary in Manchester. Answer in English only."></textarea>
      <button id="send" class="mt-3 px-4 py-2 rounded bg-indigo-600 text-white">Send</button>
      <span id="spinner" class="ml-2 hidden">…</span>
    </div>

    <!-- сюда рендерим ответ -->
    <div id="out" class="bg-white rounded-xl shadow p-4 whitespace-pre-wrap"></div>
  </div>

<script>
const byId = (id) => document.getElementById(id);
const gemSel = byId("gem");
const toolsModeSel = byId("toolsMode");
const msg = byId("msg");
const out = byId("out");
const sp = byId("spinner");

// Если фронт будет на другом домене — укажи тут адрес API (например, ngrok)
// const API_BASE = "https://<your-ngrok>.ngrok.io";
const API_BASE = "";

// Восстанавливаем выборы пользователя
(function restore() {
  const g = localStorage.getItem("gem_id");
  const m = localStorage.getItem("tools_mode");
  if (m) toolsModeSel.value = m;
  if (g) gemSel.value = g;
})();

async function loadGems() {
  const res = await fetch(API_BASE + "/gems");
  const data = await res.json();
  gemSel.innerHTML = "";
  data.forEach(g => {
    const opt = document.createElement("option");
    opt.value = g.id;
    opt.textContent = `${g.name} (${g.model ?? "default"})`;
    gemSel.appendChild(opt);
  });
  const travel = data.find(g => g.name === "Travel");
  if (travel) gemSel.value = travel.id;
}

async function send() {
  const gem_id = gemSel.value;
  const tools_mode = toolsModeSel.value;
  const content = msg.value.trim();
  if (!gem_id || !content) return;

  localStorage.setItem("gem_id", gem_id);
  localStorage.setItem("tools_mode", tools_mode);

  sp.classList.remove("hidden");
  out.innerHTML = "";
  try {
    const res = await fetch(API_BASE + "/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ gem_id, tools_mode, messages: [{role: "user", content}] })
    });
    if (!res.ok) {
      const t = await res.text();
      out.innerHTML = `<div class="text-red-600">Error ${res.status}: ${t}</div>`;
    } else {
      const data = await res.json();

      // бейдж с названием инструмента, если он использовался
      const badge = data.used_tool
        ? `<span class="inline-block text-xs px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 align-middle">used_tool: ${data.used_tool}</span>`
        : "";

      // красивый Markdown-рендер (списки, заголовки и т.п.) + безопасная очистка
      const html = DOMPurify.sanitize(marked.parse(data.content || "(empty)"));
      out.innerHTML = `${badge}<div class="mt-2">${html}</div>`;
    }
  } catch (e) {
    out.innerHTML = `<div class="text-red-600">Request failed: ${e}</div>`;
  } finally {
    sp.classList.add("hidden");
  }
}

byId("send").addEventListener("click", send);
// Отправка по Enter (Shift+Enter — новая строка)
msg.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.metaKey) {
    e.preventDefault(); send();
  }
});

loadGems();
</script>
</body>
</html>
    """