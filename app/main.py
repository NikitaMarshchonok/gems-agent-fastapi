from fastapi import FastAPI, HTTPException
from typing import Any
import re

from .models import Gem, GemCreate, GemUpdate, ChatRequest, ChatResponse, Message
from . import store
from .tools import list_tools, run_tool
from .llm import chat as llm_chat

app = FastAPI(title="Gems Agent API", version="0.1.0")

TOOLS_INSTRUCTION = (
    "You have access to the following tools: {tools}.\n"
    "When you want to call a tool, reply with ONLY this JSON (no extra text):\n"
    "{{\"tool\":\"<tool_name>\",\"input\":\"<text>\"}}\n"
    "After the tool result is provided, produce a concise final answer for the user.\n"
)


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

    # Try to detect a tool JSON like {"tool":"...","input":"..."}
    used_tool = None
    tool_input = None
    if body.tools_mode == "auto" and gem.tools:
        tool_match = re.search(r'\{\s*\\"tool\\"\s*:\s*\\"([^\\"]+)\\"\s*,\s*\\"input\\"\s*:\s*\\"([\s\S]*?)\\"\s*\}', first)
        if tool_match:
            tname = tool_match.group(1).strip()
            tinp = tool_match.group(2).strip()
            if tname in gem.tools:
                used_tool = tname
                tool_input = tinp
                tool_result = run_tool(tname, tinp)
                # Feed tool result back and ask for final answer
                convo.append({"role": "assistant", "content": first})
                convo.append({"role": "tool", "content": f"Tool {tname} result:\n{tool_result}"})
                final = llm_chat(convo, temperature=gem.temperature, model_override=gem.model)
                return ChatResponse(content=final, used_tool=used_tool, tool_input=tool_input)

    # No (valid) tool call; return first answer
    return ChatResponse(content=first, used_tool=used_tool, tool_input=tool_input)
