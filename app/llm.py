# app/llm.py
import os
import re
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# -------- Общие настройки --------
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()
EMBED_BACKEND   = os.getenv("EMBED_BACKEND", DEFAULT_BACKEND).lower()
_HTTP_TIMEOUT   = float(os.getenv("HTTP_TIMEOUT", "120"))
_EMBED_TIMEOUT  = float(os.getenv("EMBED_TIMEOUT", "120"))

# -------- Чат-модели --------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------- Эмбеддинги --------
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


# ==================== helpers ====================

def _pick_backend(requested: Optional[str]) -> str:
    b = (requested or DEFAULT_BACKEND).lower()
    if b == "openai" and not OPENAI_API_KEY:
        return "ollama"
    if b not in {"ollama", "openai"}:
        return "ollama"
    return b

# убираем управляющие символы/мусор и ограничиваем длину
_CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]+')
def _sanitize_for_embed(s: str, max_len: int = 8000) -> str:
    if not s:
        return "."
    s = _CONTROL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "."
    if len(s) > max_len:
        s = s[:max_len]
    return s


# ==================== CHAT ====================

def chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    model_override: Optional[str] = None,
) -> str:
    backend = _pick_backend(DEFAULT_BACKEND)
    if backend == "ollama":
        return _chat_ollama(messages, temperature, model_override or OLLAMA_MODEL)
    elif backend == "openai":
        return _chat_openai(messages, temperature, model_override or OPENAI_MODEL)
    raise RuntimeError(f"Unknown backend: {backend}")

def _chat_ollama(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")

def _chat_openai(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ==================== EMBEDDINGS ====================

def embed(texts: List[str], model_override: Optional[str] = None) -> List[List[float]]:
    """
    Надёжная обёртка над Ollama/OpenAI эмбеддингами.
    - Очистка текста от управляющих символов/мусора.
    - OpenAI: /v1/embeddings (batch).
    - Ollama: по одному тексту; пробуем {"prompt": ...} → {"input": ...};
      если пусто — фолбэки моделей (mxbai-embed-large → nomic-embed-text).
    """
    backend = EMBED_BACKEND
    if backend == "openai" and not OPENAI_API_KEY:
        backend = "ollama"

    sanitized = [_sanitize_for_embed(str(t or "")) for t in texts]

    if backend == "openai":
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model_override or OPENAI_EMBED_MODEL, "input": sanitized}
        r = requests.post(url, headers=headers, json=payload, timeout=_EMBED_TIMEOUT)
        r.raise_for_status()
        data = r.json()["data"]
        # гарантируем список float
        return [[float(x) for x in d["embedding"]] for d in data]

    # ---- OLLAMA ----
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    primary = model_override or OLLAMA_EMBED_MODEL
    fallbacks = [m for m in ["mxbai-embed-large", "nomic-embed-text"] if m != primary]

    def _one(model: str, text: str) -> List[float]:
        # сначала формат prompt, затем input — встречаются обе реализации
        for payload in ({"prompt": text}, {"input": text}):
            r = requests.post(url, json={"model": model, **payload}, timeout=_EMBED_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
            if isinstance(emb, list) and emb:
                return [float(x) for x in emb]
        return []

    out: List[List[float]] = []
    for t in sanitized:
        v = _one(primary, t)
        if not v:
            for fb in fallbacks:
                v = _one(fb, t)
                if v:
                    break
        if not v:
            # как последняя защита — вернём нулевой вектор, чтобы пайплайн не падал
            # (или можно raise RuntimeError — на твой выбор)
            v = [0.0] * 768
        out.append(v)
    return out


def embed_one(text: str, model_override: Optional[str] = None) -> List[float]:
    return embed([text], model_override=model_override)[0]
