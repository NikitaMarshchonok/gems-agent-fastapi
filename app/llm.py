# app/llm.py
import os, requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

# выбранный бэкенд: "ollama" или "openai"
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# -------- Ollama --------
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
# модель эмбеддингов в Ollama (обязательно: `ollama pull nomic-embed-text`)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# -------- OpenAI (опционально) --------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ==================== CHAT ====================

def chat(messages: List[Dict[str, str]],
         temperature: float = 0.2,
         model_override: Optional[str] = None) -> str:
    backend = DEFAULT_BACKEND
    if backend == "openai" and not OPENAI_API_KEY:
        backend = "ollama"
    if backend == "ollama":
        return _chat_ollama(messages, temperature, model_override or OLLAMA_MODEL)
    elif backend == "openai":
        return _chat_openai(messages, temperature, model_override or OPENAI_MODEL)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

def _chat_ollama(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature}}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")

def _chat_openai(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ==================== EMBEDDINGS ====================

def embed(texts: List[str], model_override: Optional[str] = None) -> List[List[float]]:
    """
    Возвращает эмбеддинги для списка текстов.
    - при LLM_BACKEND=openai (и наличии API-ключа) — /v1/embeddings
    - иначе — Ollama /api/embeddings (по умолчанию nomic-embed-text)
    """
    backend = DEFAULT_BACKEND
    if backend == "openai" and not OPENAI_API_KEY:
        backend = "ollama"

    if backend == "openai":
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
                   "Content-Type": "application/json"}
        payload = {"model": model_override or OPENAI_EMBED_MODEL, "input": texts}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in data]

    # Ollama embeddings
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    vecs: List[List[float]] = []
    for t in texts:
        payload = {"model": model_override or OLLAMA_EMBED_MODEL, "prompt": t}
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return vecs
