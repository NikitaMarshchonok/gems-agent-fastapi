import os, requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# OpenAI settings (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def chat(messages: List[Dict[str, str]], temperature: float = 0.2, model_override: Optional[str]=None) -> str:
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
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")

def _chat_openai(messages: List[Dict[str, str]], temperature: float, model: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
