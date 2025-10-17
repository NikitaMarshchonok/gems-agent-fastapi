# app/kb.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json, numpy as np
from pypdf import PdfReader
from .llm import embed

BASE = Path(__file__).resolve().parent.parent / "data"

def _gem_dir(gem_id: str) -> Path:
    d = BASE / gem_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            r = PdfReader(str(path))
            return "\n\n".join([p.extract_text() or "" for p in r.pages])
        except Exception:
            return ""
    # txt/md/etc
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _chunk(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    words = text.split()
    if not words:
        return []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += step
    return out

def ingest_files(gem_id: str, file_paths: List[Path]) -> Dict:
    gdir = _gem_dir(gem_id)
    fdir = gdir / "files"
    fdir.mkdir(exist_ok=True)

    chunks = []
    copied = []
    for p in file_paths:
        dst = fdir / p.name
        dst.write_bytes(p.read_bytes())
        copied.append(dst.name)
        text = _read_text(dst)
        for idx, ch in enumerate(_chunk(text)):
            chunks.append({"text": ch, "source": dst.name, "i": idx})

    # пустой корпус — создаём «пустые» артефакты, чтобы статусы не падали
    if not chunks:
        (gdir / "meta.json").write_text("[]", encoding="utf-8")
        np.savez_compressed(gdir / "index.npz", vecs=np.zeros((0, 1)))
        return {"files": copied, "chunks": 0}

    vecs = np.array(embed([c["text"] for c in chunks]), dtype=float)
    np.savez_compressed(gdir / "index.npz", vecs=vecs)
    (gdir / "meta.json").write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8"
    )
    return {"files": copied, "chunks": len(chunks)}

def has_index(gem_id: str) -> bool:
    gdir = _gem_dir(gem_id)
    return (gdir / "index.npz").exists() and (gdir / "meta.json").exists()

def list_files(gem_id: str) -> List[str]:
    fdir = _gem_dir(gem_id) / "files"
    return [p.name for p in fdir.iterdir() if p.is_file()] if fdir.exists() else []

def status(gem_id: str) -> Dict:
    gdir = _gem_dir(gem_id)
    ok = has_index(gem_id)
    chunks = 0
    if ok:
        try:
            meta = json.loads((gdir / "meta.json").read_text(encoding="utf-8"))
            chunks = len(meta)
        except Exception:
            pass
    return {"indexed": ok, "chunks": chunks, "files": list_files(gem_id)}

def query(gem_id: str, q: str, k: int = 4) -> List[Dict]:
    if not has_index(gem_id):
        return []
    gdir = _gem_dir(gem_id)
    meta = json.loads((gdir / "meta.json").read_text(encoding="utf-8"))
    vecs = np.load(gdir / "index.npz")["vecs"]
    if vecs.size == 0:
        return []
    qv = np.array(embed([q])[0], dtype=float)
    sims = (vecs @ qv) / (np.linalg.norm(vecs, axis=1) * (np.linalg.norm(qv) + 1e-8))
    idx = np.argsort(-sims)[:k]
    return [
        {"text": meta[int(i)]["text"], "source": meta[int(i)]["source"], "score": float(sims[int(i)])}
        for i in idx
    ]

def build_context(snips: List[Dict]) -> str:
    if not snips:
        return ""
    rows = [
        f"[{i}] (src: {s['source']}, score={s['score']:.3f})\n{s['text']}"
        for i, s in enumerate(snips, 1)
    ]
    return "Knowledge Base snippets (use if relevant; cite [#]):\n\n" + "\n\n".join(rows)
