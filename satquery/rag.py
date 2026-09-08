"""Agentic RAG for SatQuery — retrieves remote-sensing domain knowledge (the 19
BigEarthNet LULC class definitions + task/metadata conventions) to ground VQA and
captioning answers.

Self-contained: MiniLM embeddings via `transformers` (no faiss / sentence-transformers)
+ a numpy cosine store. The embedder is pinned to CPU so it never competes with the
VLM for the 8 GB GPU (LOW_VRAM). Lazy-loaded on first use; degrades silently to
"no context" if the index is missing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_INDEX = _ROOT / "rag" / "index"          # -> rag/index.npz + rag/index.json
_EMBED_MODEL = os.getenv("SATQUERY_RAG_EMBED", "sentence-transformers/all-MiniLM-L6-v2")
_TOP_K = int(os.getenv("SATQUERY_RAG_TOPK", "4"))
_MIN_SCORE = float(os.getenv("SATQUERY_RAG_MINSCORE", "0.15"))
_MAX_CTX_TOKENS = int(os.getenv("SATQUERY_RAG_MAXTOK", "800"))

# Retrieval helps semantic/label questions; it rarely helps pure localization.
_NO_RETRIEVE_HINTS = ("highlight", "where is", "where are", "locate", "mark ",
                      "outline", "bounding box", "<point>", "<ref>")

_STATE: dict = {"vecs": None, "docs": None, "tok": None, "model": None, "loaded": False}


def _rough_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def _lazy_load() -> bool:
    """Load index + embedder once. Returns False if unavailable (RAG disabled)."""
    if _STATE["loaded"]:
        return _STATE["vecs"] is not None
    _STATE["loaded"] = True
    try:
        npz, js = _INDEX.with_suffix(".npz"), _INDEX.with_suffix(".json")
        if not (npz.exists() and js.exists()):
            print("[rag] no index found -> RAG disabled")
            return False
        _STATE["vecs"] = np.load(npz)["vectors"].astype(np.float32)
        _STATE["docs"] = json.loads(js.read_text(encoding="utf-8"))
        import torch
        from transformers import AutoModel, AutoTokenizer
        _STATE["tok"] = AutoTokenizer.from_pretrained(_EMBED_MODEL)
        _STATE["model"] = AutoModel.from_pretrained(_EMBED_MODEL).to("cpu").eval()
        print(f"[rag] loaded {len(_STATE['docs'])} docs, embedder on CPU")
        return True
    except Exception as e:  # noqa — never break the demo over RAG
        print(f"[rag] load failed, disabling RAG: {e}")
        _STATE["vecs"] = None
        return False


def _embed(text: str) -> np.ndarray:
    import torch
    tok, model = _STATE["tok"], _STATE["model"]
    enc = tok([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb[0].cpu().numpy().astype(np.float32)


def should_retrieve(query: str) -> bool:
    q = (query or "").lower()
    return not any(h in q for h in _NO_RETRIEVE_HINTS)


def retrieve_context(query: str, trace=None) -> str:
    """Return a token-budgeted context block for the query, or '' if none/disabled."""
    if not query or not should_retrieve(query) or not _lazy_load():
        return ""
    qv = _embed(query)
    sims = _STATE["vecs"] @ qv
    k = min(_TOP_K, len(sims))
    idx = np.argsort(-sims)[:k]
    blocks, used, budget = [], [], _MAX_CTX_TOKENS
    for i in idx:
        score = float(sims[i])
        if score < _MIN_SCORE:
            continue
        doc = _STATE["docs"][int(i)]
        cost = _rough_tokens(doc["text"])
        if cost > budget:
            continue
        title = doc.get("metadata", {}).get("title", doc.get("id", ""))
        blocks.append(f"[{title}]\n{doc['text']}")
        used.append(title)
        budget -= cost
    if trace and used:
        trace.tool("rag.retrieve", k=len(used), docs=used)
    return "\n\n".join(blocks)
