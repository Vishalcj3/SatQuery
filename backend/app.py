"""SatQuery India — Geospatial Intelligence API (FastAPI).

Endpoints:
  GET  /health          -> status + whether Sentinel/LLM are configured
  POST /api/query {q}    -> interpret a plain-language question -> mode + site
  POST /api/change {...} -> live Sentinel-2 change footprint (GeoJSON) for a site

Run:  uvicorn app:app --reload --port 8000   (from the backend/ folder)
"""
import base64
import io
import os
import sys
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sentinel
from catalog import load_catalog
from interpret import interpret_query

# Make the repo-root `satquery` package importable (backend runs from backend/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="SatQuery India — Geospatial Intelligence API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CATALOG = load_catalog()


class QueryReq(BaseModel):
    q: str


class ChangeReq(BaseModel):
    lat: float
    lng: float
    bbox: Optional[List[float]] = None
    index: Optional[str] = None
    direction: Optional[str] = None
    date_before_from: Optional[str] = None
    date_before: Optional[str] = None
    date_after_from: Optional[str] = None
    date_after: Optional[str] = None
    mode: Optional[str] = None


@app.get("/health")
def health():
    return {
        "ok": True,
        "sites": len(CATALOG),
        "sentinel_configured": sentinel.available(),
        "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/api/query")
def query(r: QueryReq):
    return interpret_query(r.q, CATALOG)


@app.post("/api/change")
def change(r: ChangeReq):
    return sentinel.compute_change(r.model_dump())


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "img.png")[1] or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


@app.post("/api/vlm")
async def vlm(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
    query: str = Form(""),
    mod1: str = Form("auto"),
    mod2: str = Form("auto"),
):
    """Run the fine-tuned Qwen3-VL + RAG pipeline on 1-2 images + a query.

    Returns the routed answer, the task the agent chose, any visual evidence
    (PNG as base64), the execution trace, and extra fields (e.g. rag_used).
    The VLM loads lazily on the first call, so this request can take a while.
    """
    from satquery import handlers  # lazy — keeps /health and map endpoints light

    paths, mods = [], []
    tmp_files = []
    for up, mod in ((image1, mod1), (image2, mod2)):
        if up is not None:
            p = _save_upload(up)
            tmp_files.append(p)
            paths.append(p)
            mods.append(mod)
    try:
        result, trace = handlers.run(query, paths, mods)
        evidence_b64 = None
        ev = result.get("evidence")
        if ev is not None:
            buf = io.BytesIO()
            ev.save(buf, format="PNG")
            evidence_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {
            "answer": result.get("answer", ""),
            "task": getattr(trace, "task", None),
            "trace": trace.pretty(),
            "evidence": evidence_b64,
            "extra": result.get("extra", {}),
        }
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass
