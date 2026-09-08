"""Gemini VLM backend — Google Generative Language API (cloud).

Sends the prompt + image(s) to Gemini's generateContent endpoint. Used when
config.VLM_BACKEND == "gemini". No SDK dependency — plain REST via `requests`.

Auth: Gemini API keys (they start with 'AIza' or the newer 'AQ.') go in the
?key= query param. Set SATQUERY_GEMINI_AUTH=bearer only if you are instead using
an OAuth 2 access token, which is sent as a Bearer header (and is short-lived).
"""
from __future__ import annotations

import base64
import io
import os

import requests

from . import config

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _auth(url: str, key: str) -> tuple[str, dict]:
    """Return (url, headers) with the right auth. API keys (AIza…/AQ.…) use the
    ?key= param; an OAuth token uses a Bearer header (opt in via env)."""
    if os.getenv("SATQUERY_GEMINI_AUTH", "").lower() == "bearer":
        return url, {"Authorization": f"Bearer {key}"}
    return f"{url}?key={key}", {}


def _img_part(image) -> dict:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return {"inline_data": {"mime_type": "image/png",
                            "data": base64.b64encode(buf.getvalue()).decode()}}


def ask_gemini(images: list, prompt: str, trace=None) -> str:
    """Run Gemini on 1..N images + a text prompt, return the answer text."""
    key = config.GEMINI_API_KEY
    if not key:
        raise RuntimeError("No Gemini key. Set SATQUERY_GEMINI_KEY or gemini_api_key.txt.")

    url = _ENDPOINT.format(model=config.GEMINI_MODEL)
    url, headers = _auth(url, key)
    parts = [{"text": prompt}] + [_img_part(im) for im in images]
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0,
                             "maxOutputTokens": config.MAX_NEW_TOKENS},
    }
    if trace:
        trace.tool("gemini.generate", model=config.GEMINI_MODEL, n_images=len(images))

    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        # Surface the API's error message (never the key) so failures are debuggable.
        raise RuntimeError(f"Gemini API {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        # Blocked / empty completion -> report the finish reason if present.
        reason = (data.get("candidates", [{}])[0].get("finishReason")
                  or data.get("promptFeedback", {}).get("blockReason") or "no content")
        return f"[Gemini returned no text — {reason}]"
