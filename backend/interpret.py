"""Natural-language query interpretation.

If ANTHROPIC_API_KEY is set, use Claude to map a free-form question to the best
mode + site. Otherwise fall back to a deterministic keyword/place matcher so the
query box still works with no external dependency.
"""
import json
import os
import re

# domain keywords -> mode, used by the rule-based fallback
MODE_KEYWORDS = {
    "disaster":    ["flood", "flooded", "under water", "inundat", "cyclone", "tsunami",
                    "earthquake", "landslide", "disaster", "submerg", "villages under"],
    "agri":        ["paddy", "crop", "sown", "sowing", "fallow", "kharif", "rabi",
                    "agricultur", "farm", "field", "harvest"],
    "urban":       ["built-up", "built up", "construction", "encroach", "building",
                    "urban", "unauthor", "illegal", "concret"],
    "forestwater": ["forest", "tree cover", "canopy", "deforest", "reservoir",
                    "water level", "lake", "dam", "shrunk", "shrink", "green"],
    "defence":     ["border", "structure", "build-up", "buildup", "military",
                    "loc", "lac", "strip", "vehicle"],
}


def _rule_match(q, catalog):
    ql = q.lower()
    # 1) mode hint
    mode_scores = {}
    for mode, kws in MODE_KEYWORDS.items():
        mode_scores[mode] = sum(1 for k in kws if k in ql)
    best_mode = max(mode_scores, key=mode_scores.get) if any(mode_scores.values()) else None

    # 2) score each site by token overlap; boost sites in the hinted mode
    tokens = [t for t in re.findall(r"[a-z]{3,}", ql)]
    best, best_score = None, -1
    for site in catalog:
        score = sum(1 for t in tokens if t in site["keywords"])
        # place-name overlap weighs heavily
        if any(t in site["name"].lower() for t in tokens):
            score += 3
        if best_mode and site["mode"] == best_mode:
            score += 2
        if score > best_score:
            best, best_score = site, score

    if best is None:
        return None
    return {
        "ok": True,
        "source": "rules",
        "mode": best["mode"],
        "name": best["name"],
        "lat": best["lat"],
        "lng": best["lng"],
        "confidence": "low" if best_score <= 1 else "medium",
        "reason": f"keyword match (score {best_score})",
    }


def _llm_match(q, catalog):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except Exception:
        return None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    compact = [
        {"i": i, "mode": s["mode"], "name": s["name"], "answers": s["query"], "change": s["change"]}
        for i, s in enumerate(catalog)
    ]
    sys = (
        "You route a district officer's plain-language question to ONE monitoring site "
        "from a catalog. Reply with ONLY compact JSON: "
        '{"index": <catalog i>, "reason": "<short>"}. Pick the single best-matching site.'
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            system=sys,
            messages=[{"role": "user", "content": f"Question: {q}\n\nCatalog:\n{json.dumps(compact)}"}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0))
        s = catalog[int(data["index"])]
        return {
            "ok": True,
            "source": "llm",
            "mode": s["mode"],
            "name": s["name"],
            "lat": s["lat"],
            "lng": s["lng"],
            "confidence": "high",
            "reason": data.get("reason", "")[:200],
        }
    except Exception as e:
        print("[interpret] llm error:", e)
        return None


def interpret_query(q, catalog):
    if not q or not q.strip():
        return {"ok": False, "reason": "empty query"}
    return _llm_match(q, catalog) or _rule_match(q, catalog) or {"ok": False, "reason": "no match"}
