"""Task handlers + the MODEL_REGISTRY, plus the top-level orchestrator run().
This is where the router's decision turns into actual tool calls."""
from PIL import Image

from . import io_utils, router, vlm, grounding, change, fusion, rag
from .trace import Trace


# --- individual task handlers -------------------------------------------------
# each returns dict(answer=str, evidence=PIL.Image|None, extra=dict)

def _h_vqa(images, query, modalities, trace):
    ctx = rag.retrieve_context(query, trace=trace)
    ans = vlm.ask_vlm(images, vlm.build_prompt("vqa", query, context=ctx), trace=trace)
    return {"answer": ans, "evidence": None, "extra": {"rag_used": bool(ctx)}}


def _h_caption(images, query, modalities, trace):
    ctx = rag.retrieve_context(query, trace=trace)
    ans = vlm.ask_vlm(images, vlm.build_prompt("caption", query, context=ctx), trace=trace)
    return {"answer": ans, "evidence": None, "extra": {"rag_used": bool(ctx)}}


def _h_grounding(images, query, modalities, trace):
    overlay, info = grounding.ground(images[0], query, trace=trace)
    if info["n_boxes"] == 0:
        ans = f"No '{info['phrase'].strip('.')}' confidently located in the image."
    else:
        ans = (f"Located {info['n_boxes']} instance(s) of "
               f"'{info['phrase'].strip('.')}' (mean confidence {info['confidence']:.2f}). "
               f"See the boxes drawn on the image.")
    return {"answer": ans, "evidence": overlay, "extra": info}


def _h_change(images, query, modalities, trace):
    overlay, pct, _mask = change.change_map(images[0], images[1], trace=trace)
    desc = vlm.ask_vlm(images, vlm.build_prompt("change", query), trace=trace)
    ans = f"{desc}\n\nPixel-level change: {pct}% of the scene differs between the two dates."
    return {"answer": ans, "evidence": overlay, "extra": {"pct_changed": pct}}


def _h_fusion(images, query, modalities, trace):
    # order images so optical is first, sar second
    if modalities == ["sar", "optical"]:
        images = [images[1], images[0]]
    text, composite, parts = fusion.fuse(images[0], images[1], query, trace=trace)
    return {"answer": text, "evidence": composite, "extra": parts}


MODEL_REGISTRY = {
    "vqa": _h_vqa,
    "caption": _h_caption,
    "grounding": _h_grounding,
    "change": _h_change,
    "optical_sar_fusion": _h_fusion,
}


# --- orchestrator -------------------------------------------------------------

def run(query: str, image_paths: list[str], modality_overrides: list[str] | None = None):
    """Full pipeline: validate -> load -> classify -> dispatch -> return.

    Returns (result_dict, trace) where result_dict has answer/evidence/extra."""
    trace = Trace(query=query or "")
    modality_overrides = modality_overrides or []

    ok, msg = io_utils.validate(image_paths)
    if not ok:
        trace.log(f"validation failed: {msg}")
        return {"answer": msg, "evidence": None, "extra": {"error": True}}, trace

    trace.log(f"received {len(image_paths)} image(s)")
    images, modalities = [], []
    for i, p in enumerate(image_paths):
        override = modality_overrides[i] if i < len(modality_overrides) else "auto"
        mod = io_utils.guess_modality(p, override)
        modalities.append(mod)
        images.append(io_utils.load_image(p, mod))
        trace.log(f"image[{i}] loaded, modality={mod}")

    trace.n_images = len(images)
    trace.modalities = modalities

    task = router.classify_task(query, len(images), modalities, trace=trace)
    handler = MODEL_REGISTRY[task]
    trace.log(f"dispatching to handler '{handler.__name__}'")

    try:
        result = handler(images, query, modalities, trace)
    except Exception as e:  # noqa  — never crash the demo; surface the error
        trace.log(f"!! handler error: {type(e).__name__}: {e}")
        result = {"answer": f"Tool error ({type(e).__name__}): {e}",
                  "evidence": None, "extra": {"error": True}}

    trace.log("done")
    return result, trace
