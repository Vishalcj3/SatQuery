"""Open-vocabulary grounding via grounding-dino-tiny (HF transformers).
Text prompt -> boxes drawn on the image. No CUDA compilation, pip-only."""
import re
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from . import config, models

_STRIP = ("highlight", "where is", "where are", "the", "mark", "locate", "show me",
          "outline", "find", "detect", "point to", "please", "in the image", "?")


def phrase_from_query(query: str) -> str:
    q = (query or "").lower()
    for s in _STRIP:
        q = q.replace(s, " ")
    q = re.sub(r"\s+", " ", q).strip(" .")
    q = q or "object"
    # grounding-dino likes lowercase queries terminated with a period
    return q + "."


def _post_process(proc, outputs, input_ids, target_sizes):
    """Handle the arg rename across transformers versions (box_threshold->threshold)."""
    try:
        return proc.post_process_grounded_object_detection(
            outputs, input_ids, threshold=0.30, text_threshold=0.25,
            target_sizes=target_sizes)
    except TypeError:
        return proc.post_process_grounded_object_detection(
            outputs, input_ids, box_threshold=0.30, text_threshold=0.25,
            target_sizes=target_sizes)


def ground(image: Image.Image, query: str, trace=None):
    """Return (overlay_image, result_dict)."""
    model, proc = models.get_grounding()
    phrase = phrase_from_query(query)
    if trace:
        trace.tool("grounding_dino", model=config.GROUNDING_MODEL_ID, phrase=phrase)

    inputs = proc(images=image, text=phrase, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = _post_process(proc, outputs, inputs.input_ids,
                            target_sizes=[image.size[::-1]])[0]

    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    labels = results.get("labels") or results.get("text_labels") or [phrase] * len(boxes)

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa
        font = None
    for box, score in zip(boxes, scores):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)
        tag = f"{phrase.strip('.')} {score:.2f}"
        draw.text((x0 + 3, max(0, y0 - 12)), tag, fill=(255, 40, 40), font=font)

    conf = float(np.mean(scores)) if scores else 0.0
    if trace:
        trace.confidence = round(conf, 3)
        trace.log(f"grounding found {len(boxes)} box(es), mean conf {conf:.3f}")

    return overlay, {"phrase": phrase, "n_boxes": len(boxes),
                     "boxes": boxes, "scores": scores, "confidence": conf}
