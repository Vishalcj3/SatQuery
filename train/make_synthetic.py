"""Generate a SYNTHETIC remote-sensing-style dataset with ZERO internet.

Purpose: smoke-test the whole pipeline (fine-tune loop + all 5 app tasks) before
you've downloaded anything. These are coloured blobs, NOT real satellite imagery —
the VLM's answers on them will be generic/wrong. That's fine: you're testing that
the CODE runs, not the model's accuracy.

For the REAL submission, use real RSVQA (see DATA.md). Synthetic data is not
'RS data' and does not satisfy the fine-tune requirement on its own.

Produces:
  data/images/*.png + data/train.jsonl        (offline fine-tune smoke set)
  presets/*.png                                (so the app demos all 5 tasks now)
"""
import json
import os
import random

import numpy as np
from PIL import Image

random.seed(7)
np.random.seed(7)

S = 256  # tile size


def _bg(kind):
    if kind == "water":
        base = np.array([40, 80, 160])
    elif kind == "veg":
        base = np.array([50, 130, 60])
    else:  # bare soil
        base = np.array([170, 150, 120])
    img = np.ones((S, S, 3)) * base
    img += np.random.randn(S, S, 3) * (6 if kind == "water" else 18)
    return np.clip(img, 0, 255)


def _add_buildings(img, n):
    for _ in range(n):
        x, y = random.randint(8, S - 24), random.randint(8, S - 24)
        w, h = random.randint(8, 16), random.randint(8, 16)
        shade = random.randint(110, 180)
        img[y:y + h, x:x + w] = [shade, shade, shade]
    return img


def make_tile():
    kind = random.choice(["water", "veg", "soil", "soil"])
    img = _bg("water" if kind == "water" else ("veg" if kind == "veg" else "soil"))
    n_buildings = 0
    urban = random.random() < 0.5
    if urban:
        n_buildings = random.randint(3, 14)
        img = _add_buildings(img, n_buildings)
    has_water = kind == "water" or random.random() < 0.25
    if has_water and kind != "water":
        # a water patch
        x, y = random.randint(0, S - 80), random.randint(0, S - 80)
        img[y:y + 70, x:x + 70] = [40, 80, 160] + np.random.randn(70, 70, 3) * 5
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), {
        "urban": urban, "n_buildings": n_buildings,
        "has_water": has_water or kind == "water", "veg": kind == "veg"}


def qa_for(meta):
    pairs = [
        ("Is it a rural or an urban area?", "urban" if meta["urban"] else "rural"),
        ("Are there any water areas in the image?", "yes" if meta["has_water"] else "no"),
        ("What is the number of buildings?", str(meta["n_buildings"])),
        ("Is there vegetation present?", "yes" if meta["veg"] else "no"),
    ]
    return pairs


def to_sar(pil):
    """Fake a SAR view: grayscale roughness/edges of the optical tile."""
    import cv2
    a = np.array(pil.convert("L"))
    edges = cv2.Laplacian(a, cv2.CV_64F)
    edges = np.abs(edges)
    edges = (255 * edges / (edges.max() + 1e-6)).astype(np.uint8)
    sar = (0.5 * a + 0.5 * edges).astype(np.uint8)
    return Image.fromarray(np.stack([sar] * 3, -1))


def main(n=40):
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("presets", exist_ok=True)
    rows = []
    for i in range(n):
        img, meta = make_tile()
        p = f"data/images/syn_{i:04d}.png"
        img.save(p)
        for q, a in qa_for(meta):
            rows.append({"image": f"images/syn_{i:04d}.png", "question": q, "answer": a})
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[synthetic] {n} tiles, {len(rows)} Q/A -> data/train.jsonl")

    # preset images so the app runs all 5 tasks immediately (PLACEHOLDERS)
    vqa, _ = make_tile()
    vqa.save("presets/vqa_airport.png")
    water, _ = make_tile()
    water.save("presets/ground_water.png")
    before, mb = make_tile()
    before.save("presets/change_before.png")
    after = _add_buildings(np.array(before).astype(float), 6)
    Image.fromarray(np.clip(after, 0, 255).astype(np.uint8)).save("presets/change_after.png")
    opt, _ = make_tile()
    opt.save("presets/fusion_optical.png")
    to_sar(opt).save("presets/fusion_sar.png")
    print("[synthetic] wrote 6 placeholder preset images -> presets/  (REPLACE with real for judging)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args().n)
