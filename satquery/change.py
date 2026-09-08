"""Change detection — honest MVP version: align, absolute difference, threshold,
colour the changed pixels. Fast, explainable, no extra model. The stretch
upgrade (a pretrained ChangeFormer) can drop in behind the same interface."""
import cv2
import numpy as np
from PIL import Image

from . import config


def _pil_to_bgr(img: Image.Image, size=(512, 512)) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    arr = cv2.resize(arr, size)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def change_map(img_a: Image.Image, img_b: Image.Image, trace=None):
    """Return (overlay_pil, pct_changed, mask)."""
    if trace:
        trace.tool("change_map", method="absdiff+threshold",
                   threshold=config.CHANGE_THRESHOLD)

    a = _pil_to_bgr(img_a)
    b = _pil_to_bgr(img_b)
    a_g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(a_g, b_g)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)  # kill single-pixel speckle
    mask = (diff > config.CHANGE_THRESHOLD).astype(np.uint8)

    # clean up: open then close
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    pct = 100.0 * float(mask.mean())

    # overlay changed pixels in red on top of the 'after' image
    over = b.copy()
    over[mask == 1] = (0, 0, 255)
    blended = cv2.addWeighted(b, 0.6, over, 0.4, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    if trace:
        trace.confidence = None
        trace.log(f"change map: {pct:.2f}% of pixels changed")

    return Image.fromarray(blended_rgb), round(pct, 2), mask
