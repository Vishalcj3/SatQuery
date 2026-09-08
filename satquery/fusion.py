"""Optical-SAR handling. This is DELIBERATELY a lightweight fusion: describe each
sensor with the VLM, merge the two readings, and build a false-colour composite
for visual evidence. We label it honestly in the UI and report as a first-pass
fusion — real pixel/feature-level fusion is future work."""
import cv2
import numpy as np
from PIL import Image

from . import vlm


def false_colour_stack(optical: Image.Image, sar: Image.Image) -> Image.Image:
    """R,G = optical luminance; B = SAR. A quick, standard 'is there structure
    the optical missed' composite. Not a scientific product — a demo visual."""
    size = (512, 512)
    o = cv2.resize(np.array(optical.convert("L")), size)
    s = cv2.resize(np.array(sar.convert("L")), size)
    rgb = np.stack([o, o, s], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def fuse(optical: Image.Image, sar: Image.Image, query: str, trace=None):
    """Return (answer_text, composite_image, parts_dict)."""
    if trace:
        trace.tool("optical_sar_fusion", strategy="describe-each-then-merge")
        trace.log("running VLM on optical view")
    opt_desc = vlm.ask_vlm([optical],
                           "Describe this OPTICAL satellite image: land cover, water, built-up "
                           "areas, vegetation. Be concise.", trace=trace)
    if trace:
        trace.log("running VLM on SAR view")
    sar_desc = vlm.ask_vlm([sar],
                           "This is a SAR (radar) image of the SAME area. Dark = smooth (water, "
                           "roads), bright = rough/structures (buildings, vegetation). Describe "
                           "the structure and likely water vs built-up. Be concise.", trace=trace)

    merged = (
        f"Optical reading: {opt_desc}\n\n"
        f"SAR reading: {sar_desc}\n\n"
        f"Combined interpretation: the optical view gives colour/land-cover context while the "
        f"SAR view confirms structure and water/built-up boundaries independent of cloud or "
        f"lighting. (First-pass fusion: sensor readings merged at the description level.)"
    )
    composite = false_colour_stack(optical, sar)
    if trace:
        trace.log("fusion complete (description-level merge + false-colour composite)")
    return merged, composite, {"optical": opt_desc, "sar": sar_desc}
