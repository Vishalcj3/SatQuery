"""Image I/O for remote-sensing data: PNG/JPG and GeoTIFF, optical and SAR.

The PS explicitly requires GeoTIFF handling, so we route everything through a
loader that understands multi-band rasters, not just Pillow."""
import os
import numpy as np
from PIL import Image

from . import config


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Percentile-stretch any float/int band stack to displayable uint8."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def sar_to_db(x: np.ndarray) -> np.ndarray:
    """SAR intensity -> decibels. SAR is unreadable without this log step."""
    return 10.0 * np.log10(np.clip(x.astype(np.float32), 1e-4, None))


def read_geotiff(path: str, is_sar: bool = False) -> Image.Image:
    """Read a GeoTIFF into a displayable RGB PIL image.

    - single band (typical SAR): dB-scale then grayscale->RGB
    - >=3 bands (optical): take first 3 as RGB (Sentinel-2 order varies; good enough for MVP)
    """
    import rasterio  # imported lazily so the app still starts if rasterio is missing
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    bands = arr.shape[0]
    if bands == 1:
        band = arr[0]
        if is_sar:
            band = sar_to_db(band)
        g = _to_uint8(band)
        rgb = np.stack([g, g, g], axis=-1)
    else:
        chans = [_to_uint8(arr[i]) for i in range(min(3, bands))]
        while len(chans) < 3:
            chans.append(chans[-1])
        rgb = np.stack(chans[:3], axis=-1)
    return Image.fromarray(rgb)


def load_image(path: str, modality: str = "auto") -> Image.Image:
    """Load any supported file to an RGB PIL image, honouring the SAR path."""
    is_sar = modality == "sar"
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        img = read_geotiff(path, is_sar=is_sar)
    else:
        img = Image.open(path).convert("RGB")
    # cap size so we don't feed a 10k x 10k scene to the VLM
    w, h = img.size
    if max(w, h) > config.MAX_IMAGE_SIDE:
        s = config.MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img


def guess_modality(path: str, override: str = "auto") -> str:
    """Decide optical vs sar. Explicit override wins; otherwise sniff the filename.
    Honest MVP behaviour — we do NOT try to classify sensor from pixels."""
    if override in ("optical", "sar"):
        return override
    name = os.path.basename(path).lower()
    sar_hints = ("sar", "_s1", "s1_", "risat", "sentinel1", "sentinel-1", "_vv", "_vh", "grd")
    if any(h in name for h in sar_hints):
        return "sar"
    return "optical"


def validate(paths: list[str]) -> tuple[bool, str]:
    if not paths:
        return False, "No image uploaded. Add at least one image."
    if len(paths) > 2:
        return False, "This MVP handles 1 or 2 images. Use two for change / fusion tasks."
    for p in paths:
        if not os.path.exists(p):
            return False, f"File not found: {p}"
    return True, "ok"
