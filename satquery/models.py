"""Heavy-model lifecycle. On an 8GB card the failure mode is OOM, so this module
is the traffic cop: with LOW_VRAM it guarantees only one big model is resident.

Load is lazy (first use), so `python app.py` starts instantly and the browser
opens before any weights touch the GPU.

Uses AutoModelForImageTextToText so the SAME code runs Qwen2-VL AND Qwen2.5-VL —
switch models with the SATQUERY_VLM env var, no code change."""
import gc
import os
import torch

from . import config

_STATE: dict[str, object] = {"vlm": None, "vlm_proc": None,
                             "grd": None, "grd_proc": None,
                             "resident": None}  # 'vlm' | 'grd' | None


def _free(name: str):
    keys = {"vlm": ("vlm", "vlm_proc"), "grd": ("grd", "grd_proc")}[name]
    for k in keys:
        _STATE[k] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _claim(name: str):
    """Under LOW_VRAM, evict whatever else is resident before loading `name`."""
    if config.LOW_VRAM and _STATE["resident"] not in (None, name):
        _free(_STATE["resident"])
    _STATE["resident"] = name


def get_vlm():
    """Return (model, processor) for the VLM, loading on first call."""
    _claim("vlm")
    if _STATE["vlm"] is not None:
        return _STATE["vlm"], _STATE["vlm_proc"]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = dict(device_map=config.DEVICE if config.DEVICE == "cuda" else None)
    if config.LOAD_4BIT and config.DEVICE == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = config.DTYPE

    model = AutoModelForImageTextToText.from_pretrained(config.VLM_MODEL_ID, **kwargs)

    # Load the LoRA adapter if present — this is the RS-adaptation the PS mandates.
    if config.USE_ADAPTER and os.path.isdir(config.ADAPTER_DIR) and os.listdir(config.ADAPTER_DIR):
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.ADAPTER_DIR)
            print(f"[models] loaded LoRA adapter from {config.ADAPTER_DIR}")
        except Exception as e:  # noqa
            print(f"[models] adapter load failed, using base model: {e}")

    model.eval()
    proc = AutoProcessor.from_pretrained(
        config.VLM_MODEL_ID, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS)
    _STATE["vlm"], _STATE["vlm_proc"] = model, proc
    return model, proc


def get_grounding():
    """Return (model, processor) for grounding-dino-tiny (no compilation needed)."""
    _claim("grd")
    if _STATE["grd"] is not None:
        return _STATE["grd"], _STATE["grd_proc"]

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(config.GROUNDING_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(config.GROUNDING_MODEL_ID)
    if config.DEVICE == "cuda":
        model = model.to("cuda")
    model.eval()
    _STATE["grd"], _STATE["grd_proc"] = model, proc
    return model, proc


def free_all():
    _free("vlm")
    _free("grd")
    _STATE["resident"] = None
