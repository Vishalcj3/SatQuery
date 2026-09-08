"""Central config. Everything tunable lives here so you can change behaviour
without hunting through modules at 3am."""
import os
import torch

# --- Models ---
# Default is Qwen2.5-VL-3B (best quality that fits 8GB in 4-bit). If anything
# fights you, fall back to the 2B in fp16 (no bitsandbytes needed):
#   set SATQUERY_VLM=Qwen/Qwen2-VL-2B-Instruct
#   set SATQUERY_LOAD_4BIT=0
# Base must match the fine-tuned adapter in ./adapter (Qwen3-VL-2B). A LoRA is
# bound to its base model, so do not change this without swapping the adapter.
VLM_MODEL_ID = os.getenv("SATQUERY_VLM", "Qwen/Qwen3-VL-2B-Instruct")

# --- VLM backend: "qwen" (local, fine-tuned) or "gemini" (Google cloud API) ---
# Gemini sends images to Google's servers. The key is read from the env var or,
# failing that, a gitignored local file (gemini_api_key.txt). Never commit it.
VLM_BACKEND = os.getenv("SATQUERY_VLM_BACKEND", "gemini").lower()
GEMINI_MODEL = os.getenv("SATQUERY_GEMINI_MODEL", "gemini-3.6-flash")


def _read_gemini_key() -> str:
    key = os.getenv("SATQUERY_GEMINI_KEY", "").strip()
    if key:
        return key
    keyfile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "gemini_api_key.txt")
    try:
        with open(keyfile, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


GEMINI_API_KEY = _read_gemini_key()
GROUNDING_MODEL_ID = os.getenv("SATQUERY_GROUNDING", "IDEA-Research/grounding-dino-tiny")

# Path to a LoRA adapter produced by train/train_lora.py. If it exists and
# USE_ADAPTER is true, the app loads it on top of the base VLM.
ADAPTER_DIR = os.getenv("SATQUERY_ADAPTER", "adapter")
USE_ADAPTER = os.getenv("SATQUERY_USE_ADAPTER", "1") == "1"

# --- Hardware behaviour ---
# 4-bit is the default because the 3B in fp16 is too tight on 8GB and you need
# bitsandbytes for the fine-tune anyway. Set SATQUERY_LOAD_4BIT=0 for fp16 (2B).
LOAD_4BIT = os.getenv("SATQUERY_LOAD_4BIT", "1") == "1"

# On 8GB, keep only ONE heavy model resident at a time. When True, loading the
# grounding model frees the VLM first (and vice-versa). Single most important
# switch for surviving a live demo without an OOM.
LOW_VRAM = os.getenv("SATQUERY_LOW_VRAM", "1") == "1"

# Cap the visual tokens Qwen builds per image. RS scenes can be huge; this keeps
# VRAM bounded. 512*512 px worth of patches is plenty for a demo.
MAX_PIXELS = int(os.getenv("SATQUERY_MAX_PIXELS", str(512 * 512)))
MIN_PIXELS = int(os.getenv("SATQUERY_MIN_PIXELS", str(128 * 128)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Generation ---
MAX_NEW_TOKENS = int(os.getenv("SATQUERY_MAX_NEW_TOKENS", "256"))

# --- I/O ---
MAX_IMAGE_SIDE = 1024  # downscale huge scenes before they hit the VLM
CHANGE_THRESHOLD = 40  # 0-255 abs-diff cutoff for the change map


def summary() -> dict:
    return {
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vlm_backend": VLM_BACKEND,
        "vlm": GEMINI_MODEL if VLM_BACKEND == "gemini" else VLM_MODEL_ID,
        "gemini_key_set": bool(GEMINI_API_KEY) if VLM_BACKEND == "gemini" else None,
        "load_4bit": LOAD_4BIT,
        "low_vram": LOW_VRAM,
        "adapter": ADAPTER_DIR if (USE_ADAPTER and VLM_BACKEND == "qwen") else None,
    }
