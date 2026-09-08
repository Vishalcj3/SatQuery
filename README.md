# 🛰️ SatQuery AI — MVP

> **New here? Follow `GUIDE.md` — the 20-hour step-by-step.** This README is reference.
> Default model is **Qwen2.5-VL-3B-Instruct** (4-bit). Fall back to the 2B in fp16 with
> `set SATQUERY_VLM=Qwen/Qwen2-VL-2B-Instruct` + `set SATQUERY_LOAD_4BIT=0`.


Agentic remote-sensing assistant. You upload satellite image(s) + a plain-language
query; an agent router picks the right tool (VQA, captioning, grounding, change,
optical–SAR fusion), runs a **pretrained** RS-capable VLM plus a **small LoRA
adaptation**, and returns the answer **with visual evidence and an execution trace**.

## What this is — and isn't (read this)

This is an **MVP**, scoped to one person, one night, an RTX 3070 (8 GB). It proves the
pipeline end-to-end. It does **not** win the benchmark and does **not** attempt full
optical–SAR pixel fusion or large-scale fine-tuning. Honest scope:

| Capability | What runs here | Status |
|---|---|---|
| Single-image VQA | Qwen2-VL-2B (+ your LoRA) | real |
| Captioning | same VLM | real |
| Grounding | grounding-dino-tiny (text→box) | real |
| Change analysis | pixel absdiff + threshold **+** VLM description | first-pass |
| Optical–SAR fusion | describe-each-then-merge + false-colour composite | first-pass, labelled honestly |
| RS adaptation | LoRA fine-tune on a few hundred RS-VQA pairs | real, small |

Change detection and fusion are deliberately shallow and **say so** in the report.
"MVP uses pretrained RS models + a small LoRA adaptation; scaling the fine-tune,
proper optical–SAR fusion, and Cartosat/RISAT domain tuning are future work."

## Setup (Windows / RTX 3070)

```powershell
# 1. fresh venv using whatever Python you have (3.10 is fine)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"   # must end in ...\.venv\Scripts\python.exe

# 2. Install torch WITH CUDA first (cu121 is safe on a 3070). No torchvision (unused).
python -m pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu121 --timeout 300 --retries 10

# 3. Everything else
pip install -r requirements.txt --timeout 300 --retries 10

# 4. Sanity-check the GPU is seen
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run the app

```powershell
python app.py
```
Opens at http://127.0.0.1:7860. Models load lazily on first query (the app starts
instantly). First query downloads Qwen2-VL-2B (~4 GB) — do this **once before the
demo** so weights are cached.

Env switches (all optional):
- `set SATQUERY_LOAD_4BIT=1` — 4-bit inference if you're VRAM-starved (needs bitsandbytes)
- `set SATQUERY_LOW_VRAM=0` — keep both VLM + grounding resident (only if you have headroom)
- `set SATQUERY_USE_ADAPTER=0` — run base model without the LoRA

## The mandatory fine-tune

```powershell
# option A: verified clean dataset (columns image/question/answer, split=validation)
python train/prepare_data.py --hf dmarsili/RSVQA-LR-2k --split validation --n 600

# option B (bulletproof fallback): make data/raw/ with images + a qa.csv
#   qa.csv columns: image,question,answer   (image = filename in data/raw/)
python train/prepare_data.py --from-folder data/raw

# then train (batch 1 + grad-accum, fits 8GB; ~20-40 min for a few hundred samples)
python train/train_lora.py --epochs 2

# prove it worked — base vs adapted, side by side (screenshot for your writeup)
python train/eval_before_after.py --n 3
```
The app auto-loads `./adapter` if it exists.

## Demo script (what to show, in order)

1. **VQA** — optical scene, "How many runways are visible?" → answer + trace.
2. **Grounding** — "Highlight the water body" → boxed image.
3. **Change** — before/after pair, "What changed?" → red change overlay + % + description.
4. **Fusion** — optical + SAR pair → merged reading + composite (say "first-pass fusion").
5. Point at the **execution-trace panel** each time: "you ask, it decides, it shows its work."
6. Click **Download report** → open the HTML → Ctrl+P → PDF.

Pre-load demo images into `presets/` (see `presets/README.md`) so nothing waits on
uploads while judges watch. **Record a screen capture of all five working tonight** —
if the live demo breaks tomorrow, you still have proof.

## Windows gotchas (the ones that eat hours)

- **`torch.cuda.is_available()` is False** → you installed CPU torch. Reinstall with the
  `--index-url .../cu121` line above, in the activated venv.
- **bitsandbytes import error** → you only need it for `train_lora.py`. Inference runs in
  fp16 without it. If training won't install bnb, `pip install bitsandbytes>=0.44`.
- **rasterio install fails** → `pip install rasterio` ships Windows wheels for 3.9+; if it
  still fails, upgrade pip (`python -m pip install -U pip`).
- **OOM during a query** → keep `SATQUERY_LOW_VRAM=1` (default). Close other GPU apps
  (browsers with hardware accel, games). The 3070 has 8 GB; Qwen-2B fp16 is ~5 GB.
- **OOM during training** → lower `--n`, set `--max-side 448`, or `--limit 200`.

## Layout

```
app.py                     Gradio entry point
satquery/
  config.py                all switches
  router.py                classify_task — the agent's decision (pure rules)
  handlers.py              MODEL_REGISTRY + orchestrator run()
  models.py                lazy load + 8GB single-resident traffic cop
  vlm.py                   Qwen2-VL inference (VQA/caption/change-desc)
  grounding.py             grounding-dino-tiny (no compile)
  change.py                pixel-diff change map
  fusion.py                optical-SAR describe-then-merge
  io_utils.py              GeoTIFF + SAR dB + modality sniffing
  report.py                standalone HTML report
  trace.py                 execution trace (the gradeable audit log)
train/
  prepare_data.py          build data/train.jsonl
  train_lora.py            QLoRA fine-tune
  eval_before_after.py     before/after proof
presets/                   demo images + presets.json
```
