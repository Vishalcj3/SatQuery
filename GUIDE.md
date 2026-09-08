# SatQuery — 20-hour build guide (RTX 3070, Windows, solo)

Follow in order. Each phase ends in a **checkpoint** — if you hit it, you have a
working demo even if you stop there. Do the fine-tune EARLY (phase 2), not last.

Rule of the night: **get the model downloading and the fine-tune running before
you polish anything.** Downloads and training are the only things you can't speed
up by trying harder.

---

## Models you will install (this is the full list)

| Model | HF id | Size (download) | Role |
|---|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | `Qwen/Qwen2.5-VL-3B-Instruct` | ~7 GB | the VLM (VQA, caption, change desc, fusion) |
| **grounding-dino-tiny** | `IDEA-Research/grounding-dino-tiny` | ~0.7 GB | text→box grounding |

Dataset: **`dmarsili/RSVQA-LR-2k`** (small, clean, ~few hundred MB).

That's everything. **No Kimi, no Llama-70B, no GeoChat.** They do not fit 8 GB —
Kimi needs a server with ~100× your VRAM and can't see images anyway. The 3B is
the best real choice for a card that must run inference AND train on itself.

None of these are gated — no HuggingFace login needed.

---

## Phase 0 — Environment (H 0–2). The riskiest phase. Don't rush it.

```powershell
cd C:\Users\visha\Documents\SatQuery
# MUST be Python 3.10 (or 3.11/3.12). PyTorch has NO wheels for 3.13/3.14 — a newer
# python is the #1 cause of "torch is not a supported wheel". Use the full path to be sure:
C:\Users\visha\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.version)"   # MUST print 3.10.x
# if that errors "running scripts is disabled":
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   (then re-run Activate.ps1)

# CONFIRM you're in the venv — must end in ...\SatQuery\.venv\Scripts\python.exe
python -c "import sys; print(sys.executable)"

python -m pip install -U pip
# torch WITH cuda first. NO torchvision (unused). Long timeout so a 2.5GB
# download survives a network hiccup:
pip install torch --index-url https://download.pytorch.org/whl/cu121 --timeout 300 --retries 10
pip install -r requirements.txt --timeout 300 --retries 10
```

**If torch keeps timing out** (connection can't hold a 2.5 GB stream): download the
wheel in a browser instead (browsers resume, pip doesn't). Open
`https://download.pytorch.org/whl/torch/`, Ctrl+F for `cu121-cp310` AND `win_amd64`,
grab a 2.x file, then `pip install C:\path\to\that_file.whl`.

**Checkpoint 0a — GPU is really seen:**
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True NVIDIA GeForce RTX 3070`. If `False`, you installed CPU torch —
redo the torch line inside the activated venv. **Nothing works until this is True.**

**Checkpoint 0b — bitsandbytes imports:**
```powershell
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```
If this errors: `pip install -U bitsandbytes`. You need it for both 4-bit
inference and the fine-tune.

**Start the model download now (it runs while you read on):**
```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct')"
```

---

## Phase 1 — App runs on the base model (H 2–3)

```powershell
python app.py
```
Opens http://127.0.0.1:7860. Upload one optical image, ask "Describe this satellite
image." First query is slow (weights load). You should get an answer + a trace.

**Checkpoint 1: real answer from the base 3B in your browser.** Your pipeline is alive.

If it OOMs here: you're already in 4-bit; close Chrome/games (they eat VRAM), and
set a tighter cap: `set SATQUERY_MAX_PIXELS=200704` then rerun.

---

## Phase 2 — The mandatory fine-tune (H 3–7). START THIS EARLY.

```powershell
# 1. pull the data (verified id + split)
python train/prepare_data.py --hf dmarsili/RSVQA-LR-2k --split validation --n 600

# 2. train (gradient checkpointing + 4-bit; ~45–90 min for 600 samples, 2 epochs)
python train/train_lora.py --epochs 2
```
Watch the loss print every 5 steps. It should drift downward. When it finishes you
get `adapter/` — the app auto-loads it next launch.

**Checkpoint 2: `adapter/` exists.** The "≥1 component adapted on RS data"
requirement is genuinely met.

OOM during training? In order: `--n 300` → `set SATQUERY_MAX_PIXELS=200704` →
`--max-side 448`. If it still won't fit, add `--limit 200`. Something small that
finishes beats something ambitious that crashes.

Dataset id fails? Fall back: put ~30 RS images + a `qa.csv` (columns
`image,question,answer`) in `data/raw\`, then
`python train/prepare_data.py --from-folder data\raw`.

---

## Phase 3 — Prove the fine-tune worked (H 7–8)

```powershell
python train/eval_before_after.py --n 3 --load-4bit
```
Prints base vs tuned answers on the same 3 questions. **Screenshot this** — it's
your evidence slide. Then restart the app; the trace/UI now runs on your adapter.

**Checkpoint 3: a before/after screenshot showing the adapted model answering RS
questions differently.**

---

## Phase 4 — The other four tasks + demo data (H 8–12)

You need real images for change and optical–SAR. Get these now:
- **Change (before/after):** LEVIR-CD samples — bi-temporal building pairs.
  GitHub `justchenhao/LEVIR` → grab 3–4 A/B pairs.
- **Optical–SAR pair:** any Sentinel-1 (SAR) + Sentinel-2 (optical) tile of the
  same area. Sentinel Hub EO Browser, or the CDVQA/BigEarthNet samples.
- **Grounding/VQA:** reuse RSVQA images from `data/images\`.

Drop them in `presets\` and match names to `presets/presets.json` (edit either side).

Test each of the five in the browser: VQA, caption, grounding, change, fusion.

**Checkpoint 4: all five task types produce answer + evidence + trace.** Complete MVP.

---

## Phase 5 — Polish + report (H 12–15)

- Confirm each preset loads instantly (no upload wait during the demo).
- Check the **execution-trace panel** is visible and readable — that panel is your
  strongest "this is an agent" evidence.
- Run one query, click **Download report**, open the HTML, Ctrl+P → save as PDF.
  That PDF goes in your submission.

**Checkpoint 5: presets + trace + downloadable report all clean.**

---

## Phase 6 — Backup + writeup (H 15–18), then buffer (H 18–20)

- **Record a screen capture of all five tasks working.** If the live demo dies
  tomorrow, you play the recording. Do not skip this.
- Writeup, stated plainly: "MVP = pretrained Qwen2.5-VL-3B + a small LoRA
  adaptation on RSVQA-LR; grounding via grounding-dino-tiny; change detection is
  pixel-diff; optical–SAR is a first-pass description-level fusion. Scaling the
  fine-tune, a pretrained change-detection net, proper feature-level fusion, and
  Cartosat/RISAT domain tuning are next." Honesty reads as competence.
- Buffer for whatever broke.

---

## If everything is on fire (fallback demo)

Cut, in this order: SAR fusion → change detection → LoRA (run base model with
`set SATQUERY_USE_ADAPTER=0`). A rock-solid **VQA + grounding** demo with a clean
trace panel beats a broken five-task demo. Every time.
