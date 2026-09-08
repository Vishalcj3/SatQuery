# Data — synthetic (already here) vs real (you download on Windows)

## What's already in your folder (works offline, right now)

- `data/train.jsonl` + `data/images/syn_*.png` — 40 synthetic tiles, 160 Q/A.
- `presets/*.png` — 6 placeholder images so the app demos all 5 tasks immediately.

**These are coloured blobs, not satellite imagery.** Use them ONLY to confirm the
pipeline runs (fine-tune loop starts, app routes tasks, no OOM). The VLM's answers
on them will be generic and grounding won't find much — that's expected. The
pixel-diff change map is the one thing that looks right on synthetic pairs.

**Do NOT submit on synthetic data.** Synthetic blobs are not "RS data" and won't
satisfy the fine-tune requirement or impress a judge. Download the real data below.

## Smoke-test first (5 min, before downloading anything)

```powershell
python train/train_lora.py --epochs 1 --limit 40   # does the loop run + save adapter?
python app.py                                       # do all 5 presets route + render?
```
If both work, your code is de-risked. NOW get real data.

## Real data (run on your Windows machine — your home internet reaches HF fine)

### 1. RSVQA-LR — your real fine-tune + VQA/grounding demo images
```powershell
python train/prepare_data.py --hf dmarsili/RSVQA-LR-2k --split validation --n 600
```
This overwrites `data/train.jsonl` with real RSVQA and fills `data/images/` with
real Sentinel-2 tiles. Then retrain:
```powershell
python train/train_lora.py --epochs 2
```
Copy 2 of the real tiles into `presets/` as `vqa_airport.png` and `ground_water.png`.

### 2. LEVIR-CD — real before/after pair for the change task
- Repo: github.com/justchenhao/LEVIR-CD  (or search "LEVIR-CD" on HuggingFace/Kaggle)
- Grab one bi-temporal pair (an `A/` image + its matching `B/` image).
- Save as `presets/change_before.png` and `presets/change_after.png`.

### 3. Optical–SAR pair — for the fusion task (the fiddly one)
No account-free one-click source. Fastest honest options:
- **Sentinel Hub EO Browser** (browser, free account): pick an area, download a
  Sentinel-2 true-colour (optical) AND a Sentinel-1 (SAR) image of the SAME area/date.
- **BigEarthNet-MM sample** or **CDVQA** repos have paired S1/S2 patches if you'd
  rather script it.
- Save as `presets/fusion_optical.png` and `presets/fusion_sar.png`, and in the app
  set image-2's sensor to **SAR**.

If you run out of time on #3, demo fusion with any two different-looking tiles and
say plainly "fusion is a first-pass, description-level merge" — don't fake a result.

## Optional: a strong RS baseline to compare against (not required)
`google/paligemma-3b-ft-rsvqa-lr-224` is a PaliGemma already fine-tuned on RSVQA-LR.
It's gated (accept the license on HF) and a different architecture, so it does NOT
count as your fine-tune — but it's a good "here's what a bigger tuned model does"
comparison line in your writeup if you have spare time.
