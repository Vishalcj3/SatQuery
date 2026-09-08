"""Build a tiny (image, question, answer) training set for the LoRA fine-tune.

Two paths, because HF dataset ids for RS-VQA drift and you do NOT want to
discover a broken id at 2am:

  1) --hf dmarsili/RSVQA-LR-2k --split validation   VERIFIED clean set
        (columns: image / question / answer, 2000 rows, split is 'validation')
  2) --from-folder data/raw     scan a folder + a qa.csv you fill in yourself

Both write:  data/train.jsonl  (+ copies images into data/images/)
Each line: {"image": "images/xxx.png", "question": "...", "answer": "..."}

If --hf fails (wrong id / offline), it TELLS you and exits cleanly — fall back
to --from-folder using the LEVIR/RSVQA samples you grabbed for the demo."""
import argparse
import csv
import json
import os
import shutil

OUT_DIR = "data"
IMG_DIR = os.path.join(OUT_DIR, "images")
JSONL = os.path.join(OUT_DIR, "train.jsonl")

# HF ids for RS VQA. First one is VERIFIED (image/question/answer, split=validation).
KNOWN_HF = ["dmarsili/RSVQA-LR-2k", "dmarsili/RSVQA-HR-2k", "saaketht/rsvqa_lq"]


def _ensure_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)


def from_hf(dataset_id, split, n, image_key, q_key, a_key):
    from datasets import load_dataset
    print(f"[prepare] loading {dataset_id} split={split}[:{n}] ...")
    ds = load_dataset(dataset_id, split=f"{split}[:{n}]")
    print(f"[prepare] columns: {ds.column_names}")
    rows = []
    for i, ex in enumerate(ds):
        img = ex.get(image_key)
        q = ex.get(q_key)
        a = ex.get(a_key)
        if img is None or q is None or a is None:
            continue
        path = os.path.join(IMG_DIR, f"hf_{i:05d}.png")
        img.convert("RGB").save(path)
        rows.append({"image": os.path.relpath(path, OUT_DIR).replace("\\", "/"),
                     "question": str(q), "answer": str(a)})
    return rows


def from_folder(folder):
    """folder/ should contain images + a qa.csv with columns: image,question,answer
    (image = filename inside folder)."""
    qa = os.path.join(folder, "qa.csv")
    if not os.path.exists(qa):
        raise SystemExit(f"[prepare] no qa.csv in {folder}. Create one with columns "
                         f"image,question,answer (image = filename in that folder).")
    rows = []
    with open(qa, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src = os.path.join(folder, r["image"])
            if not os.path.exists(src):
                print(f"[prepare] skip missing {src}")
                continue
            dst = os.path.join(IMG_DIR, os.path.basename(src))
            shutil.copy(src, dst)
            rows.append({"image": os.path.relpath(dst, OUT_DIR).replace("\\", "/"),
                         "question": r["question"], "answer": r["answer"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", help="HuggingFace dataset id, e.g. RSVQA/RSVQA-LR")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--image-key", default="image")
    ap.add_argument("--q-key", default="question")
    ap.add_argument("--a-key", default="answer")
    ap.add_argument("--from-folder", help="folder with images + qa.csv")
    args = ap.parse_args()

    _ensure_dirs()

    if args.from_folder:
        rows = from_folder(args.from_folder)
    elif args.hf:
        try:
            rows = from_hf(args.hf, args.split, args.n, args.image_key, args.q_key, args.a_key)
        except Exception as e:  # noqa
            print(f"[prepare] HF load FAILED: {e}")
            print(f"[prepare] known ids to try: {KNOWN_HF}")
            print("[prepare] fall back to: python train/prepare_data.py --from-folder data/raw")
            raise SystemExit(1)
    else:
        raise SystemExit("Pass --hf <id> or --from-folder <dir>. See --help.")

    if not rows:
        raise SystemExit("[prepare] no rows produced — check the column keys.")

    with open(JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[prepare] wrote {len(rows)} examples -> {JSONL}")


if __name__ == "__main__":
    main()
