"""Before/after proof for the report. Runs the SAME questions through the base
model and the LoRA-adapted model and prints them side by side. Screenshot this
for your writeup — it's your evidence of RS adaptation."""
import argparse
import json
import os

import torch
from PIL import Image

MIN_PIXELS = int(os.getenv("SATQUERY_MIN_PIXELS", str(128 * 128)))
MAX_PIXELS = int(os.getenv("SATQUERY_MAX_PIXELS", str(512 * 512)))


def gen(model, processor, img, question, max_new=64):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": f"You are a remote-sensing analyst. Answer concisely.\nQuestion: {question}"}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--adapter", default="adapter")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--load-4bit", action="store_true", help="load base in 4-bit (saves VRAM)")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    kw = dict(torch_dtype=torch.float16, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    print("loading base model...")
    base = AutoModelForImageTextToText.from_pretrained(args.model, **kw).eval()
    for r in rows:
        img = Image.open(os.path.join(args.data_root, r["image"])).convert("RGB")
        print("\n" + "=" * 60)
        print("Q:", r["question"]); print("GOLD:", r["answer"])
        print("BASE:", gen(base, proc, img, r["question"]))

    print("\nloading adapter...")
    tuned = PeftModel.from_pretrained(base, args.adapter).eval()
    for r in rows:
        img = Image.open(os.path.join(args.data_root, r["image"])).convert("RGB")
        print("\n" + "-" * 60)
        print("Q:", r["question"]); print("GOLD:", r["answer"])
        print("TUNED:", gen(tuned, proc, img, r["question"]))


if __name__ == "__main__":
    main()
