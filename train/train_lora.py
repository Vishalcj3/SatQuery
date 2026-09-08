"""QLoRA fine-tune of the VLM on the RS-VQA mini set. This is the mandatory
'>=1 component adapted on RS data' proof.

Design choices for 8GB:
  - 4-bit base (QLoRA) + LoRA on attention proj layers
  - gradient checkpointing (trades compute for memory — essential on 8GB)
  - batch size 1 + gradient accumulation (no padding headaches)
  - prompt/image tokens masked to -100 so loss is only on the answer
  - capped visual tokens via min/max pixels

Run:
  python train/prepare_data.py --hf dmarsili/RSVQA-LR-2k --split validation --n 600
  python train/train_lora.py --epochs 2

Then start the app — it auto-loads ./adapter if present.

If you OOM: --n 300, --max-side 448, or set env SATQUERY_MAX_PIXELS=200704 (=448*448)."""
import argparse
import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MIN_PIXELS = int(os.getenv("SATQUERY_MIN_PIXELS", str(128 * 128)))
MAX_PIXELS = int(os.getenv("SATQUERY_MAX_PIXELS", str(512 * 512)))


def load_rows(path, data_root):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["image"] = os.path.join(data_root, r["image"])
            rows.append(r)
    return rows


class RSVQADataset(Dataset):
    def __init__(self, rows, max_side=512):
        self.rows = rows
        self.max_side = max_side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        w, h = img.size
        if max(w, h) > self.max_side:
            s = self.max_side / max(w, h)
            img = img.resize((int(w * s), int(h * s)))
        return img, r["question"], str(r["answer"])


def build_collate(processor):
    """batch size 1 collate -> model inputs with a masked label tensor."""
    def collate(batch):
        img, question, answer = batch[0]
        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text":
                f"You are a remote-sensing analyst. Answer concisely.\nQuestion: {question}"}]}]

        prompt_text = processor.apply_chat_template(
            user_msg, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + answer + "<|im_end|>\n"

        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(user_msg)

        prompt_inputs = processor(text=[prompt_text], images=image_inputs, return_tensors="pt")
        full_inputs = processor(text=[full_text], images=image_inputs, return_tensors="pt")

        labels = full_inputs["input_ids"].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100  # don't train on the prompt/image tokens
        full_inputs["labels"] = labels
        return full_inputs
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap #examples (0=all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  model={args.model}")

    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    load_kwargs = dict(torch_dtype=torch.float16)
    if device == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        load_kwargs["device_map"] = device

    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    if device == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # needed for grad ckpt + frozen base
    model.config.use_cache = False

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = load_rows(args.data, args.data_root)
    if args.limit:
        rows = rows[:args.limit]
    random.shuffle(rows)
    print(f"[train] {len(rows)} examples")

    ds = RSVQADataset(rows, max_side=args.max_side)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=build_collate(processor))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        opt.zero_grad()
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"[train] epoch {epoch} step {step} avg_loss {running/(i+1):.4f}")
        print(f"[train] epoch {epoch} done, mean loss {running/max(1,len(dl)):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"[train] saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
