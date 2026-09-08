"""VLM inference wrapper around Qwen2-VL. One function, ask_vlm(), powers VQA,
captioning and change-description — only the prompt changes."""
import torch
from PIL import Image

from . import config, models


# Prompts tuned for remote sensing. Short + directive beats flowery.
PROMPTS = {
    "vqa": "You are a remote-sensing image analyst. Answer the question about this "
           "satellite image concisely and factually. If counting, give a number.\n\nQuestion: {q}",
    "caption": "You are a remote-sensing image analyst. Describe the land cover and notable "
               "features in this satellite image in 2-3 sentences.",
    "change": "You are a remote-sensing analyst comparing two satellite images of the SAME "
              "location at different times (image 1 = earlier, image 2 = later). Describe what "
              "changed. Be specific about what appeared, disappeared, or grew.\n\nUser note: {q}",
}


def ask_vlm(images: list[Image.Image], prompt: str, trace=None) -> str:
    """Run the configured VLM backend on 1..N images + a prompt -> answer text."""
    if config.VLM_BACKEND == "gemini":
        from . import gemini_vlm
        return gemini_vlm.ask_gemini(images, prompt, trace=trace)

    from qwen_vl_utils import process_vision_info

    model, proc = models.get_vlm()
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors="pt").to(model.device)

    if trace:
        trace.tool("vlm.generate", model=config.VLM_MODEL_ID, n_images=len(images),
                   max_new_tokens=config.MAX_NEW_TOKENS)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                             do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    answer = proc.batch_decode(trimmed, skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)[0].strip()
    return answer


def build_prompt(task: str, query: str, context: str = "") -> str:
    tmpl = PROMPTS.get(task, PROMPTS["vqa"])
    prompt = tmpl.format(q=query) if "{q}" in tmpl else tmpl
    if context:
        # RAG: ground the answer with retrieved domain knowledge (class defs, etc.).
        prompt = ("Reference knowledge (use only if relevant to the image):\n"
                  f"{context}\n\n{prompt}")
    return prompt
