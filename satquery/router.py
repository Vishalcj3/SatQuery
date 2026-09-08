"""The agentic router. Pure rules — no model, so it never breaks live.

classify_task() reads the query + how many images + their modalities, and
returns ONE task name. handlers.MODEL_REGISTRY maps that name to a function.
The whole 'agent' requirement is: decide, dispatch, record."""
from .trace import Trace

TASKS = ("vqa", "caption", "grounding", "change", "optical_sar_fusion")

_GROUNDING_KW = ("highlight", "where is", "where are", "mark", "locate", "show me the",
                 "outline", "find the", "detect", "point to")
_CAPTION_KW = ("describe", "caption", "summarise", "summarize", "land cover",
               "land-cover", "what is in", "what's in", "overview")
_CHANGE_KW = ("change", "changed", "difference", "differ", "increase", "increased",
              "grow", "grew", "expand", "expanded", "before", "after", "new ",
              "removed", "appeared", "disappeared", "over time")


def classify_task(query: str, n_images: int, modalities: list[str], trace: Trace | None = None) -> str:
    q = (query or "").lower().strip()
    modalities = modalities or []

    # 2 images that are explicitly one optical + one SAR -> fusion
    if n_images == 2 and sorted(modalities) == ["optical", "sar"]:
        task = "optical_sar_fusion"
    # any other 2-image case -> change analysis (VQA over the pair or diff)
    elif n_images == 2:
        task = "change"
    # single image, intent-driven
    elif any(k in q for k in _GROUNDING_KW):
        task = "grounding"
    elif any(k in q for k in _CAPTION_KW) and "how many" not in q:
        task = "caption"
    elif any(k in q for k in _CHANGE_KW) and n_images == 1:
        # user asked about change but only gave one image -> fall back to VQA, note it
        task = "vqa"
        if trace:
            trace.log("! change-style query with a single image; downgraded to VQA")
    else:
        task = "vqa"

    if trace:
        trace.task = task
        trace.log(f"router classified task = '{task}' (n_images={n_images}, modalities={modalities})")
    return task
