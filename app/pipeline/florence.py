from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from PIL import Image

from app.pipeline.classes import canonical_label
from app.pipeline.geometry import Region
from app.settings import settings

log = logging.getLogger(__name__)

_PROMPT = (
    "roof, gable, gutter, window, sliding window, vent, louver, "
    "wall, siding, foundation, downspout, pipe"
)


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = torch.device(settings.device)
    dtype = torch.float32 if settings.device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        settings.florence_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(settings.florence_id, trust_remote_code=True)
    return model, processor, device


def propose_boxes(bgr: np.ndarray) -> list[Region]:
    if not settings.enable_florence:
        return []
    try:
        import torch

        model, processor, device = _load()
    except Exception:
        log.exception("Florence-2 unavailable")
        return []

    rgb = bgr[:, :, ::-1]
    image = Image.fromarray(rgb)
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = processor(text=task + _PROMPT, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=2,
            do_sample=False,
        )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=(image.width, image.height)
    )
    block = parsed.get(task) or {}
    bboxes = block.get("bboxes") or []
    labels = block.get("labels") or []
    h, w = bgr.shape[:2]
    out: list[Region] = []
    for box, raw in zip(bboxes, labels):
        label = canonical_label(str(raw))
        if not label:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        out.append(
            Region(
                label=label,
                mask=mask,
                score=0.8,
                source="florence",
                box=(x1, y1, x2, y2),
            )
        )
    log.info("Florence-2 proposed %d boxes", len(out))
    return out
