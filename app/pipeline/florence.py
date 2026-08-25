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
    import sys
    import os
    from pathlib import Path

    stub = Path(__file__).resolve().parents[2] / "stubs"
    if stub.exists() and str(stub) not in sys.path:
        sys.path.insert(0, str(stub))

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4))))
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
    orig_w, orig_h = image.size
    max_side = max(256, int(settings.florence_max_side))
    scale = min(1.0, max_side / max(orig_w, orig_h))
    if scale < 1:
        image = image.resize((max(1, int(orig_w * scale)), max(1, int(orig_h * scale))), Image.BILINEAR)
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = processor(text=task + _PROMPT, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=192,
            num_beams=1,
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
    env_area = h * w
    out: list[Region] = []
    for box, raw in zip(bboxes, labels):
        label = canonical_label(str(raw))
        if not label:
            continue
        x1, y1, x2, y2 = [int(v / scale) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > 0.22 * env_area:
            continue
        if label in {"window", "vent"} and box_area > 0.08 * env_area:
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
