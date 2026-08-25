from __future__ import annotations

import json
import logging

from app.pipeline.critic import Issue

log = logging.getLogger(__name__)

_PROMPT = """You are checking an architectural elevation segmentation overlay.
Original drawing is the CAD linework. Overlay colors:
- red = roof, green = wall floor 1, cyan = wall floor 2
- magenta = foundation, yellow = window, brown = vent, gray = pipe
Return JSON only: {"issues":[{"kind":"coverage|missing|topology|overlap","message":"...","label":"roof|wall_l1|wall_l2|window|vent|foundation|pipe|null"}]}
If the overlay looks correct, return {"issues":[]}."""


def critique_overlay(bgr: np.ndarray, overlay: np.ndarray) -> list[Issue]:
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import torch

    from app.settings import settings

    device = torch.device(settings.device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        settings.vl_model,
        torch_dtype=torch.float32,
    ).to(device)
    processor = AutoProcessor.from_pretrained(settings.vl_model)

    orig = Image.fromarray(bgr[:, :, ::-1])
    over = Image.fromarray(overlay[:, :, ::-1])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": orig},
                {"type": "image", "image": over},
                {"type": "text", "text": _PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[orig, over], return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=400)
    raw = processor.batch_decode(out, skip_special_tokens=True)[0]
    start = raw.rfind("{")
    end = raw.rfind("}")
    if start < 0 or end < 0:
        return []
    data = json.loads(raw[start : end + 1])
    issues = []
    for item in data.get("issues") or []:
        issues.append(
            Issue(
                kind=str(item.get("kind") or "coverage"),
                message=str(item.get("message") or ""),
                label=item.get("label"),
            )
        )
    return issues
