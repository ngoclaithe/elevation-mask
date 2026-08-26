from __future__ import annotations

import base64
import json
import logging
import cv2
import numpy as np
import requests

from app.pipeline.critic import Issue
from app.settings import settings

log = logging.getLogger(__name__)

_PROMPT = """You are an expert architectural CAD QA inspector reviewing an elevation segmentation overlay.
Image 1: Original architectural CAD elevation drawing (black & white linework).
Image 2: Color segmentation overlay:
- Red = Roof (only sloped/tiled roof surfaces)
- Cyan = Wall Floor 2 (wall facade of 2nd story, parapet tower, gable wall)
- Green = Wall Floor 1 (wall facade of 1st story)
- Magenta = Foundation (bottom concrete plinth above ground line)
- Yellow = Window / Glass Door (rectangular frames with sashes)
- Brown = Vent / Louver (small square vents)
- Gray = Pipe (vertical downspout pipes)

Carefully compare Image 1 and Image 2:
1. Check if any windows, vents, or doors were missed or misclassified.
2. Check if roof color (Red) incorrectly spills into walls or if wall color spills into roof.
3. Check if Floor 1 / Floor 2 wall boundary is correct.
4. Check if any text/dimension lines outside the building are mistakenly colored.

Return JSON ONLY in this format:
{
  "pass": true | false,
  "summary": "...",
  "issues": [
    {
      "kind": "wrong_class" | "missing" | "leak" | "coverage",
      "box_2d": [ymin, xmin, ymax, xmax],
      "current_label": "roof" | "wall_l1" | "wall_l2" | "window" | "vent" | "foundation" | "pipe" | "none",
      "correct_label": "roof" | "wall_l1" | "wall_l2" | "window" | "vent" | "foundation" | "pipe" | "none",
      "reason": "..."
    }
  ]
}
Note: Coordinates in box_2d are normalized between 0 and 1000 (integers). If no issues, return {"pass": true, "summary": "Segmentation is accurate", "issues": []}.
"""


def critique_with_gemini(bgr: np.ndarray, overlay: np.ndarray) -> list[Issue]:
    api_key = settings.gemini_api_key
    if not api_key:
        return []

    try:
        _, buf_orig = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        _, buf_over = cv2.imencode(".png", overlay)
        orig_b64 = base64.b64encode(buf_orig).decode("utf-8")
        over_b64 = base64.b64encode(buf_over).decode("utf-8")

        model_name = settings.gemini_model or "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": orig_b64}},
                        {"inline_data": {"mime_type": "image/png", "data": over_b64}},
                        {"text": _PROMPT},
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        res = requests.post(url, json=payload, timeout=25)
        if res.status_code != 200:
            log.warning("Gemini Critic failed with status %d: %s", res.status_code, res.text)
            return []

        data = res.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)

        h, w = bgr.shape[:2]
        issues: list[Issue] = []

        for item in result.get("issues", []):
            kind = str(item.get("kind") or "coverage")
            reason = str(item.get("reason") or item.get("message") or "")
            corr_label = item.get("correct_label")
            curr_label = item.get("current_label")

            box_norm = item.get("box_2d")
            mask = None
            if box_norm and len(box_norm) == 4:
                ymin, xmin, ymax, xmax = box_norm
                px1 = max(0, min(w - 1, int(xmin * w / 1000.0)))
                py1 = max(0, min(h - 1, int(ymin * h / 1000.0)))
                px2 = max(px1 + 1, min(w, int(xmax * w / 1000.0)))
                py2 = max(py1 + 1, min(h, int(ymax * h / 1000.0)))
                mask = np.zeros((h, w), np.uint8)
                mask[py1:py2, px1:px2] = 255

            issues.append(
                Issue(
                    kind=kind,
                    message=f"{reason} (curr={curr_label}, corr={corr_label})",
                    label=corr_label if corr_label != "none" else None,
                    mask=mask,
                )
            )

        log.info("Gemini Critic returned %d issues (pass=%s)", len(issues), result.get("pass"))
        return issues
    except Exception:
        log.exception("Error during Gemini critic call")
        return []
