from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.classes import CLASSES
from app.settings import settings


def render_overlay(bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    overlay = bgr.copy()
    color = np.zeros_like(bgr)
    for name, cls in CLASSES.items():
        mask = masks.get(name)
        if mask is None:
            continue
        color[mask > 0] = cls.bgr
    blended = cv2.addWeighted(overlay, 1.0 - settings.overlay_alpha, color, settings.overlay_alpha, 0)
    ink = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lines = ink < 80
    blended[lines] = bgr[lines]
    return blended
