from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.classes import CLASSES
from app.pipeline.snap import resolve_priority
from app.settings import settings


def solidify_mask(mask: np.ndarray, kernel: int = 9, min_area: int | None = None) -> np.ndarray:
    """Turn speckled pixels into connected filled polygons."""
    if mask is None or int(np.count_nonzero(mask)) == 0:
        return mask if mask is not None else np.zeros((1, 1), np.uint8)
    k = max(3, kernel | 1)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(closed)
    if min_area is None:
        min_area = max(40, int(closed.shape[0] * closed.shape[1] * 0.00015))
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(1.0, 0.002 * peri), True)
        cv2.drawContours(out, [approx], -1, 255, thickness=cv2.FILLED)
    holes = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return holes


def solidify_masks(masks: dict[str, np.ndarray], envelope: np.ndarray | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        if name in {"window", "vent", "pipe"}:
            solidified = solidify_mask(mask, kernel=3, min_area=20)
        else:
            solidified = solidify_mask(mask)
        if envelope is not None:
            solidified = cv2.bitwise_and(solidified, envelope)
        out[name] = solidified
    return resolve_priority(out, envelope)


def render_mask_layer(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Colored mask on transparent background (BGRA). Original drawing is not in this image."""
    h, w = next(iter(masks.values())).shape[:2]
    layer = np.zeros((h, w, 4), np.uint8)
    alpha = int(np.clip(settings.overlay_alpha, 0.2, 0.85) * 255)
    for cls in sorted(CLASSES.values(), key=lambda c: c.priority):
        mask = masks.get(cls.name)
        if mask is None or int(np.count_nonzero(mask)) == 0:
            continue
        hit = mask > 0
        layer[hit, 0] = cls.bgr[0]
        layer[hit, 1] = cls.bgr[1]
        layer[hit, 2] = cls.bgr[2]
        layer[hit, 3] = alpha
    return layer


def composite_overlay(original_bgr: np.ndarray, mask_bgra: np.ndarray) -> np.ndarray:
    """Keep the CAD drawing intact; put the mask layer on top."""
    base = original_bgr.copy()
    alpha = mask_bgra[:, :, 3:4].astype(np.float32) / 255.0
    color = mask_bgra[:, :, :3].astype(np.float32)
    src = base.astype(np.float32)
    out = src * (1.0 - alpha) + color * alpha
    ink = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) < 90
    result = out.astype(np.uint8)
    result[ink] = original_bgr[ink]
    return result


def render_overlay(bgr: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    layer = render_mask_layer(masks)
    return composite_overlay(bgr, layer)
