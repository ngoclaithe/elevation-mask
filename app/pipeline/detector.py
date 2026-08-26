from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

from app.pipeline.classes import canonical_label
from app.pipeline.geometry import Region
from app.settings import settings

log = logging.getLogger(__name__)

_CLASSES = [
    "window",
    "sliding window",
    "roof",
    "tile roof",
    "gable",
    "vent",
    "louver",
    "foundation",
    "door",
]


@lru_cache(maxsize=1)
def _load():
    from ultralytics import YOLOWorld

    model = YOLOWorld(settings.yolo_world_id)
    model.set_classes(_CLASSES)
    return model


def _thicken_cad(bgr: np.ndarray) -> np.ndarray:
    """Photo detectors miss 1px CAD ink; thicken strokes so frames read as objects."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 90).astype(np.uint8) * 255
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    out = np.full_like(bgr, 255)
    out[ink > 0] = (20, 20, 20)
    return out


def propose_boxes(bgr: np.ndarray, enabled: bool | None = None) -> list[Region]:
    is_enabled = settings.enable_yolo_world if enabled is None else enabled
    if not is_enabled:
        return []
    try:
        model = _load()
    except Exception:
        log.exception("YOLO-World unavailable")
        return []

    h, w = bgr.shape[:2]
    thick = _thicken_cad(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 90).astype(np.uint8) * 255
    inv = cv2.cvtColor(255 - ink, cv2.COLOR_GRAY2BGR)
    views = [bgr, thick, inv]
    regions: list[Region] = []
    try:
        for img in views:
            results = model.predict(img, conf=settings.yolo_conf, verbose=False, device=settings.device)
            if not results:
                continue
            res = results[0]
            if res.boxes is None:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for box, cid, score in zip(xyxy, cls_ids, confs):
                raw = _CLASSES[int(cid)] if 0 <= int(cid) < len(_CLASSES) else ""
                label = canonical_label(raw)
                if label is None:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw < 10 or bh < 10:
                    continue
                if label in {"window", "vent"} and (bw > 0.35 * w or bh > 0.40 * h):
                    continue
                mask = np.zeros((h, w), np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                regions.append(
                    Region(label=label, mask=mask, score=float(score), source="yolo", box=(x1, y1, x2, y2))
                )
    except Exception:
        log.exception("YOLO-World predict failed")
        return []
    log.info("yolo-world kept %s regions", len(regions))
    return regions
