from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from app.pipeline.geometry import Region
from app.settings import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load():
    from ultralytics import SAM

    return SAM(settings.sam_weights)


def refine_regions(bgr: np.ndarray, regions: list[Region]) -> list[Region]:
    if not settings.enable_sam:
        return regions
    boxes = [r for r in regions if r.box and r.source in {"florence", "yolo"}]
    if not boxes:
        return regions
    try:
        model = _load()
    except Exception:
        log.exception("SAM unavailable")
        return regions

    refined: list[Region] = []
    kept_ids = set()
    rgb = bgr[:, :, ::-1]
    for region in boxes:
        x1, y1, x2, y2 = region.box  # type: ignore[misc]
        try:
            results = model.predict(
                rgb,
                bboxes=[x1, y1, x2, y2],
                verbose=False,
            )
        except Exception:
            log.exception("SAM predict failed")
            refined.append(region)
            continue
        mask = None
        if results and results[0].masks is not None:
            m = results[0].masks.data[0].cpu().numpy()
            mask = (m > 0.5).astype(np.uint8) * 255
            if mask.shape != bgr.shape[:2]:
                import cv2

                mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask is None or int(np.count_nonzero(mask)) < 20:
            refined.append(region)
            continue
        refined.append(
            Region(
                label=region.label,
                mask=mask,
                score=max(region.score, 0.85),
                source="sam",
                box=region.box,
            )
        )
        kept_ids.add(id(region))
    for region in regions:
        if id(region) not in kept_ids:
            refined.append(region)
    return refined
