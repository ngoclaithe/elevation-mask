from __future__ import annotations

import numpy as np

from app.pipeline.classes import CLASSES


def compute_areas(
    masks: dict[str, np.ndarray],
    envelope: np.ndarray,
    scale_mm_per_px: float | None = None,
) -> dict[str, dict]:
    env_area = max(int(np.count_nonzero(envelope)), 1)
    out: dict[str, dict] = {}
    for name, cls in CLASSES.items():
        mask = masks.get(name)
        pixels = int(np.count_nonzero(mask)) if mask is not None else 0
        item = {
            "pixels": pixels,
            "percent_of_envelope": round(100.0 * pixels / env_area, 3),
            "count_area": cls.count_area,
        }
        if scale_mm_per_px is not None:
            mm2 = pixels * (scale_mm_per_px ** 2)
            item["square_mm"] = round(mm2, 2)
            item["square_m"] = round(mm2 / 1_000_000.0, 6)
        out[name] = item
    out["_envelope"] = {
        "pixels": env_area,
        "percent_of_envelope": 100.0,
        "count_area": False,
    }
    return out
