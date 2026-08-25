from __future__ import annotations

import logging

import numpy as np

from app.pipeline.area import compute_areas
from app.pipeline.critic import Issue, critique
from app.pipeline.fixer import apply_fixes
from app.pipeline.florence import propose_boxes
from app.pipeline.geometry import Region, perceive
from app.pipeline.render import render_overlay
from app.pipeline.sam_seg import refine_regions
from app.pipeline.snap import snap_regions
from app.settings import settings

log = logging.getLogger(__name__)


def _vl_issues(bgr: np.ndarray, overlay: np.ndarray) -> list[Issue]:
    if not settings.enable_vl_critic:
        return []
    try:
        from app.pipeline.qwen_vl import critique_overlay

        return critique_overlay(bgr, overlay)
    except Exception:
        log.exception("VL critic failed")
        return []


def _as_regions(masks: dict[str, np.ndarray]) -> list[Region]:
    return [
        Region(label=name, mask=mask, score=1.0, source="fixer")
        for name, mask in masks.items()
        if int(np.count_nonzero(mask)) > 0
    ]


def run_agent(bgr: np.ndarray, max_iters: int | None = None) -> dict:
    iters = max_iters if max_iters is not None else settings.max_iters
    perceived = perceive(bgr)
    florence = propose_boxes(perceived.bgr)
    merged = refine_regions(perceived.bgr, list(perceived.regions) + florence)
    masks = snap_regions(perceived, merged)

    overlay = render_overlay(perceived.bgr, masks)
    last_issues: list[Issue] = []
    trace: list[dict] = []

    for step in range(iters):
        last_issues = critique(perceived, masks)
        overlay = render_overlay(perceived.bgr, masks)
        last_issues.extend(_vl_issues(perceived.bgr, overlay))
        trace.append(
            {
                "iter": step,
                "issues": [i.__dict__ for i in last_issues],
                "florence_boxes": len(florence),
            }
        )
        if not last_issues:
            break
        masks = apply_fixes(perceived, masks, last_issues)
        masks = snap_regions(perceived, _as_regions(masks) + florence)

    return {
        "masks": masks,
        "overlay": overlay,
        "areas": compute_areas(masks, perceived.envelope),
        "trace": trace,
        "envelope_pixels": int(np.count_nonzero(perceived.envelope)),
        "meta": {
            "eave_y": perceived.eave_y,
            "floor_y": perceived.floor_y,
            "foundation_y": perceived.foundation_y,
            "florence_boxes": len(florence),
            "iters": len(trace),
            "open_issues": [i.__dict__ for i in last_issues],
        },
    }
