from __future__ import annotations

import logging
import time

import numpy as np

from app.pipeline.area import compute_areas
from app.pipeline.critic import Issue, critique
from app.pipeline.fixer import apply_fixes
from app.pipeline.geometry import PerceiveResult, Region, perceive
from app.pipeline.render import composite_overlay, render_mask_layer, solidify_masks
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


def _florence_boxes(bgr: np.ndarray) -> list[Region]:
    if not settings.enable_florence:
        return []
    from app.pipeline.florence import propose_boxes

    return propose_boxes(bgr)


def _filter_florence(regions: list[Region], perceived: PerceiveResult) -> list[Region]:
    env_area = max(int(np.count_nonzero(perceived.envelope)), 1)
    kept: list[Region] = []
    for region in regions:
        if region.box is None:
            continue
        x1, y1, x2, y2 = region.box
        cy = 0.5 * (y1 + y2)
        if region.label == "roof":
            mask = region.mask.copy()
            mask[perceived.eave_y :, :] = 0
            if int(np.count_nonzero(mask)) < 40:
                continue
            kept.append(
                Region(label="roof", mask=mask, score=region.score, source=region.source, box=region.box)
            )
            continue
        if region.label in {"window", "vent"}:
            if cy < perceived.eave_y or cy > perceived.foundation_y:
                continue
        if region.label.startswith("wall") and (x2 - x1) * (y2 - y1) > 0.3 * env_area:
            continue
        kept.append(region)
    return kept


def run_agent(bgr: np.ndarray, max_iters: int | None = None) -> dict:
    iters = max_iters if max_iters is not None else settings.max_iters
    t0 = time.perf_counter()
    perceived = perceive(bgr)
    t_perceive = time.perf_counter()
    florence = _filter_florence(_florence_boxes(perceived.bgr), perceived)
    t_florence = time.perf_counter()
    merged = refine_regions(perceived.bgr, list(perceived.regions) + florence)
    t_sam = time.perf_counter()
    masks = snap_regions(perceived, merged)

    last_issues: list[Issue] = []
    trace: list[dict] = []
    overlay = composite_overlay(perceived.bgr, render_mask_layer(solidify_masks(masks, perceived.envelope)))

    for step in range(iters):
        last_issues = critique(perceived, masks)
        overlay = composite_overlay(perceived.bgr, render_mask_layer(solidify_masks(masks, perceived.envelope)))
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
        masks = snap_regions(perceived, _as_regions(masks))

    masks = solidify_masks(masks, perceived.envelope)
    mask_layer = render_mask_layer(masks)
    overlay = composite_overlay(bgr, mask_layer)

    t_end = time.perf_counter()
    timing = {
        "perceive_ms": round((t_perceive - t0) * 1000),
        "florence_ms": round((t_florence - t_perceive) * 1000),
        "sam_ms": round((t_sam - t_florence) * 1000),
        "rest_ms": round((t_end - t_sam) * 1000),
        "total_ms": round((t_end - t0) * 1000),
    }
    log.info("agent timing %s", timing)
    return {
        "masks": masks,
        "mask_layer": mask_layer,
        "overlay": overlay,
        "areas": compute_areas(masks, perceived.envelope),
        "trace": trace,
        "envelope_pixels": int(np.count_nonzero(perceived.envelope)),
        "meta": {
            "geometry": "silhouette",
            "eave_y": perceived.eave_y,
            "floor_y": perceived.floor_y,
            "foundation_y": perceived.foundation_y,
            "florence_boxes": len(florence),
            "iters": len(trace),
            "open_issues": [i.__dict__ for i in last_issues],
            "timing": timing,
        },
    }
