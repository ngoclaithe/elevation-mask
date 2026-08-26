from __future__ import annotations

import logging
import time

import numpy as np

from app.pipeline.area import compute_areas
from app.pipeline.critic import Issue, critique
from app.pipeline.detector import propose_boxes
from app.pipeline.fixer import apply_fixes
from app.pipeline.geometry import PerceiveResult, Region, perceive
from app.pipeline.render import composite_overlay, render_mask_layer, solidify_masks
from app.pipeline.sam_seg import refine_regions
from app.pipeline.snap import snap_regions
from app.settings import settings

log = logging.getLogger(__name__)


def _vl_issues(bgr: np.ndarray, overlay: np.ndarray, enabled: bool | None = None) -> list[Issue]:
    is_enabled = settings.enable_vl_critic if enabled is None else enabled
    if not is_enabled:
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


def _filter_detections(regions: list[Region], perceived: PerceiveResult) -> list[Region]:
    env_area = max(int(np.count_nonzero(perceived.envelope)), 1)
    kept: list[Region] = []
    for region in regions:
        if region.box is None:
            continue
        x1, y1, x2, y2 = region.box
        area = max(1, (x2 - x1) * (y2 - y1))
        if region.label in {"window", "vent"} and area > 0.08 * env_area:
            continue
        if region.label.startswith("wall") and area > 0.35 * env_area:
            continue
        if region.label == "roof" and area > 0.50 * env_area:
            continue
        if (x2 - x1) > 0.85 * perceived.envelope.shape[1] and region.label != "roof":
            continue
        mask = region.mask
        if perceived.envelope is not None:
            mask = np.where(perceived.envelope > 0, mask, 0).astype(np.uint8)
            if int(np.count_nonzero(mask)) < 40:
                continue
        kept.append(
            Region(
                label=region.label,
                mask=mask,
                score=region.score,
                source=region.source,
                box=region.box,
            )
        )
    return kept


def run_agent(
    bgr: np.ndarray,
    max_iters: int | None = None,
    enable_florence: bool | None = None,
    enable_yolo_world: bool | None = None,
    enable_sam: bool | None = None,
    enable_vl_critic: bool | None = None,
) -> dict:
    iters = max_iters if max_iters is not None else settings.max_iters
    t0 = time.perf_counter()
    perceived = perceive(bgr)
    t_perceive = time.perf_counter()

    proposed: list[Region] = []
    proposed.extend(propose_boxes(perceived.bgr, enabled=enable_yolo_world))
    is_florence = settings.enable_florence if enable_florence is None else enable_florence
    if is_florence:
        from app.pipeline.florence import propose_boxes as florence_propose

        proposed.extend(florence_propose(perceived.bgr, enabled=True))

    detections = _filter_detections(proposed, perceived)
    t_detect = time.perf_counter()
    merged = refine_regions(perceived.bgr, list(perceived.regions) + detections, enabled=enable_sam)
    t_sam = time.perf_counter()
    masks = snap_regions(perceived, merged)

    last_issues: list[Issue] = []
    trace: list[dict] = []
    overlay = composite_overlay(perceived.bgr, render_mask_layer(solidify_masks(masks, perceived.envelope)))

    for step in range(iters):
        last_issues = critique(perceived, masks)
        overlay = composite_overlay(perceived.bgr, render_mask_layer(solidify_masks(masks, perceived.envelope)))
        last_issues.extend(_vl_issues(perceived.bgr, overlay, enabled=enable_vl_critic))
        trace.append(
            {
                "iter": step,
                "issues": [i.to_dict() for i in last_issues],
                "detector_boxes": len(detections),
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
        "detector_ms": round((t_detect - t_perceive) * 1000),
        "sam_ms": round((t_sam - t_detect) * 1000),
        "rest_ms": round((t_end - t_sam) * 1000),
        "total_ms": round((t_end - t0) * 1000),
    }
    log.info("agent timing %s detector=%s", timing, len(detections))
    is_yolo = settings.enable_yolo_world if enable_yolo_world is None else enable_yolo_world
    detector_name = "yolo-world" if is_yolo else ("florence" if is_florence else "none")
    return {
        "masks": masks,
        "mask_layer": mask_layer,
        "overlay": overlay,
        "areas": compute_areas(masks, perceived.envelope),
        "trace": trace,
        "envelope_pixels": int(np.count_nonzero(perceived.envelope)),
        "meta": {
            "detector": detector_name,
            "geometry": "leftover-prior",
            "eave_y": perceived.eave_y,
            "floor_y": perceived.floor_y,
            "foundation_y": perceived.foundation_y,
            "detector_boxes": len(detections),
            "iters": len(trace),
            "faces": len(perceived.faces),
            "open_issues": [i.to_dict() for i in last_issues],
            "timing": timing,
        },
    }
