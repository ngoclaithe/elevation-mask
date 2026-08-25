from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.classes import CLASSES
from app.pipeline.geometry import PerceiveResult, Region


def _clip(mask: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(mask, envelope)


def _fill_to_ink(mask: np.ndarray, envelope: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Expand a blob until it hits CAD ink, staying inside the envelope."""
    h, w = mask.shape
    seed = mask.copy()
    if int(np.count_nonzero(seed)) == 0:
        return seed
    blocked = ((ink > 0) | (envelope == 0)).astype(np.uint8) * 255
    # Walk from mask centroid via flood on non-ink paper, then keep only
    # the connected paper component overlapping the original mask a lot.
    paper = np.where((envelope > 0) & (ink == 0), 255, 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(paper)
    best = seed
    best_overlap = 0
    seed_bin = seed > 0
    for i in range(1, n):
        comp = labels == i
        overlap = int(np.count_nonzero(comp & seed_bin))
        if overlap > best_overlap:
            best_overlap = overlap
            best = (comp.astype(np.uint8) * 255)
    if best_overlap < 10:
        return _clip(seed, envelope)
    merged = cv2.bitwise_or(seed, best)
    # Do not jump over blocked ink into a huge wall when the seed is a window.
    seed_area = int(np.count_nonzero(seed))
    merged_area = int(np.count_nonzero(merged))
    if seed_area > 0 and merged_area > seed_area * 8:
        return _clip(seed, envelope)
    _ = blocked
    _ = (h, w)
    return _clip(merged, envelope)


def snap_regions(perceived: PerceiveResult, regions: list[Region]) -> dict[str, np.ndarray]:
    h, w = perceived.envelope.shape
    stacked: dict[str, np.ndarray] = {name: np.zeros((h, w), np.uint8) for name in CLASSES}
    for region in regions:
        if region.label not in stacked:
            continue
        snapped = region.mask
        if region.source in {"florence", "sam"}:
            snapped = _fill_to_ink(region.mask, perceived.envelope, perceived.ink)
        else:
            snapped = _clip(region.mask, perceived.envelope)
        stacked[region.label] = cv2.bitwise_or(stacked[region.label], snapped)

    # Subtract higher-priority classes from lower ones.
    ordered = sorted(CLASSES.values(), key=lambda c: c.priority, reverse=True)
    claimed = np.zeros((h, w), np.uint8)
    out: dict[str, np.ndarray] = {}
    for cls in ordered:
        mask = cv2.bitwise_and(stacked[cls.name], cv2.bitwise_not(claimed))
        mask = cv2.bitwise_and(mask, perceived.envelope)
        out[cls.name] = mask
        if cls.name in {"window", "vent"}:
            claimed = cv2.bitwise_or(claimed, mask)
        else:
            claimed = cv2.bitwise_or(claimed, mask)
    return out
