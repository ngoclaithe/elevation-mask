from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.classes import CLASSES
from app.pipeline.geometry import PerceiveResult, Region


def _clip(mask: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(mask, envelope)


def _fill_to_ink(mask: np.ndarray, envelope: np.ndarray, ink: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Expand a detection inside its local bounding box up to sealed CAD ink lines."""
    h, w = mask.shape
    seed = mask.copy()
    seed_area = int(np.count_nonzero(seed))
    if seed_area == 0:
        return seed

    env_area = max(int(np.count_nonzero(envelope)), 1)
    max_allowed = min(int(max(seed_area * 1.4, 200)), int(0.045 * env_area))

    # Determine bounding ROI with a small margin
    if box is not None:
        bx1, by1, bx2, by2 = box
    else:
        ys, xs = np.where(seed > 0)
        bx1, bx2 = int(xs.min()), int(xs.max())
        by1, by2 = int(ys.min()), int(ys.max())

    margin = 8
    rx1, ry1 = max(0, bx1 - margin), max(0, by1 - margin)
    rx2, ry2 = min(w, bx2 + margin + 1), min(h, by2 + margin + 1)

    # Local ROI processing to prevent leakage across whole building
    roi_env = envelope[ry1:ry2, rx1:rx2]
    roi_ink = ink[ry1:ry2, rx1:rx2]
    roi_seed = seed[ry1:ry2, rx1:rx2]

    # Morphological close on ink to seal tiny 1px drafting gaps
    sealed_ink = cv2.dilate(roi_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    roi_paper = np.where((roi_env > 0) & (sealed_ink == 0), 255, 0).astype(np.uint8)

    n, labels = cv2.connectedComponents(roi_paper)
    best_roi = roi_seed
    best_overlap = 0
    seed_bin = roi_seed > 0

    for i in range(1, n):
        comp = labels == i
        overlap = int(np.count_nonzero(comp & seed_bin))
        comp_area = int(np.count_nonzero(comp))
        if overlap > best_overlap and comp_area <= max_allowed:
            best_overlap = overlap
            best_roi = (comp.astype(np.uint8) * 255)

    out = seed.copy()
    if best_overlap > 10:
        out[ry1:ry2, rx1:rx2] = cv2.bitwise_or(roi_seed, best_roi)

    out_area = int(np.count_nonzero(out))
    if out_area > max_allowed * 1.5:
        return _clip(seed, envelope)
    return _clip(out, envelope)


def resolve_priority(masks: dict[str, np.ndarray], envelope: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Higher-priority classes punch holes in lower ones (windows stay on top of walls)."""
    if not masks:
        return masks
    h, w = next(iter(masks.values())).shape[:2]
    claimed = np.zeros((h, w), np.uint8)
    out: dict[str, np.ndarray] = {}
    for cls in sorted(CLASSES.values(), key=lambda c: c.priority, reverse=True):
        mask = masks.get(cls.name)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(claimed))
        if envelope is not None:
            mask = cv2.bitwise_and(mask, envelope)
        out[cls.name] = mask
        claimed = cv2.bitwise_or(claimed, mask)
    return out


def snap_regions(perceived: PerceiveResult, regions: list[Region]) -> dict[str, np.ndarray]:
    h, w = perceived.envelope.shape
    stacked: dict[str, np.ndarray] = {name: np.zeros((h, w), np.uint8) for name in CLASSES}

    # Match regions to perceived geometric faces for pixel-perfect CAD alignment
    used_face_ids = set()
    for region in regions:
        if region.label not in stacked:
            continue

        # Only discrete object classes can be proposed by object detectors
        if region.source in {"yolo", "florence", "sam"}:
            if region.label not in {"window", "vent", "pipe"}:
                continue

        if region.label in {"window", "vent"}:
            # Check overlap with true CAD opening faces
            matched_face = None
            for face in perceived.faces:
                if id(face) in used_face_ids:
                    continue
                overlap = int(np.count_nonzero((region.mask > 0) & (face.mask > 0)))
                face_area = int(np.count_nonzero(face.mask > 0))
                if face_area > 0 and overlap / face_area >= 0.25:
                    matched_face = face
                    used_face_ids.add(id(face))
                    break

            if matched_face is not None:
                snapped = matched_face.mask
            else:
                snapped = _fill_to_ink(region.mask, perceived.envelope, perceived.ink, region.box)
        elif region.label == "pipe":
            snapped = _clip(region.mask, perceived.envelope)
        else:
            snapped = _clip(region.mask, perceived.envelope)

        stacked[region.label] = cv2.bitwise_or(stacked[region.label], snapped)

    # Ensure all detected geometric faces are included if not yet claimed
    for face in perceived.faces:
        if face.label in stacked and int(np.count_nonzero(stacked[face.label] & face.mask)) == 0:
            stacked[face.label] = cv2.bitwise_or(stacked[face.label], face.mask)

    # Subtract higher-priority classes from lower ones.
    ordered = sorted(CLASSES.values(), key=lambda c: c.priority, reverse=True)
    claimed = np.zeros((h, w), np.uint8)
    out: dict[str, np.ndarray] = {}
    for cls in ordered:
        mask = cv2.bitwise_and(stacked[cls.name], cv2.bitwise_not(claimed))
        mask = cv2.bitwise_and(mask, perceived.envelope)
        out[cls.name] = mask
        claimed = cv2.bitwise_or(claimed, mask)

    # Fill remaining building body with geometric priors (wall L1, wall L2, roof, foundation)
    leftover = cv2.bitwise_and(perceived.envelope, cv2.bitwise_not(claimed))
    if int(np.count_nonzero(leftover)):
        fill_names = ["wall_l2", "wall_l1", "foundation", "roof"]
        for name in fill_names:
            prior = perceived.geometry.get(name)
            if prior is None:
                continue
            fill = cv2.bitwise_and(leftover, prior)
            out[name] = cv2.bitwise_or(out[name], fill)

    return resolve_priority(out, perceived.envelope)
