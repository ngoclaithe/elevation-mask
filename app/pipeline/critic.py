from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.classes import CLASSES
from app.pipeline.geometry import PerceiveResult


@dataclass
class Issue:
    kind: str
    message: str
    label: str | None = None
    mask: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, "label": self.label}


def _majority_label(masks: dict[str, np.ndarray], face_mask: np.ndarray) -> str | None:
    best_name = None
    best_n = 0
    face = face_mask > 0
    n_face = int(np.count_nonzero(face))
    if n_face < 20:
        return None
    for name, mask in masks.items():
        n = int(np.count_nonzero(face & (mask > 0)))
        if n > best_n:
            best_n = n
            best_name = name
    if best_n < 0.40 * n_face:
        return None
    return best_name


def critique(perceived: PerceiveResult, masks: dict[str, np.ndarray]) -> list[Issue]:
    issues: list[Issue] = []
    env = perceived.envelope > 0
    env_area = int(np.count_nonzero(env))
    if env_area < 100:
        issues.append(Issue("no_building", "Could not find a building envelope"))
        return issues

    # 1. Check if geometric opening faces are properly labeled
    for face in perceived.faces:
        current = _majority_label(masks, face.mask)
        if current == face.label:
            continue
        if face.label in {"window", "vent"} and current in {"wall_l1", "wall_l2", None}:
            issues.append(
                Issue("reclassify", f"{face.label} face labeled {current}", face.label, face.mask)
            )
        elif face.label == "pipe" and current in {"wall_l1", "wall_l2", None}:
            issues.append(Issue("reclassify", "pipe masked by wall", "pipe", face.mask))
        elif face.label == "roof" and current in {"wall_l1", "wall_l2", None}:
            issues.append(Issue("reclassify", f"roof hatch labeled {current}", "roof", face.mask))

    # 2. Check for Window Leak / Overgrowth
    win_mask = masks.get("window", np.zeros_like(perceived.envelope))
    win_area = int(np.count_nonzero(win_mask))
    if win_area / env_area > 0.18:
        issues.append(
            Issue(
                "window_overgrowth",
                f"Window area is abnormally high ({win_area / env_area:.1%})",
                "window",
                win_mask,
            )
        )
    else:
        # Check individual window components for giant anomalous blobs
        n_w, l_w, stats_w, _ = cv2.connectedComponentsWithStats((win_mask > 0).astype(np.uint8))
        for i in range(1, n_w):
            comp_area = int(stats_w[i, cv2.CC_STAT_AREA])
            if comp_area > 0.045 * env_area:
                comp_mask = (l_w == i).astype(np.uint8) * 255
                issues.append(
                    Issue(
                        "window_overgrowth",
                        f"Single window component is too large ({comp_area / env_area:.1%})",
                        "window",
                        comp_mask,
                    )
                )
                break

    # 3. Check for coverage gaps
    labeled = np.zeros(env.shape, dtype=bool)
    for mask in masks.values():
        labeled |= mask > 0
    unlabeled = int(np.count_nonzero(env & ~labeled))
    if unlabeled / env_area > 0.02:
        issues.append(Issue("coverage", f"Unlabeled envelope {unlabeled / env_area:.1%}"))

    # 4. Check essential classes
    if int(np.count_nonzero(masks.get("roof", np.zeros_like(perceived.envelope)))) < env_area * 0.015:
        issues.append(Issue("missing", "Missing roof", "roof"))
    if int(np.count_nonzero(masks.get("wall_l1", np.zeros_like(perceived.envelope)))) < env_area * 0.015:
        issues.append(Issue("missing", "Missing first-floor wall", "wall_l1"))

    # 5. Check overlapping masks
    names = [c.name for c in CLASSES.values() if c.count_area]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = int(np.count_nonzero((masks[a] > 0) & (masks[b] > 0)))
            if overlap / env_area > 0.005:
                issues.append(Issue("overlap", f"{a} overlaps {b}", a))

    return issues
