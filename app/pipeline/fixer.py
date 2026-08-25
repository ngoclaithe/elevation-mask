from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.critic import Issue
from app.pipeline.geometry import PerceiveResult


def apply_fixes(
    perceived: PerceiveResult,
    masks: dict[str, np.ndarray],
    issues: list[Issue],
) -> dict[str, np.ndarray]:
    env = perceived.envelope
    labeled = np.zeros(env.shape, dtype=np.uint8)
    for mask in masks.values():
        labeled = cv2.bitwise_or(labeled, mask)
    holes = cv2.bitwise_and(env, cv2.bitwise_not(labeled))

    for issue in issues:
        if issue.kind == "coverage":
            for name in ("roof", "wall_l2", "wall_l1", "foundation"):
                prior = perceived.geometry.get(name)
                if prior is None:
                    continue
                masks[name] = cv2.bitwise_or(masks[name], cv2.bitwise_and(holes, prior))
        elif issue.kind == "missing" and issue.label:
            prior = perceived.geometry.get(issue.label)
            if prior is not None:
                masks[issue.label] = cv2.bitwise_or(masks[issue.label], prior)
        elif issue.kind == "topology" and issue.label in {"window", "vent"}:
            roof = perceived.geometry.get("roof", np.zeros_like(env))
            found = perceived.geometry.get("foundation", np.zeros_like(env))
            keep = cv2.bitwise_and(env, cv2.bitwise_not(cv2.bitwise_or(roof, found)))
            masks[issue.label] = cv2.bitwise_and(masks[issue.label], keep)
        elif issue.kind == "topology" and issue.label == "foundation":
            prior = perceived.geometry.get("foundation")
            if prior is not None:
                masks["foundation"] = prior.copy()
    return masks
