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
            # Unlabeled pixels above eave → roof; between eave and floor → L2;
            # between floor and foundation → L1; else foundation.
            h = env.shape[0]
            roof_band = np.zeros_like(holes)
            roof_band[: perceived.eave_y] = holes[: perceived.eave_y]
            l2_band = np.zeros_like(holes)
            l2_band[perceived.eave_y : perceived.floor_y] = holes[perceived.eave_y : perceived.floor_y]
            l1_band = np.zeros_like(holes)
            l1_band[perceived.floor_y : perceived.foundation_y] = holes[
                perceived.floor_y : perceived.foundation_y
            ]
            found_band = np.zeros_like(holes)
            found_band[perceived.foundation_y : h] = holes[perceived.foundation_y : h]
            masks["roof"] = cv2.bitwise_or(masks["roof"], roof_band)
            masks["wall_l2"] = cv2.bitwise_or(masks["wall_l2"], l2_band)
            masks["wall_l1"] = cv2.bitwise_or(masks["wall_l1"], l1_band)
            masks["foundation"] = cv2.bitwise_or(masks["foundation"], found_band)
        elif issue.kind == "missing" and issue.label:
            if issue.label == "roof":
                fill = env.copy()
                fill[perceived.eave_y :, :] = 0
                masks["roof"] = cv2.bitwise_or(masks["roof"], fill)
            elif issue.label == "wall_l1":
                fill = env.copy()
                fill[: perceived.floor_y, :] = 0
                fill[perceived.foundation_y :, :] = 0
                masks["wall_l1"] = cv2.bitwise_or(masks["wall_l1"], fill)
        elif issue.kind == "topology" and issue.label in {"window", "vent"}:
            band = env.copy()
            band[: perceived.eave_y] = 0
            band[perceived.foundation_y :] = 0
            masks[issue.label] = cv2.bitwise_and(masks[issue.label], band)
        elif issue.kind == "topology" and issue.label == "foundation":
            fill = env.copy()
            fill[: perceived.foundation_y, :] = 0
            masks["foundation"] = fill
    return masks
