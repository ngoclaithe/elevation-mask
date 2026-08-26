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
        if issue.kind == "window_overgrowth":
            # Reclaim bloated window areas and reset to true CAD opening faces
            true_windows = np.zeros_like(env)
            for face in perceived.faces:
                if face.label == "window":
                    true_windows = cv2.bitwise_or(true_windows, face.mask)
            bloat = cv2.bitwise_and(masks.get("window", np.zeros_like(env)), cv2.bitwise_not(true_windows))
            masks["window"] = true_windows

            # Reassign bloated area to corresponding wall layer
            if int(np.count_nonzero(bloat)) > 0:
                h, w = env.shape
                yy = np.arange(h)[:, None]
                two_wall = cv2.bitwise_and(bloat, (yy < perceived.floor_y).astype(np.uint8) * 255)
                one_wall = cv2.bitwise_and(bloat, (yy >= perceived.floor_y).astype(np.uint8) * 255)
                if "wall_l2" in masks:
                    masks["wall_l2"] = cv2.bitwise_or(masks["wall_l2"], two_wall)
                if "wall_l1" in masks:
                    masks["wall_l1"] = cv2.bitwise_or(masks["wall_l1"], one_wall)

        elif issue.kind in {"wrong_class", "reclassify", "leak"} and issue.mask is not None:
            bit = issue.mask
            if bit.shape[:2] != env.shape[:2]:
                bit = cv2.resize(bit, (env.shape[1], env.shape[0]), interpolation=cv2.INTER_NEAREST)
            bit = cv2.bitwise_and(bit.astype(np.uint8), env)
            if issue.label in masks:
                masks[issue.label] = cv2.bitwise_or(masks[issue.label], bit)
                for name in list(masks.keys()):
                    if name != issue.label:
                        masks[name] = cv2.bitwise_and(masks[name], cv2.bitwise_not(bit))
            else:
                # If label is None or 'none', restore to wall
                h, w = env.shape
                yy = np.arange(h)[:, None]
                two_wall = cv2.bitwise_and(bit, (yy < perceived.floor_y).astype(np.uint8) * 255)
                one_wall = cv2.bitwise_and(bit, (yy >= perceived.floor_y).astype(np.uint8) * 255)
                for name in list(masks.keys()):
                    masks[name] = cv2.bitwise_and(masks[name], cv2.bitwise_not(bit))
                if "wall_l2" in masks:
                    masks["wall_l2"] = cv2.bitwise_or(masks["wall_l2"], two_wall)
                if "wall_l1" in masks:
                    masks["wall_l1"] = cv2.bitwise_or(masks["wall_l1"], one_wall)

        elif issue.kind == "missing":
            if issue.mask is not None and issue.label in masks:
                bit = issue.mask
                if bit.shape[:2] != env.shape[:2]:
                    bit = cv2.resize(bit, (env.shape[1], env.shape[0]), interpolation=cv2.INTER_NEAREST)
                bit = cv2.bitwise_and(bit.astype(np.uint8), env)
                masks[issue.label] = cv2.bitwise_or(masks[issue.label], bit)
                for name in list(masks.keys()):
                    if name != issue.label:
                        masks[name] = cv2.bitwise_and(masks[name], cv2.bitwise_not(bit))
            elif issue.label:
                prior = perceived.geometry.get(issue.label)
                if prior is not None:
                    masks[issue.label] = cv2.bitwise_or(masks[issue.label], prior)

        elif issue.kind == "coverage":
            if issue.mask is not None and issue.label in masks:
                bit = issue.mask
                if bit.shape[:2] != env.shape[:2]:
                    bit = cv2.resize(bit, (env.shape[1], env.shape[0]), interpolation=cv2.INTER_NEAREST)
                bit = cv2.bitwise_and(bit.astype(np.uint8), env)
                masks[issue.label] = cv2.bitwise_or(masks[issue.label], bit)
                for name in list(masks.keys()):
                    if name != issue.label:
                        masks[name] = cv2.bitwise_and(masks[name], cv2.bitwise_not(bit))
            else:
                for name in ("roof", "wall_l2", "wall_l1", "foundation"):
                    prior = perceived.geometry.get(name)
                    if prior is None:
                        continue
                    masks[name] = cv2.bitwise_or(masks[name], cv2.bitwise_and(holes, prior))

    return masks
