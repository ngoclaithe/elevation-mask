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


def critique(perceived: PerceiveResult, masks: dict[str, np.ndarray]) -> list[Issue]:
    issues: list[Issue] = []
    env = perceived.envelope > 0
    env_area = int(np.count_nonzero(env))
    if env_area < 100:
        issues.append(Issue("no_building", "Could not find a building envelope"))
        return issues

    labeled = np.zeros(env.shape, dtype=bool)
    for name, mask in masks.items():
        labeled |= mask > 0
    unlabeled = int(np.count_nonzero(env & ~labeled))
    if unlabeled / env_area > 0.03:
        issues.append(
            Issue(
                "coverage",
                f"Unlabeled envelope {unlabeled / env_area:.1%}",
            )
        )

    if int(np.count_nonzero(masks.get("roof", np.zeros_like(perceived.envelope)))) < env_area * 0.02:
        issues.append(Issue("missing", "Missing roof", "roof"))
    if int(np.count_nonzero(masks.get("wall_l1", np.zeros_like(perceived.envelope)))) < env_area * 0.02:
        issues.append(Issue("missing", "Missing first-floor wall", "wall_l1"))

    # Windows/vents are subtracted from walls by priority, so they must not
    # be tested as a subset of wall masks. They should sit in the wall band.
    for hole in ("window", "vent"):
        m = masks.get(hole)
        if m is None or int(np.count_nonzero(m)) == 0:
            continue
        roof = perceived.geometry.get("roof")
        found = perceived.geometry.get("foundation")
        if roof is not None and int(np.count_nonzero((m > 0) & (roof > 0))) > 0.6 * int(np.count_nonzero(m)):
            issues.append(Issue("topology", f"{hole} mostly on roof", hole))
        elif found is not None and int(np.count_nonzero((m > 0) & (found > 0))) > 0.6 * int(
            np.count_nonzero(m)
        ):
            issues.append(Issue("topology", f"{hole} mostly on foundation", hole))

    foundation = masks.get("foundation")
    if foundation is not None and int(np.count_nonzero(foundation)):
        ys = np.where(foundation > 0)[0]
        env_bottom = np.where(env)[0].max() if np.any(env) else 0
        if ys.size and abs(int(ys.max()) - int(env_bottom)) > 12:
            issues.append(Issue("topology", "Foundation not at envelope bottom", "foundation"))

    # Overlap after priority should already be 0; flag if geometry still collides.
    names = [c.name for c in CLASSES.values() if c.count_area]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = int(np.count_nonzero((masks[a] > 0) & (masks[b] > 0)))
            if overlap / env_area > 0.005:
                issues.append(Issue("overlap", f"{a} overlaps {b}", a))
    return issues
