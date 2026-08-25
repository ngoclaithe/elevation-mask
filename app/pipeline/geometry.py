from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    label: str
    mask: np.ndarray
    score: float
    source: str
    box: tuple[int, int, int, int] | None = None


@dataclass
class PerceiveResult:
    bgr: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    envelope: np.ndarray
    eave_y: int
    floor_y: int
    foundation_y: int
    regions: list[Region] = field(default_factory=list)
    geometry: dict[str, np.ndarray] = field(default_factory=dict)


def _ensure_ink_black(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = 255 - bw
    return bw


def _envelope(paper: np.ndarray) -> np.ndarray:
    h, w = paper.shape
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    sealed = np.where(ink > 0, 0, 255).astype(np.uint8)
    flood = sealed.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, y), 128)
    env = np.where(flood != 128, 255, 0).astype(np.uint8)
    env = cv2.erode(env, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((env > 0).astype(np.uint8))
    if n <= 2:
        return env
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = stats[keep, cv2.CC_STAT_AREA]
    out = np.zeros_like(env)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.25 * largest:
            out[labels == i] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    return out


def _band_ink_rows(ink: np.ndarray, envelope: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = max(0, y0)
    y1 = min(ink.shape[0], max(y0 + 1, y1))
    band = ink[y0:y1] & envelope[y0:y1]
    return np.sum(band > 0, axis=1)


def _peak_row(scores: np.ndarray, y0: int) -> int:
    if scores.size == 0:
        return y0
    return int(y0 + np.argmax(scores))


def _column_profile(envelope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = envelope.shape
    env = envelope > 0
    has = env.any(axis=0)
    top = np.argmax(env, axis=0).astype(np.int32)
    bot = (h - 1 - np.argmax(env[::-1], axis=0)).astype(np.int32)
    top = np.where(has, top, -1).astype(np.int32)
    bot = np.where(has, bot, -1).astype(np.int32)
    return top, bot


def _fill_missing(arr: np.ndarray) -> np.ndarray:
    valid = arr >= 0
    if not valid.any():
        return arr.copy()
    idx = np.arange(len(arr))
    out = arr.astype(np.float64)
    out[~valid] = np.interp(idx[~valid], idx[valid], arr[valid].astype(np.float64))
    return np.round(out).astype(np.int32)


def _median_smooth(arr: np.ndarray, k: int = 15) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    padded = np.pad(arr.astype(np.int32), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1).astype(np.int32)


def _runs_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, hit in enumerate(mask):
        if hit and start is None:
            start = i
        elif (not hit) and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _horizontal_ink(ink: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    lines = ((ink > 0) & (envelope > 0)).astype(np.uint8) * 255
    return cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1)))


def _hatch_eave(
    horiz: np.ndarray,
    envelope: np.ndarray,
    top_s: np.ndarray,
    col_mask: np.ndarray,
    max_depth: int,
) -> int | None:
    xs = np.flatnonzero(col_mask)
    if xs.size < 8:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = int(np.median(top_s[col_mask]))
    y1 = min(envelope.shape[0], y0 + max(8, max_depth))
    energy = np.zeros(max(y1 - y0, 1), np.float32)
    for i, y in enumerate(range(y0, y1)):
        n = int(np.count_nonzero(envelope[y, x0:x1]))
        if n < 8:
            continue
        energy[i] = float(np.count_nonzero(horiz[y, x0:x1])) / n
    if energy.size == 0 or float(energy.max()) < 0.08:
        return None
    thresh = max(0.10, float(energy.max()) * 0.45)
    end = 0
    in_band = False
    for i, val in enumerate(energy):
        if val >= thresh:
            in_band = True
            end = i
        elif in_band and val < thresh * 0.55:
            break
    if not in_band:
        return None
    return int(y0 + end + 1)


def _component_eave(
    top_s: np.ndarray,
    heights: np.ndarray,
    col_mask: np.ndarray,
    horiz: np.ndarray,
    envelope: np.ndarray,
    max_h: int,
) -> int:
    tops = top_s[col_mask]
    hs = heights[col_mask]
    med_top = float(np.median(tops))
    spread = float(np.percentile(tops, 90) - np.percentile(tops, 10))
    med_h = float(np.median(hs))
    max_depth = int(max(8, min(0.36 * med_h, 0.28 * max_h)))
    hatch = _hatch_eave(horiz, envelope, top_s, col_mask, max_depth)
    if spread > 0.10 * max_h:
        eave = int(np.percentile(tops, 88))
        if hatch is not None:
            eave = max(eave, hatch)
        return eave
    inliers = col_mask & (top_s <= int(med_top + 0.08 * max_h))
    base = int(np.median(top_s[inliers])) if inliers.any() else int(med_top)
    if hatch is not None and hatch > base + 3:
        return int(min(hatch, base + max_depth))
    return int(base + max(8, 0.13 * med_h))


def _geometry_from_silhouette(
    ink: np.ndarray, envelope: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Roof/walls follow the building outline per column — no global horizontal bands."""
    h, w = envelope.shape
    top, bot = _column_profile(envelope)
    valid = top >= 0
    if not valid.any():
        z = np.zeros((h, w), np.uint8)
        return z, z, z, z, h // 4, h // 2, int(h * 0.92)

    top_s = _median_smooth(_fill_missing(top), 15)
    bot_s = _median_smooth(_fill_missing(bot), 15)
    heights = np.where(valid, np.maximum(bot_s - top_s, 0), 0)
    max_h = int(heights.max()) if int(heights.max()) > 0 else 1
    horiz = _horizontal_ink(ink, envelope)

    two = valid & (heights >= int(0.70 * max_h))
    one = valid & (heights >= int(0.38 * max_h)) & ~two
    stub = valid & ~two & ~one

    eave = np.zeros(w, np.int32)
    for story_mask in (two, one):
        for x0, x1 in _runs_1d(story_mask):
            if x1 - x0 < 12:
                continue
            local = np.zeros(w, dtype=bool)
            local[x0:x1] = story_mask[x0:x1]
            if int(local.sum()) < 12:
                continue
            local_eave = _component_eave(top_s, heights, local, horiz, envelope, max_h)
            eave[local] = local_eave
    for x0, x1 in _runs_1d(stub):
        local = np.zeros(w, dtype=bool)
        local[x0:x1] = stub[x0:x1]
        if not local.any():
            continue
        eave[local] = (top_s[local] + np.maximum(3, (0.10 * heights[local]).astype(np.int32))).astype(np.int32)

    min_eave = top_s + np.maximum(3, (0.06 * heights).astype(np.int32))
    depth = np.where(two, 0.28, 0.18) * heights
    max_eave = top_s + np.maximum(4, depth.astype(np.int32))
    eave = np.clip(eave, min_eave, np.maximum(min_eave, max_eave))
    eave = np.where(valid, eave, 0)

    found_depth = max(3, int(0.032 * max_h))
    found = np.maximum(eave + 5, bot_s - found_depth)
    found = np.minimum(found, bot_s)
    found = np.where(valid, found, 0)

    if two.any():
        y0 = int(np.median(eave[two]))
        y1 = int(np.median(found[two]))
        wall_h = max(y1 - y0, 1)
        floor_lo = y0 + int(wall_h * 0.30)
        floor_hi = y0 + int(wall_h * 0.68)
        floor_y = _peak_row(_band_ink_rows(ink, envelope, floor_lo, floor_hi), floor_lo)
        if abs(floor_y - y0) < wall_h * 0.18:
            floor_y = y0 + int(wall_h * 0.48)
    else:
        floor_y = int(np.median(eave[valid]) + 0.5 * (np.median(found[valid]) - np.median(eave[valid])))

    yy = np.arange(h, dtype=np.int32)[:, None]
    envb = envelope > 0
    top_b = top_s[None, :]
    eave_b = eave[None, :]
    found_b = found[None, :]
    bot_b = bot_s[None, :]
    two_b = two[None, :]
    low_b = (one | stub)[None, :]
    valid_b = valid[None, :]

    roof = (envb & valid_b & (yy >= top_b) & (yy < eave_b)).astype(np.uint8) * 255
    foundation = (envb & valid_b & (yy >= found_b) & (yy <= bot_b)).astype(np.uint8) * 255
    wall_l2 = (envb & two_b & (yy >= eave_b) & (yy < floor_y) & (yy < found_b)).astype(np.uint8) * 255
    wall_l1 = (
        (envb & low_b & (yy >= eave_b) & (yy < found_b))
        | (envb & two_b & (yy >= floor_y) & (yy < found_b))
    ).astype(np.uint8) * 255

    close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    roof = cv2.morphologyEx(roof, cv2.MORPH_CLOSE, close)
    wall_l1 = cv2.morphologyEx(wall_l1, cv2.MORPH_CLOSE, close)
    wall_l2 = cv2.morphologyEx(wall_l2, cv2.MORPH_CLOSE, close)
    foundation = cv2.morphologyEx(foundation, cv2.MORPH_CLOSE, close)

    eave_y = int(np.median(eave[two])) if two.any() else int(np.median(eave[valid]))
    foundation_y = int(np.median(found[valid]))
    return roof, wall_l1, wall_l2, foundation, eave_y, int(floor_y), foundation_y


def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _region(label: str, mask: np.ndarray, score: float, source: str) -> Region | None:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 20:
        return None
    return Region(label=label, mask=mask, score=score, source=source, box=_box(mask))


def _overlap_frac(mask: np.ndarray, other: np.ndarray) -> float:
    n = int(np.count_nonzero(mask))
    if n == 0:
        return 0.0
    return float(np.count_nonzero((mask > 0) & (other > 0))) / n


def _inner_rectangles(
    ink: np.ndarray,
    envelope: np.ndarray,
    building_h: int,
    roof_mask: np.ndarray,
    foundation_mask: np.ndarray,
) -> list[tuple[np.ndarray, float, str]]:
    """Fill CAD window/vent frames (bounding rects), not the ink strokes themselves."""
    h, w = ink.shape
    lines = ((ink > 0) & (envelope > 0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    out: list[tuple[np.ndarray, float, str]] = []
    env_area = max(int(np.count_nonzero(envelope)), 1)
    max_win_h = max(18, int(building_h * 0.28))
    max_win_w = max(24, int(w * 0.18))
    max_win_area = int(env_area * 0.035)

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 10:
            continue
        area = bw * bh
        if area < 90 or area > max_win_area or bh > max_win_h or bw > max_win_w:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4:
            continue
        rectness = area / max(cv2.contourArea(cnt), 1)
        if rectness < 0.55:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.32 or aspect > 4.2:
            continue
        pad = 1
        if bw <= 2 * pad or bh <= 2 * pad:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (x + pad, y + pad), (x + bw - pad, y + bh - pad), 255, -1)
        mask = cv2.bitwise_and(mask, envelope)
        if int(np.count_nonzero(mask)) < 60:
            continue
        if _overlap_frac(mask, roof_mask) > 0.35:
            continue
        if _overlap_frac(mask, foundation_mask) > 0.45:
            continue
        roi = lines[y : y + bh, x : x + bw]
        horiz = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, bw // 4), 1)))
        vert = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, bh // 4))))
        horiz_score = float(np.mean(horiz)) / 255.0
        vert_score = float(np.mean(vert)) / 255.0
        child = hierarchy[i][2]
        n_children = 0
        large_child = False
        while child != -1:
            n_children += 1
            cx, cy, cw, ch = cv2.boundingRect(contours[child])
            if cw >= 10 and ch >= 10 and (cw * ch) >= 0.18 * area:
                large_child = True
                break
            child = hierarchy[child][0]
        if large_child:
            continue
        is_small = max(bw, bh) <= 36 and 0.7 <= aspect <= 1.45
        if is_small and horiz_score >= 0.12:
            kind = "vent"
            score = 0.78
        elif n_children >= 1 or (horiz_score >= 0.04 and vert_score >= 0.04):
            kind = "window"
            score = 0.82
        else:
            continue
        out.append((mask, score, kind))

    # Drop openings fully contained in a larger kept opening.
    kept: list[tuple[np.ndarray, float, str]] = []
    boxes = [cv2.boundingRect(m) for m, _, _ in out]
    for i, item in enumerate(out):
        xi, yi, wi, hi = boxes[i]
        contained = False
        for j, other in enumerate(out):
            if i == j or other[2] == "vent":
                continue
            xj, yj, wj, hj = boxes[j]
            if wi * hi >= wj * hj:
                continue
            if xi >= xj and yi >= yj and xi + wi <= xj + wj and yi + hi <= yj + hj:
                contained = True
                break
        if not contained:
            kept.append(item)
    return kept


def perceive(bgr: np.ndarray) -> PerceiveResult:
    if bgr.ndim == 2:
        gray = bgr
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    paper = _ensure_ink_black(gray)
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    envelope = _envelope(paper)
    roof_mask, wall_l1, wall_l2, foundation, eave_y, floor_y, foundation_y = _geometry_from_silhouette(
        ink, envelope
    )

    geometry = {
        "roof": roof_mask,
        "wall_l1": wall_l1,
        "wall_l2": wall_l2,
        "foundation": foundation,
    }

    regions: list[Region] = []
    for label, mask, score in (
        ("roof", roof_mask, 0.78),
        ("wall_l2", wall_l2, 0.74),
        ("wall_l1", wall_l1, 0.74),
        ("foundation", foundation, 0.72),
    ):
        r = _region(label, mask, score, "geometry")
        if r:
            regions.append(r)

    h = envelope.shape[0]
    ys, _ = np.where(envelope > 0)
    building_h = int(ys.max() - ys.min()) if ys.size else h
    for mask, score, kind in _inner_rectangles(ink, envelope, building_h, roof_mask, foundation):
        r = _region(kind, mask, score, "geometry")
        if r:
            regions.append(r)

    return PerceiveResult(
        bgr=bgr,
        gray=gray,
        ink=ink,
        envelope=envelope,
        eave_y=eave_y,
        floor_y=floor_y,
        foundation_y=foundation_y,
        regions=regions,
        geometry=geometry,
    )
