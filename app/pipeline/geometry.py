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
class Face:
    mask: np.ndarray
    label: str
    score: float
    box: tuple[int, int, int, int]


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
    faces: list[Face] = field(default_factory=list)
    hatch: np.ndarray | None = None


def _ensure_ink_black(gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bw) < 127:
        bw = 255 - bw
    return bw


def _envelope_flood(paper: np.ndarray) -> np.ndarray:
    h, w = paper.shape
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    sealed = np.where(ink > 0, 0, 255).astype(np.uint8)
    flood = sealed.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, y), 128)
    env = np.where(flood != 128, 255, 0).astype(np.uint8)
    env = cv2.erode(env, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    env = cv2.morphologyEx(env, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 51)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((env > 0).astype(np.uint8))
    if n <= 2:
        return env
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = stats[keep, cv2.CC_STAT_AREA]
    out = np.zeros_like(env)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.25 * largest:
            out[labels == i] = 255
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


def _envelope_mass(paper: np.ndarray) -> np.ndarray:
    """Building body from ink mass, ignoring page-wide dimension/level guides."""
    h, w = paper.shape
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    dashed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1)))
    guides = cv2.morphologyEx(
        dashed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(48, int(0.50 * w)), 1))
    )
    core = cv2.bitwise_and(ink, cv2.bitwise_not(guides))
    thick = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((thick > 0).astype(np.uint8))
    keep = np.zeros_like(ink)
    min_h = max(24, int(0.18 * h))
    min_a = max(400, int(0.012 * h * w))
    for i in range(1, n):
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if bh < min_h or area < min_a:
            continue
        if bh < 18 and bw > 0.45 * w:
            continue
        keep[labels == i] = 255
    if int(np.count_nonzero(keep)) < min_a:
        keep = thick
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
    inv = np.where(keep > 0, 0, 255).astype(np.uint8)
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, y), 128)
    env = np.where(flood != 128, 255, 0).astype(np.uint8)
    env = cv2.erode(env, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((env > 0).astype(np.uint8))
    if n <= 2:
        return env
    keep_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = stats[keep_i, cv2.CC_STAT_AREA]
    out = np.zeros_like(env)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 0.20 * largest:
            out[labels == i] = 255
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))


def _envelope(paper: np.ndarray) -> np.ndarray:
    flood = _envelope_flood(paper)
    mass = _envelope_mass(paper)
    frac_f = float(flood.mean()) / 255.0
    frac_m = float(mass.mean()) / 255.0
    if frac_m < 0.18 and frac_f > frac_m:
        return flood
    if 0.18 <= frac_m <= 0.58:
        return mass
    if 0.18 <= frac_f <= 0.70:
        return flood
    return flood if frac_f < frac_m else mass


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


def _median_smooth(arr: np.ndarray, k: int = 15) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    padded = np.pad(arr.astype(np.int32), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1).astype(np.int32)


def _line_maps(ink: np.ndarray, envelope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = ink.shape
    lines = ((ink > 0) & (envelope > 0)).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(
        lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 70), 1))
    )
    vert = cv2.morphologyEx(
        lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 70)))
    )
    return horiz, vert


def _band_ink_rows(ink: np.ndarray, envelope: np.ndarray, y0: int, y1: int) -> np.ndarray:
    y0 = max(0, y0)
    y1 = min(ink.shape[0], max(y0 + 1, y1))
    band = ink[y0:y1] & envelope[y0:y1]
    return np.sum(band > 0, axis=1)


def _peak_row(scores: np.ndarray, y0: int) -> int:
    if scores.size == 0:
        return y0
    return int(y0 + np.argmax(scores))


def _roof_from_hatch(
    horiz: np.ndarray,
    envelope: np.ndarray,
    top_s: np.ndarray,
    heights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Roof = hatched tile bands that sit on a wall, plus the void up to the ridge."""
    h, w = envelope.shape
    hatch = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    hatch = cv2.morphologyEx(hatch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    hatch = cv2.bitwise_and(hatch, envelope)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((hatch > 0).astype(np.uint8))
    roof_hatch = np.zeros_like(hatch)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 80:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(labels == i)
        dist = ys.astype(np.int32) - top_s[xs]
        med_dist = float(np.median(dist)) if dist.size else 1e9
        med_h = float(np.median(heights[xs])) if xs.size else 1.0
        below_y0 = min(h, y + bh)
        below_y1 = min(h, y + bh + max(10, bh // 3))
        below = envelope[below_y0:below_y1, x : x + bw]
        horiz_below = horiz[below_y0:below_y1, x : x + bw]
        below_n = max(int(np.count_nonzero(below)), 1)
        below_hatch = float(np.count_nonzero(horiz_below)) / below_n
        near_top = med_dist <= 0.28 * max(med_h, 1.0)
        sits_on_wall = below_hatch < 0.10 and int(np.count_nonzero(below)) > 20
        wide_band = bw >= 20 and bh <= int(0.38 * h)
        if bh > int(0.50 * h):
            continue
        if not ((near_top or sits_on_wall) and (wide_band or near_top)):
            continue
        roof_hatch[labels == i] = 255

    h, w = envelope.shape
    has = roof_hatch > 0
    hatch_bot = np.full(w, -1, np.int32)
    gap_max = 6
    for x in range(w):
        ys = np.flatnonzero(has[:, x])
        if ys.size == 0:
            continue
        y1 = int(ys[0])
        for y in ys:
            if y <= y1 + gap_max:
                y1 = int(y)
            else:
                break
        hatch_bot[x] = y1

    max_h = int(heights.max()) if int(heights.max()) > 0 else 1
    two = (heights >= int(0.70 * max_h)) & (top_s >= 0)
    one = (heights >= int(0.38 * max_h)) & ~two & (top_s >= 0)
    eave = np.zeros(w, np.int32)
    for story in (two, one):
        for x0, x1 in _runs_1d(story):
            local = np.zeros(w, dtype=bool)
            local[x0:x1] = story[x0:x1]
            hb = hatch_bot[local & (hatch_bot >= 0)]
            if hb.size >= 8:
                eave_val = int(np.percentile(hb, 82))
            else:
                eave_val = int(np.percentile(top_s[local], 85) + max(8, 0.12 * np.median(heights[local])))
            eave[local] = eave_val

    min_eave = top_s + np.maximum(4, (0.06 * heights).astype(np.int32))
    max_eave = top_s + np.maximum(6, np.where(two, 0.32, 0.20) * heights).astype(np.int32)
    eave = np.clip(eave, min_eave, np.maximum(min_eave, max_eave))
    eave = np.where(top_s >= 0, eave, 0)
    yy = np.arange(h)[:, None]
    roof = ((envelope > 0) & (top_s[None, :] >= 0) & (yy >= top_s[None, :]) & (yy < eave[None, :])).astype(np.uint8) * 255
    return roof, roof_hatch


def _foundation(envelope: np.ndarray, bot_s: np.ndarray, heights: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = envelope.shape
    max_h = int(heights.max()) if heights.size else h
    depth = max(3, int(0.028 * max_h))
    found_line = np.where(valid, np.maximum(0, bot_s - depth), h)
    yy = np.arange(h)[:, None]
    foundation = ((envelope > 0) & valid[None, :] & (yy >= found_line[None, :]) & (yy <= bot_s[None, :])).astype(
        np.uint8
    ) * 255
    foundation_y = int(np.median(found_line[valid])) if valid.any() else int(h * 0.92)
    return foundation, foundation_y


def _overlap_frac(mask: np.ndarray, other: np.ndarray) -> float:
    n = int(np.count_nonzero(mask))
    if n == 0:
        return 0.0
    return float(np.count_nonzero((mask > 0) & (other > 0))) / n


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


def _pipe_faces(
    ink: np.ndarray,
    envelope: np.ndarray,
    roof: np.ndarray,
    foundation: np.ndarray,
    building_h: int,
) -> list[tuple[np.ndarray, float, str]]:
    """Vertical downspout / pipes running along walls."""
    h, w = ink.shape
    env_area = max(int(np.count_nonzero(envelope)), 1)
    vert = cv2.morphologyEx(
        ink & envelope,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, building_h // 20))),
    )
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((vert > 0).astype(np.uint8))
    pipes: list[tuple[np.ndarray, float, str]] = []
    for i in range(1, n):
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if bh < max(28, int(building_h * 0.12)) or bw > 22 or bw < 2:
            continue
        if bh / max(bw, 1) < 3.5:
            continue
        if area > 0.015 * env_area:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        mask = cv2.bitwise_and(mask, envelope)
        pipes.append((mask, 0.75, "pipe"))
    return pipes


def _opening_faces(
    ink: np.ndarray,
    envelope: np.ndarray,
    roof: np.ndarray,
    foundation: np.ndarray,
    building_h: int,
) -> list[tuple[np.ndarray, float, str]]:
    """Paper holes inside walls = window/vent interiors, bounded by CAD ink."""
    h, w = ink.shape
    barrier = cv2.dilate(((ink > 0) & (envelope > 0)).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    paper = ((envelope > 0) & (barrier == 0) & (roof == 0) & (foundation == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper)
    env_area = max(int(np.count_nonzero(envelope)), 1)
    out: list[tuple[np.ndarray, float, str]] = []
    max_win_h = max(16, int(building_h * 0.32))
    max_win_w = max(20, int(w * 0.22))
    max_area = int(env_area * 0.035)
    min_area = max(50, int(env_area * 0.0002))

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        if area < min_area or area > max_area:
            continue
        if bw < 8 or bh < 8 or bh > max_win_h or bw > max_win_w:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.22 or aspect > 5.0:
            continue
        if area / max(bw * bh, 1) < 0.50:
            continue
        mask = (labels == i).astype(np.uint8) * 255
        if _overlap_frac(mask, roof) > 0.15:
            continue
        roi = ink[y : y + bh, x : x + bw]
        hs = float(
            np.mean(
                cv2.morphologyEx(
                    roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(4, bw // 4), 1))
                )
            )
        ) / 255.0
        small = max(bw, bh) <= 45 and 0.60 <= aspect <= 1.65
        if small and hs >= 0.06:
            out.append((mask, 0.85, "vent"))
        else:
            out.append((mask, 0.82, "window"))
    return out


def perceive(bgr: np.ndarray) -> PerceiveResult:
    if bgr.ndim == 2:
        gray = bgr
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    paper = _ensure_ink_black(gray)
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    envelope = _envelope(paper)
    h, w = envelope.shape
    top, bot = _column_profile(envelope)
    valid = top >= 0
    top_s = _median_smooth(_fill_missing(top), 15)
    bot_s = _median_smooth(_fill_missing(bot), 15)
    heights = np.where(valid, np.maximum(bot_s - top_s, 0), 0)
    if valid.any():
        ground = int(np.percentile(bot_s[valid], 88))
        for x in np.flatnonzero(valid):
            if heights[x] >= int(0.28 * max(int(heights.max()), 1)) and bot_s[x] < ground - 12:
                envelope[int(bot_s[x]) : ground + 1, x] = 255
        top, bot = _column_profile(envelope)
        valid = top >= 0
        top_s = _median_smooth(_fill_missing(top), 15)
        bot_s = _median_smooth(_fill_missing(bot), 15)
        heights = np.where(valid, np.maximum(bot_s - top_s, 0), 0)
    max_h = int(heights.max()) if int(heights.max()) > 0 else 1
    horiz, _vert = _line_maps(ink, envelope)

    roof, hatch = _roof_from_hatch(horiz, envelope, top_s, heights)
    foundation, foundation_y = _foundation(envelope, bot_s, heights, valid)

    two = valid & (heights >= int(0.68 * max_h))
    one = valid & ~two

    if two.any():
        has_roof = roof > 0
        roof_bottom = np.where(has_roof.any(axis=0), h - 1 - np.argmax(has_roof[::-1], axis=0), 0)
        eave_y = int(np.median(roof_bottom[two & has_roof.any(axis=0)])) if (two & has_roof.any(axis=0)).any() else int(
            np.median(top_s[two])
        )
        y0 = eave_y
        y1 = int(np.median(np.where(valid, np.maximum(0, bot_s - 4), 0)[two]))
        wall_h = max(y1 - y0, 1)
        floor_lo = y0 + int(wall_h * 0.32)
        floor_hi = y0 + int(wall_h * 0.68)
        floor_y = _peak_row(_band_ink_rows(ink, envelope, floor_lo, floor_hi), floor_lo)
        if abs(floor_y - y0) < wall_h * 0.16 or abs(floor_y - y1) < wall_h * 0.16:
            floor_y = y0 + int(wall_h * 0.50)
    else:
        eave_y = int(np.median(top_s[valid])) if valid.any() else h // 4
        floor_y = int(h * 0.55)

    yy = np.arange(h, dtype=np.int32)[:, None]
    envb = envelope > 0
    two_b = two[None, :]
    one_b = one[None, :]
    wall = envb & (roof == 0) & (foundation == 0)
    wall_l2 = (wall & two_b & (yy < floor_y)).astype(np.uint8) * 255
    wall_l1 = ((wall & one_b) | (wall & two_b & (yy >= floor_y))).astype(np.uint8) * 255

    ys, _ = np.where(envelope > 0)
    building_h = int(ys.max() - ys.min()) if ys.size else h
    openings = _opening_faces(ink, envelope, roof, foundation, building_h)
    pipes = _pipe_faces(ink, envelope, roof, foundation, building_h)

    geometry = {
        "roof": roof,
        "wall_l1": wall_l1,
        "wall_l2": wall_l2,
        "foundation": foundation,
    }

    regions: list[Region] = []
    faces: list[Face] = []
    for mask, score, kind in openings + pipes:
        r = _region(kind, mask, score, "geometry")
        if r and r.box:
            regions.append(r)
            faces.append(Face(mask=mask, label=kind, score=score, box=r.box))

    return PerceiveResult(
        bgr=bgr,
        gray=gray,
        ink=ink,
        envelope=envelope,
        eave_y=int(eave_y),
        floor_y=int(floor_y),
        foundation_y=int(foundation_y),
        regions=regions,
        geometry=geometry,
        faces=faces,
        hatch=hatch,
    )
