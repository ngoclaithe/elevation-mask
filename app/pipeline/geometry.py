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


def _ink_density_by_row(ink: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    h = ink.shape[0]
    dens = np.zeros(h, np.float32)
    for y in range(h):
        n = int(np.count_nonzero(envelope[y]))
        if n < 12:
            continue
        dens[y] = float(np.count_nonzero((ink[y] > 0) & (envelope[y] > 0))) / n
    kernel = np.ones(9, np.float32) / 9.0
    return np.convolve(dens, kernel, mode="same")


def _band_ink_rows(ink: np.ndarray, envelope: np.ndarray, y0: int, y1: int) -> np.ndarray:
    band = ink[y0:y1] & envelope[y0:y1]
    return np.sum(band > 0, axis=1)


def _peak_row(scores: np.ndarray, y0: int) -> int:
    if scores.size == 0:
        return y0
    return int(y0 + np.argmax(scores))


def _row_width(envelope: np.ndarray, y: int) -> int:
    xs = np.where(envelope[y] > 0)[0]
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min())


def _horizontal_line_peaks(ink: np.ndarray, envelope: np.ndarray) -> tuple[int, int, int]:
    ys, xs = np.where(envelope > 0)
    if ys.size == 0:
        h = envelope.shape[0]
        return h // 4, h // 2, int(h * 0.92)
    top, bot = int(ys.min()), int(ys.max())
    height = max(bot - top, 1)
    dens = _ink_density_by_row(ink, envelope)
    widths = np.array([_row_width(envelope, y) for y in range(envelope.shape[0])])
    mid = top + int(height * 0.50)
    thresh = max(0.18, float(np.median(dens[top:bot]) + 0.06))
    band_end = None
    in_band = False
    for y in range(top, mid):
        if dens[y] >= thresh:
            in_band = True
            band_end = y
        elif in_band and dens[y] < thresh * 0.7:
            break
    hatch = (
        band_end is not None
        and (band_end - top) > height * 0.12
        and float(np.mean(dens[top:band_end])) > 0.20
    )
    if hatch:
        eave_y = int(band_end)
    else:
        upper_hi = top + int(height * 0.45)
        upper_max = max(int(widths[top:upper_hi].max()), 1)
        eave_y = top + int(height * 0.22)
        for y in range(top + int(height * 0.08), upper_hi):
            if widths[y] >= 0.90 * upper_max:
                eave_y = y
                break
        lo = max(top, eave_y - int(height * 0.06))
        hi = min(upper_hi, eave_y + int(height * 0.08))
        eave_y = _peak_row(_band_ink_rows(ink, envelope, lo, hi), lo)
    eave_y = int(np.clip(eave_y, top + int(height * 0.12), top + int(height * 0.40)))

    found_lo = top + int(height * 0.86)
    foundation_y = _peak_row(_band_ink_rows(ink, envelope, found_lo, bot + 1), found_lo)
    if foundation_y <= eave_y:
        foundation_y = int(bot - max(4, height * 0.03))

    wall_h = max(foundation_y - eave_y, 1)
    floor_lo = eave_y + int(wall_h * 0.28)
    floor_hi = eave_y + int(wall_h * 0.68)
    floor_y = _peak_row(_band_ink_rows(ink, envelope, floor_lo, floor_hi), floor_lo)
    if abs(floor_y - eave_y) < wall_h * 0.2:
        floor_y = eave_y + int(wall_h * 0.45)
    return eave_y, floor_y, foundation_y


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


def _inner_rectangles(
    ink: np.ndarray,
    envelope: np.ndarray,
    building_h: int,
    eave_y: int,
    foundation_y: int,
) -> list[tuple[np.ndarray, float, str]]:
    """Find window/vent candidates from nested line rectangles."""
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
    max_win_area = int(env_area * 0.08)

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 12 or bh < 12:
            continue
        if y + bh < eave_y + 4 or y > foundation_y - 4:
            continue
        area = bw * bh
        if area < 120 or area > max_win_area or bh > max_win_h:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            continue
        if len(approx) < 4:
            continue
        rectness = area / max(cv2.contourArea(cnt), 1)
        if rectness < 0.72:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 4.2:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.bitwise_and(mask, envelope)
        roi = ink[y : y + bh, x : x + bw]
        horiz = 0
        for row in range(0, bh, max(1, bh // 8)):
            horiz += int(np.mean(roi[row]) > 20)
        has_child = hierarchy[i][2] != -1
        if bh <= 32 and bw <= 32 and horiz >= 3 and 0.7 <= aspect <= 1.5:
            kind = "vent"
            score = 0.74
        elif has_child and 0.55 <= aspect <= 3.8:
            kind = "window"
            score = 0.78
        elif 0.9 <= aspect <= 2.8 and env_area * 0.008 <= area <= env_area * 0.07:
            kind = "window"
            score = 0.58
        elif 0.32 <= aspect <= 0.7 and bh > bw and env_area * 0.004 <= area <= env_area * 0.04:
            kind = "window"
            score = 0.55
        else:
            continue
        out.append((mask, score, kind))
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
    eave_y, floor_y, foundation_y = _horizontal_line_peaks(ink, envelope)

    h, w = envelope.shape
    roof_mask = envelope.copy()
    roof_mask[eave_y:, :] = 0
    wall_mask = envelope.copy()
    wall_mask[:eave_y, :] = 0
    wall_mask[foundation_y:, :] = 0
    wall_l2 = wall_mask.copy()
    wall_l2[floor_y:, :] = 0
    wall_l1 = wall_mask.copy()
    wall_l1[:floor_y, :] = 0
    foundation = envelope.copy()
    foundation[:foundation_y, :] = 0

    regions: list[Region] = []
    for label, mask, score in (
        ("roof", roof_mask, 0.62),
        ("wall_l2", wall_l2, 0.6),
        ("wall_l1", wall_l1, 0.6),
        ("foundation", foundation, 0.65),
    ):
        r = _region(label, mask, score, "geometry")
        if r:
            regions.append(r)

    ys, xs = np.where(envelope > 0)
    building_h = int(ys.max() - ys.min()) if ys.size else h
    for mask, score, kind in _inner_rectangles(ink, envelope, building_h, eave_y, foundation_y):
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
    )
