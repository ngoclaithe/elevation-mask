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
    flood = paper.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, y), 128)
    env = np.where(flood != 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    env = cv2.morphologyEx(env, cv2.MORPH_CLOSE, kernel)
    return env


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
    widths = np.array([_row_width(envelope, y) for y in range(envelope.shape[0])])
    max_w = max(int(widths.max()), 1)

    # Eave = first row from the ridge where the silhouette is already "wall-wide".
    eave_y = top
    for y in range(top, bot):
        if widths[y] >= 0.55 * max_w:
            eave_y = y
            break

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
    max_win_area = int(env_area * 0.10)

    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 10:
            continue
        area = bw * bh
        if area < 90 or area > max_win_area or bh > max_win_h:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4:
            continue
        rectness = area / max(cv2.contourArea(cnt), 1)
        if rectness < 0.7:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 5:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.bitwise_and(mask, envelope)
        roi = ink[y : y + bh, x : x + bw]
        horiz = 0
        for row in range(0, bh, max(1, bh // 8)):
            horiz += int(np.mean(roi[row]) > 20)
        has_child = hierarchy[i][2] != -1
        if bh <= 36 and bw <= 36 and horiz >= 3 and 0.6 <= aspect <= 1.6:
            kind = "vent"
            score = 0.74
        elif has_child and 0.5 <= aspect <= 4.0:
            kind = "window"
            score = 0.78
        elif 0.7 <= aspect <= 3.2 and area < env_area * 0.04:
            kind = "window"
            score = 0.52
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
    for mask, score, kind in _inner_rectangles(ink, envelope, building_h):
        r = _region(kind, mask, score, "geometry")
        if r:
            regions.append(r)

    # Thin vertical pipes on the right/left envelope edge.
    ys, xs = np.where(envelope > 0)
    if xs.size:
        left, right = int(xs.min()), int(xs.max())
        for x0 in (left, right - 6):
            strip = np.zeros((h, w), np.uint8)
            x1 = min(w, max(0, x0) + 7)
            strip[:, max(0, x0) : x1] = envelope[:, max(0, x0) : x1]
            strip[:eave_y] = 0
            r = _region("pipe", strip, 0.4, "geometry")
            if r and int(np.count_nonzero(r.mask)) < (h * w * 0.02):
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
