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
def _box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _region(label: str, mask: np.ndarray, score: float, source: str) -> Region | None:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 10:
        return None
    return Region(label=label, mask=mask, score=score, source=source, box=_box(mask))


def _clean_envelope(ink: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Extract clean building envelope, strictly excluding dimension lines, text, and leader marks."""
    h, w = ink.shape
    h_lines_med = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
    v_lines_med = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))
    
    # 1. Find outermost vertical wall boundaries (excluding thin dimension arrows)
    v_proj = np.sum(v_lines_med > 0, axis=0)
    v_dense = np.flatnonzero(v_proj > 25)
    if v_dense.size > 0:
        min_x = max(0, int(v_dense.min()) - 6)
        max_x = min(w, int(v_dense.max()) + 6)
    else:
        min_x, max_x = 0, w

    # 2. Find bottom ground line
    h_proj = np.sum(h_lines_med[:, min_x:max_x] > 0, axis=1)
    bottom_y_candidates = np.flatnonzero(h_proj[int(h * 0.70):] > (max_x - min_x) * 0.35)
    if bottom_y_candidates.size > 0:
        ground_y = int(h * 0.70) + int(bottom_y_candidates.max())
    else:
        ground_y = int(h * 0.92)

    # 3. Seal and flood fill building body within structural boundaries
    core_ink = ink[:ground_y + 2, min_x:max_x].copy()
    sealed = cv2.dilate(core_ink, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    
    pad = 2
    sh, sw = sealed.shape
    flood = np.pad(sealed, pad, mode='constant', constant_values=0)
    ff_mask = np.zeros((sh + pad*2 + 2, sw + pad*2 + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 128)
    
    env_roi = (flood[pad:-pad, pad:-pad] != 128).astype(np.uint8) * 255
    env_roi = cv2.morphologyEx(env_roi, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    
    envelope = np.zeros((h, w), np.uint8)
    envelope[:ground_y + 2, min_x:max_x] = env_roi
    
    return envelope, min_x, max_x, ground_y


def _roof_from_hatch(ink: np.ndarray, envelope: np.ndarray, ground_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect sloped roof tile surfaces with parallel hatching."""
    h, w = envelope.shape
    h_hatch = cv2.morphologyEx(ink & envelope, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1)))
    roof_dense = cv2.morphologyEx(h_hatch, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15)))
    roof_dense = cv2.dilate(roof_dense, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    n_r, l_r, s_r, _ = cv2.connectedComponentsWithStats((roof_dense > 0).astype(np.uint8))
    roof = np.zeros((h, w), np.uint8)
    for i in range(1, n_r):
        area = int(s_r[i, cv2.CC_STAT_AREA])
        by = int(s_r[i, cv2.CC_STAT_TOP])
        bh = int(s_r[i, cv2.CC_STAT_HEIGHT])
        bw_ = int(s_r[i, cv2.CC_STAT_WIDTH])
        if area < 400 or bw_ < 35 or bh < 15:
            continue
        comp_mask = (l_r == i).astype(np.uint8) * 255
        h_in_comp = cv2.morphologyEx(ink & comp_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (12, 1)))
        line_rows = np.where(np.sum(h_in_comp > 0, axis=1) > 15)[0]
        if len(line_rows) >= 5 and by < 0.65 * ground_y:
            # Check average line spacing: roof tiles have pitch < 22px
            diffs = np.diff(line_rows)
            valid_pitch = diffs[(diffs > 2) & (diffs < 22)]
            if len(valid_pitch) >= 4:
                # Fill tile region solidly ONLY within this component
                solid_comp = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
                cnts, _ = cv2.findContours(solid_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) > 200:
                        cv2.drawContours(roof, [c], -1, 255, -1)
                        
    roof = cv2.bitwise_and(roof, envelope)
    return roof, h_hatch


def _pipe_faces(
    ink: np.ndarray,
    envelope: np.ndarray,
    building_h: int,
) -> list[tuple[np.ndarray, float, str]]:
    """Vertical downspout / pipes running along walls."""
    h, w = ink.shape
    env_area = max(int(np.count_nonzero(envelope)), 1)
    v_pipes = cv2.morphologyEx(ink & envelope, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35)))
    v_pipes = cv2.dilate(v_pipes, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1)))
    n_p, l_p, s_p, _ = cv2.connectedComponentsWithStats((v_pipes > 0).astype(np.uint8))
    pipes: list[tuple[np.ndarray, float, str]] = []
    for i in range(1, n_p):
        bh = int(s_p[i, cv2.CC_STAT_HEIGHT])
        bw = int(s_p[i, cv2.CC_STAT_WIDTH])
        area = int(s_p[i, cv2.CC_STAT_AREA])
        if bh >= 45 and bw <= 18 and bh / max(bw, 1) >= 4.0 and area < 0.015 * env_area:
            mask = (l_p == i).astype(np.uint8) * 255
            mask = cv2.bitwise_and(mask, envelope)
            pipes.append((mask, 0.75, "pipe"))
    return pipes


def _opening_faces(
    ink: np.ndarray,
    envelope: np.ndarray,
    roof: np.ndarray,
) -> list[tuple[np.ndarray, float, str]]:
    """Paper holes inside walls = window/vent interiors, bounded by CAD ink."""
    h, w = ink.shape
    sealed_cad = cv2.dilate(ink & envelope, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    paper_holes = ((envelope > 0) & (sealed_cad == 0) & (roof == 0)).astype(np.uint8) * 255
    
    n_w, l_w, s_w, _ = cv2.connectedComponentsWithStats(paper_holes)
    env_area = max(int(np.count_nonzero(envelope)), 1)
    out: list[tuple[np.ndarray, float, str]] = []

    for i in range(1, n_w):
        area = int(s_w[i, cv2.CC_STAT_AREA])
        bw_ = int(s_w[i, cv2.CC_STAT_WIDTH])
        bh_ = int(s_w[i, cv2.CC_STAT_HEIGHT])
        
        if bw_ < 8 or bh_ < 8:
            continue
        aspect = bw_ / max(bh_, 1)
        rect_ratio = area / max(bw_ * bh_, 1)
        
        if 0.18 <= aspect <= 5.5 and rect_ratio >= 0.55:
            mask = (l_w == i).astype(np.uint8) * 255
            if 30 <= area <= 500 and max(bw_, bh_) <= 40 and 0.6 <= aspect <= 1.6:
                out.append((mask, 0.88, "vent"))
            elif 500 < area <= 0.035 * env_area and bh_ >= 15 and bw_ >= 12:
                out.append((mask, 0.85, "window"))
    return out


def perceive(bgr: np.ndarray) -> PerceiveResult:
    if bgr.ndim == 2:
        gray = bgr
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    paper = _ensure_ink_black(gray)
    ink = np.where(paper == 0, 255, 0).astype(np.uint8)
    h, w = ink.shape
    
    # 1. Clean building envelope
    envelope, min_x, max_x, ground_y = _clean_envelope(ink)
    
    # 2. Roof from tile hatching
    roof, hatch = _roof_from_hatch(ink, envelope, ground_y)
    
    # 3. Openings & Pipes
    ys, _ = np.where(envelope > 0)
    building_h = int(ys.max() - ys.min()) if ys.size else h
    openings = _opening_faces(ink, envelope, roof)
    pipes = _pipe_faces(ink, envelope, building_h)
    
    # 4. Foundation
    foundation = np.zeros((h, w), np.uint8)
    f_height = max(12, int(h * 0.032))
    f_top = max(0, ground_y - f_height)
    foundation[f_top:ground_y + 1, min_x:max_x] = 255
    foundation = cv2.bitwise_and(foundation, envelope)
    
    # Exclude windows and vents from foundation
    win_vent_all = np.zeros((h, w), np.uint8)
    for m, _, _ in openings:
        win_vent_all = cv2.bitwise_or(win_vent_all, m)
    foundation = cv2.bitwise_and(foundation, cv2.bitwise_not(win_vent_all))
    
    # 5. Wall Division (Floor L1 vs L2)
    pipe_all = np.zeros((h, w), np.uint8)
    for m, _, _ in pipes:
        pipe_all = cv2.bitwise_or(pipe_all, m)
        
    wall_space = envelope & ~roof & ~foundation & ~win_vent_all & ~pipe_all
    ys_w, _ = np.where(wall_space > 0)
    top_wall_y = int(ys_w.min()) if ys_w.size > 0 else h // 4
    bot_wall_y = int(ys_w.max()) if ys_w.size > 0 else int(ground_y)
    wall_h = max(bot_wall_y - top_wall_y, 1)
    
    h_lines_med = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
    search_y0 = top_wall_y + int(wall_h * 0.35)
    search_y1 = top_wall_y + int(wall_h * 0.65)
    h_floor = np.sum(h_lines_med[search_y0:search_y1, min_x:max_x] > 0, axis=1)
    if h_floor.size > 0 and h_floor.max() > 0:
        floor_y = search_y0 + int(np.argmax(h_floor))
    else:
        floor_y = top_wall_y + int(wall_h * 0.50)
        
    yy = np.arange(h)[:, None]
    wall_l2 = np.where(wall_space & (yy < floor_y), 255, 0).astype(np.uint8)
    wall_l1 = np.where(wall_space & (yy >= floor_y), 255, 0).astype(np.uint8)
    
    eave_y = int(top_wall_y)
    foundation_y = int(f_top)

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
