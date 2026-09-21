"""语义矢量化（确定性核心）：检测圆 + 提取直线 -> 生成 DrawingIR。

设计原则：精确的事交给算法，坐标是「算」出来的，不是模型猜的。
参数可调，供 Leader Agent 在「校验回环」中优化。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from core.assemble import assemble_lines
from emit.ir import DrawingIR, Entity, LayerSpec
from tools.image_io import imread
from vectorize.arcs import detect_arcs
from vectorize.ellipses import detect_ellipses, mask_ellipses
from vectorize.cleaners import (
    classify_dimension_lines,
    detect_border,
    detect_hatching,
    detect_titleblock,
    drop_dimension_lines,
    mask_polygon,
)
from vectorize.preprocess import skeleton, split_ink, to_ink

DEFAULTS: dict[str, float] = {
    "hough_thresh": 15,
    "min_line_len": 30,
    "max_line_gap": 8,
    "merge_angle_tol": 4.0,
    "merge_dist_tol": 6.0,
    "merge_gap_tol": 20.0,
    "circle_min_r": 20,
    "circle_max_r": 0,  # 0 = 自动（按图幅推算）
    "circle_param2": 55,
    "circle_min_coverage": 0.60,  # 圆周上需有 60% 的点落在墨迹上
    "circle_tol": 4,
    "circle_big_r": 150,  # 大圆要求更高覆盖度（抑制把椭圆当圆）
    # --- 尺寸线剔除 ---
    "drop_dimensions": True,
    "dimension_action": "drop",  # drop=删除(实测更优) / layer=归到dim图层保留
    "dim_line_tol": 30,
    "hatch_min_region": 0.004,
    # --- 圆弧检测 ---
    "arc_min_len": 30,
    "arc_max_rms": 2.5,
    "arc_min_span_deg": 35,
    "arc_min_r": 15,
    "arc_max_r": 1500,
    "line_pad": 9,
    # --- 椭圆检测 ---
    "detect_ellipses": True,
    "ellipse_min_pts": 30,
    "ellipse_min_axis": 25,
    "ellipse_max_axis": 800,
    "ellipse_min_ecc": 1.15,
    "ellipse_max_resid": 0.10,
    "ellipse_min_cov": 0.70,
    "ellipse_tol": 5,
    "ellipse_pad": 6,
    # --- 图元装配 ---
    "assemble": True,
    # --- 图纸清洗 ---
    "clean_border": True,
    "clean_hatch": True,
    "border_area_ratio": 0.75,
    "border_edge_margin": 0.10,
    "hatch_min_len": 6,
    "hatch_max_len": 90,
    "hatch_min_count": 40,
    "hatch_cell": 40,
    "hatch_thr": 6,
}


def _params(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
    dist = p1[0] * nx + p1[1] * ny
    return ang, dist, length


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _merge_group(group, gap_tol):
    ang, dist, _ = _params(group[0][0], group[0][1])
    nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
    px, py = nx * dist, ny * dist
    ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))

    intervals = []
    for p1, p2 in group:
        t1 = p1[0] * ux + p1[1] * uy
        t2 = p2[0] * ux + p2[1] * uy
        intervals.append((min(t1, t2), max(t1, t2)))
    intervals.sort()

    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a - merged[-1][1] <= gap_tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [((px + ux * a, py + uy * a), (px + ux * b, py + uy * b)) for a, b in merged]


def merge_segments(segments, p):
    groups: list[list] = []
    for s in segments:
        ang, dist, length = _params(*s)
        if length < p["min_line_len"]:
            continue
        placed = False
        for g in groups:
            ga, gd, _ = _params(*g[0])
            if _angle_diff(ang, ga) <= p["merge_angle_tol"] and abs(dist - gd) <= p["merge_dist_tol"]:
                g.append(s)
                placed = True
                break
        if not placed:
            groups.append([s])
    merged = []
    for g in groups:
        merged.extend(_merge_group(g, p["merge_gap_tol"]))
    return [s for s in merged if _params(*s)[2] >= p["min_line_len"]]


def _circle_coverage(ink_dil: np.ndarray, cx: float, cy: float, r: float, n: int = 72) -> float:
    """圆周上采样，统计落在墨迹上的比例。"""
    h, w = ink_dil.shape[:2]
    hits = 0
    for i in range(n):
        a = 2 * math.pi * i / n
        x = int(round(cx + r * math.cos(a)))
        y = int(round(cy + r * math.sin(a)))
        if 0 <= x < w and 0 <= y < h and ink_dil[y, x]:
            hits += 1
    return hits / n


def detect_circles(gray: np.ndarray, p: dict, ink: np.ndarray | None = None) -> list[Entity]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    max_r = int(p["circle_max_r"]) or max(40, min(h, w) // 4)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(60, max_r // 3),
        param1=120, param2=int(p["circle_param2"]),
        minRadius=int(p["circle_min_r"]), maxRadius=max_r,
    )
    if circles is None:
        return []

    ink_dil = None
    if ink is not None:
        k = 2 * int(p["circle_tol"]) + 1
        ink_dil = cv2.dilate(ink, np.ones((k, k), np.uint8))

    out: list[Entity] = []
    for x, y, r in np.round(circles[0]).astype(int):
        min_cov = p["circle_min_coverage"] + (0.15 if r > p["circle_big_r"] else 0.0)
        if ink_dil is not None and _circle_coverage(ink_dil, x, y, r) < min_cov:
            continue  # 假圆：圆周上没多少墨迹
        out.append(Entity(type="circle", center=(float(x), float(y)), radius=float(r)))
    return out


def _mask_circles(shape, circles: list[Entity], pad: int = 8) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for c in circles:
        cv2.circle(mask, (int(c.center[0]), int(c.center[1])), int(c.radius), 255, pad)
    return mask


def _drop_on_circles(lines: list[Entity], circles: list[Entity], tol: float = 14.0) -> list[Entity]:
    """滤掉「贴着圆、且中间也在圆上」的弧线碎段（弦线会被保留）。"""
    if not circles:
        return lines
    out = []
    for e in lines:
        near = False
        mid = ((e.start[0] + e.end[0]) / 2, (e.start[1] + e.end[1]) / 2)
        for c in circles:
            d1 = abs(math.hypot(e.start[0] - c.center[0], e.start[1] - c.center[1]) - c.radius)
            d2 = abs(math.hypot(e.end[0] - c.center[0], e.end[1] - c.center[1]) - c.radius)
            dm = abs(math.hypot(mid[0] - c.center[0], mid[1] - c.center[1]) - c.radius)
            if d1 <= tol and d2 <= tol and dm <= tol:
                near = True
                break
        if not near:
            out.append(e)
    return out


def detect_lines(ink_without_circles: np.ndarray, p: dict) -> list[Entity]:
    skel = skeleton(ink_without_circles)
    raw = cv2.HoughLinesP(skel, 1, np.pi / 180, threshold=int(p["hough_thresh"]),
                          minLineLength=int(p["min_line_len"]), maxLineGap=int(p["max_line_gap"]))
    segs = []
    if raw is not None:
        for l in raw:
            x1, y1, x2, y2 = np.asarray(l).reshape(-1)[:4]
            segs.append(((float(x1), float(y1)), (float(x2), float(y2))))
    merged = merge_segments(segs, p)
    return [Entity(type="line", start=s[0], end=s[1]) for s in merged]


def vectorize(path: str, params: dict | None = None, meta: dict | None = None,
              ocr: list[dict] | None = None) -> DrawingIR:
    p = {**DEFAULTS, **(params or {})}
    bgr = imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    ink = to_ink(gray)
    if p.get("use_color_split", False):
        black_ink, _color_ink = split_ink(bgr)
        if black_ink.sum() > 0.001 * black_ink.size * 255:
            ink = black_ink

    # 剔除图框 / 标题栏
    if p.get("clean_border", True):
        border = detect_border(ink, p["border_area_ratio"], p["border_edge_margin"])
        if border is not None:
            tb = detect_titleblock(ink, border)
            if tb is not None:
                ink[int(tb[1]):int(tb[3]), int(tb[0]):int(tb[2])] = 0
            ink[mask_polygon(ink.shape, border) > 0] = 0

    # 剔除剖面线（密集平行短线）
    if p.get("clean_hatch", True):
        ink[detect_hatching(ink, p) > 0] = 0

    circles = detect_circles(gray, p, ink)
    residual = ink.copy()
    residual[_mask_circles(ink.shape, circles) > 0] = 0

    ellipses: list[Entity] = []
    if p.get("detect_ellipses", True):
        ellipses = detect_ellipses(residual, p)
        if ellipses:
            residual[mask_ellipses(ink.shape, ellipses, int(p["ellipse_pad"])) > 0] = 0

    lines = _drop_on_circles(detect_lines(residual, p), circles)
    if ocr and p.get("drop_dimensions", True):
        if p.get("dimension_action", "layer") == "layer":
            lines = classify_dimension_lines(lines, ocr, float(p["dim_line_tol"]))
        else:
            lines = drop_dimension_lines(lines, ocr, float(p["dim_line_tol"]))

    rects: list[Entity] = []
    if p.get("assemble", True):
        lines, rects = assemble_lines(lines, p)

    # 剔除直线后，对剩余墨迹做圆弧拟合
    line_mask = np.zeros(ink.shape, np.uint8)
    for e in lines:
        cv2.line(line_mask, (int(e.start[0]), int(e.start[1])),
                 (int(e.end[0]), int(e.end[1])), 255, int(p["line_pad"]))
    arc_residual = residual.copy()
    arc_residual[line_mask > 0] = 0
    arcs = detect_arcs(arc_residual, p)

    layers = [LayerSpec(name="outline"), LayerSpec(name="dim", color=8),
              LayerSpec(name="text", color=3)]
    return DrawingIR(width=float(w), height=float(h), layers=layers,
                     entities=[*lines, *rects, *arcs, *circles, *ellipses],
                     meta={"source": str(path), "params": p,
                           "rects": len(rects), "ellipses": len(ellipses), **(meta or {})})
