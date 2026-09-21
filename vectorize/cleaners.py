"""图纸清洗：剔除图框/标题栏、剖面线（斜线填充）等"非几何"干扰。"""
from __future__ import annotations

import math

import cv2
import numpy as np


def detect_border(ink: np.ndarray, area_ratio: float = 0.5, edge_margin: float = 0.15):
    """找覆盖大部分画面、且贴近四边的矩形 = 图框。返回 4x2 顶点或 None。"""
    h, w = ink.shape[:2]
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < area_ratio * h * w:
            break
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype(float)
        xs, ys = pts[:, 0], pts[:, 1]
        if (xs.min() < edge_margin * w and xs.max() > (1 - edge_margin) * w
                and ys.min() < edge_margin * h and ys.max() > (1 - edge_margin) * h):
            return pts
    return None


def mask_polygon(shape, pts: np.ndarray, thickness: int = 8) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], True, 255, thickness)
    return m


def detect_titleblock(ink: np.ndarray, border: np.ndarray):
    """在图框右下角找最大的矩形 = 标题栏。返回 (x0,y0,x1,y1) 或 None。"""
    xs, ys = border[:, 0], border[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rx0 = int(x0 + (x1 - x0) * 0.50)
    ry0 = int(y0 + (y1 - y0) * 0.72)
    region = ink[ry0:int(y1), rx0:int(x1)]
    if region.size == 0:
        return None
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    region_area = region.shape[0] * region.shape[1]
    for c in contours:
        a = cv2.contourArea(c)
        if 0.10 * region_area < a > best_area:
            best, best_area = c, a
    if best is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(best)
    if bw < 0.2 * region.shape[1] and bh < 0.2 * region.shape[0]:
        return None
    return (rx0 + bx, ry0 + by, rx0 + bx + bw, ry0 + by + bh)


def detect_hatching(ink: np.ndarray, p: dict) -> np.ndarray:
    """检测密集平行短线簇（剖面线）。支持多个不同方向，并按区域连通成片。"""
    lines = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=15,
                            minLineLength=int(p.get("hatch_min_len", 6)),
                            maxLineGap=3)
    if lines is None:
        return np.zeros_like(ink)

    short = []
    for l in lines:
        x1, y1, x2, y2 = np.asarray(l).reshape(-1)[:4]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < p.get("hatch_min_len", 6) or length > p.get("hatch_max_len", 90):
            continue
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        short.append((int(ang // 5) % 36, (x1 + x2) / 2, (y1 + y2) / 2))
    if len(short) < p.get("hatch_min_count", 40):
        return np.zeros_like(ink)

    # 找出所有"够密集"的方向（允许多个剖面区不同角度）
    ang_hist = np.zeros(36)
    for b, _mx, _my in short:
        ang_hist[b] += 1
    peak = ang_hist.max()
    strong = {b for b in range(36) if ang_hist[b] >= max(peak * 0.35, 15)}
    # 邻近方向合并（±1 bin）
    strong |= {((b + d) % 36) for b in list(strong) for d in (-1, 1)}

    cell = int(p.get("hatch_cell", 40))
    thr = int(p.get("hatch_thr", 6))
    h, w = ink.shape[:2]
    grid = np.zeros((h // cell + 2, w // cell + 2))
    for b, mx, my in short:
        if b in strong:
            grid[int(my // cell), int(mx // cell)] += 1

    mask = np.zeros_like(ink)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] >= thr:
                mask[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = 255

    # 区域生长：补满缝隙 + 只保留成片的区域
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((cell, cell), np.uint8))
    mask = cv2.dilate(mask, np.ones((cell // 2 + 1, cell // 2 + 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(ink)
    min_area = p.get("hatch_min_region", 0.004) * h * w
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def classify_dimension_lines(lines: list, ocr: list[dict], tol: float = 30.0, layer: str = "dim") -> list:
    """把靠近数字标注的线段归到 dim 图层（不删除，保留可开关）。"""
    import re

    nums = [o for o in ocr if re.search(r"\d", o.get("text", ""))]
    if not nums:
        return lines
    out = []
    for e in lines:
        mx = (e.start[0] + e.end[0]) / 2
        my = (e.start[1] + e.end[1]) / 2
        near = False
        for o in nums:
            x0, y0, x1, y1 = o["box"]
            dx = max(x0 - mx, 0.0, mx - x1)
            dy = max(y0 - my, 0.0, my - y1)
            if (dx * dx + dy * dy) ** 0.5 < tol:
                near = True
                break
        if near:
            out.append(e.model_copy(update={"layer": layer}))
        else:
            out.append(e)
    return out


def drop_dimension_lines(lines: list, ocr: list[dict], tol: float = 30.0) -> list:
    """剔除靠近数字标注的线段（尺寸线/尺寸界线）。"""
    import re

    nums = [o for o in ocr if re.search(r"\d", o.get("text", ""))]
    if not nums:
        return lines
    out = []
    for e in lines:
        mx = (e.start[0] + e.end[0]) / 2
        my = (e.start[1] + e.end[1]) / 2
        near = False
        for o in nums:
            x0, y0, x1, y1 = o["box"]
            dx = max(x0 - mx, 0.0, mx - x1)
            dy = max(y0 - my, 0.0, my - y1)
            if (dx * dx + dy * dy) ** 0.5 < tol:
                near = True
                break
        if not near:
            out.append(e)
    return out
