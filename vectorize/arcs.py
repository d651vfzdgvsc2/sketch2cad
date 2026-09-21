"""圆弧检测：直线和整圆剔除后，对剩余墨迹做圆拟合，提取圆弧。

用于圆角矩形(R10)、腰形槽两端半圆、俯视图两端等真实工程图元素。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from emit.ir import Entity


def kasa_fit(pts: np.ndarray):
    """最小二乘圆拟合（Kasa 法）。返回 (cx, cy, r, rms)。"""
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    a = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    rms = float(np.sqrt((resid ** 2).mean()))
    return cx, cy, r, rms


def arc_span(pts: np.ndarray, cx: float, cy: float):
    """用「最大角度缺口」法求圆弧的起止角与张角（弧度，图像坐标 y 向下）。"""
    angs = np.sort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    if len(angs) < 3:
        return 0.0, 0.0, 0.0
    gaps = np.diff(np.concatenate([angs, [angs[0] + 2 * math.pi]]))
    gi = int(np.argmax(gaps))
    start = float(angs[(gi + 1) % len(angs)])
    end = float(angs[gi])
    span = 2 * math.pi - float(gaps[gi])
    return start, end, span


def detect_arcs(ink_residual: np.ndarray, p: dict) -> list[Entity]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink_residual, connectivity=8)
    out: list[Entity] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < p["arc_min_len"]:
            continue
        ys, xs = np.where(labels == i)
        pts = np.c_[xs, ys].astype(float)
        cx, cy, r, rms = kasa_fit(pts)
        if r < p["arc_min_r"] or r > p["arc_max_r"]:
            continue
        if rms > p["arc_max_rms"] or rms > 0.12 * r:
            continue
        start, end, span = arc_span(pts, cx, cy)
        if span < math.radians(p["arc_min_span_deg"]):
            continue
        if r * span < p["arc_min_len"]:
            continue
        out.append(Entity(type="arc", center=(cx, cy), radius=r,
                          start_angle=math.degrees(start), end_angle=math.degrees(end)))
    return out
