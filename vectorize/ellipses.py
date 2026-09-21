"""椭圆检测：轮廓 + 最小二乘椭圆拟合，识别透视角的椭圆（如轴端、斜面圆）。"""
from __future__ import annotations

import math

import cv2
import numpy as np

from emit.ir import Entity


def _residual(pts: np.ndarray, cx: float, cy: float, a: float, b: float, rot_deg: float) -> float:
    """把点映射到椭圆单位圆，返回半径的离散度（越接近 0 越像椭圆）。"""
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    r = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    return float(np.std(r))


def _coverage(ink_dil: np.ndarray, cx: float, cy: float, a: float, b: float,
              rot_deg: float, n: int = 72) -> float:
    """沿椭圆采样，统计落在墨迹上的比例（去假椭圆）。"""
    h, w = ink_dil.shape[:2]
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)
    hits = 0
    for i in range(n):
        ang = 2 * math.pi * i / n
        lx, ly = a * math.cos(ang), b * math.sin(ang)
        x = int(round(cx + lx * ct - ly * st))
        y = int(round(cy + lx * st + ly * ct))
        if 0 <= x < w and 0 <= y < h and ink_dil[y, x]:
            hits += 1
    return hits / n


def detect_ellipses(ink: np.ndarray, p: dict) -> list[Entity]:
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    k = 2 * int(p.get("ellipse_tol", 5)) + 1
    ink_dil = cv2.dilate(ink, np.ones((k, k), np.uint8))
    out: list[Entity] = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < p["ellipse_min_pts"]:
            continue
        try:
            (cx, cy), (d1, d2), ang = cv2.fitEllipse(pts.astype(np.float32))
        except Exception:  # noqa: BLE001
            continue
        if d1 < 1 or d2 < 1:
            continue
        a, b, rot = d1 / 2.0, d2 / 2.0, ang
        if a < b:  # 保证 a=半长轴
            a, b = b, a
            rot = (ang + 90.0) % 180.0
        if b < p["ellipse_min_axis"] or a > p["ellipse_max_axis"]:
            continue
        if a / b < p["ellipse_min_ecc"]:  # 太圆 -> 交给圆检测
            continue
        if _residual(pts, cx, cy, a, b, rot) > p["ellipse_max_resid"]:
            continue
        if _coverage(ink_dil, cx, cy, a, b, rot) < p.get("ellipse_min_cov", 0.70):
            continue
        out.append(Entity(type="ellipse", center=(float(cx), float(cy)),
                          major=float(a), minor=float(b), rotation=float(rot)))
    return out


def mask_ellipses(shape, ellipses: list[Entity], pad: int = 6) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for e in ellipses:
        cv2.ellipse(m, (int(e.center[0]), int(e.center[1])),
                    (int(e.major), int(e.minor)), e.rotation, 0, 360, 255, pad)
    return m
