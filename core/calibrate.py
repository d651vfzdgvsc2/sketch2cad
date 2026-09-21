"""用尺寸标注反推比例尺：让输出 DXF 具备真实尺寸（mm）。

思路：把 OCR 到的数字，和它最近的一条线段长度做比值（value / length_px），
收集多个估计后取中位数，抗离群。
"""
from __future__ import annotations

import math
import re

from emit.ir import DrawingIR, Entity


def _point_seg_dist(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _parse_number(text: str) -> float | None:
    nums = re.findall(r"\d+\.?\d*", text or "")
    if len(nums) == 1:
        try:
            return float(nums[0])
        except ValueError:
            return None
    return None


def estimate_scale(ocr: list[dict], ir: DrawingIR, tol: float = 50.0,
                   min_len: float = 8.0) -> dict:
    lines = [e for e in ir.entities if e.type == "line"]
    est: list[float] = []
    for o in ocr:
        v = _parse_number(o.get("text", ""))
        if not v or v <= 0:
            continue
        c = o["center"]
        best, best_d = None, tol
        for e in lines:
            d = _point_seg_dist(c, e.start, e.end)
            if d < best_d:
                best_d, best = d, e
        if best is None:
            continue
        length = math.hypot(best.end[0] - best.start[0], best.end[1] - best.start[1])
        if length < min_len:
            continue
        est.append(v / length)
    if not est:
        return {"mm_per_px": None, "n": 0, "n_inliers": 0, "estimates": []}

    est_sorted = sorted(est)
    med = est_sorted[len(est_sorted) // 2]
    inliers = sorted(x for x in est if 0.4 * med <= x <= 2.5 * med)
    if inliers:
        med = inliers[len(inliers) // 2]
    return {"mm_per_px": round(med, 5), "n": len(est), "n_inliers": len(inliers),
            "estimates": [round(x, 4) for x in est]}


def scale_ir(ir: DrawingIR, factor: float) -> DrawingIR:
    """整体缩放（像素 -> 真实尺寸）。"""
    def s(pt):
        return (pt[0] * factor, pt[1] * factor)

    ents: list[Entity] = []
    for e in ir.entities:
        d = e.model_dump()
        if e.type == "line":
            d["start"], d["end"] = s(e.start), s(e.end)
        elif e.type == "polyline":
            d["points"] = [s(p) for p in e.points]
        elif e.type in ("circle", "arc"):
            d["center"], d["radius"] = s(e.center), e.radius * factor
        elif e.type == "ellipse":
            d["center"] = s(e.center)
            d["major"], d["minor"] = e.major * factor, e.minor * factor
        elif e.type == "text":
            d["pos"], d["height"] = s(e.pos), (e.height or 2.5) * factor
        ents.append(Entity(**d))
    return DrawingIR(width=ir.width * factor, height=ir.height * factor,
                     layers=ir.layers, entities=ents,
                     meta={**ir.meta, "scale_mm_per_px": factor})
