"""用尺寸标注反推比例尺：让输出 DXF 具备真实尺寸（mm）。

需要尺寸文字与两端界线的关联证据，以及多个独立一致的比例；冲突时不自动校准。
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
    from engineering.calibration import parse_dimension
    parsed = parse_dimension(text)
    return parsed["value"] if parsed["kind"] == "linear" else None


def estimate_scale(ocr: list[dict], ir: DrawingIR, tol: float = 50.0,
                   min_len: float = 8.0) -> dict:
    from engineering.calibration import estimate_scale as conservative_estimate
    return conservative_estimate(ocr, ir, tol, min_len)


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
