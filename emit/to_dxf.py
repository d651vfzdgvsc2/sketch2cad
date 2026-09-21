"""IR -> DXF 写出（确定性代码，坐标在此处统一翻转 y 轴）。"""
from __future__ import annotations

import math
from pathlib import Path

import ezdxf

from .ir import DrawingIR, Entity


def _flip(pt, height: float) -> tuple[float, float]:
    return (float(pt[0]), float(height - pt[1]))


def _add_entity(msp, e: Entity, height: float) -> None:
    layer = {"layer": e.layer}
    if e.type == "line":
        msp.add_line(_flip(e.start, height), _flip(e.end, height), dxfattribs=layer)
    elif e.type == "polyline":
        pts = [_flip(p, height) for p in e.points]
        msp.add_lwpolyline(pts, close=bool(e.closed), dxfattribs=layer)
    elif e.type == "circle":
        msp.add_circle(_flip(e.center, height), float(e.radius), dxfattribs=layer)
    elif e.type == "arc":
        cx, cy = _flip(e.center, height)
        sa = -float(e.end_angle)
        ea = -float(e.start_angle)
        msp.add_arc((cx, cy), float(e.radius), sa, ea, dxfattribs=layer)
    elif e.type == "ellipse":
        cx, cy = _flip(e.center, height)
        major, minor = float(e.major), float(e.minor)
        ang = math.radians(-e.rotation)  # 图像角度 -> CAD 角度（y 轴翻转）
        msp.add_ellipse((cx, cy), (major * math.cos(ang), major * math.sin(ang)),
                        ratio=max(1e-6, min(1.0, minor / major)), dxfattribs=layer)
    elif e.type == "text":
        pos = _flip(e.pos, height)
        txt = msp.add_text(e.content, height=float(e.height or 2.5),
                           dxfattribs={**layer, "rotation": e.rotation})
        try:
            txt.set_placement(pos, align=ezdxf.enums.TextEntityAlignment.LEFT)
        except Exception:  # noqa: BLE001
            txt.dxf.insert = pos


def ir_to_doc(ir: DrawingIR):
    doc = ezdxf.new("R2010")
    for layer in ir.layers:
        if layer.name not in doc.layers:
            doc.layers.add(layer.name, color=layer.color)
    msp = doc.modelspace()
    for e in ir.entities:
        _add_entity(msp, e, ir.height)
    return doc


def ir_to_dxf(ir: DrawingIR, path: str | Path) -> str:
    doc = ir_to_doc(ir)
    doc.saveas(str(path))
    return str(path)
