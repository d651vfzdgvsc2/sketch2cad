"""假山 / 有机形状矢量化通道：Potrace 轮廓拟合 -> IR -> DXF。

适用于不规则自由曲线（假山、植物、地形），不适合机械工程图。
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import potrace

from emit.ir import DrawingIR, Entity, LayerSpec
from emit.to_dxf import ir_to_dxf
from render.render_dxf import render_dxf_to_image
from tools.image_io import imread

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "rockery"


def _pt(p) -> tuple[float, float]:
    try:
        return (float(p.x), float(p.y))
    except AttributeError:
        return (float(p[0]), float(p[1]))


def _bezier(p0, c1, c2, p1, n: int = 8):
    out = []
    for t in np.linspace(0, 1, n)[1:]:
        mt = 1 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p1[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p1[1]
        out.append((float(x), float(y)))
    return out


def _curve_points(curve, n: int = 8) -> list[tuple[float, float]]:
    pts = [_pt(curve.start_point)]
    for seg in curve:
        if getattr(seg, "is_corner", False):
            pts.append(_pt(seg.c))
            pts.append(_pt(seg.end_point))
        else:
            pts.extend(_bezier(pts[-1], _pt(seg.c1), _pt(seg.c2), _pt(seg.end_point), n))
    return pts


def trace_drawing(image: str, max_side: int = 1600, min_size: float = 12.0) -> tuple[DrawingIR, dict]:
    gray = imread(image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image)
    h0, w0 = gray.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    t = time.time()
    path = potrace.Bitmap(gray < 128).trace()
    H, W = gray.shape
    inv = 1.0 / scale

    entities: list[Entity] = []
    skipped_border = 0
    for curve in path:
        pts = _curve_points(curve)
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        if bw > 0.9 * W and bh > 0.9 * H:  # 整幅图边框
            skipped_border += 1
            continue
        if max(bw, bh) < min_size / max(scale, 1e-6):
            continue
        scaled = [(round(x * inv, 2), round(y * inv, 2)) for x, y in pts]
        entities.append(Entity(type="polyline", points=scaled, closed=True, layer="rockery"))

    ir = DrawingIR(width=float(w0), height=float(h0), entities=entities,
                   layers=[LayerSpec(name="rockery", color=7)],
                   meta={"source": image, "engine": "potrace"})
    stats = {"curves": len(path), "entities": len(entities),
             "skipped_border": skipped_border, "secs": round(time.time() - t, 2)}
    return ir, stats


def run_rockery(image: str, out_dir: str | Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else TMP
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem

    ir, stats = trace_drawing(image)
    dxf = out / f"{stem}_rockery.dxf"
    png = out / f"{stem}_rockery.png"
    ir_to_dxf(ir, dxf)
    render_dxf_to_image(dxf, png, width=ir.width, height=ir.height)
    (out / f"{stem}_rockery.json").write_text(ir.to_json(), encoding="utf-8")

    return {"image": image, "dxf": str(dxf), "png": str(png),
            "n_entities": len(ir.entities), **stats}
