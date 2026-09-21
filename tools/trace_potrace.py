"""用 Potrace（纯 Python）把线稿轮廓拟合成平滑曲线并渲染，用于假山等有机形状。

用法：python -m tools.trace_potrace <图片> [--max 1000] [--out path.png]
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import potrace
from PIL import Image, ImageDraw

from tools.image_io import imread


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


def trace(image_path: str, out_png: str, max_side: int = 1000) -> dict:
    gray = imread(image_path, cv2.IMREAD_GRAYSCALE)
    h0, w0 = gray.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    binary = gray < 128
    t = time.time()
    path = potrace.Bitmap(binary).trace()
    n_curves = len(path)
    n_segments = sum(len(c) for c in path)

    h, w = gray.shape
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for curve in path:
        pts = [_pt(curve.start_point)]
        for seg in curve:
            if getattr(seg, "is_corner", False):
                pts.append(_pt(seg.c))
                pts.append(_pt(seg.end_point))
            else:
                pts.extend(_bezier(pts[-1], _pt(seg.c1), _pt(seg.c2), _pt(seg.end_point)))
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw > 0.9 * w and bh > 0.9 * h:
            continue  # 跳过整幅图边框（背景轮廓）
        d.polygon(pts, fill=0)
    img.save(out_png)

    return {"curves": n_curves, "segments": n_segments,
            "secs": round(time.time() - t, 1), "size": f"{w0}x{h0}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = args.out or (args.image.rsplit(".", 1)[0] + "_potrace.png")
    print(trace(args.image, out, args.max), "->", out)
