"""本地 OCR（RapidOCR，免费、离线）：识别图纸上的文字/尺寸/图框。

支持旋转文字：对 0°/90°/270° 三个方向分别识别，再把框映射回原图并去重。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidocr_onnxruntime import RapidOCR

from tools.image_io import imread


@lru_cache(maxsize=1)
def _engine() -> RapidOCR:
    return RapidOCR()


def _map_box_back(box, k: int, w: int, h: int):
    """把 rot90(k=1/3) 后的框坐标映射回原图。box: 4x2。"""
    pts = []
    for x, y in box:
        if k == 0:
            ox, oy = x, y
        elif k == 1:  # CCW 90
            ox, oy = (w - 1) - y, x
        else:  # k == 3, CW 90
            ox, oy = y, (h - 1) - x
        pts.append((float(ox), float(oy)))
    return pts


def _dedup(items: list[dict], center_tol: float = 20.0) -> list[dict]:
    out: list[dict] = []
    for it in sorted(items, key=lambda d: -d["score"]):
        dup = False
        for kept in out:
            if kept["text"] == it["text"]:
                dx = abs(kept["center"][0] - it["center"][0])
                dy = abs(kept["center"][1] - it["center"][1])
                if dx < center_tol and dy < center_tol:
                    dup = True
                    break
        if not dup:
            out.append(it)
    return out


def run_ocr(image_path: str | Path, rotations: tuple[int, ...] = (0, 1, 3)) -> list[dict]:
    img = imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    collected: list[dict] = []
    for k in rotations:
        rot = img if k == 0 else np.ascontiguousarray(np.rot90(img, k=k))
        result, _elapse = _engine()(rot)
        for item in result or []:
            box, text, score = item[0], item[1], item[2]
            pts = _map_box_back(box, k, w, h)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            collected.append({
                "text": str(text),
                "score": round(float(score), 4),
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "center": [round((min(xs) + max(xs)) / 2, 1), round((min(ys) + max(ys)) / 2, 1)],
            })
    return _dedup(collected)
