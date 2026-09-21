"""多视图分割：把一张图纸上并排的多个视图，按空白切成若干区域，分别矢量化再合并。

思路：对墨迹做形态学膨胀（把同一视图内的线连成一片），再做连通域，
每个足够大的连通域 ≈ 一个视图；对每个区域裁剪后单独矢量化，最后把坐标平移回去。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from emit.ir import DrawingIR, Entity
from tools.image_io import imread, imwrite
from tools.ocr import run_ocr
from vectorize.preprocess import to_ink
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "mv"


def segment_views(ink: np.ndarray, dilate_px: int = 50, min_area_ratio: float = 0.02,
                  margin: int = 15) -> list[tuple[int, int, int, int]]:
    """返回视图区域列表 [(x, y, w, h), ...]。"""
    h, w = ink.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
    dil = cv2.dilate(ink, k)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    regions = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_ratio * h * w:
            continue
        x, y, rw, rh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        x = max(0, x - margin)
        y = max(0, y - margin)
        rw = min(w - x, rw + 2 * margin)
        rh = min(h - y, rh + 2 * margin)
        regions.append((int(x), int(y), int(rw), int(rh)))
    # 按阅读顺序排序（上到下、左到右）
    regions.sort(key=lambda r: (r[1] // max(1, dilate_px), r[0]))
    return regions


def _offset_entity(e: Entity, dx: float, dy: float) -> Entity:
    d = e.model_dump()

    def o(pt):
        return (pt[0] + dx, pt[1] + dy)

    if e.type == "line":
        d["start"], d["end"] = o(e.start), o(e.end)
    elif e.type == "polyline":
        d["points"] = [o(p) for p in e.points]
    elif e.type in ("circle", "arc", "ellipse"):
        d["center"] = o(e.center)
    elif e.type == "text":
        d["pos"] = o(e.pos)
    return Entity(**d)


def vectorize_multiview(image: str, params: dict | None = None,
                        dilate_px: int = 50, min_area_ratio: float = 0.02) -> tuple[DrawingIR, int]:
    """多视图分割 + 分区矢量化。返回 (合并后的 IR, 视图数)。"""
    bgr = imread(image)
    if bgr is None:
        raise FileNotFoundError(image)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = to_ink(gray)

    regions = segment_views(ink, dilate_px=dilate_px, min_area_ratio=min_area_ratio)
    if len(regions) <= 1:
        return vectorize(image, params=params, ocr=run_ocr(image)), 1

    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    all_entities: list[Entity] = []
    for idx, (x, y, rw, rh) in enumerate(regions):
        crop = bgr[y:y + rh, x:x + rw]
        crop_path = TMP / f"{stem}_v{idx}.png"
        imwrite(crop_path, crop)
        ocr = run_ocr(crop_path)
        ir = vectorize(str(crop_path), params=params, ocr=ocr)
        for e in ir.entities:
            all_entities.append(_offset_entity(e, x, y))

    merged = DrawingIR(width=float(w), height=float(h),
                       entities=all_entities,
                       meta={"source": image, "views": len(regions)})
    return merged, len(regions)
