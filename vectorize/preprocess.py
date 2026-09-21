"""线稿预处理：灰度 -> 二值墨迹掩码 -> 去噪 -> 骨架化。"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from tools.image_io import imread


def load_gray(path: str) -> np.ndarray:
    g = imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(f"读不到图片: {path}")
    return g


def _remove_small(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def to_ink(gray: np.ndarray, min_area: int = 8) -> np.ndarray:
    """统一成「白底黑线」再取墨迹掩码：线条=255，背景=0。"""
    g = gray
    if float(g.mean()) < 127:  # 深色背景 -> 反相
        g = 255 - g
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _remove_small(bw, min_area)


def skeleton(ink: np.ndarray) -> np.ndarray:
    """把粗线细化成 1 像素中心线。"""
    return (skeletonize(ink > 0).astype(np.uint8)) * 255


def split_ink(bgr: np.ndarray, min_area: int = 8):
    """按颜色把墨迹分成「黑色几何」和「彩色中心线」。

    工程图里红色/彩色点划线通常是中心线/辅助线，属于标注而非几何，
    混进几何会生成大量碎线段。返回 (black_ink, color_ink)，均为 0/255。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)

    colored = ((s > 70) & (v < 240)).astype(np.uint8) * 255
    black = ((s <= 70) & (v < 160)).astype(np.uint8) * 255

    return _remove_small(black, min_area), _remove_small(colored, min_area)
