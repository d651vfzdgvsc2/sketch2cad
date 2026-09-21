"""评测指标：图像级（IoU / Dice / SSIM / Chamfer）+ 分辨率容差。"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim


def to_ink_mask(img: np.ndarray) -> np.ndarray:
    """转为布尔掩码：True = 线条（墨迹）。自动兼容黑底/白底。"""
    gray = img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 127:  # 深色背景 -> 反相
        gray = 255 - gray
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw > 127


def _dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def iou(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    a, b = _dilate(a.astype(bool), tol), _dilate(b.astype(bool), tol)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def dice(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    a, b = _dilate(a.astype(bool), tol), _dilate(b.astype(bool), tol)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    return float(_ssim(ga, gb, data_range=255))


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """双向平均最近距离（像素），越小越好。"""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("inf")
    dt_b = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    dt_a = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dt_b[a].mean() + dt_a[b].mean()) / 2)


def compare(img_pred: np.ndarray, img_gt: np.ndarray, tol: int = 2) -> dict[str, float]:
    """两张图（BGR/灰度）的全套指标。"""
    a, b = to_ink_mask(img_pred), to_ink_mask(img_gt)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0])) > 0
    return {
        "iou": round(iou(a, b, tol), 4),
        "dice": round(dice(a, b, tol), 4),
        "ssim": round(ssim(img_pred, img_gt), 4),
        "chamfer_px": round(chamfer(a, b), 3),
        "ink_ratio_pred": round(float(a.mean()), 5),
        "ink_ratio_gt": round(float(b.mean()), 5),
    }
