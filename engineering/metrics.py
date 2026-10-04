"""Strict, foreground-aware engineering evaluation in original pixel coordinates."""
from __future__ import annotations

import math
import cv2
import numpy as np
from skimage.morphology import skeletonize

from eval.metrics import compare as legacy_compare, to_ink_mask


def compare(pred, reference, tol=2, ignore_mask=None):
    if pred.shape[:2] != reference.shape[:2]:
        raise ValueError("Engineering evaluation requires identical canvas sizes; resize is forbidden")
    if ignore_mask is not None:
        pred, reference = pred.copy(), reference.copy()
        pred[ignore_mask > 0] = 255
        reference[ignore_mask > 0] = 255
    result = legacy_compare(pred, reference, tol=tol)
    a, b = (skeletonize(to_ink_mask(im)) for im in (pred, reference))
    if not a.any() or not b.any():
        result["chamfer_px"] = None
        result.update(precision=0., recall=0., f1=0., p95_px=None,
                      centerline_chamfer_px=None, normalized_chamfer=None, valid=False)
        result["score"] = 0.
        return result
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    forward, reverse = db[a], da[b]
    precision, recall = float((forward <= tol).mean()), float((reverse <= tol).mean())
    f1 = 2*precision*recall/max(precision+recall, 1e-12)
    distance = float((forward.mean()+reverse.mean())/2)
    p95 = float(max(np.percentile(forward, 95), np.percentile(reverse, 95)))
    result.update(precision=round(precision, 6), recall=round(recall, 6), f1=round(f1, 6),
                  centerline_chamfer_px=round(distance, 4), p95_px=round(p95, 4),
                  normalized_chamfer=round(distance/math.hypot(*a.shape), 8),
                  valid=True, metric_version="engineering-fixed-canvas-v1", tolerance_px=tol)
    result["score"] = combined_score(result)
    return result


def combined_score(sc):
    if not sc.get("valid", True):
        return 0.
    f1 = float(sc.get("f1", sc.get("dice", 0)) or 0)
    distance = sc.get("centerline_chamfer_px", sc.get("chamfer_px"))
    p95 = sc.get("p95_px", distance)
    if distance is None or p95 is None or not math.isfinite(float(distance)):
        return 0.
    return round(.7*f1 + .2*math.exp(-float(distance)/3) + .1*math.exp(-float(p95)/6), 6)
