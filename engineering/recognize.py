"""Fit library templates to measured contours; accept only pixel-supported fits."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from engineering.primitives import Feature, build
from vectorize.arcs import kasa_fit


def _box_distance(points, cx, cy, w, h, radius, angle):
    a = math.radians(angle)
    local = (points-(cx, cy)) @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = np.abs(local) - (np.array([w, h])/2-radius)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0)-radius


def _fit_options(points):
    candidates = []
    cx, cy, radius, rms = kasa_fit(points)
    if radius > 2:
        fit = least_squares(lambda p: np.linalg.norm(points-p[:2], axis=1)-p[2],
                            [cx, cy, radius], max_nfev=40)
        c = fit.x
        candidates.append(("circle", {"center": c[:2].tolist(), "radius": float(c[2])}, fit.fun))
    (cx, cy), (w, h), angle = cv2.minAreaRect(points.astype(np.float32))
    if h > w:
        w, h, angle = h, w, angle+90
    if min(w, h) < 4:
        return candidates
    bounds = ([cx-w*.15, cy-w*.15, w*.7, h*.7, angle-12],
              [cx+w*.15, cy+w*.15, w*1.3, h*1.3, angle+12])
    fit = least_squares(lambda p: _box_distance(points, *p[:4], 0, p[4]),
                        [cx, cy, w, h, angle], bounds=bounds, max_nfev=50)
    x = fit.x
    candidates.append(("rectangle", {"center": x[:2].tolist(), "width": x[2],
                                     "height": x[3], "rotation": x[4]}, fit.fun))
    # Radius parameter is a fraction of min(width,height)/2: bounds remain valid.
    initial = [cx, cy, w, h, angle, .2]
    lo, hi = list(bounds[0])+[0.001], list(bounds[1])+[1.]
    fit = least_squares(lambda p: _box_distance(points, *p[:4], min(p[2:4])/2*p[5], p[4]),
                        initial, bounds=(lo, hi), max_nfev=70)
    x = fit.x
    radius = min(x[2:4])/2*x[5]
    if radius > 1.5:
        candidates.append(("rounded_rectangle", {"center": x[:2].tolist(), "width": x[2],
                         "height": x[3], "rotation": x[4], "radius": radius}, fit.fun))
    if w/h > 1.3:
        # Parameterize slot by positive center-to-center span + diameter.
        fit = least_squares(lambda p: _box_distance(points, p[0], p[1], p[2]+p[3], p[3], p[3]/2, p[4]),
                            [cx, cy, w-h, h, angle],
                            bounds=([cx-w*.15, cy-w*.15, .1, h*.7, angle-12],
                                    [cx+w*.15, cy+w*.15, w*1.3, h*1.3, angle+12]), max_nfev=60)
        x = fit.x
        candidates.append(("slot", {"center": x[:2].tolist(), "length": x[2]+x[3],
                                    "diameter": x[3], "rotation": x[4]}, fit.fun))
    if len(points) >= 10:
        (ex, ey), (aw, ah), rotation = cv2.fitEllipse(points.astype(np.float32))
        if aw < ah:
            aw, ah, rotation = ah, aw, rotation+90
        if ah > 4 and 1.06 < aw/ah < 20:
            a = math.radians(rotation)
            loc = (points-(ex, ey)) @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            residual = (np.sqrt((loc[:, 0]/(aw/2))**2+(loc[:, 1]/(ah/2))**2)-1)*(ah/2)
            candidates.append(("ellipse", {"center": [ex, ey], "major": aw/2, "minor": ah/2,
                                            "rotation": rotation}, residual))
    return candidates


def recognize_features(ink, skel, annotation_mask=None):
    """No image-specific coordinates or fixture names; contours measured at runtime."""
    height, width = ink.shape
    if not skel.any():
        return [], []
    dt, indices = distance_transform_edt(~skel, return_indices=True)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates, rejected = [], []
    contours = sorted(contours, key=lambda c: (-cv2.contourArea(c), cv2.boundingRect(c)))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 9 or cv2.contourArea(contour) < 45 or len(contour) < 20:
            continue
        raw = contour.reshape(-1, 2)
        if annotation_mask is not None and (annotation_mask[raw[:, 1], raw[:, 0]] > 0).mean() > .3:
            continue
        # Snap both sides of a stroke to its measured centerline before fitting.
        pts = np.column_stack((indices[1, raw[:, 1], raw[:, 0]], indices[0, raw[:, 1], raw[:, 0]]))
        pts = np.unique(pts, axis=0).astype(float)
        if len(pts) > 500:
            # Stable, evenly spaced samples by polar order; not first-N truncation.
            center = pts.mean(axis=0)
            order = np.argsort(np.arctan2(pts[:, 1]-center[1], pts[:, 0]-center[0]))
            pts = pts[order[np.linspace(0, len(pts)-1, 500).astype(int)]]
        options = []
        for kind, params, residual in _fit_options(pts):
            rms = float(np.sqrt(np.mean(residual**2)))
            if rms > .95 or np.percentile(np.abs(residual), 95) > 1.7:
                continue
            try:
                feature = Feature(kind, {k: float(v) if isinstance(v, np.floating) else v
                                         for k, v in params.items()}, build(kind, params), rms=rms)
            except ValueError:
                continue
            samples = feature.samples()
            xy = np.rint(samples).astype(int)
            inside = (xy[:, 0] >= 0)&(xy[:, 0] < width)&(xy[:, 1] >= 0)&(xy[:, 1] < height)
            if not inside.all():
                continue
            distances = dt[xy[:, 1], xy[:, 0]]
            feature.support = float((distances <= 1.6).mean())
            # Full perimeter evidence prevents fitting a complete circle to a semicircle.
            if feature.support < .97 or np.percentile(distances, 99) > 2.3:
                continue
            backward, _ = cKDTree(samples).query(pts)
            if np.percentile(backward, 95) > 1.7:
                continue
            feature.evidence = {"bbox_px": [x, y, w, h], "contour_samples": len(pts),
                                "p95_fit_px": round(float(np.percentile(backward, 95)), 3)}
            complexity = {"circle": 0., "rectangle": .05, "ellipse": .10,
                          "slot": .07, "rounded_rectangle": .18}[kind]
            options.append((rms+complexity, feature))
        if options:
            candidates.append(min(options, key=lambda pair: pair[0])[1])
    # Nested stroke contours yield the same measured feature: retain only one.
    accepted, trees = [], []
    for candidate in sorted(candidates, key=lambda f: (f.rms, -f.support)):
        samples = candidate.samples()
        duplicate = False
        for other, tree in zip(accepted, trees):
            if math.dist(candidate.params["center"], other.params["center"]) > 3:
                continue
            ds, _ = tree.query(samples)
            if np.percentile(ds, 95) < 2:
                duplicate = True
                break
        if not duplicate and trees:
            nearest = np.minimum.reduce([tree.query(samples)[0] for tree in trees])
            if (nearest < 1.25).mean() > .60:
                duplicate = True  # Do not duplicate shared rectangle boundaries.
        if not duplicate:
            candidate.id = f"feature_{len(accepted):04d}"
            accepted.append(candidate)
            trees.append(cKDTree(samples))
    return accepted, rejected


def assign_views(ink, features, annotation_mask=None):
    """Conservative connected drawing regions after excluding frame/text influences.

    Coordinates always stay in the full image. Unknown view assignments never
    authorize relations across regions; the original pixels are not erased.
    """
    work = ink.copy()
    from vectorize.cleaners import detect_border
    border = detect_border(work, .75, .1)
    if border is not None:
        cv2.polylines(work, [np.rint(border).astype(np.int32)], True, 0, 8)
    if annotation_mask is not None:
        work[annotation_mask > 0] = 0
    size = max(3, int(min(work.shape)*.012))
    expanded = cv2.dilate(work, np.ones((size, size), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    regions = []
    for x, y, w, h, area in stats[1:]:
        if area >= ink.size*.001 and w > size*2 and h > size*2:
            regions.append([int(x), int(y), int(w), int(h)])
    regions.sort(key=lambda b: (b[1]//max(1, size), b[0]))
    for feature in features:
        fx, fy, fw, fh = feature.evidence["bbox_px"]
        fits = [(i, w*h) for i, (x, y, w, h) in enumerate(regions)
                if x <= fx+2 and y <= fy+2 and x+w >= fx+fw-2 and y+h >= fy+fh-2]
        feature.evidence["view_id"] = f"view_{max(fits, key=lambda v: v[1])[0]:03d}" if fits else "unknown"
    return [{"id": f"view_{i:03d}", "bbox_px": box, "transform_to_image": [1, 0, 0, 1, 0, 0],
             "status": "candidate_region"} for i, box in enumerate(regions)]


def measured_patterns(features):
    """Retrieve compound patterns using measured members; do not force/snatch holes."""
    circles = [f for f in features if f.kind == "circle"]
    result = []
    consumed = set()
    for seed in circles:
        if seed.id in consumed:
            continue
        members = [f for f in circles if f.id not in consumed and
                   f.evidence.get("view_id") == seed.evidence.get("view_id") and
                   seed.evidence.get("view_id", "unknown") != "unknown" and
                   abs(f.params["radius"]-seed.params["radius"]) <= max(.7, seed.params["radius"]*.03)]
        if len(members) >= 2:
            # Require row/column alignment and proximity in units of hole diameter.
            group = [seed]
            pending = [f for f in members if f.id != seed.id]
            while pending:
                additions = []
                for f in pending:
                    for member in group:
                        delta = np.abs(np.array(f.params["center"])-member.params["center"])
                        if delta.min() <= 2 and delta.max() <= seed.params["radius"]*100:
                            additions.append(f)
                            break
                if not additions:
                    break
                group.extend(additions)
                pending = [f for f in pending if f not in additions]
            if len(group) >= 2:
                consumed.update(f.id for f in group)
                result.append({"template": "hole_array", "entity_ids": [f.id for f in group],
                               "params": {"centers": [f.params["center"] for f in group],
                                          "radius": float(np.median([f.params["radius"] for f in group]))},
                               "status": "measured_relation_only"})
    for i, a in enumerate(circles):
        for b in circles[i+1:]:
            if (a.evidence.get("view_id") == b.evidence.get("view_id") and
                    a.evidence.get("view_id", "unknown") != "unknown" and
                    math.dist(a.params["center"], b.params["center"]) < 1.5):
                result.append({"template": "concentric_circles", "entity_ids": [a.id, b.id],
                               "status": "measured_relation_only"})
    return result
