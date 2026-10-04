"""Topology-preserving engineering tracing and local geometric fitting."""
from __future__ import annotations

import math
import cv2
import numpy as np
from skimage.morphology import skeletonize

from emit.ir import Entity
from vectorize.arcs import kasa_fit


def ink_mask(image):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 127:
        gray = 255-gray
    # No region erasure: hatch, annotation, thin walls and frame stay available.
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 2)*255
    return keep[labels]


def trace_paths(mask):
    ys, xs = np.where(mask)
    points = set(zip(xs.tolist(), ys.tolist()))
    graph = {}
    for x, y in sorted(points):
        neighbors = []
        for dx, dy in ((-1, 0), (0, -1), (0, 1), (1, 0),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = x+dx, y+dy
            if q not in points:
                continue
            # Do not create a diagonal shortcut around an orthogonal corner.
            if dx and dy and ((x+dx, y) in points or (x, y+dy) in points):
                continue
            neighbors.append(q)
        graph[(x, y)] = neighbors
    used, paths = set(), []
    edge = lambda a, b: tuple(sorted((a, b)))

    def walk(a, b):
        path = [a, b]
        used.add(edge(a, b))
        prev, cur = a, b
        while len(graph[cur]) == 2:
            nxt = next(q for q in graph[cur] if q != prev)
            if edge(cur, nxt) in used:
                break
            used.add(edge(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for p, neighbors in graph.items():
        if len(neighbors) != 2:
            for q in neighbors:
                if edge(p, q) not in used:
                    paths.append(walk(p, q))
    for p, neighbors in graph.items():
        for q in neighbors:
            if edge(p, q) not in used:
                paths.append(walk(p, q))
    return paths, graph


def prune_spurs(mask, max_length=3):
    """Remove only short leaf-to-junction branches, never entire open strokes."""
    result = mask.copy().astype(bool)
    paths, graph = trace_paths(result)
    for path in paths:
        a, b = len(graph[path[0]]), len(graph[path[-1]])
        if a != 1:
            path, a, b = path[::-1], b, a
        if a == 1 and b >= 3 and sum(math.dist(p, q) for p, q in zip(path, path[1:])) <= max_length:
            for x, y in path[:-1]:
                result[y, x] = False
    return result


def fit_path(path, layer="source_ink", analytic=True):
    pts = np.asarray(path, np.float64)
    if len(pts) < 2:
        return []
    closed = tuple(path[0]) == tuple(path[-1])
    chord = float(np.linalg.norm(pts[-1]-pts[0]))
    if not closed and chord > 0:
        v = (pts[-1]-pts[0])/chord
        residual = np.abs((pts[:, 0]-pts[0, 0])*v[1] - (pts[:, 1]-pts[0, 1])*v[0])
        projection = (pts-pts[0])@v
        if residual.max() <= .65 and np.diff(projection).min(initial=0) >= -.1:
            return [Entity(type="line", start=tuple(pts[0]), end=tuple(pts[-1]), layer=layer)]
    if analytic and len(pts) >= 12:
        cx, cy, radius, rms = kasa_fit(pts)
        bbox = float(np.ptp(pts, axis=0).max())
        if 2 <= radius <= 10*max(bbox, 1) and rms < .55:
            angles = np.unwrap(np.arctan2(pts[:, 1]-cy, pts[:, 0]-cx))
            span = angles[-1]-angles[0]
            endpoint_error = max(abs(math.dist(p, (cx, cy))-radius) for p in (pts[0], pts[-1]))
            radial = np.abs(np.linalg.norm(pts-(cx, cy), axis=1)-radius)
            if radial.max() < 1.2 and endpoint_error < .8:
                if closed and abs(span) > 6.0:
                    return [Entity(type="circle", center=(cx, cy), radius=radius, layer=layer)]
                if not closed and math.radians(18) < abs(span) < 2*math.pi-.1:
                    # Reject backtracking paths; arc must follow the actual stroke.
                    delta = np.diff(angles)*np.sign(span)
                    if (delta < -.02).mean() < .03:
                        a, b = (angles[0], angles[-1]) if span > 0 else (angles[-1], angles[0])
                        return [Entity(type="arc", center=(cx, cy), radius=radius,
                                       start_angle=math.degrees(a), end_angle=math.degrees(b), layer=layer)]
    simplified = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), .45, closed).reshape(-1, 2)
    if len(simplified) < 2:
        return []
    return [Entity(type="polyline", points=[tuple(map(float, p)) for p in simplified],
                   closed=closed, layer=layer)]


def trace_entities(skel, analytic=True, annotation_mask=None):
    paths, _ = trace_paths(skel)
    entities = []
    for path in paths:
        layer = "source_ink"
        if annotation_mask is not None:
            xy = np.asarray(path)
            if (annotation_mask[xy[:, 1], xy[:, 0]] > 0).mean() > .5:
                layer = "annotation"
        entities.extend(fit_path(path, layer, analytic=analytic))
    return entities


def skeleton(ink):
    return skeletonize(ink > 0)
