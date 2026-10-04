"""Assemble continuous engineering strokes without deleting hatch/detail strokes."""
from __future__ import annotations

import math
from collections import defaultdict
import cv2
import numpy as np
from scipy.spatial import cKDTree

from emit.ir import Entity
from engineering.primitives import sample_entity
from engineering.trace import trace_paths, prune_spurs
from vectorize.arcs import kasa_fit


def heal_scan_strokes(ink):
    """Fill enclosed one-pixel white cracks within a heavier scanned stroke.

    No open gap is closed and no region with an inscribed radius >= 2px is
    filled. Restrict to cracks bordered by thick ink to retain thin geometry.
    """
    background = 255-ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8)
    depth = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    maxima = np.zeros(n)
    np.maximum.at(maxima, labels.ravel(), depth.ravel())
    thick = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    near_thick = cv2.dilate((thick >= 1.8).astype(np.uint8), np.ones((5, 5), np.uint8))
    result = ink.copy()
    for i in np.flatnonzero((maxima > 0) & (maxima < 2)):
        x, y, w, h, _ = stats[i]
        if x == 0 or y == 0 or x+w == ink.shape[1] or y+h == ink.shape[0]: continue
        region = labels[y:y+h, x:x+w] == i
        if near_thick[y:y+h, x:x+w][region].mean() > .9:
            result[y:y+h, x:x+w][region] = 255
    return result


def repair_color_crossings(ink, colored):
    """Restore only short straight black strokes occluded by colored ink."""
    allowed = cv2.dilate(colored.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = ink.copy()
    for kernel in (np.ones((1, 9), np.uint8), np.ones((9, 1), np.uint8)):
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
        result[allowed] |= closed[allowed]
    return result


def continuous_paths(skel):
    paths, graph = trace_paths(prune_spurs(skel, max_length=2.0))
    # Collapse tiny junction connectors, but bound the size of each cluster.
    ends = sorted(set(p[0] for p in paths) | set(p[-1] for p in paths))
    parent = {p: p for p in ends}
    members = {p: [p] for p in ends}
    def root(p):
        while parent[p] != p: p = parent[p]
        return p
    for path in sorted(paths, key=len):
        a, b = root(path[0]), root(path[-1])
        if a == b or len(path) > 4 or len(graph[path[0]]) < 3 or len(graph[path[-1]]) < 3:
            continue
        group = members[a]+members[b]
        if max(math.dist(p, q) for p in group for q in group) <= 2.5:
            parent[b] = a
            members[a] = group
    centres = {p: tuple(np.mean(members[root(p)], axis=0)) for p in ends}
    chains = {}
    for i, path in enumerate(paths):
        path = list(path)
        path[0], path[-1] = centres[path[0]], centres[path[-1]]
        if max(np.ptp(np.array(path), axis=0)) < .5: continue
        chains[i] = path
    # Pair near-tangent branches at existing junctions. Never bridge an empty
    # interval: dashed/hidden lines retain their deliberate gaps.
    def tangent(path, end):
        seq = path if end == 0 else path[::-1]
        a = np.array(seq[0]); b = np.array(seq[min(8, len(seq)-1)])
        d = b-a
        return d/max(np.linalg.norm(d), 1e-9)
    while True:
        junctions = defaultdict(list)
        for i, path in chains.items():
            if path[0] == path[-1]: continue
            junctions[path[0]].append((i, 0))
            junctions[path[-1]].append((i, 1))
        options = []
        for point, group in junctions.items():
            for j, (i, ei) in enumerate(group):
                for k, ek in group[j+1:]:
                    if i == k: continue
                    dot = float(tangent(chains[i], ei)@tangent(chains[k], ek))
                    if dot < -math.cos(math.radians(22)):
                        options.append((dot, i, ei, k, ek))
        if not options: break
        touched = set()
        for _, i, ei, k, ek in sorted(options):
            if i in touched or k in touched: continue
            a = chains[i] if ei == 1 else chains[i][::-1]
            b = chains[k] if ek == 0 else chains[k][::-1]
            chains[i] = a+b[1:]
            del chains[k]
            touched.update((i, k))
    return list(chains.values())


def fit_chain(path, layer='outline'):
    pts = np.asarray(path, dtype=float)
    if len(pts) < 2: return []
    closed = math.dist(pts[0], pts[-1]) < .01
    chord = np.linalg.norm(pts[-1]-pts[0])
    if not closed and chord > 0:
        direction = (pts[-1]-pts[0])/chord
        distance = np.abs((pts[:, 0]-pts[0, 0])*direction[1]
                          -(pts[:, 1]-pts[0, 1])*direction[0])
        travel = (pts-pts[0])@direction
        if distance.max() <= 1.35 and np.diff(travel).min(initial=0) >= -1.05:
            return [Entity(type='line', start=tuple(pts[0]), end=tuple(pts[-1]), layer=layer)]
    if len(pts) >= 12:
        cx, cy, r, rms = kasa_fit(pts)
        if r >= 3 and r < max(np.ptp(pts, axis=0))*5 and rms < 1.0:
            residual = np.abs(np.linalg.norm(pts-(cx, cy), axis=1)-r)
            theta = np.unwrap(np.arctan2(pts[:, 1]-cy, pts[:, 0]-cx))
            span = theta[-1]-theta[0]
            if residual.max() < 2.0 and (np.diff(theta)*np.sign(span) < -.04).mean() < .04:
                if closed and abs(span) > 6:
                    return [Entity(type='circle', center=(cx, cy), radius=r, layer=layer)]
                if not closed and math.radians(25) < abs(span) < math.pi*1.98:
                    a, b = (theta[0], theta[-1]) if span > 0 else (theta[-1], theta[0])
                    return [Entity(type='arc', center=(cx, cy), radius=r,
                                   start_angle=math.degrees(a), end_angle=math.degrees(b), layer=layer)]
    pts = cv2.approxPolyDP(pts.astype(np.float32), 1.2, closed).reshape(-1, 2)
    if len(pts) < 2: return []
    if len(pts) == 2 and not closed:
        return [Entity(type='line', start=tuple(pts[0]), end=tuple(pts[1]), layer=layer)]
    return [Entity(type='polyline', points=[tuple(p) for p in pts], closed=closed, layer=layer)]


def assemble_paths(paths, features):
    """Replace supported runs with templates while retaining attachment points.

    Split on *edges*, retaining a common boundary point. The former raster
    subtraction dropped pixels at both ends and produced disconnected residuals.
    """
    if not features:
        return [e for p in paths for e in fit_chain(p)]
    samples = np.concatenate([f.samples() for f in features])
    tree = cKDTree(samples)
    result = [e for f in features for e in f.entities]
    for path in paths:
        points = np.array(path, dtype=float)
        distances, _ = tree.query(points)
        covered = (distances[:-1] <= 2.) & (distances[1:] <= 2.)
        # A single crossing is not an explained feature segment.
        changes = np.r_[0, np.flatnonzero(np.diff(covered))+1, len(covered)]
        for lo, hi in zip(changes, changes[1:]):
            if covered[lo] and hi-lo < 4: covered[lo:hi] = False
        changes = np.r_[0, np.flatnonzero(np.diff(covered))+1, len(covered)]
        for lo, hi in zip(changes, changes[1:]):
            if covered[lo]: continue
            segment = points[lo:hi+1].copy()
            # Keep the boundary of the retained segment attached to template.
            for index, attached in ((0, lo > 0), (-1, hi < len(covered))):
                if attached:
                    _, nearest = tree.query(segment[index])
                    segment[index] = samples[nearest]
            result.extend(fit_chain(segment))
    return result


def close_fitted_junctions(entities, tolerance=1.5):
    """Snap line/polyline endpoints to analytic arcs, not the reverse."""
    curves = [e for e in entities if e.type == 'arc']
    anchors = []
    for arc in curves:
        for angle in (arc.start_angle, arc.end_angle):
            angle = math.radians(angle)
            anchors.append(np.array(arc.center)+arc.radius*np.array([math.cos(angle), math.sin(angle)]))
    if not anchors: return entities
    tree = cKDTree(anchors)
    for e in entities:
        if e.type == 'line':
            for field in ('start', 'end'):
                p = getattr(e, field); distance, index = tree.query(p)
                if distance <= tolerance: setattr(e, field, tuple(anchors[index]))
        elif e.type == 'polyline' and not e.closed:
            for index in (0, -1):
                distance, nearest = tree.query(e.points[index])
                if distance <= tolerance: e.points[index] = tuple(anchors[nearest])
    return entities


def structure_report(entities):
    lengths, vertices, counts = [], 0, defaultdict(int)
    for e in entities:
        counts[e.type] += 1
        if e.type == 'line': lengths.append(math.dist(e.start, e.end))
        elif e.type == 'polyline':
            vertices += len(e.points)
            p = e.points+[e.points[0]] if e.closed else e.points
            lengths.extend(math.dist(a, b) for a, b in zip(p, p[1:]))
    return dict(entity_types=dict(counts), polyline_vertices=vertices,
                segments_under_3px=sum(v < 3 for v in lengths),
                short_segment_fraction=sum(v < 3 for v in lengths)/max(1, len(lengths)),
                native_texts=counts['text'], geometry_entities=len(entities)-counts['text'])


def assemble_centerlines(entities):
    """Represent regular collinear dash trains by native CAD line types.

    Irregular or curved marks remain separate. Large blank gaps split trains,
    so separate drawing views are not joined into one centre line.
    """
    groups, other = [], []
    for entity in entities:
        points = sample_entity(entity)
        if not len(points): continue
        span = np.ptp(points, axis=0)
        axis = int(span[1] > span[0])
        if span[1-axis] > 2.5 or span[axis] < 3:
            other.append(entity); continue
        offset = float(np.median(points[:, 1-axis]))
        group = next((g for g in groups if g['axis'] == axis and abs(g['offset']-offset) < 2), None)
        if group is None:
            group = dict(axis=axis, offset=offset, members=[]); groups.append(group)
        group['members'].append((float(points[:, axis].min()), float(points[:, axis].max()), entity))
    patterns = {}
    for group in groups:
        members = sorted(group['members'], key=lambda p: p[0])
        median_length = float(np.median([b-a for a, b, _ in members]))
        runs = [[]]
        for member in members:
            if runs[-1] and member[0]-runs[-1][-1][1] > max(20, median_length*4): runs.append([])
            runs[-1].append(member)
        for run in runs:
            if len(run) < 4:
                other.extend(e for _, _, e in run); continue
            gaps = [b[0]-a[1] for a, b in zip(run, run[1:])]
            dash = float(np.median([b-a for a, b, _ in run]))
            gap = float(np.median(gaps))
            if gap <= 1 or max(gaps) > gap*3 or min(gaps) < gap*.25:
                other.extend(e for _, _, e in run); continue
            layer = f'centerline_{len(patterns):03d}'
            # Align the first and final dash; don't extend line beyond evidence.
            length = run[-1][1]-run[0][0]
            period = (length-dash)/max(1, len(run)-1)
            gap = max(1., period-dash)
            patterns[layer] = [dash+gap, dash, -gap]
            a, b = [group['offset']]*2, [group['offset']]*2
            a[group['axis']], b[group['axis']] = run[0][0], run[-1][1]
            other.append(Entity(type='line', start=tuple(a), end=tuple(b), layer=layer))
    for entity in other:
        if not entity.layer.startswith('centerline_'): entity.layer = 'centerline'
    return other, patterns


def choose_assemblies(paths, features):
    """Adopt a template only if it reduces CAD complexity, not just raster error."""
    def cost(entities):
        return sum(max(1., len(e.points)*.6) if e.type == 'polyline' else 1. for e in entities)
    selected = []
    entities = assemble_paths(paths, [])
    current = cost(entities)
    for feature in sorted(features, key=lambda f: -len(f.samples())):
        proposed = assemble_paths(paths, selected+[feature])
        score = cost(proposed)
        if score < current-.1:
            entities, current = proposed, score
            selected.append(feature)
    return entities, selected
