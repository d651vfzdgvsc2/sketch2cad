"""Post-assembly analytic geometry review, checked against source ink."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from emit.ir import Entity
from engineering.primitives import sample_entity
from engineering.trace import ink_mask
from vectorize.arcs import kasa_fit


def review_geometry(ir, image):
    """Replace circular polylines only when their entire perimeter is supported.

    Fit dense edge samples, not only polygon vertices: a regular polygon must
    not become a circle merely because all its vertices are on a circle.
    """
    ink = ink_mask(image)
    distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    revised = ir.model_copy(deep=True)
    changes, rejected = [], []
    for i, entity in enumerate(ir.entities):
        if entity.type != 'polyline' or entity.layer.startswith(('text', 'centerline')):
            continue
        points = sample_entity(entity, spacing=.8)
        if len(points) < 20 or len(entity.points) < 8: continue
        cx, cy, radius, rms = kasa_fit(points)
        if radius < 5 or radius > np.ptp(points, axis=0).max()*2: continue
        fit = least_squares(lambda p: np.linalg.norm(points-p[:2], axis=1)-p[2],
                            (cx, cy, radius), max_nfev=40)
        cx, cy, radius = fit.x
        residual = np.abs(fit.fun)
        tolerance = min(2.5, max(1.2, radius*.025))
        theta = np.unwrap(np.arctan2(points[:, 1]-cy, points[:, 0]-cx))
        span = theta[-1]-theta[0]
        if abs(span) < math.radians(50) or residual.mean() > tolerance*.7 or residual.max() > tolerance*2:
            continue
        if (np.diff(theta)*np.sign(span) < -.03).mean() > .02: continue
        closed = entity.closed or (math.dist(points[0], points[-1]) <= 3 and abs(span) > math.radians(355))
        if closed and abs(span) < math.radians(350): continue
        if closed:
            candidate = Entity(type='circle', center=(cx, cy), radius=radius, layer=entity.layer)
        else:
            a, b = (theta[0], theta[-1]) if span > 0 else (theta[-1], theta[0])
            candidate = Entity(type='arc', center=(cx, cy), radius=radius, layer=entity.layer,
                               start_angle=math.degrees(a), end_angle=math.degrees(b))
        samples = sample_entity(candidate, spacing=.5)
        xy = np.rint(samples).astype(int)
        if ((xy < 0).any() or (xy[:, 0] >= ink.shape[1]).any() or (xy[:, 1] >= ink.shape[0]).any()): continue
        support = float((distance[xy[:, 1], xy[:, 0]] <= 2.).mean())
        # Independent raster evidence; require near complete coverage.
        deviation = float(cKDTree(points).query(samples)[0].max())
        if support < .98 or deviation > tolerance*2:
            rejected.append(dict(entity_id=i, reason='insufficient_perimeter_evidence', support=support))
            continue
        # Preserve endpoints of other entities that touched the original chain.
        # The maximum allowed displacement is the same bound as the fit.
        old_tree = cKDTree(points)
        attachments = []
        for j, other in enumerate(revised.entities):
            if j == i: continue
            ends = [('start', other.start), ('end', other.end)] if other.type == 'line' else []
            if other.type == 'polyline' and not other.closed:
                ends = [(0, other.points[0]), (-1, other.points[-1])]
            for field, point in ends:
                if old_tree.query(point)[0] > .8: continue
                delta = np.array(point)-(cx, cy)
                length = float(np.linalg.norm(delta))
                if not length or abs(length-radius) > tolerance*2: continue
                projected = tuple(np.array((cx, cy))+delta*radius/length)
                if isinstance(field, str): setattr(other, field, projected)
                else: other.points[field] = projected
                attachments.append(j)
        revised.entities[i] = candidate
        changes.append(dict(entity_id=i, before='polyline', after=candidate.type,
                            vertices=len(entity.points), radius_px=float(radius),
                            fit_mean_px=float(residual.mean()), max_deviation_px=deviation,
                            perimeter_support=support, attached_entities=sorted(set(attachments))))
    straighten_and_merge(revised, distance, changes)
    revised.meta['geometry_review'] = dict(changes=changes, rejected=rejected)
    return revised


def _straight(points, layer, distance):
    """Only straighten almost-collinear measured paths, never shortcut steps."""
    points=np.asarray(points,dtype=float)
    a,b=points[0],points[-1];length=float(np.linalg.norm(b-a))
    if length<15:return None
    direction=(b-a)/length
    projection=(points-a)@direction
    deviation=np.abs(direction[0]*(points-a)[:,1]-direction[1]*(points-a)[:,0])
    if deviation.max()>.55 or (np.diff(projection)<-.05).any():return None
    deltas=np.diff(points,axis=0);sizes=np.linalg.norm(deltas,axis=1)
    valid=sizes>.1
    if not valid.any() or ((deltas[valid]@direction)/sizes[valid]<math.cos(math.radians(2))).any():return None
    candidate=Entity(type='line',start=tuple(a),end=tuple(b),layer=layer)
    xy=np.rint(sample_entity(candidate,spacing=.4)).astype(int)
    if (xy<0).any() or (xy[:,0]>=distance.shape[1]).any() or (xy[:,1]>=distance.shape[0]).any():return None
    if (distance[xy[:,1],xy[:,0]]<=1.2).mean()<.995:return None
    return candidate


def straighten_and_merge(ir, distance, changes):
    colors=ir.meta.get('stroke_colors_rgb', [None]*len(ir.entities))
    if len(colors)!=len(ir.entities):colors=[None]*len(ir.entities)
    colors=list(colors)
    for index,e in enumerate(ir.entities):
        if e.type!='polyline' or e.closed or e.layer.startswith(('text','centerline')):continue
        candidate=_straight(e.points,e.layer,distance)
        if candidate:
            ir.entities[index]=candidate
            changes.append(dict(entity_id=index,before='polyline',after='line',vertices=len(e.points),evidence='collinear_path_and_source_ink'))
    # Revisit endpoints after each merge; no bridging of intentional dash gaps.
    removed=set();merged=True
    while merged:
        merged=False
        ids=[i for i,e in enumerate(ir.entities) if i not in removed and e.type=='line' and not e.layer.startswith(('text','centerline'))]
        endpoints=[p for i in ids for p in (ir.entities[i].start,ir.entities[i].end)]
        if not endpoints:break
        tree=cKDTree(endpoints)
        for ia,ib in sorted(tree.query_pairs(.55)):
            i,j=ids[ia//2],ids[ib//2]
            if i==j:continue
            a,b=ir.entities[i],ir.entities[j]
            if a.layer!=b.layer or colors[i]!=colors[j]:continue
            # Shared endpoints only, with strict collinearity; preserve curve
            # junctions and short arrowheads as separate measured entities.
            pa=(a.start,a.end);pb=(b.start,b.end)
            outer_a,join_a=pa[1-ia%2],pa[ia%2]
            join_b,outer_b=pb[ib%2],pb[1-ib%2]
            if math.dist(*pa)<5 or math.dist(*pb)<5:continue
            if math.dist(join_a,join_b)>1e-6:
                # A tiny pixel gap still requires ink at the gap itself.
                mid=np.rint((np.array(join_a)+join_b)/2).astype(int)
                if not (0<=mid[0]<distance.shape[1] and 0<=mid[1]<distance.shape[0]) or distance[mid[1],mid[0]]>.5:continue
            candidate=_straight([outer_a,join_a,join_b,outer_b],a.layer,distance)
            if candidate:
                ir.entities[i]=candidate;removed.add(j);merged=True
                changes.append(dict(entity_id=i,consumed_entity_id=j,before='line_fragments',after='line',evidence='shared_collinear_endpoints_and_source_ink'))
                break
    if removed:
        ir.entities=[e for i,e in enumerate(ir.entities) if i not in removed]
        ir.meta['stroke_colors_rgb']=[c for i,c in enumerate(colors) if i not in removed]
