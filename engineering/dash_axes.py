"""Align source-supported dash trains without joining views or bending contours."""
from __future__ import annotations

import math
import cv2
import numpy as np

from emit.ir import Entity
from engineering.linework import (_straight_record, _protected, _near_text,
                                  _supported, _preserve_associations)
from engineering.dimensions import dimension_candidates
from engineering.primitives import sample_entity
from engineering.trace import ink_mask


def review_dash_axes(ir, image):
    revised = ir.model_copy(deep=True)
    candidates = dimension_candidates(ir)
    samples = {i: sample_entity(e, .8) for i, e in enumerate(ir.entities) if e.type != 'text'}
    protected = _protected(ir, candidates, samples)
    ink = ink_mask(image)
    distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    records = [r for i, p in samples.items()
               if (r := _straight_record(i, ir.entities[i], p)) is not None
               and i not in protected and not _near_text(p, ir)]
    changes = {}; removed = set(); trains = []
    # Existing native axes have stronger evidence than small tracing remnants.
    for i, e in enumerate(ir.entities):
        if e.type != 'line' or not e.layer.startswith('centerline_'): continue
        a, b = np.array(e.start), np.array(e.end)
        length = np.linalg.norm(b-a); u = (b-a)/max(length, 1e-9)
        n = np.array([-u[1], u[0]])
        for j, p in samples.items():
            raw = ir.entities[j]
            if j in protected or raw.layer != 'centerline' or raw.type not in ('line', 'polyline'): continue
            if raw.closed and len(raw.points) > 2: continue
            t = (p-a)@u
            if t.min() < -1 or t.max() > length+1 or np.abs((p-a)@n).max() > 3: continue
            projected = a+np.outer(t,u)
            if not raw.closed and t.max()-t.min() > .5 and _supported(projected,distance,2.) >= .8:
                changes[j] = Entity(type='line',start=tuple(a+u*t.min()),end=tuple(a+u*t.max()),layer=raw.layer)
            # A remnant is consumed only inside an ON interval of the native
            # pattern, not inside a deliberate OFF gap or at another crossing.
            pattern = ir.meta.get('linetype_patterns', {}).get(e.layer, [])
            if len(pattern) != 3: continue
            phase = np.mod(t, pattern[0])
            if ((phase <= pattern[1]+2) | (phase >= pattern[0]-2)).mean() < .85: continue
            removed.add(j)
    available = [r for r in records if r['id'] not in removed and
                 not r['layer'].startswith('centerline_') and 2 <= r['length'] <= 85]
    used = set()
    for seed in sorted(available, key=lambda r: -r['length']):
        if seed['id'] in used: continue
        u = seed['v'].copy()
        # Short tilted dashes cannot define a reliable train angle. Propose
        # the nearby engineering axis first; the whole run/raster validates it.
        if abs(u[1]) < math.sin(math.radians(8)): u=np.array([np.sign(u[0]),0.])
        elif abs(u[0]) < math.sin(math.radians(8)): u=np.array([0.,np.sign(u[1])])
        n = np.array([-u[1], u[0]])
        peers = [r for r in available if r['id'] not in used and r['layer'] == seed['layer']
                 and abs(r['v']@u) >= math.cos(math.radians(14))
                 and abs((r['center']-seed['center'])@n) <= 4.5]
        peers.sort(key=lambda r: r['center']@u)
        runs = [[]]
        typical = np.median([r['length'] for r in peers])
        for r in peers:
            if runs[-1] and (r['center']-runs[-1][-1]['center'])@u > max(35, typical*5): runs.append([])
            runs[-1].append(r)
        for run in runs:
            if len(run) < 4: continue
            centres = np.array([r['center'] for r in run]); origin = centres.mean(axis=0)
            _, _, vt = np.linalg.svd(centres-origin, full_matrices=False)
            axis = vt[0]
            if axis@u < 0: axis = -axis
            if axis@u < math.cos(math.radians(10)): continue
            normal = np.array([-axis[1], axis[0]])
            # Snap only an already measured horizontal/vertical axis.
            if abs(axis[1]) < .015: axis = np.array([np.sign(axis[0]), 0.])
            elif abs(axis[0]) < .015: axis = np.array([0., np.sign(axis[1])])
            normal = np.array([-axis[1], axis[0]])
            intervals = []
            aligned = []
            for r in run:
                p = samples[r['id']]; t = (p-origin)@axis
                if np.abs((p-origin)@normal).max() > 2.8: break
                a, b = origin+axis*t.min(), origin+axis*t.max()
                projected = np.linspace(a, b, max(3, math.ceil(np.linalg.norm(b-a))))
                if _supported(projected, distance, 2.) < .8: break
                intervals.append((float(t.min()), float(t.max())))
                aligned.append((r['id'], a, b))
            else:
                gaps = np.array([b[0]-a[1] for a,b in zip(intervals, intervals[1:])])
                lengths = np.array([b-a for a,b in intervals]); gap = float(np.median(gaps))
                if gap < 2 or gaps.min() < gap*.35: continue
                # Missing dashes at a contour crossing are real gaps. Permit
                # whole absent periods, preserving all original endpoints.
                period=float(np.median(lengths))+gap
                multiples=(gaps-gap)/period
                if gaps.max()>gap+period*2.5 or (np.abs(multiples-np.rint(multiples))>.25).any(): continue
                if lengths.max()/max(1, lengths.min()) > 3: continue
                # Blank gap evidence distinguishes dashes from a broken solid
                # contour or neighboring hatch strokes.
                mids = np.array([origin+axis*((a[1]+b[0])/2) for a,b in zip(intervals, intervals[1:])])
                if _supported(mids, distance, 1.) > .35: continue
                for j,a,b in aligned:
                    changes[j] = Entity(type='line', start=tuple(a), end=tuple(b), layer=ir.entities[j].layer)
                    used.add(j)
                trains.append(dict(entities=[r['id'] for r in run], axis=axis.tolist(),
                                   max_displacement_px=float(np.abs((centres-origin)@normal).max())))
    colors = ir.meta.get('stroke_colors_rgb', [None]*len(ir.entities))
    if len(colors) != len(ir.entities): colors = [None]*len(ir.entities)
    revised.entities = [changes.get(i,e) for i,e in enumerate(revised.entities) if i not in removed]
    revised.meta['stroke_colors_rgb'] = [v for i,v in enumerate(colors) if i not in removed]
    _preserve_associations(ir, revised, candidates)
    revised.meta['dash_axis_review'] = dict(trains=trains, aligned_segments=sum(i not in removed for i in changes),
        native_axis_remnants_removed=len(removed), policy='common measured axis; original ON/OFF intervals retained')
    return revised
