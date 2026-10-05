"""Source-backed cleanup of engineering strokes, with annotation ownership intact.

Short and unlabelled is a review candidate, never sufficient evidence of noise.
No shared rockery/vectorization code is used or modified by this module.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from emit.ir import Entity
from engineering.dimensions import dimension_candidates, candidate_fingerprint
from engineering.leaders import leader_plans
from engineering.primitives import sample_entity
from engineering.trace import ink_mask


def _length(points):
    return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())


def _signature(candidate):
    """Measured geometry identity, independent of renumbered IR entity indices."""
    fields=('text_index','text','kind','p1','p2','measured_line','center','mpoint','angular_rays','text_position','intermediate_witness_partition')
    return json.dumps({k:candidate[k] for k in fields if k in candidate},sort_keys=True)


def _preserve_associations(old, revised, candidates):
    association=old.meta.get('dimension_association')
    if not association:return
    # Never make an already stale association valid by renumbering it.
    if association.get('candidate_fingerprint')!=candidate_fingerprint(candidates):
        revised.meta.pop('dimension_association',None);return
    new=dimension_candidates(revised)
    by_id={c['id']:c for c in candidates};new_ids={}
    for c in new:new_ids.setdefault(_signature(c),[]).append(c['id'])
    decisions={}
    for tid,choice in association.get('decisions',{}).items():
        if choice=='rejected':decisions[tid]=choice;continue
        c=by_id.get(choice)
        if c:
            matched=new_ids.get(_signature(c),[])
            if len(matched)==1:decisions[tid]=matched[0]
    revised.meta['dimension_association']={**association,'decisions':decisions,
        'candidate_fingerprint':candidate_fingerprint(new),
        'linework_remap':'same measured endpoints and text; no coordinate inference'}


def _protected(ir,candidates,samples):
    ids=set();tips=[]
    for c in candidates:
        ids.update(p['entity'] for p in c.get('line_parts',[]) if p['entity']>=0)
        ids.update(j for group in c.get('witnesses',[]) for j in group)
        ids.update(c.get('annotation_entities',[]))
        if 'circle_entity' in c:ids.add(c['circle_entity'])
        tips.extend((np.array(p),max(6.,c['text_height']*.6)) for p in (c['p1'],c['p2']))
    for p in leader_plans(ir):
        ids.update(p['line_entities']);tips.append((np.array(p['points'][0]),max(6.,p['height']*.6)))
    for i,points in samples.items():
        if any(np.linalg.norm(points-point,axis=1).min()<=radius for point,radius in tips):ids.add(i)
    return ids


def _near_text(points,ir):
    boxes=[a['box'] for key in ('annotations','ocr','ocr_before_review')
           for a in ir.meta.get(key,[]) if isinstance(a,dict) and 'box' in a]
    for e in ir.entities:
        if e.type=='text':
            width=max(e.height or 8,len(e.content)*(e.height or 8)*.6)
            boxes.append([e.pos[0]-width/2,e.pos[1]-(e.height or 8),e.pos[0]+width/2,e.pos[1]+(e.height or 8)])
    return any(((points[:,0]>=b[0]-5)&(points[:,0]<=b[2]+5)&
                (points[:,1]>=b[1]-5)&(points[:,1]<=b[3]+5)).any() for b in boxes)


def _straight_record(i,e,points):
    if e.type not in ('line','polyline') or e.closed or len(points)<2:return None
    a,b=points[0],points[-1];length=float(np.linalg.norm(b-a))
    if length<.1:return None
    v=(b-a)/length
    deviation=np.abs((points-a)@np.array([-v[1],v[0]])).max()
    if deviation>1.2:return None
    return dict(id=i,a=a,b=b,v=v,length=length,center=(a+b)/2,layer=e.layer)


def _repeated_detail(record,records):
    """Protect periodic dashes and parallel hatch/texture; do not close gaps."""
    peers=[];train=[]
    for other in records:
        if other['id']==record['id']:continue
        delta=other['center']-record['center'];distance=float(np.linalg.norm(delta))
        if distance>max(70,record['length']*12):continue
        if abs(float(record['v']@other['v']))<math.cos(math.radians(6)):continue
        along=abs(float(delta@record['v']));across=abs(float(delta@np.array([-record['v'][1],record['v'][0]])))
        if across<=2.2 and along>(record['length']+other['length'])*.35:
            train.append(other)
        if distance<=70 and .2<=other['length']/record['length']<=5:peers.append(other)
    if len(train)>=2:
        offsets=sorted([0.]+[float((p['center']-record['center'])@record['v']) for p in train])
        gaps=np.diff(offsets)
        if len(gaps)>=2 and min(gaps)>2 and max(gaps)/min(gaps)<=3.5:return True
    # An array of parallel thin marks is meaningful even with unequal lengths.
    return len(peers)>=3


def _supported(points,distance,tolerance=1.5):
    xy=np.rint(points).astype(int)
    if ((xy<0).any() or (xy[:,0]>=distance.shape[1]).any() or (xy[:,1]>=distance.shape[0]).any()):return 0.
    return float((distance[xy[:,1],xy[:,0]]<=tolerance).mean())


def review_linework(ir,image):
    revised=ir.model_copy(deep=True)
    candidates=dimension_candidates(ir)
    ink=ink_mask(image)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) if image.ndim==3 else None
    # JPEG darkens dash ends: the tracing threshold treats some dark red pixels
    # as black. This mask is for review only, never a new tracing threshold.
    colored=((hsv[:,:,1]>45)&(hsv[:,:,2]>30)&(np.ptp(image.astype(np.int16),axis=2)>22)) if hsv is not None else np.zeros(ink.shape,bool)
    black=ink.copy();black[colored]=0
    black_distance=cv2.distanceTransform(255-black,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    color_distance=cv2.distanceTransform((~(colored&(ink>0))).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    pen=cv2.distanceTransform(ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    pen_width=max(1.,float(np.percentile(pen[pen>0],70))*2) if ink.any() else 1.
    heights=[e.height for e in ir.entities if e.type=='text' and e.height]
    short_limit=min(16.,max(8.,float(np.median(heights))*.5 if heights else pen_width*4))
    speck_limit=min(3.,max(1.5,pen_width*.7))
    samples={i:sample_entity(e,.7) for i,e in enumerate(ir.entities) if e.type!='text'}
    protected=_protected(ir,candidates,samples)
    colors=list(ir.meta.get('stroke_colors_rgb',[None]*len(ir.entities)))
    if len(colors)!=len(ir.entities):colors=[None]*len(ir.entities)
    curves=[ps for i,ps in samples.items() if
            ir.entities[i].type in ('circle','ellipse') or ir.entities[i].closed or
            (ir.entities[i].type=='arc' and _supported(ps,black_distance,1.5)>.5)]
    curve_tree=cKDTree(np.concatenate(curves)) if curves else None
    all_points=np.concatenate(list(samples.values())) if samples else np.empty((0,2))
    owners=np.concatenate([np.full(len(ps),i) for i,ps in samples.items()]) if samples else np.array([],int)
    tree=cKDTree(all_points) if len(all_points) else None
    records=[r for i,ps in samples.items() if (r:=_straight_record(i,ir.entities[i],ps)) is not None]
    by_id={r['id']:r for r in records}
    removed={};uncertain=[];straightened=[]
    _,components,stats,_=cv2.connectedComponentsWithStats(ink,8)
    _,color_components,color_stats,_=cv2.connectedComponentsWithStats((colored&(ink>0)).astype(np.uint8),8)
    for i,points in samples.items():
        e=ir.entities[i]
        if e.type not in ('line','polyline','arc') or e.closed or len(points)<2:continue
        length=_length(points)
        if length>short_limit:continue
        if i in protected or e.layer.startswith(('text','dimensions','centerline_')) or _near_text(points,ir):continue
        if curve_tree is not None and curve_tree.query(points)[0].min()<=3:continue
        xy=np.rint(points).astype(int)
        if ((xy<0).any() or (xy[:,0]>=ink.shape[1]).any() or (xy[:,1]>=ink.shape[0]).any()):continue
        # Black skeleton artifacts at colored crossings/dash caps are not
        # drawing details when the original has only colored ink there.
        black_support=_supported(points,black_distance,.9)
        color_support=_supported(points,color_distance,1.2)
        color_ids=color_components[xy[:,1],xy[:,0]]
        color_counts=Counter(int(c) for c in color_ids if c)
        color_component=color_stats[color_counts.most_common(1)[0][0]] if color_counts else None
        source_is_straight_dash=(color_component is not None and
            max(color_component[2:4])>=3*max(1,min(color_component[2:4])))
        if not e.layer.startswith('centerline') and black_support<.15 and color_support>.9 and (e.type!='arc' or source_is_straight_dash):
            removed[i]=dict(reason='black_fragment_from_colored_stroke',length_px=length,
                           black_support=black_support,colored_support=color_support);continue
        if e.type=='arc' or e.layer.startswith('centerline'):continue # keep true curves/dash geometry
        record=by_id.get(i)
        if record and _repeated_detail(record,records):continue
        near=[{int(owners[j]) for j in tree.query_ball_point(p,1.6) if owners[j]!=i} if tree else set() for p in (points[0],points[-1])]
        # Preserve even a tiny connector when both ends belong to geometry.
        if all(near) and len(near[0]|near[1])>1:continue
        component_ids=components[xy[:,1],xy[:,0]]
        counts=Counter(int(c) for c in component_ids if c)
        component=stats[counts.most_common(1)[0][0]] if counts else None
        if component is not None and not any(near):
            x,y,w,h,area=map(int,component)
            # A bounded speck cannot be an entire long dash/edge or a glyph.
            if length<=speck_limit and max(w,h)<=max(4.,speck_limit+1) and area<=max(12.,pen_width*speck_limit*1.5):
                neighbors={int(owners[j]) for j in tree.query_ball_point(points.mean(axis=0),max(8,short_limit)) if owners[j]!=i} if tree else set()
                if len(neighbors)<2:
                    removed[i]=dict(reason='isolated_compact_speck',length_px=length,source_component=[x,y,w,h,area]);continue
        # A terminal twig wholly inside the thickness of a measured long line
        # does not express a separate edge. Tiny steps with two ends survive.
        if length<=speck_limit and any(near) and record:
            touching=near[0]|near[1]
            for j in sorted(touching):
                parent=by_id.get(j)
                if not parent or parent['length']<max(25,short_limit*3) or j in protected:continue
                offsets=np.abs((points-parent['a'])@np.array([-parent['v'][1],parent['v'][0]]))
                if offsets.max()<=min(1.8,pen_width*.6) and abs(float(record['v']@parent['v']))<math.cos(math.radians(20)):
                    removed[i]=dict(reason='terminal_twig_inside_parent_stroke',length_px=length,parent_entity=j,
                                   maximum_offset_px=float(offsets.max()));break
            if i in removed:continue
        uncertain.append(dict(entity_id=i,length_px=length,reason='short_unlabelled_without_sufficient_noise_evidence'))
    # Relax only the straight-path fit; retain measured endpoints, arbitrary
    # slopes, genuine corners, connections and all intentional gaps.
    for i,points in samples.items():
        e=ir.entities[i]
        if i in removed or i in protected or e.type!='polyline' or e.closed or e.layer.startswith(('centerline','text')):continue
        if _near_text(points,ir) or len(e.points)<3:continue
        a,b=points[0],points[-1];length=float(np.linalg.norm(b-a))
        if length<30:continue
        v=(b-a)/length;projection=(points-a)@v
        deviation=np.abs((points-a)@np.array([-v[1],v[0]]))
        if deviation.max()>1.1 or np.diff(projection).min(initial=0)<-.1 or _length(points)>length*1.008:continue
        candidate=Entity(type='line',layer=e.layer,start=tuple(a),end=tuple(b))
        support=_supported(sample_entity(candidate,.4),black_distance,1.25)
        if support<.99:continue
        revised.entities[i]=candidate
        straightened.append(dict(entity_id=i,before='polyline',after='line',maximum_deviation_px=float(deviation.max()),
                                 source_support=support,endpoints_unchanged=True))
    revised.entities=[e for i,e in enumerate(revised.entities) if i not in removed]
    revised.meta['stroke_colors_rgb']=[c for i,c in enumerate(colors) if i not in removed]
    if removed or straightened:_preserve_associations(ir,revised,candidates)
    revised.meta['linework_review']=dict(removed=[dict(entity_id=i,source_entity=ir.entities[i].model_dump(exclude_none=True),
        source_position=samples[i].mean(axis=0).tolist(),**r) for i,r in removed.items()],
        straightened=straightened,retained_uncertain=uncertain,short_candidate_limit_px=short_limit,
        measured_pen_width_px=pen_width,protected_annotation_entities=len(protected),
        policy='global source-backed noise filtering; short/unlabelled alone never deletes an entity; dash trains, hatching, curves, text and annotation candidates protected')
    return revised
