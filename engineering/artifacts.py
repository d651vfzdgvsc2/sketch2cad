"""Conservative review of duplicate and unsupported engineering entities."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from engineering.primitives import sample_entity
from engineering.trace import ink_mask


def review_artifacts(ir,image):
    revised=ir.model_copy(deep=True)
    ink=ink_mask(image)
    distance=cv2.distanceTransform(255-ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    removed={};seen={};annotations=ir.meta.get('annotations',[])
    colors=ir.meta.get('stroke_colors_rgb',[None]*len(ir.entities))
    if len(colors)!=len(ir.entities):colors=[None]*len(ir.entities)
    samples={i:sample_entity(e,spacing=.8) for i,e in enumerate(ir.entities) if e.type!='text'}
    endpoints=[];owners=[]
    for i,e in enumerate(ir.entities):
        if e.type=='line':ends=(e.start,e.end)
        elif e.type=='polyline' and not e.closed:ends=(e.points[0],e.points[-1])
        else:ends=[]
        endpoints.extend(ends);owners.extend([i]*len(ends))
    tree=cKDTree(endpoints) if endpoints else None
    for i,e in enumerate(ir.entities):
        if e.type=='text' or e.layer.startswith('centerline'):continue
        # Exact duplicate semantics/style only; no approximate line erasure.
        record=e.model_dump(exclude_none=True)
        if e.type=='line':
            record['start'],record['end']=sorted((record['start'],record['end']))
        import json
        key=json.dumps([record,colors[i]],sort_keys=True)
        if key in seen:
            removed[i]=dict(reason='exact_duplicate',retained_entity=seen[key]);continue
        seen[key]=i
        points=samples.get(i,np.empty((0,2)))
        if len(points)<2 or e.type not in ('line','polyline'):continue
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length>24:continue # long features are never deleted as speckles
        xy=np.rint(points).astype(int)
        if (xy<0).any() or (xy[:,0]>=ink.shape[1]).any() or (xy[:,1]>=ink.shape[0]).any():continue
        unsupported=float((distance[xy[:,1],xy[:,0]]>3).mean())
        if unsupported<.98:continue
        if any(((points[:,0]>=a['box'][0]-4)&(points[:,0]<=a['box'][2]+4)&
                (points[:,1]>=a['box'][1]-4)&(points[:,1]<=a['box'][3]+4)).any() for a in annotations):continue
        attached=any(owners[j]!=i and owners[j] not in removed
                     for p in (points[0],points[-1]) for j in (tree.query_ball_point(p,3) if tree else []))
        if attached:continue
        removed[i]=dict(reason='short_isolated_entity_without_source_ink',length_px=length,
                        unsupported_fraction=unsupported)
    if removed:
        revised.entities=[e for i,e in enumerate(revised.entities) if i not in removed]
        revised.meta['stroke_colors_rgb']=[c for i,c in enumerate(colors) if i not in removed]
    revised.meta['artifact_review']=dict(removed=[dict(entity_id=i,**r) for i,r in removed.items()],
        policy='exact duplicates or short isolated strokes with >=98% samples >3px from source ink; annotations and connected details protected')
    return revised
