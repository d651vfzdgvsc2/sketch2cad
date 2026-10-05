"""Measured extension/arrow ownership for one selectable DIMENSION object.

Only annotation portions are consumed. A witness that continues through the
part is split at its first contour junction; its contour remainder survives.
Measured graphics live in the dimension's anonymous block, never a loose GROUP.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np


def part_key(part):return (part['entity'],part.get('edge'))


def claims_overlap(a,b):
    if part_key(a)!=part_key(b):return False
    lo,hi=a.get('interval',(0.,1.));start,end=b.get('interval',(0.,1.))
    return min(hi,end)-max(lo,start)>1e-6


def ownership_parts(plan):
    return plan['line_parts']+[dict(entity=i,edge='primitive') for i in plan.get('annotation_entities',[])]


def segment_distance(point,a,b):
    v=b-a;length=float(v@v)
    return float(np.linalg.norm(point-(a+np.clip((point-a)@v/length,0,1)*v))) if length else float(np.linalg.norm(point-a))


def witness_portion(segment,endpoint,normal,segments,excluded):
    a,b=segment['a'],segment['b'];v=b-a;length=float(np.linalg.norm(v))
    if not length:return None
    unit=v/length;t=float(np.clip((endpoint-a)@unit,0,length))
    # Move towards the part (opposite the text side). Stop at the first
    # non-collinear junction, instead of claiming the whole part edge.
    left,right=t,length-t
    if abs(left-right)>2:direction=unit if right>left else -unit
    else:direction=unit if float(unit@normal)<0 else -unit
    maximum=length-t if float(direction@unit)>0 else t
    if maximum<2:return None
    anchor=endpoint+direction*maximum
    junctions=[]
    for other in segments:
        if part_key(other) in excluded:continue
        c,d=other['a'],other['b'];ov=d-c;size=float(np.linalg.norm(ov))
        if size<5 or abs(float(ov/size@direction))>.96:continue
        # Endpoints/interior intersection on this witness.
        determinant=float(direction[0]*ov[1]-direction[1]*ov[0])
        if abs(determinant)<1e-8:continue
        delta=c-endpoint
        along=float((delta[0]*ov[1]-delta[1]*ov[0])/determinant)
        fraction=float((delta[0]*direction[1]-delta[1]*direction[0])/determinant)
        if 3<along<=maximum+1 and -.025<=fraction<=1.025:junctions.append(along)
    if junctions:anchor=endpoint+direction*min(junctions)
    # Include the measured short overshoot beyond the dimension line.
    outer=a if float((a-endpoint)@direction)<float((b-endpoint)@direction) else b
    start=float(np.clip((anchor-a)@unit,0,length))/length
    end=float(np.clip((outer-a)@unit,0,length))/length
    lo,hi=sorted((start,end))
    return dict(entity=segment['entity'],edge=segment['edge'],interval=[lo,hi],
                a=(a+v*lo).tolist(),b=(a+v*hi).tolist(),anchor=anchor.tolist(),
                role='extension',preserves_shared_remainder=lo>1e-6 or hi<1-1e-6)


def collect_graphics(plan,segments):
    a,b=np.array(plan['p1']),np.array(plan['p2']);length=float(np.linalg.norm(b-a))
    if not length:return []
    direction=(b-a)/length;normal=np.array([-direction[1],direction[0]])
    side=float((np.array(plan['text_position'])-(a+b)/2)@normal)
    if side<0:normal=-normal
    # Centered text has no side evidence: use nearest witness ends while
    # retaining exact source geometry in the dimension block.
    excluded={part_key(p) for p in plan['line_parts']}
    graphics=[];chosen=set()
    for side_index,(endpoint,group) in enumerate(zip((a,b),plan.get('witness_parts',[[],[]]))):
        matches=[s for s in segments if part_key(s) in {part_key(p) for p in group}]
        matches.sort(key=lambda s:segment_distance(endpoint,s['a'],s['b']))
        if not matches:continue
        witness=matches[0];key=part_key(witness)
        if key in chosen:continue
        chosen.add(key)
        portion=witness_portion(witness,endpoint,normal,segments,excluded|{key})
        if portion:
            portion['endpoint_index']=side_index
            # Measured overshoot beyond the tip controls native DIMEXE.
            inner=np.array(portion['anchor'])-endpoint
            if np.linalg.norm(inner):
                unit=inner/np.linalg.norm(inner)
                portion['overshoot']=max(0.,-min(float((np.array(portion[k])-endpoint)@unit) for k in ('a','b')))
            graphics.append(portion)
    # Attached residual arrow/tick strokes: only consume isolated short paths
    # at a dimension endpoint, pointing into the measured span.
    limit=min(18,max(4,plan['text_height']*.65))
    for endpoint,inward in ((a,direction),(b,-direction)):
        for s in segments:
            key=part_key(s)
            if key in excluded or key in chosen:continue
            c,d=s['a'],s['b'];size=float(np.linalg.norm(d-c))
            if not .8<=size<=limit:continue
            if min(np.linalg.norm(c-endpoint),np.linalg.norm(d-endpoint))>2.8:continue
            other=d if np.linalg.norm(c-endpoint)<np.linalg.norm(d-endpoint) else c
            delta=other-endpoint;along=float(delta@inward)
            across=abs(float(inward[0]*delta[1]-inward[1]*delta[0]))
            if not 0<=along<=limit or not .5<=across<=limit*.7:continue
            # A shared corner or branch belongs to the part; do not steal it.
            attached=any(part_key(o) not in excluded|{key} and part_key(o) not in chosen and
                         min(np.linalg.norm(other-o['a']),np.linalg.norm(other-o['b']))<1.2
                         for o in segments)
            if attached:continue
            graphics.append(dict(entity=s['entity'],edge=s['edge'],interval=[0.,1.],
                                 a=c.tolist(),b=d.tolist(),role='arrow'))
            chosen.add(key)
    return graphics


def add_measured_graphics(block,graphics,flip):
    for g in graphics:
        block.add_line(flip(g['a']),flip(g['b']),dxfattribs={'layer':'0','color':0})


def arrow_size(plan,image):
    """Require both filled source arrowheads; do not add arrows to plain lines."""
    if image is None:return 0.
    a,b=np.array(plan['p1']),np.array(plan['p2']);length=float(np.linalg.norm(b-a))
    if length<6:return 0.
    direction=(b-a)/length;normal=np.array([-direction[1],direction[0]])
    size=min(18.,max(3.,plan['text_height']*.55),length*.25)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Skeleton tips can be displaced by a pixel in a scanned arrowhead.
    # Tolerate that displacement only after the width-profile arrow gate.
    supported=cv2.dilate((gray<150).astype(np.uint8),np.ones((3,3),np.uint8))
    supports=[]
    for tip,inward in ((a,direction),(b,-direction)):
        widths=[]
        for along in (size*.25,size*.75,size*2):
            points=[tip+inward*along+normal*offset for offset in range(-8,9)]
            coords=np.rint(points).astype(int)
            if (coords<0).any() or (coords[:,0]>=gray.shape[1]).any() or (coords[:,1]>=gray.shape[0]).any():return 0.
            widths.append(int((gray[coords[:,1],coords[:,0]]<150).sum()))
        # Constant-width source strokes are not arrowheads. A real filled
        # head broadens behind the tip, then narrows back to the shaft.
        if widths[1]<3 or widths[1]-widths[2]<2:return 0.
        samples=[]
        for along in np.linspace(size*.4,size*.9,7):
            for across in (-1.,1.):
                samples.append(tip+inward*along+normal*across*max(1.2,along*.18))
        xy=np.rint(samples).astype(int)
        if (xy<0).any() or (xy[:,0]>=gray.shape[1]).any() or (xy[:,1]>=gray.shape[0]).any():return 0.
        supports.append(float(supported[xy[:,1],xy[:,0]].mean()))
    return size if min(supports)>=.65 else 0.


def remaining_intervals(consumed):
    merged=[]
    for lo,hi in sorted(consumed,key=lambda interval:(interval[0],interval[1])):
        if not 0<=lo<=hi<=1:raise ValueError('Invalid consumed interval')
        if merged and lo<=merged[-1][1]+1e-6:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    cursor=0.;result=[]
    for lo,hi in merged:
        if lo-cursor>1e-6:result.append((cursor,lo))
        cursor=max(cursor,hi)
    if 1-cursor>1e-6:result.append((cursor,1.))
    return result
