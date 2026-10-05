"""Conservative, engineering-only conversion of measured section strokes to HATCH.

The shared IR stays unchanged. Plans are rebuilt on each export; only owned
polyline edges are replaced, and uncertain boundaries retain their strokes.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np

from engineering.primitives import sample_entity
from engineering.linework import _protected, _near_text
from engineering.dimensions import dimension_candidates
from engineering.trace import ink_mask


def _inside(p, contour, margin=0):
    return cv2.pointPolygonTest(contour, tuple(map(float,p)), True) >= -margin


def _clusters(values, tolerance=1.8):
    groups = []
    for x in sorted(values):
        if not groups or x-np.median(groups[-1]) > tolerance: groups.append([])
        groups[-1].append(x)
    return np.array([np.median(g) for g in groups])


def _raster_pattern(ink,mask,angle):
    """Measure the periodic thin stroke family, independent of fitted edges."""
    pen=cv2.distanceTransform(ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    inner=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    y,x=np.where(inner&(ink>0)&(pen<=1.5))
    if len(x)<100:return None
    xy=np.array([x,y]).T.astype(float)
    if len(xy)>2500:xy=xy[np.linspace(0,len(xy)-1,2500).astype(int)]
    best=(0.,None)
    def search(angles,spacings):
        nonlocal best
        for theta in angles:
            rad=math.radians(theta);offsets=xy@np.array([-math.sin(rad),math.cos(rad)])
            for spacing in spacings:
                z=np.exp(2j*np.pi*offsets/spacing).mean()
                if abs(z)>best[0]:best=(float(abs(z)),(float(theta),float(spacing),float(np.angle(z)*spacing/(2*np.pi)%spacing)))
    search(np.arange(angle-3,angle+3.01,.25),np.arange(3.5,30,.25))
    if best[1] is None:return None
    a,s,_=best[1]
    search(np.arange(a-.25,a+.251,.05),np.arange(max(3,s-.25),s+.251,.05))
    return best[1] if best[0]>=.28 else None


def plan_hatches(ir, image):
    if image is None: return []
    ink = ink_mask(image)
    pen=cv2.distanceTransform(ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    samples = {i:sample_entity(e, 1.) for i,e in enumerate(ir.entities) if e.type != 'text'}
    protected = _protected(ir, dimension_candidates(ir), samples)
    segments = []
    for i,e in enumerate(ir.entities):
        if i in protected or e.layer.startswith(('centerline','text','dimensions')): continue
        if e.type == 'line': points = [e.start,e.end]
        elif e.type == 'polyline': points = e.points+[e.points[0]] if e.closed else e.points
        else: continue
        for edge,(a,b) in enumerate(zip(points,points[1:])):
            a,b = np.array(a),np.array(b); delta=b-a; length=np.linalg.norm(delta)
            if length < 10 or _near_text(np.array([a,b]),ir): continue
            xy=np.rint(np.linspace(a,b,max(3,math.ceil(length)))).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,ink.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,ink.shape[0]-1)
            if (pen[xy[:,1],xy[:,0]]>1.8).mean()>.25:continue
            angle = math.degrees(math.atan2(delta[1],delta[0]))%180
            segments.append(dict(entity=i, edge=edge if e.type=='polyline' else None,
                a=a,b=b,length=float(length),angle=angle,mid=(a+b)/2))
    boundaries = []
    for i,e in enumerate(ir.entities):
        if e.type!='polyline' or not e.closed or len(e.points)<3: continue
        contour = np.array(e.points,np.float32)
        area=abs(cv2.contourArea(contour))
        if area<500 or area>ir.width*ir.height*.7: continue # drawing frame
        # Self intersections and nearly collapsed loops are not fill boundaries.
        try:
            from shapely.geometry import Polygon
            if not Polygon(e.points).is_valid: continue
        except ImportError:
            # Without a topology validator use only convex measured boundaries.
            if not cv2.isContourConvex(contour): continue
        mask=np.zeros(ink.shape,np.uint8);cv2.fillPoly(mask,[np.rint(contour).astype(int)],1)
        boundaries.append((area,i,contour,mask))
    plans=[];claimed=set()
    for area, boundary, contour, mask in sorted(boundaries):
        local=[s for s in segments if (s['entity'],s['edge']) not in claimed
               and all(_inside(p,contour,1.5) for p in (s['a'],s['b'],s['mid']))]
        if len(local)<6: continue
        # Arbitrary hatch angles are allowed. Axis-aligned strokes are usually
        # contours/dimensions and require a separate pattern recognizer.
        bins={}
        for s in local:
            if min(s['angle'],180-s['angle'],abs(s['angle']-90))<8: continue
            bins.setdefault(round(s['angle']/4),[]).append(s)
        if not bins: continue
        seed=max(bins.values(),key=lambda g:sum(s['length'] for s in g))
        angle=float(np.median([s['angle'] for s in seed]))
        raster=_raster_pattern(ink,mask,angle)
        if raster is None:continue
        angle,spacing,phase=raster
        radians=math.radians(angle);u=np.array([math.cos(radians),math.sin(radians)]);n=np.array([-u[1],u[0]])
        members=[s for s in local if abs(s['angle']-angle)<4]
        if len(members)<6: continue
        offsets=_clusters([s['mid']@n for s in members])
        if len(offsets)<6: continue
        if (offsets[-1]-offsets[0])/max(1,np.ptp(contour@n))<.65: continue
        grid=np.indices(ink.shape);positions=grid[1]*n[0]+grid[0]*n[1]
        stripe=np.abs((positions-phase+spacing/2)%spacing-spacing/2)<.65
        interior=cv2.erode(mask,np.ones((7,7),np.uint8))>0
        holes=[];holemask=np.zeros(ink.shape,np.uint8)
        # Only measured closed contours whose interior lacks this hatch family
        # become islands. A circle crossed by source hatching stays hatched.
        for j,e in enumerate(ir.entities):
            if j==boundary or e.type not in ('circle','polyline') or (e.type=='polyline' and not e.closed): continue
            p=sample_entity(e,1.)
            if not len(p) or not all(_inside(q,contour) for q in p): continue
            hm=np.zeros(ink.shape,np.uint8)
            cv2.fillPoly(hm,[np.rint(p).astype(int)],1)
            inner=cv2.erode(hm,np.ones((5,5),np.uint8))>0
            if inner.sum()<30 or inner.sum()>area*.5: continue
            on=inner&stripe
            if on.sum()<10: continue
            near=cv2.dilate(ink,np.ones((3,3),np.uint8))>0
            if near[on].mean()>.2: continue
            if (ink[inner]>0).mean()>.12: continue
            holemask[hm>0]=1
            holes.append(dict(type=e.type,points=p.tolist(),
                **({'center':list(e.center),'radius':e.radius} if e.type=='circle' else {})))
        interior &= holemask==0
        near=cv2.dilate(ink,np.ones((5,5),np.uint8))>0
        if (interior&stripe).sum()<80 or near[interior&stripe].mean()<.72: continue
        # Unexplained blank pockets veto a fill rather than covering unknown
        # holes in fragmented geometry. Thin dash axes do not create pockets.
        white=(ink==0)&(mask>0)&(holemask==0)
        depth=cv2.distanceTransform(white.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        if depth[interior].max(initial=0)>spacing*1.15: continue
        # Enough of the source hatch family must be accounted for. A sparse
        # parallel feature array or threads are not an area fill.
        if sum(s['length'] for s in members)*spacing < area*.45: continue
        plans.append(dict(boundary_entity=boundary,outer=contour.tolist(),holes=holes,
            angle=angle,spacing=spacing,phase=phase,
            parts=[dict(entity=s['entity'],edge=s['edge']) for s in members],
            source_grid_support=float(near[interior&stripe].mean())))
        claimed.update((s['entity'],s['edge']) for s in members)
    return plans


def add_native_hatches(doc, ir, original):
    from tools.image_io import imread
    source=ir.meta.get('source')
    image=imread(source) if source and Path(source).is_file() else None
    plans=plan_hatches(ir,image);msp=doc.modelspace();converted=[];errors=[]
    if 'section_hatch' not in doc.layers: doc.layers.new('section_hatch',dxfattribs={'color':7,'lineweight':18})
    def flip(p): return (float(p[0]),float(ir.height-p[1]))
    for plan in plans:
        ids={p['entity'] for p in plan['parts']}
        if any(not original[i].is_alive for i in ids): continue
        before={e.dxf.handle for e in msp}
        try:
            hatch=msp.add_hatch(color=7,dxfattribs={'layer':'section_hatch'})
            angle=-plan['angle'];rad=math.radians(angle)
            # Explicit pattern definition keeps measured spacing and phase.
            normal=(-math.sin(rad),math.cos(rad))
            base_image=np.array([-math.sin(math.radians(plan['angle'])),math.cos(math.radians(plan['angle']))])*plan['phase']
            hatch.set_pattern_fill('ENG_MEASURED',angle=0,scale=1,
                definition=[[angle,flip(base_image),
                    (normal[0]*plan['spacing'],normal[1]*plan['spacing']),[]]])
            hatch.paths.add_polyline_path([flip(p) for p in plan['outer']],is_closed=True,flags=1)
            for hole in plan['holes']:
                if hole['type']=='circle':
                    path=hatch.paths.add_edge_path(flags=0)
                    path.add_arc(flip(hole['center']),hole['radius'],0,360)
                else: hatch.paths.add_polyline_path([flip(p) for p in hole['points']],is_closed=True,flags=0)
            # Retain non-hatch edges of mixed open polylines, with original style.
            replacements=[]
            for i in ids:
                e=ir.entities[i];old=original[i]
                owned={p['edge'] for p in plan['parts'] if p['entity']==i}
                if e.type=='polyline':
                    attrs={k:v for k,v in old.dxfattribs().items() if k in ('layer','color','true_color','lineweight','linetype','ltscale')}
                    points=e.points+[e.points[0]] if e.closed else e.points
                    for edge,(a,b) in enumerate(zip(points,points[1:])):
                        if edge not in owned: replacements.append((a,b,attrs))
            for a,b,attrs in replacements: msp.add_line(flip(a),flip(b),dxfattribs=attrs)
            # All construction succeeds before removing any source primitive.
            for i in ids: msp.delete_entity(original[i])
            converted.append(dict(**plan,handle=hatch.dxf.handle))
        except Exception as exc:
            for e in list(msp):
                if e.dxf.handle not in before: msp.delete_entity(e)
            errors.append(dict(boundary=plan['boundary_entity'],reason=type(exc).__name__))
    return dict(native_hatches=len(converted),converted=converted,errors=errors,
        replaced_edges=sum(len(p['parts']) for p in converted),
        policy='closed source-supported boundaries; measured pattern; confirmed blank islands; uncertain regions retained')
