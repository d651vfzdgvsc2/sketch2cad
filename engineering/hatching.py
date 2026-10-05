"""Conservative, engineering-only conversion of measured section strokes to HATCH.

The shared IR stays unchanged. Only the selected, repaired drawing is filled;
owned polyline edges are replaced and uncertain boundaries retain their strokes.
"""
from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np

from engineering.primitives import sample_entity
from engineering.linework import _near_text
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


def _raster_pattern(ink,mask,angle,pen=None):
    """Measure the periodic thin stroke family, independent of fitted edges."""
    if pen is None:pen=cv2.distanceTransform(ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
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


def _section_boundaries(ir, ink, pen):
    """Existing CAD loops first, then source-closed thick section contours.

    A raster fallback cannot bridge an open outline: no closing/dilation is
    applied before testing enclosure. One-pixel expansion locates the contour
    nearer the stroke centre, after an enclosed region has been measured.
    """
    boundaries=[]
    for i,e in enumerate(ir.entities):
        if e.type!='polyline' or not e.closed or len(e.points)<3:continue
        contour=np.array(e.points,np.float32);area=abs(cv2.contourArea(contour))
        if area<120 or area>ir.width*ir.height*.7:continue
        try:
            from shapely.geometry import Polygon
            if not Polygon(e.points).is_valid:continue
        except ImportError:
            if not cv2.isContourConvex(contour):continue
        mask=np.zeros(ink.shape,np.uint8);cv2.fillPoly(mask,[np.rint(contour).astype(int)],1)
        boundaries.append((area,i,contour,mask,'cad_closed_contour'))
    # Distance cores remove thin hatch/axis strokes without inventing closure.
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,21),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((21,1),np.uint8))
    thick=((pen>=1.8)|(horizontal>0)|(vertical>0)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(1-thick,8)
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        if area<120 or area>ir.width*ir.height*.7 or min(w,h)<10:continue
        if x==0 or y==0 or x+w==ink.shape[1] or y+h==ink.shape[0]:continue
        region=(labels==label).astype(np.uint8)
        contours,_=cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=1:continue
        contour=cv2.approxPolyDP(contours[0],.8,True).reshape(-1,2).astype(np.float32)
        if len(contour)<3:continue
        mask=np.zeros(ink.shape,np.uint8);cv2.fillPoly(mask,[np.rint(contour).astype(int)],1)
        boundaries.append((abs(cv2.contourArea(contour)),None,contour,mask,'source_closed_thick_contour'))
    # Native contour candidates retain priority, to avoid breaking one known
    # region into overlapping raster pieces around holes and internal details.
    return sorted(boundaries,key=lambda b:(b[1] is None,b[0]))


def _raster_family(ink,mask,pen):
    """Measure source strokes even when tracing fragmented most diagonals."""
    thin=((ink>0)&(pen<=2.1)&(cv2.erode(mask,np.ones((5,5),np.uint8))>0)).astype(np.uint8)*255
    y,x=np.where(mask>0)
    if not len(x):return []
    x0,x1=int(x.min()),int(x.max())+1;y0,y1=int(y.min()),int(y.max())+1
    found=cv2.HoughLinesP(thin[y0:y1,x0:x1],1,np.pi/360,threshold=8,minLineLength=8,maxLineGap=4)
    rows=[]
    for row in [] if found is None else np.asarray(found).reshape(-1,4):
        a,b=row.reshape(2,2).astype(float)+[x0,y0];delta=b-a
        angle=math.degrees(math.atan2(delta[1],delta[0]))%180
        if min(angle,180-angle,abs(angle-90))<8:continue
        rows.append(dict(a=a,b=b,mid=(a+b)/2,angle=angle,length=float(np.linalg.norm(delta))))
    return rows


def plan_hatches(ir, image):
    if image is None: return []
    ink = ink_mask(image)
    pen=cv2.distanceTransform(ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Protect measured annotation EDGES, not every edge of a polyline that
    # happens to touch a dimension. Mixed contour/hatch paths are common.
    protected=set()
    for candidate in dimension_candidates(ir):
        for part in candidate.get('line_parts',[]):protected.add((part['entity'],part.get('edge')))
        for group in candidate.get('witness_parts',[]):
            for part in group:protected.add((part['entity'],part.get('edge')))
    segments = [];shorts=[]
    for i,e in enumerate(ir.entities):
        if e.layer.startswith(('centerline','text','dimensions')): continue
        if e.type == 'line': points = [e.start,e.end]
        elif e.type == 'polyline': points = e.points+[e.points[0]] if e.closed else e.points
        else: continue
        for edge,(a,b) in enumerate(zip(points,points[1:])):
            if (i,edge if e.type=='polyline' else None) in protected:continue
            a,b = np.array(a),np.array(b); delta=b-a; length=np.linalg.norm(delta)
            if length < .5 or _near_text(np.array([a,b]),ir): continue
            xy=np.rint(np.linspace(a,b,max(3,math.ceil(length)))).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,ink.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,ink.shape[0]-1)
            if (pen[xy[:,1],xy[:,0]]>1.8).mean()>.25:continue
            angle = math.degrees(math.atan2(delta[1],delta[0]))%180
            record=dict(entity=i, edge=edge if e.type=='polyline' else None,
                a=a,b=b,length=float(length),angle=angle,mid=(a+b)/2)
            (segments if length>=6 else shorts).append(record)
    boundaries = _section_boundaries(ir,ink,pen)
    gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if gray.mean()<127:gray=255-gray
    faint=(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,10)>0)&(gray<225)
    plans=[];claimed=set()
    filled=np.zeros(ink.shape,np.uint8)
    for area, boundary, contour, mask, boundary_source in boundaries:
        if (filled[mask>0]>0).mean()>.15:continue
        local=[s for s in segments if (s['entity'],s['edge']) not in claimed
               and all(_inside(p,contour,3. if boundary is None else 1.5) for p in (s['a'],s['b'],s['mid']))]
        hatch_ink=ink
        hatch_pen=pen
        if boundary is None:
            hatch_ink=ink.copy();hatch_ink[(mask>0)&faint]=255
            hatch_pen=cv2.distanceTransform(hatch_ink,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        evidence=_raster_family(hatch_ink,mask,hatch_pen) if boundary is None else local
        if len(evidence)<4: continue
        # Arbitrary hatch angles are allowed. Axis-aligned strokes are usually
        # contours/dimensions and require a separate pattern recognizer.
        bins={}
        for s in evidence:
            if min(s['angle'],180-s['angle'],abs(s['angle']-90))<8: continue
            bins.setdefault(round(s['angle']/4),[]).append(s)
        if not bins: continue
        seed=max(bins.values(),key=lambda g:sum(s['length'] for s in g))
        angle=float(np.median([s['angle'] for s in seed]))
        # A partly crossed or mixed pattern cannot be replaced by one uniform
        # hatch family. Preserve it until a multi-pattern recognizer exists.
        opposite=[s for s in evidence if 65<min(abs(s['angle']-angle),180-abs(s['angle']-angle))<115]
        if len(opposite)>=4 and sum(s['length'] for s in opposite)>.12*sum(s['length'] for s in seed):continue
        raster=_raster_pattern(hatch_ink,mask,angle,hatch_pen)
        if raster is None:continue
        angle,spacing,phase=raster
        radians=math.radians(angle);u=np.array([math.cos(radians),math.sin(radians)]);n=np.array([-u[1],u[0]])
        measured=[s for s in evidence if abs(s['angle']-angle)<4]
        members=[s for s in local if min(abs(s['angle']-angle),180-abs(s['angle']-angle))<5]
        if len(measured)<4: continue
        offsets=_clusters([s['mid']@n for s in measured])
        if len(offsets)<3: continue
        # Tiny pale sections can yield only the central repeated strokes. The
        # full-area source-grid and blank-pocket tests below still must pass.
        if (offsets[-1]-offsets[0])/max(1,np.ptp(contour@n))<(.25 if boundary is None and area<2500 else .55): continue
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
            near=cv2.dilate(hatch_ink,np.ones((3,3),np.uint8))>0
            if near[on].mean()>.2: continue
            if (hatch_ink[inner]>0).mean()>.12: continue
            holemask[hm>0]=1
            holes.append(dict(type=e.type,points=p.tolist(),
                **({'center':list(e.center),'radius':e.radius} if e.type=='circle' else {})))
        interior &= holemask==0
        near=cv2.dilate(hatch_ink,np.ones((5,5),np.uint8))>0
        if (interior&stripe).sum()<40 or near[interior&stripe].mean()<.72: continue
        # Unexplained blank pockets veto a fill rather than covering unknown
        # holes in fragmented geometry. Thin dash axes do not create pockets.
        white=(hatch_ink==0)&(mask>0)&(holemask==0)
        depth=cv2.distanceTransform(white.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        if depth[interior].max(initial=0)>spacing*1.15: continue
        # Enough of the source hatch family must be accounted for. A sparse
        # parallel feature array or threads are not an area fill.
        if sum(s['length'] for s in measured)*spacing < area*(.2 if boundary is None and area<2500 else .45): continue
        # Small collinear hatch fragments belong to the same measured pattern.
        # A tiny hook additionally needs contact with an accepted hatch edge;
        # shortness alone never authorizes deletion of detail or contours.
        endpoints=np.array([p for s in members for p in (s['a'],s['b'])]).reshape(-1,2)
        remnants=[]
        source_distance=cv2.distanceTransform((hatch_ink==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        for s in shorts:
            if (s['entity'],s['edge']) in claimed or s['entity']==boundary:continue
            if not all(_inside(p,contour,3.) for p in (s['a'],s['mid'],s['b'])):continue
            on_grid=np.max(np.abs((np.array([s['a'],s['b']])@n-phase+spacing/2)%spacing-spacing/2))<=1.8
            aligned=min(abs(s['angle']-angle),180-abs(s['angle']-angle))<5
            touches=bool(len(endpoints)) and min(np.linalg.norm(endpoints-s['a'],axis=1).min(),np.linalg.norm(endpoints-s['b'],axis=1).min())<=1.6
            sampled=np.rint(np.linspace(s['a'],s['b'],max(3,math.ceil(s['length']*2)))).astype(int)
            sampled[:,0]=np.clip(sampled[:,0],0,ink.shape[1]-1);sampled[:,1]=np.clip(sampled[:,1],0,ink.shape[0]-1)
            unsupported=(source_distance[sampled[:,1],sampled[:,0]]<=1.5).mean()<.8
            if on_grid and (aligned or (s['length']<=5 and touches and unsupported)):
                # Keep even tiny pieces tangent to a measured area boundary.
                if min(cv2.pointPolygonTest(contour,tuple(map(float,s['mid'])),True),
                       cv2.pointPolygonTest(contour,tuple(map(float,s['a'])),True))<1.5 and not aligned:continue
                remnants.append(s)
        members+=remnants
        curves=[]
        for i,e in enumerate(ir.entities):
            if e.type!='arc' or e.layer.startswith(('centerline','dimensions')):continue
            points=sample_entity(e,.5)
            if len(points)<3 or not all(_inside(p,contour,3.) for p in points):continue
            if _near_text(points,ir):continue
            chord=points[-1]-points[0]
            chord_angle=math.degrees(math.atan2(chord[1],chord[0]))%180
            if min(abs(chord_angle-angle),180-abs(chord_angle-angle))>12:continue
            if np.linalg.norm(np.diff(points,axis=0),axis=1).sum()>spacing*7:continue
            xy=np.rint(points).astype(int);xy[:,0]=np.clip(xy[:,0],0,ink.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,ink.shape[0]-1)
            support=float((source_distance[xy[:,1],xy[:,0]]<=1.5).mean())
            grid_distance=np.abs((points@n-phase+spacing/2)%spacing-spacing/2)
            boundary_distance=np.array([cv2.pointPolygonTest(contour,tuple(map(float,p)),True) for p in points])
            represented=float(((grid_distance<=1.8)|(boundary_distance<=3)).mean())
            # Junction fits may be source-supported because thick contour ink
            # hides the bend. Consume only curves already represented almost
            # entirely by this measured hatch and its closed area boundary.
            if support>=.8 and (represented<.9 or boundary_distance.max()<3):continue
            if np.min(np.abs((points[[0,-1]]@n-phase+spacing/2)%spacing-spacing/2))>2.5:continue
            curves.append(dict(entity=i,source_support=support,represented_by_hatch_and_boundary=represented))
        plans.append(dict(boundary_entity=boundary,boundary_source=boundary_source,outer=contour.tolist(),holes=holes,
            angle=angle,spacing=spacing,phase=phase,
            parts=[dict(entity=s['entity'],edge=s['edge']) for s in members],
            short_remnants=len(remnants),
            rejected_curves=curves,
            source_grid_support=float(near[interior&stripe].mean())))
        claimed.update((s['entity'],s['edge']) for s in members)
        filled[mask>0]=1
    return plans


def add_native_hatches(doc, ir, original):
    from tools.image_io import imread
    source=ir.meta.get('source')
    image=imread(source) if source and Path(source).is_file() else None
    plans=plan_hatches(ir,image);msp=doc.modelspace();converted=[];errors=[]
    if 'section_hatch' not in doc.layers: doc.layers.new('section_hatch',dxfattribs={'color':7,'lineweight':18})
    def flip(p): return (float(p[0]),float(ir.height-p[1]))
    accepted=[]
    for plan in plans:
        ids={p['entity'] for p in plan['parts']}|{p['entity'] for p in plan['rejected_curves']}
        if any(original[i] is None or not original[i].is_alive for i in ids):
            errors.append(dict(boundary=plan['boundary_entity'],reason='source_primitive_missing'))
            continue
        accepted.append(plan)
    # All edge splits and source-backed remnant removal precede HATCH creation.
    # The caller works on a disposable in-memory document until audit/save.
    owned={}
    for plan in accepted:
        for part in plan['parts']:owned.setdefault(part['entity'],set()).add(part['edge'])
        for part in plan['rejected_curves']:owned.setdefault(part['entity'],set()).add(None)
    for i,edges in owned.items():
        e=ir.entities[i];old=original[i]
        if e.type!='polyline':continue
        attrs={k:v for k,v in old.dxfattribs().items() if k in ('layer','color','true_color','lineweight','linetype','ltscale')}
        points=e.points+[e.points[0]] if e.closed else e.points
        for edge,(a,b) in enumerate(zip(points,points[1:])):
            if edge not in edges:msp.add_line(flip(a),flip(b),dxfattribs=attrs)
    for i in owned:msp.delete_entity(original[i])
    for plan in accepted:
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
            converted.append(dict(**plan,handle=hatch.dxf.handle))
        except Exception:
            raise # caller retains the previously saved, unfilled selected DXF
    # HATCH is created after all drawing operations, but displays underneath
    # contours, dimensions and text. Creation order and redraw order differ.
    ordered=sorted(msp,key=lambda e:e.dxftype()!='HATCH')
    msp.set_redraw_order((e.dxf.handle,format(i+1,'X')) for i,e in enumerate(ordered))
    return dict(native_hatches=len(converted),converted=converted,errors=errors,
        replaced_edges=sum(len(p['parts']) for p in converted),
        short_remnants_removed=sum(p['short_remnants'] for p in converted),
        unsupported_hatch_arcs_removed=sum(len(p['rejected_curves']) for p in converted),
        policy='final selected drawing only; source-closed contours; measured pattern; confirmed blank islands; hatch drawn behind annotations')


def finalize_hatches(ir, source, target):
    """Fill the actual selected DXF without regenerating annotations/geometry."""
    import ezdxf
    doc=ezdxf.readfile(str(source))
    if len(doc.modelspace().query('HATCH')):raise ValueError('Drawing already finalized')
    # Dimension creation may split a mixed polyline into surviving LINEs.
    # Read actual selected geometry; never regenerate the accepted dimensions
    # or guess which old IR handle now owns a fragment.
    from emit.ir import Entity
    geometry=[];original=[]
    def flip(p):return (float(p[0]),float(ir.height-p[1]))
    for e in doc.modelspace():
        kind=e.dxftype();layer=e.dxf.layer
        if layer.startswith(('text','dimensions')):continue
        if kind=='LINE':item=Entity(type='line',start=flip(e.dxf.start),end=flip(e.dxf.end),layer=layer)
        elif kind=='LWPOLYLINE':
            if e.has_arc:continue
            item=Entity(type='polyline',points=[flip(p) for p in e.get_points('xy')],closed=e.closed,layer=layer)
        elif kind=='CIRCLE':item=Entity(type='circle',center=flip(e.dxf.center),radius=e.dxf.radius,layer=layer)
        elif kind=='ARC':item=Entity(type='arc',center=flip(e.dxf.center),radius=e.dxf.radius,
            start_angle=-e.dxf.end_angle,end_angle=-e.dxf.start_angle,layer=layer)
        else:continue
        geometry.append(item);original.append(e)
    for e in ir.entities:
        if e.type=='text':geometry.append(e);original.append(None)
    selected_ir=ir.model_copy(deep=True);selected_ir.entities=geometry
    selected_ir.meta.pop('dimension_association',None)
    report=add_native_hatches(doc,selected_ir,original)
    report['entity_index_space']='actual selected DXF geometry'
    if doc.audit().has_errors:raise ValueError('Final hatch DXF failed audit')
    doc.saveas(str(target))
    ir.meta['native_hatches']=report
    return report
