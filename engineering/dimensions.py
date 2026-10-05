"""Convert evidence-linked annotations to native DXF DIMENSION entities.

The source labels are explicit text overrides in pixel-space drawings. They
are never advertised as measurements or used as proof of manufacturing scale.
Measured annotation graphics belong to the DIMENSION's anonymous block.
Part edges shared with witnesses remain outside the annotation object.
"""
from __future__ import annotations

import math
import re
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from engineering.calibration import parse_dimension
from engineering.coordinates import image_to_cad
from engineering.dimension_graphics import (collect_graphics,add_measured_graphics,
                                           remaining_intervals,part_key,arrow_size,claims_overlap,ownership_parts)


def _segments(ir, include_centerlines=False):
    result=[]
    for index,e in enumerate(ir.entities):
        if e.layer.startswith('text') or (e.layer.startswith('centerline') and not include_centerlines): continue
        if e.type=='line':
            result.append(dict(entity=index,edge=None,a=np.array(e.start),b=np.array(e.end)))
        elif e.type=='polyline':
            pts=e.points+[e.points[0]] if e.closed else e.points
            result.extend(dict(entity=index,edge=j,a=np.array(a),b=np.array(b))
                          for j,(a,b) in enumerate(zip(pts,pts[1:])))
    return result


def _split_segments(segments, center, height, record):
    """Reconnect a dimension interrupted by its label, not arbitrary gaps."""
    nearby=[];result=[]
    for i,s in enumerate(segments):
        if float(np.linalg.norm(s['b']-s['a']))<max(8,height*.5):continue
        if min(float(np.linalg.norm(p-center)) for p in (s['a'],s['b']))<max(record.get('width_px',height)*1.5,height*3):
            nearby.append((i,s))
    for n,(i,a) in enumerate(nearby):
        for j,b in nearby[n+1:]:
            if a['entity']==b['entity'] and a['edge']==b['edge']:continue
            # Choose nearest endpoints; outer endpoints must be on opposite
            # sides of the text, with two measured witness lines checked later.
            ends_a=(a['a'],a['b']);ends_b=(b['a'],b['b'])
            u,v=min(((u,v) for u in (0,1) for v in (0,1)),key=lambda uv:np.linalg.norm(ends_a[uv[0]]-ends_b[uv[1]]))
            p,q=ends_a[u],ends_b[v];start,end=ends_a[1-u],ends_b[1-v]
            span=float(np.linalg.norm(end-start));gap=float(np.linalg.norm(q-p))
            if span<30 or gap<3 or gap>max(record.get('width_px',height)+height*2,height*3):continue
            direction=(end-start)/span
            cross=lambda point:abs(float(direction[0]*(point-start)[1]-direction[1]*(point-start)[0]))
            if max(cross(p),cross(q))>1.2 or cross(center)>height*.75:continue
            cp=float((center-start)@direction);pp=float((p-start)@direction);qp=float((q-start)@direction)
            if not pp-height*.25<=cp<=qp+height*.25:continue
            result.append(dict(entity=a['entity'],edge=a['edge'],a=start,b=end,
                               parts=[dict(entity=s['entity'],edge=s['edge']) for s in (a,b)],split=True))
    return result


def _raster_segments(ir,center,height,record,segments,image=None):
    """Recover a short dimension crossbar lost while skeletonizing arrowheads.

    Only a measured source stroke outside the text box is eligible; both
    perpendicular witnesses still have to be present in the vector drawing.
    """
    source=ir.meta.get('source')
    if not source or not Path(source).is_file():return []
    if image is None:
        from tools.image_io import imread
        image=imread(source)
    h,w=image.shape[:2]
    radius=max(height*3,record.get('width_px',height)*1.5)
    xa,ya=max(0,int(center[0]-radius)),max(0,int(center[1]-radius))
    xb,yb=min(w,int(center[0]+radius)+1),min(h,int(center[1]+radius)+1)
    gray=cv2.cvtColor(image[ya:yb,xa:xb],cv2.COLOR_BGR2GRAY)
    ink=(gray<140).astype(np.uint8)*255
    # Detector padding sometimes includes the short crossbar itself. Keep
    # only strokes sufficiently offset from the locked text center, and
    # require two vector witnesses below; never classify an isolated glyph.
    result=[]
    for axis in (0,1):
        kernel=np.ones((1,7) if axis==0 else (7,1),np.uint8)
        strokes=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
        _,_,stats,_=cv2.connectedComponentsWithStats(strokes,8)
        for x,y,cw,ch,area in stats[1:]:
            if (ch>max(8,height*.3) if axis==0 else cw>max(8,height*.3)):continue
            if axis==0:a,b=np.array([xa+x,ya+y+(ch-1)/2]),np.array([xa+x+cw-1,ya+y+(ch-1)/2])
            else:a,b=np.array([xa+x+(cw-1)/2,ya+y]),np.array([xa+x+(cw-1)/2,ya+y+ch-1])
            if abs(float(center[1-axis]-a[1-axis]))<height*.3:continue
            if np.linalg.norm(b-a)<6:continue
            # Do not add a competing detection of an existing vector line.
            if any(np.linalg.norm(s['b']-s['a'])>np.linalg.norm(b-a)*.8 and
                   max(_point_segment(p,s['a'],s['b']) for p in (a,b))<3 for s in segments):continue
            # Trim arrow widths to the two nearest measured witness crossings.
            v=(b-a)/np.linalg.norm(b-a);crossings=[]
            for s in segments:
                d=s['b']-s['a'];size=float(np.linalg.norm(d))
                if size<4 or abs(float(d/size@v))>.15:continue
                normal=np.array([-v[1],v[0]]);distance=float((a-s['a'])@normal)
                fraction=distance/float(d@normal)
                if -.04<=fraction<=1.04:
                    p=s['a']+d*np.clip(fraction,0,1);along=float((p-a)@v)
                    if -4<=along<=float(np.linalg.norm(b-a))+4:crossings.append((along,p))
            crossings.sort(key=lambda c:c[0])
            if len(crossings)<2:continue
            a,b=crossings[0][1],crossings[-1][1]
            result.append(dict(entity=-1,edge=None,a=a,b=b,parts=[],raster_recovered=True))
    return result


def _point_segment(p,a,b):
    v=b-a;size=float(v@v)
    return float(np.linalg.norm(p-(a+np.clip((p-a)@v/size,0,1)*v))) if size else float(np.linalg.norm(p-a))


def _partition_segments(segments,witnesses,height,center=None):
    """Measured intermediate witnesses split adjacent dimensions on one stroke."""
    result=[]
    for s in segments:
        a,b=s['a'],s['b'];v=b-a;length=float(np.linalg.norm(v))
        if length<height*2:continue
        if center is not None and _point_segment(center,a,b)>height*4:continue
        direction=v/length;cuts=[]
        for other in witnesses:
            if part_key(s)==part_key(other):continue
            c,d=other['a'],other['b'];u=d-c;size=float(np.linalg.norm(u))
            if size<max(4,height*.3) or abs(float(u/size@direction))>.12:continue
            cross=float(v[0]*u[1]-v[1]*u[0])
            if abs(cross)<1e-8:continue
            delta=c-a
            t=float((delta[0]*u[1]-delta[1]*u[0])/cross)
            f=float((delta[0]*v[1]-delta[1]*v[0])/cross)
            if height*.4/length<t<1-height*.4/length and -2/size<=f<=1+2/size:
                if not any(abs(t-q)*length<3 for q in cuts):cuts.append(t)
        limits=[0.,*sorted(cuts),1.]
        if len(limits)<=2:continue
        for lo,hi in zip(limits,limits[1:]):
            if (hi-lo)*length<max(6,height*.3):continue
            result.append(dict(entity=s['entity'],edge=s['edge'],a=a+v*lo,b=a+v*hi,
                parts=[dict(entity=s['entity'],edge=s['edge'],interval=[lo,hi])],subspan=True))
    return result


def dimension_candidates(ir):
    segments=_segments(ir)
    witness_segments=_segments(ir,include_centerlines=True)
    texts=[(i,e) for i,e in enumerate(ir.entities) if e.type=='text']
    records={a['text_index']:a for a in ir.meta.get('annotations',[]) if 'text_index' in a}
    plans=[]
    source=ir.meta.get('source');image=None
    if source and Path(source).is_file():
        from tools.image_io import imread
        image=imread(source)
    for text_index,(entity_index,text) in enumerate(texts):
        parsed=parse_dimension(text.content)
        # Keep the accepted OCR string exactly; middle-dot decimal display is
        # a source convention, not permission to rewrite its visible text.
        if parsed['kind']!='linear':
            parsed=parse_dimension(text.content.replace('·','.'))
        plain=parsed['kind']=='linear'
        if not plain:
            value=re.fullmatch(r'[QΩØøΦφ⌀M]\s*(\d+(?:\.\d+)?)',text.content)
            if not value:continue
            parsed={'kind':'linear','value':float(value[1])}
        if not parsed['value']:continue
        record=records.get(text_index,{})
        if record.get('score',0)<.85 or record.get('review_status')=='numeric_conflict':continue
        center=np.array(text.pos); height=text.height or 10
        choices=[]
        local_segments=segments+_split_segments(segments,center,height,record)+_raster_segments(ir,center,height,record,segments,image)+_partition_segments(segments,witness_segments,height,center)
        for seg_id,segment in enumerate(local_segments):
            a,b=segment['a'],segment['b']; length=float(np.linalg.norm(b-a))
            if length<max(6,height*.3):continue
            direction=(b-a)/length
            # Engineering drawings use both aligned and unidirectional text:
            # a vertical dimension can legitimately have horizontal numerals.
            projection=float((center-a)@direction)
            offset=abs(float(direction[0]*(center-a)[1]-direction[1]*(center-a)[0]))
            min_offset=0 if segment.get('split') else height*.3
            maximum_offset=height*(4 if abs(text.rotation)%180>45 else 2.5)
            if not .15*length<=projection<=.85*length or not min_offset<=offset<=maximum_offset:continue
            witnesses=[]
            snapped=[]
            for point in (a,b):
                found=[]
                positions=[]
                for witness_id,other in enumerate(witness_segments):
                    if (other['entity']==segment['entity'] and other['edge']==segment['edge']) or any(other['entity']==p['entity'] and other['edge']==p['edge'] for p in segment.get('parts',[])):continue
                    c,d=other['a'],other['b']; size=float(np.linalg.norm(d-c))
                    if size<max(4,height*.15):continue
                    vector=(d-c)/size
                    if abs(float(vector@direction))>.12:continue
                    along=float((point-c)@vector)
                    across=abs(float(vector[0]*(point-c)[1]-vector[1]*(point-c)[0]))
                    tolerance=max(2.5,min(14.,height*.55))
                    is_centerline=ir.entities[other['entity']].layer.startswith('centerline')
                    if is_centerline:tolerance=max(tolerance,height*1.5)
                    if -2.5<=along<=size+2.5 and across<=tolerance:
                        found.append(witness_id)
                        # Project a crossing back onto the existing dimension
                        # line, allowing the measured arrowhead-width trim.
                        cross=float(direction[0]*vector[1]-direction[1]*vector[0])
                        delta=c-point
                        shift=float((delta[0]*vector[1]-delta[1]*vector[0])/cross)
                        positions.append((abs(shift),point if is_centerline and abs(shift)>height*.55 else point+direction*shift))
                witnesses.append(found)
                snapped.append(min(positions,key=lambda p:p[0])[1] if positions else point)
            if not witnesses[0] or not witnesses[1] or set(witnesses[0])&set(witnesses[1]):continue
            choices.append((offset,seg_id,witnesses,snapped))
        if not choices:continue
        choices.sort(key=lambda x:x[0])
        ambiguous=len(choices)>1 and choices[1][0]-choices[0][0]<height*.5
        for rank,(offset,seg_id,witnesses,snapped) in enumerate(choices[:4]):
            segment=local_segments[seg_id]
            plans.append(dict(id=f'd{len(plans)}',kind='linear',text_entity=entity_index,text_index=text_index,
                          line_entity=segment['entity'],line_edge=segment['edge'],
                          line_parts=segment.get('parts',[dict(entity=segment['entity'],edge=segment['edge'])]),
                          p1=snapped[0].tolist(),p2=snapped[1].tolist(),
                          measured_line=[segment['a'].tolist(),segment['b'].tolist()],
                          text=text.content,text_position=list(text.pos),text_height=height,
                          text_rotation=text.rotation,text_width=record.get('width_px'),
                          witnesses=[[witness_segments[j]['entity'] for j in group] for group in witnesses],
                          witness_parts=[[dict(entity=witness_segments[j]['entity'],edge=witness_segments[j]['edge']) for j in group] for group in witnesses],
                          evidence='two_perpendicular_witnesses_and_aligned_label',
                          label_distance_px=offset,deterministic_choice=plain and rank==0 and not ambiguous and not segment.get('subspan'),
                          intermediate_witness_partition=bool(segment.get('subspan')),
                          requires_semantic_confirmation=not plain,
                          source_label_override=True,associative_to_geometry=False))
    from engineering.curved_dimensions import curved_candidates
    for plan in curved_candidates(ir,segments):
        plan['id']=f'd{len(plans)}';plans.append(plan)
    return plans


def plan_dimensions(ir):
    candidates=dimension_candidates(ir)
    # Store IDs, never model-generated plans/coordinates. Rebuild evidence at
    # export so stale, unknown or conflicting associations cannot mutate DXF.
    association=ir.meta.get('dimension_association', {})
    decisions=association.get('decisions', {})
    if association.get('candidate_fingerprint') and association['candidate_fingerprint']!=candidate_fingerprint(candidates):
        decisions={} # geometry changed: old numbered choices are invalid
    plans=[];used=[];texts=set()
    for plan in candidates:
        decision=decisions.get(str(plan['text_index']))
        if decision is not None:
            if decision != plan['id']:continue
            plan['association']='validated_model_selection'
        elif not plan['deterministic_choice']:continue
        else:plan['association']='deterministic_witness_evidence'
        claims=ownership_parts(plan)
        if any(claims_overlap(p,q) for p in claims for q in used) or plan['text_index'] in texts:continue
        used.extend(claims);texts.add(plan['text_index']);plans.append(plan)
    return plans


def candidate_fingerprint(candidates):
    return hashlib.sha256(json.dumps(candidates,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def add_native_dimensions(doc, ir):
    """Transactional conversion per dimension; failed plans leave originals."""
    msp=doc.modelspace(); original=list(msp)
    if 'dimensions' not in doc.layers:doc.layers.new('dimensions',dxfattribs={'color':7,'lineweight':18})
    if 'ENG_DIMENSION' not in doc.appids:doc.appids.new('ENG_DIMENSION')
    plans=plan_dimensions(ir)
    def flip(p):return image_to_cad(p,ir.height)
    converted=[]; errors=[];consumed={};removed_texts=set();removed_annotations=set()
    segments=_segments(ir)
    source=ir.meta.get('source');source_image=None
    if source and Path(source).is_file():
        from tools.image_io import imread
        source_image=imread(source)
    for plan in plans:
        a,b=flip(plan['p1']),flip(plan['p2'])
        # normalize extension direction so standard dimension renderer has a
        # stable local coordinate system for left/right and vertical labels.
        angle=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))%180
        direction=np.array((math.cos(math.radians(angle)),math.sin(math.radians(angle))))
        swapped=float((np.array(b)-a)@direction)<0
        if swapped:a,b=b,a
        before_handles={e.dxf.handle for e in msp}
        try:
            size=arrow_size(plan,source_image)
            graphics=collect_graphics(plan,segments) if plan['kind']=='linear' else []
            by_endpoint={g['endpoint_index']:g for g in graphics if g['role']=='extension'}
            # Native renderer receives measured extension origins, rather than
            # baseline points with suppressed extensions. This remains a true
            # regeneratable linear DIMENSION, including its extension lines.
            origin1,origin2=a,b
            for endpoint_index,basepoint in ((int(swapped),a),(int(not swapped),b)):
                g=by_endpoint.get(endpoint_index)
                if g:
                    anchor=np.array(flip(g['anchor']));basepoint=np.array(basepoint)
                    anchor=anchor-direction*float((anchor-basepoint)@direction)
                    if endpoint_index==int(swapped):origin1=tuple(anchor)
                    else:origin2=tuple(anchor)
            overshoot=float(np.median([g.get('overshoot',0.) for g in by_endpoint.values()])) if by_endpoint else 0.
            override={'dimtxt':plan['text_height'],'dimtxsty':'ENG_LATIN',
                      'dimse1':0 if int(swapped) in by_endpoint else 1,
                      'dimse2':0 if int(not swapped) in by_endpoint else 1,
                      'dimexo':0.,'dimexe':overshoot,
                      'dimblk':'' if size else 'NONE','dimasz':size,
                      'dimgap':max(1,plan['text_height']*.1),'dimtad':0,'dimtmove':2,
                      'dimclrd':7,'dimclrt':7,'dimlwd':18,'dimlwe':18}
            attributes={'layer':'dimensions'}
            if plan['kind']=='linear':
                dim=msp.add_linear_dim(base=a,p1=origin1,p2=origin2,angle=angle,text=plan['text'],
                    text_rotation=plan['text_rotation'],override=override,dxfattribs=attributes)
                dim.set_location(flip(plan['text_position']),leader=False,relative=False)
            elif plan['kind'] in ('radius','diameter'):
                create=msp.add_radius_dim if plan['kind']=='radius' else msp.add_diameter_dim
                dim=create(center=flip(plan['center']),mpoint=flip(plan['mpoint']),
                    radius=ir.entities[plan['circle_entity']].radius,
                    text=plan['text'],override=override,dxfattribs=attributes)
                dim.user_location_override(flip(plan['text_position']))
            else:
                rays=plan['angular_rays']
                # Image Y reversal reverses angular orientation; swap rays.
                dim=msp.add_angular_dim_2l(base=flip(plan['angle_base']),
                    line1=tuple(flip(p) for p in rays[1]),line2=tuple(flip(p) for p in rays[0]),
                    location=flip(plan['text_position']),text=plan['text'],text_rotation=plan['text_rotation'],
                    override=override,dxfattribs=attributes)
            dim.render()
            dim.dimension.set_xdata('ENG_DIMENSION',[(1000,'source_image_label_override'),
                                                   (1000,'pixel_coordinates_not_verified_measurements')])
            # Match measured width rather than stretching the rest of the drawing.
            block=doc.blocks.get(dim.dimension.dxf.geometry)
            add_measured_graphics(block,[g for g in graphics if g['role']=='arrow' and not size],flip)
            from ezdxf.fonts import fonts
            font=fonts.make_font('times.ttf',plan['text_height'])
            nominal=font.text_width(plan['text'])
            factor=max(.2,min(5.,plan['text_width']/nominal)) if nominal and plan.get('text_width') else 1.
            for item in block.query('MTEXT'):
                item.text='\\W'+format(factor,'.6f')+';'+plan['text']
            plan['handle']=dim.dimension.dxf.handle
            plan['native_arrow_size_px']=size
            plan['owned_graphics']=graphics
            plan['single_selectable_object']=True
            plan['geometry_block']=dim.dimension.dxf.geometry
            converted.append(plan)
            removed_texts.add(plan['text_entity'])
            removed_annotations.update(plan.get('annotation_entities',[]))
            for part in plan['line_parts']:
                consumed.setdefault(part_key(part),[]).append(tuple(part.get('interval',(0.,1.))))
            for part in graphics:consumed.setdefault(part_key(part),[]).append(part['interval'])
        except Exception as exc:
            for entity in list(msp):
                if entity.dxf.handle not in before_handles:msp.delete_entity(entity)
            errors.append(dict(text=plan['text'],reason=type(exc).__name__))
    for index in removed_texts:msp.delete_entity(original[index])
    for index in removed_annotations:msp.delete_entity(original[index])
    for index in sorted({key[0] for key in consumed}):
        source=ir.entities[index];old=original[index]
        attrs=old.dxfattribs()
        attrs={k:v for k,v in attrs.items() if k in ('layer','color','true_color','lineweight','linetype','ltscale')}
        if source.type=='line':parts=[(None,source.start,source.end)]
        else:
            points=source.points+[source.points[0]] if source.closed else source.points
            parts=[(edge,a,b) for edge,(a,b) in enumerate(zip(points,points[1:]))]
        for edge,a,b in parts:
            a,b=np.array(a),np.array(b)
            for lo,hi in remaining_intervals(consumed.get((index,edge),[])):
                if float(np.linalg.norm((b-a)*(hi-lo)))>.05:
                    msp.add_line(flip(a+(b-a)*lo),flip(a+(b-a)*hi),dxfattribs=attrs)
        msp.delete_entity(old)
    from engineering.leaders import add_native_leaders
    leaders=add_native_leaders(doc,ir,original,removed_texts,{k[0] for k in consumed})
    return dict(converted=converted,errors=errors,native_dimensions=len(converted),
                leaders=leaders,native_leaders=leaders['native_leaders'],
                owned_extension_segments=sum(g['role']=='extension' for p in converted for g in p['owned_graphics']),
                owned_arrow_segments=sum(g['role']=='arrow' for p in converted for g in p['owned_graphics']),
                policy='DIMENSION owns text, dimension line and measured annotation graphics; shared contour portions retained')
