"""Convert evidence-linked annotations to native DXF DIMENSION entities.

The source labels are explicit text overrides in pixel-space drawings. They
are never advertised as measurements or used as proof of manufacturing scale.
Witness geometry remains intact; only the replaced dimension line/text is
consumed. Dimension text is native MTEXT inside its generated anonymous block.
"""
from __future__ import annotations

import math
import numpy as np

from engineering.calibration import parse_dimension
from engineering.coordinates import image_to_cad


def _segments(ir):
    result=[]
    for index,e in enumerate(ir.entities):
        if e.layer.startswith(('centerline','text')): continue
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


def dimension_candidates(ir):
    segments=_segments(ir)
    texts=[(i,e) for i,e in enumerate(ir.entities) if e.type=='text']
    records={a['text_index']:a for a in ir.meta.get('annotations',[]) if 'text_index' in a}
    plans=[]
    for text_index,(entity_index,text) in enumerate(texts):
        parsed=parse_dimension(text.content)
        if parsed['kind']!='linear' or not parsed['value']:continue
        record=records.get(text_index,{})
        if record.get('score',0)<.85 or record.get('review_status')=='numeric_conflict':continue
        center=np.array(text.pos); height=text.height or 10
        choices=[]
        local_segments=segments+_split_segments(segments,center,height,record)
        for seg_id,segment in enumerate(local_segments):
            a,b=segment['a'],segment['b']; length=float(np.linalg.norm(b-a))
            if length<max(20,height*.8):continue
            direction=(b-a)/length
            # Engineering drawings use both aligned and unidirectional text:
            # a vertical dimension can legitimately have horizontal numerals.
            projection=float((center-a)@direction)
            offset=abs(float(direction[0]*(center-a)[1]-direction[1]*(center-a)[0]))
            min_offset=0 if segment.get('split') else height*.3
            if not .25*length<=projection<=.75*length or not min_offset<=offset<=height*5:continue
            witnesses=[]
            for point in (a,b):
                found=[]
                for witness_id,other in enumerate(segments):
                    if witness_id==seg_id or any(other['entity']==p['entity'] and other['edge']==p['edge'] for p in segment.get('parts',[])):continue
                    c,d=other['a'],other['b']; size=float(np.linalg.norm(d-c))
                    if size<max(6,height*.5):continue
                    vector=(d-c)/size
                    if abs(float(vector@direction))>.12:continue
                    along=float((point-c)@vector)
                    across=abs(float(vector[0]*(point-c)[1]-vector[1]*(point-c)[0]))
                    if -2.5<=along<=size+2.5 and across<=2.5:
                        found.append(witness_id)
                witnesses.append(found)
            if not witnesses[0] or not witnesses[1] or set(witnesses[0])&set(witnesses[1]):continue
            choices.append((offset,seg_id,witnesses))
        if not choices:continue
        choices.sort(key=lambda x:x[0])
        ambiguous=len(choices)>1 and choices[1][0]-choices[0][0]<height*.5
        for rank,(offset,seg_id,witnesses) in enumerate(choices[:4]):
            segment=local_segments[seg_id]
            plans.append(dict(id=f'd{len(plans)}',kind='linear',text_entity=entity_index,text_index=text_index,
                          line_entity=segment['entity'],line_edge=segment['edge'],
                          line_parts=segment.get('parts',[dict(entity=segment['entity'],edge=segment['edge'])]),
                          p1=segment['a'].tolist(),p2=segment['b'].tolist(),
                          text=text.content,text_position=list(text.pos),text_height=height,
                          text_rotation=text.rotation,text_width=record.get('width_px'),
                          witnesses=[[segments[j]['entity'] for j in group] for group in witnesses],
                          evidence='two_perpendicular_witnesses_and_aligned_label',
                          label_distance_px=offset,deterministic_choice=rank==0 and not ambiguous,
                          source_label_override=True,associative_to_geometry=False))
    return plans


def plan_dimensions(ir):
    candidates=dimension_candidates(ir)
    # Store IDs, never model-generated plans/coordinates. Rebuild evidence at
    # export so stale, unknown or conflicting associations cannot mutate DXF.
    decisions=ir.meta.get('dimension_association', {}).get('decisions', {})
    plans=[];used=set();texts=set()
    for plan in candidates:
        decision=decisions.get(str(plan['text_index']))
        if decision is not None:
            if decision != plan['id']:continue
            plan['association']='validated_model_selection'
        elif not plan['deterministic_choice']:continue
        else:plan['association']='deterministic_witness_evidence'
        keys={(p['entity'],p['edge']) for p in plan['line_parts']}
        if keys&used or plan['text_index'] in texts:continue
        used.update(keys);texts.add(plan['text_index']);plans.append(plan)
    return plans


def add_native_dimensions(doc, ir):
    """Transactional conversion per dimension; failed plans leave originals."""
    msp=doc.modelspace(); original=list(msp)
    if 'dimensions' not in doc.layers:doc.layers.new('dimensions',dxfattribs={'color':7,'lineweight':18})
    if 'ENG_DIMENSION' not in doc.appids:doc.appids.new('ENG_DIMENSION')
    plans=plan_dimensions(ir)
    def flip(p):return image_to_cad(p,ir.height)
    converted=[]; errors=[]; removed_edges={}; removed_lines=set(); removed_texts=set()
    for plan in plans:
        a,b=flip(plan['p1']),flip(plan['p2'])
        # normalize extension direction so standard dimension renderer has a
        # stable local coordinate system for left/right and vertical labels.
        angle=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))%180
        direction=np.array((math.cos(math.radians(angle)),math.sin(math.radians(angle))))
        if float((np.array(b)-a)@direction)<0:a,b=b,a
        before_handles={e.dxf.handle for e in msp}
        try:
            override={'dimtxt':plan['text_height'],'dimtxsty':'ENG_LATIN',
                      'dimse1':1,'dimse2':1,'dimblk':'NONE','dimasz':0,
                      'dimgap':max(1,plan['text_height']*.1),'dimtad':0,'dimtmove':2,
                      'dimclrd':7,'dimclrt':7,'dimlwd':18,'dimlwe':18}
            dim=msp.add_linear_dim(base=a,p1=a,p2=b,angle=angle,text=plan['text'],
                                   text_rotation=plan['text_rotation'],
                                   override=override,dxfattribs={'layer':'dimensions'})
            dim.set_location(flip(plan['text_position']),leader=False,relative=False)
            dim.render()
            dim.dimension.set_xdata('ENG_DIMENSION',[(1000,'source_image_label_override'),
                                                   (1000,'pixel_coordinates_not_verified_measurements')])
            # Match measured width rather than stretching the rest of the drawing.
            block=doc.blocks.get(dim.dimension.dxf.geometry)
            from ezdxf.fonts import fonts
            font=fonts.make_font('times.ttf',plan['text_height'])
            nominal=font.text_width(plan['text'])
            factor=max(.2,min(5.,plan['text_width']/nominal)) if nominal and plan.get('text_width') else 1.
            for item in block.query('MTEXT'):
                item.text='\\W'+format(factor,'.6f')+';'+plan['text']
            plan['handle']=dim.dimension.dxf.handle
            converted.append(plan)
            removed_texts.add(plan['text_entity'])
            for part in plan['line_parts']:
                if part['edge'] is None:removed_lines.add(part['entity'])
                else:removed_edges.setdefault(part['entity'],set()).add(part['edge'])
        except Exception as exc:
            for entity in list(msp):
                if entity.dxf.handle not in before_handles:msp.delete_entity(entity)
            errors.append(dict(text=plan['text'],reason=type(exc).__name__))
    for index in removed_texts|removed_lines:msp.delete_entity(original[index])
    for index,edges in removed_edges.items():
        source=ir.entities[index]
        points=source.points+[source.points[0]] if source.closed else source.points
        for edge,(a,b) in enumerate(zip(points,points[1:])):
            if edge not in edges:msp.add_line(flip(a),flip(b),dxfattribs={'layer':source.layer})
        msp.delete_entity(original[index])
    return dict(converted=converted,errors=errors,native_dimensions=len(converted),
                policy='source labels retained as explicit overrides; unlinked annotations remain native TEXT')
