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


def plan_dimensions(ir):
    segments=_segments(ir)
    texts=[(i,e) for i,e in enumerate(ir.entities) if e.type=='text']
    records={a['text_index']:a for a in ir.meta.get('annotations',[]) if 'text_index' in a}
    plans=[]; used=set()
    for text_index,(entity_index,text) in enumerate(texts):
        parsed=parse_dimension(text.content)
        if parsed['kind']!='linear' or not parsed['value']:continue
        record=records.get(text_index,{})
        if record.get('score',0)<.85:continue
        center=np.array(text.pos); height=text.height or 10
        choices=[]
        for seg_id,segment in enumerate(segments):
            a,b=segment['a'],segment['b']; length=float(np.linalg.norm(b-a))
            if length<max(20,height*.8):continue
            direction=(b-a)/length
            # Engineering drawings use both aligned and unidirectional text:
            # a vertical dimension can legitimately have horizontal numerals.
            projection=float((center-a)@direction)
            offset=abs(float(direction[0]*(center-a)[1]-direction[1]*(center-a)[0]))
            if not .25*length<=projection<=.75*length or not height*.3<=offset<=height*5:continue
            witnesses=[]
            for point in (a,b):
                found=[]
                for witness_id,other in enumerate(segments):
                    if witness_id==seg_id:continue
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
        offset,seg_id,witnesses=choices[0]
        if len(choices)>1 and choices[1][0]-offset<height*.5:continue
        if seg_id in used:continue
        segment=segments[seg_id]; used.add(seg_id)
        plans.append(dict(kind='linear',text_entity=entity_index,text_index=text_index,
                          line_entity=segment['entity'],line_edge=segment['edge'],
                          p1=segment['a'].tolist(),p2=segment['b'].tolist(),
                          text=text.content,text_position=list(text.pos),text_height=height,
                          text_rotation=text.rotation,text_width=record.get('width_px'),
                          witnesses=[[segments[j]['entity'] for j in group] for group in witnesses],
                          evidence='two_perpendicular_witnesses_and_aligned_label',
                          source_label_override=True,associative_to_geometry=False))
    return plans


def add_native_dimensions(doc, ir):
    """Transactional conversion per dimension; failed plans leave originals."""
    msp=doc.modelspace(); original=list(msp)
    if 'dimensions' not in doc.layers:doc.layers.new('dimensions',dxfattribs={'color':7,'lineweight':18})
    if 'ENG_DIMENSION' not in doc.appids:doc.appids.new('ENG_DIMENSION')
    plans=plan_dimensions(ir)
    converted=[]; errors=[]; removed_edges={}; removed_lines=set(); removed_texts=set()
    for plan in plans:
        def flip(p):return (float(p[0]),float(ir.height-p[1]))
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
            if plan['line_edge'] is None:removed_lines.add(plan['line_entity'])
            else:removed_edges.setdefault(plan['line_entity'],set()).add(plan['line_edge'])
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
