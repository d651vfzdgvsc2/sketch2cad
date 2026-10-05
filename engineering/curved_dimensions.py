"""Measured radial/diametric/angular candidates, always requiring semantic review."""
import math
import re
import numpy as np
from engineering.calibration import parse_dimension
from engineering.dimension_graphics import segment_distance


def curved_candidates(ir,segments):
    texts=[(i,e) for i,e in enumerate(ir.entities) if e.type=='text']
    records={a['text_index']:a for a in ir.meta.get('annotations',[]) if 'text_index' in a}
    plans=[]
    for text_index,(entity_index,text) in enumerate(texts):
        parsed=parse_dimension(text.content)
        angle_match=re.fullmatch(r'(\d+(?:\.\d+)?)\s*°',text.content.strip())
        if parsed['kind'] not in ('radius','diameter') and not angle_match:continue
        record=records.get(text_index,{})
        if record.get('score',0)<.85 or record.get('review_status')=='numeric_conflict':continue
        height=text.height or 10;label=np.array(text.pos)
        base=dict(text_entity=entity_index,text_index=text_index,text=text.content,
            text_position=list(text.pos),text_height=height,text_rotation=text.rotation,
            text_width=record.get('width_px'),witnesses=[],witness_parts=[],
            deterministic_choice=False,requires_semantic_confirmation=True,
            source_label_override=True,associative_to_geometry=False,annotation_entities=[])
        if angle_match:
            value=float(angle_match[1])
            if not 0<value<180:continue
            for index,arc in enumerate(ir.entities):
                if arc.type!='arc' or arc.layer.startswith('centerline'):continue
                sweep=(arc.end_angle-arc.start_angle)%360
                if abs(sweep-value)>2:continue
                c=np.array(arc.center);r=arc.radius
                if abs(np.linalg.norm(label-c)-r)>height*1.5:continue
                points=[c+r*np.array([math.cos(math.radians(a)),math.sin(math.radians(a))]) for a in (arc.start_angle,arc.end_angle)]
                rays=[]
                for p in points:
                    candidates=[s for s in segments if segment_distance(c,s['a'],s['b'])<3 and
                        segment_distance(p,s['a'],s['b'])<3 and np.linalg.norm(s['b']-s['a'])>=r*.8]
                    if not candidates:break
                    rays.append(candidates[0])
                if len(rays)!=2 or rays[0]['entity']==rays[1]['entity']:continue
                mid=math.radians(arc.start_angle+sweep/2)
                plans.append(dict(base,kind='angular',p1=points[0].tolist(),p2=points[1].tolist(),
                    measured_line=[points[0].tolist(),points[1].tolist()],center=c.tolist(),
                    angle_base=(c+r*np.array([math.cos(mid),math.sin(mid)])).tolist(),
                    line_parts=[],annotation_entities=[index],
                    angular_rays=[[s['a'].tolist(),s['b'].tolist()] for s in rays],
                    evidence='measured_arc_and_two_radial_witnesses',label_distance_px=abs(np.linalg.norm(label-c)-r)))
            continue
        for index,circle in enumerate(ir.entities):
            if circle.type not in ('circle','arc') or circle.layer.startswith('centerline'):continue
            c=np.array(circle.center);r=circle.radius
            if r<3:continue
            for s in segments:
                a,b=s['a'],s['b'];v=b-a;length=float(np.linalg.norm(v))
                if length<height:continue
                if segment_distance(label,a,b)>height*2:continue
                # A true diameter chord has both ends on the circle and passes
                # through the measured centre; ordinary nearby underlines fail.
                if parsed['kind']=='diameter':
                    if circle.type!='circle' or max(abs(np.linalg.norm(p-c)-r) for p in (a,b))>3:continue
                    if np.linalg.norm((a+b)/2-c)>3:continue
                    tip=a
                else:
                    tip,tail=min(((a,b),(b,a)),key=lambda pair:abs(np.linalg.norm(pair[0]-c)-r))
                    if abs(np.linalg.norm(tip-c)-r)>3 or np.linalg.norm(tail-c)<r+height*.5:continue
                    if abs(float(v[0]*(c-a)[1]-v[1]*(c-a)[0]))/length>2.5:continue
                    if circle.type=='arc':
                        angle=math.degrees(math.atan2(*(tip-c)[::-1]))%360
                        if (angle-circle.start_angle)%360>(circle.end_angle-circle.start_angle)%360+1:continue
                plans.append(dict(base,kind=parsed['kind'],center=c.tolist(),circle_entity=index,
                    p1=a.tolist(),p2=b.tolist(),measured_line=[a.tolist(),b.tolist()],mpoint=tip.tolist(),
                    line_parts=[dict(entity=s['entity'],edge=s['edge'])],
                    evidence='source_dimension_stroke_on_measured_circle',label_distance_px=segment_distance(label,a,b)))
    return plans
