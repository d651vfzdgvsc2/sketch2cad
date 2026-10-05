"""Native MLEADER for measured open leader paths and their accepted text."""
from __future__ import annotations

import re
import math
import numpy as np
from engineering.coordinates import image_to_cad


def measured_paths(ir,center,height):
    """Join only unique touching fragments; never bridge a dashed/open gap."""
    source=[]
    for i,e in enumerate(ir.entities):
        if e.type not in ('line','polyline') or e.closed or e.layer.startswith('centerline'):continue
        pts=np.array([e.start,e.end] if e.type=='line' else e.points,float)
        if len(pts)>8:continue
        source.append((i,pts))
    result=[(i,pts,[i]) for i,pts in source]
    for i,pts in source:
        if len(pts)!=2 or min(np.linalg.norm(p-center) for p in pts)>height*4:continue
        for reverse in (False,True):
            path=pts[::-1] if reverse else pts.copy();owned=[i]
            for _ in range(3):
                choices=[]
                for j,other in source:
                    if j in owned:continue
                    for flip in (False,True):
                        nextpts=other[::-1] if flip else other
                        if np.linalg.norm(path[-1]-nextpts[0])<=1.5 and np.linalg.norm(nextpts[-1]-center)<height*10:
                            choices.append((j,nextpts))
                if len(choices)!=1:break # shared branch belongs to part, not leader
                j,nextpts=choices[0];path=np.vstack([path,nextpts[1:]]);owned.append(j)
                if len(path)>8:break
                if abs(float((path[1]-path[0])[0]*(path[-1]-path[-2])[1]-(path[1]-path[0])[1]*(path[-1]-path[-2])[0]))>2:
                    result.append((i,path.copy(),list(owned)))
    return result


def leader_plans(ir,used_texts=()):
    plans=[];used=set();texts=[(i,e) for i,e in enumerate(ir.entities) if e.type=='text']
    records={r['text_index']:r for r in ir.meta.get('annotations',[]) if 'text_index' in r}
    for text_index,(entity_index,text) in enumerate(texts):
        if entity_index in used_texts or abs(text.rotation)%180>10:continue
        # Engineering notes/radius/diameter only; ordinary title-block text is
        # never grouped with a nearby table border.
        if not re.search(r'^(?:R\d|[QΩØΦ⌀]\d|\d+\s*[×x]|M\d|深\d)',text.content):continue
        record=records.get(text_index,{})
        if record.get('score',0)<.85 or record.get('review_status')=='numeric_conflict':continue
        height=text.height or 10;center=np.array(text.pos);choices=[]
        for j,pts,owned in measured_paths(ir,center,height):
            if set(owned)&used:continue
            for reverse in (False,True):
                p=pts[::-1] if reverse else pts
                a,b=p[-2],p[-1];v=b-a;length=float(np.linalg.norm(v))
                if length<height or abs(v[1])>max(2,length*.03):continue
                projection=float((center-a)@v/(length*length))
                offset=float(center[1]-(a[1]+b[1])/2)
                if not .0<=projection<=1.15 or not -height*1.4<=offset<=-height*.2:continue
                # Require a genuine bend and bounded size, not a part outline.
                first=p[-2]-p[0]
                if len(pts)>2 and abs(float(first[1]))<1.5:continue
                if np.linalg.norm(p[0]-center)>height*10:continue
                if len(pts)==2:
                    # A straight underline must terminate on measured part
                    # geometry. An isolated/table underline is not a leader.
                    attached=False
                    for k,other in enumerate(ir.entities):
                        if k==j or other.type=='text' or other.layer.startswith('centerline'):continue
                        if other.type=='line':ends=[other.start,other.end]
                        elif other.type=='polyline':ends=other.points
                        else:continue
                        if any(np.linalg.norm(p[0]-q)<3 for q in ends):attached=True;break
                    if not attached:continue
                choices.append((abs(offset),j,p,owned))
        if not choices:continue
        choices.sort(key=lambda c:c[0])
        # The same measured merged path may be discovered from either end.
        unique={}
        for choice in choices:
            key=(tuple(sorted(choice[3])),tuple(np.round(choice[2].ravel(),3)))
            unique.setdefault(key,choice)
        choices=sorted(unique.values(),key=lambda c:c[0])
        choices=[c for c in choices if not any(set(c[3])<set(other[3]) for other in choices)]
        if len(choices)>1 and choices[1][0]-choices[0][0]<height*.25:continue
        _,j,p,owned=choices[0];used.update(owned)
        plans.append(dict(text_entity=entity_index,line_entity=j,text_index=text_index,
                          line_entities=owned,
                          text=text.content,position=list(text.pos),height=height,
                          width=record.get('width_px'),points=p.tolist(),
                          evidence='measured_bent_open_path_with_text_over_underline'))
    return plans


def add_native_leaders(doc,ir,original,used_texts=(),used_lines=()):
    from ezdxf.math import Vec2
    from ezdxf.render.mleader import ConnectionSide,TextAlignment,HorizontalConnection
    from ezdxf.fonts import fonts
    msp=doc.modelspace();converted=[];errors=[]
    for plan in leader_plans(ir,used_texts):
        if set(plan['line_entities'])&set(used_lines):continue
        handles={e.dxf.handle for e in msp}
        try:
            points=[Vec2(image_to_cad(p,ir.height)) for p in plan['points']]
            font=fonts.make_font('times.ttf',plan['height'])
            nominal=font.text_width(plan['text'])
            factor=max(.2,min(5.,plan['width']/nominal)) if nominal and plan.get('width') else 1.
            builder=msp.add_multileader_mtext(dxfattribs={'layer':'dimensions'})
            builder.set_content('\\W'+format(factor,'.6f')+';'+plan['text'],char_height=plan['height'],
                                alignment=TextAlignment.left,style='ENG_LATIN')
            builder.set_connection_properties(landing_gap=0,dogleg_length=0)
            builder.set_connection_types(left=HorizontalConnection.bottom_of_bottom_line,
                                         right=HorizontalConnection.bottom_of_bottom_line)
            builder.set_arrow_properties('NONE',size=0)
            # Preserve locked center using measured text width; the builder's
            # insertion is the left/top text anchor, not its center.
            position=image_to_cad(plan['position'],ir.height)
            insert=Vec2(position[0]-(plan['width'] or nominal)/2,position[1]+plan['height']/2)
            side=ConnectionSide.left if points[0].x<insert.x else ConnectionSide.right
            # Builder accepts render-UCS coordinates for leader vertices.
            builder.add_leader_line(side,points)
            builder.build(insert)
            entity=builder.multileader
            # Exact measured landing/underline stays in the MLEADER context;
            # default style attachment must not introduce a diagonal shortcut
            # from the knee into the middle of the text.
            from ezdxf.math import Vec3
            entity.context.leaders[0].last_leader_point=Vec3(points[-1])
            entity.update_proxy_graphic()
            entity.set_xdata('ENG_DIMENSION',[(1000,'measured_leader_and_source_text_single_object')])
            plan['handle']=entity.dxf.handle;plan['single_selectable_object']=True
            converted.append(plan)
            msp.delete_entity(original[plan['text_entity']])
            for index in plan['line_entities']:msp.delete_entity(original[index])
        except Exception as exc:
            for e in list(msp):
                if e.dxf.handle not in handles:msp.delete_entity(e)
            errors.append(dict(text=plan['text'],reason=type(exc).__name__))
    return dict(native_leaders=len(converted),converted=converted,errors=errors)
