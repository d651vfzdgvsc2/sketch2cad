"""Bounded source-region rereading and reversible CAD annotation repairs."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import cv2
import ezdxf
import numpy as np

from emit.ir import Entity
from engineering.text import extract_annotations,select_annotations,map_crop_quad,_overlap
from engineering.ocr_preprocess import ocr_views,suspicious_line_box
from engineering.trace import ink_mask
from engineering.primitives import sample_entity
from engineering.metrics import compare,combined_score
from tools.image_io import imread,imwrite


def table_text_regions(image,annotations):
    """Closed table cells beside known labels expose faint detector misses."""
    ink=ink_mask(image);h,w=ink.shape
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(45,w//30)),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(35,h//30),1),np.uint8))
    _,_,hs,_=cv2.connectedComponentsWithStats(horizontal,8)
    _,_,vs,_=cv2.connectedComponentsWithStats(vertical,8)
    rows=[(float(y+(ch-1)/2),int(x),int(x+cw-1)) for x,y,cw,ch,_ in hs[1:] if cw>60 and ch<7]
    columns=[(float(x+(cw-1)/2),int(y),int(y+ch-1)) for x,y,cw,ch,_ in vs[1:] if ch>35 and cw<7]
    rows.sort();regions=[]
    for n,(top,left,right) in enumerate(rows):
        for bottom,left2,right2 in rows[n+1:]:
            if bottom-top<15:continue
            if bottom-top>90:break
            lo,hi=max(left,left2),min(right,right2)
            if hi-lo<60:continue
            xs=sorted(x for x,ya,yb in columns if lo-2<=x<=hi+2 and ya<=top+3 and yb>=bottom-3)
            distinct=[]
            for x in xs:
                if not distinct or x-distinct[-1]>4:distinct.append(x)
            # This must be a table row, not just a rectangle near a dimension.
            known=[a for a in annotations if 'text_index' in a and lo<a['box'][0]<hi and top<a['box'][1]<bottom]
            if len(distinct)<3 or not known:continue
            for x0,x1 in zip(distinct,distinct[1:]):
                if not 22<x1-x0<240:continue
                if any(x0<(a['box'][0]+a['box'][2])/2<x1 for a in known):continue
                xa,ya,xb,yb=map(int,(x0+4,top+4,x1-3,bottom-3))
                if xa>=xb or ya>=yb:continue
                crop=image[ya:yb,xa:xb];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                # Some short internal dividers do not span both row borders.
                # Exclude their thin edge-connected components from glyph bounds.
                _,component_labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
                for index,(cx,cy,cw,ch,area) in enumerate(stats[1:],1):
                    edge=cx==0 or cy==0 or cx+cw==binary.shape[1] or cy+ch==binary.shape[0]
                    if edge and max(cw,ch)>5*max(1,min(cw,ch)):
                        binary[component_labels==index]=0
                yy,xx=np.where(binary>0)
                if len(xx)<8:continue
                # Weak JPEG edge dust is not a missing cell label.
                if xx.max()-xx.min()<5 or yy.max()-yy.min()<5:continue
                box=[max(xa,xa+int(xx.min())-3),max(ya,ya+int(yy.min())-3),
                     min(xb,xa+int(xx.max())+4),min(yb,ya+int(yy.max())+4)]
                if any(_overlap(box,r['box'])>.5 for r in regions):continue
                regions.append(dict(box=box,component_count=2,area=len(xx),source='unlabelled_table_cell'))
    return regions


def missing_text_regions(image,annotations,limit=12):
    """Program-measured groups of small source components outside native labels."""
    clean=255-ocr_views(image)['clean']
    h,w=clean.shape
    heights=[min(a['box'][2]-a['box'][0],a['box'][3]-a['box'][1]) for a in annotations if 'text_index' in a]
    typical=float(np.median(heights)) if heights else 20.
    for a in annotations:
        if 'text_index' not in a:continue
        x0,y0,x1,y1=a['box']
        clean[max(0,int(y0)-3):min(h,math.ceil(y1)+4),max(0,int(x0)-3):min(w,math.ceil(x1)+4)]=0
    n,labels,stats,_=cv2.connectedComponentsWithStats(clean,8)
    small=np.zeros_like(clean)
    ids=[]
    for i,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if area<7 or min(cw,ch)<2 or max(cw,ch)>max(70,typical*2.5):continue
        if max(cw,ch)/min(cw,ch)>8:continue
        small[y:y+ch,x:x+cw][labels[y:y+ch,x:x+cw]==i]=255;ids.append(i)
    regions=table_text_regions(image,annotations)
    gap=max(5,min(15,round(typical*.5)))
    for kernel in (np.ones((3,gap),np.uint8),np.ones((gap,3),np.uint8)):
        groups=cv2.dilate(small,kernel)
        _,_,stats,_=cv2.connectedComponentsWithStats(groups,8)
        for x,y,cw,ch,area in stats[1:]:
            if min(cw,ch)<5 or min(cw,ch)>max(70,typical*2.5) or max(cw,ch)>320:continue
            if max(cw,ch)/min(cw,ch)>10:continue # long dashed axes, not a word
            box=list(map(int,[max(0,x-8),max(0,y-8),min(w,x+cw+8),min(h,y+ch+8)]))
            if any(_overlap(box,r['box'])>.5 for r in regions):continue
            source_components=set(np.unique(labels[y:y+ch,x:x+cw]))&set(ids)
            # Several aligned components are more likely text than one hole.
            regions.append(dict(box=box,component_count=len(source_components),area=int(area)))
    regions.sort(key=lambda r:(r.get('source')!='unlabelled_table_cell',-min(r['component_count'],8),r['area']))
    return [dict(id=f'g{i}',**r) for i,r in enumerate(regions[:limit])]


def reread_regions(image,regions,existing,out,engine=None):
    from tools.ocr import _engine
    engine=engine or _engine()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    detections=[];warnings=[]
    for region in regions:
        before=len(detections)
        xa,ya,xb,yb=region['box'];crop=image[ya:yb,xa:xb]
        if not crop.size:continue
        views=ocr_views(crop)
        for channel in ('original','clean'):
            raw=crop if channel=='original' else views['clean']
            zoom=cv2.resize(raw,None,fx=3.,fy=3.,interpolation=cv2.INTER_CUBIC)
            imwrite(out/f"{region['id']}_{channel}.png",zoom)
            for k in (0,1):
                try:result,_=engine(zoom if not k else np.ascontiguousarray(np.rot90(zoom,k)))
                except Exception as exc:
                    warnings.append(f"{region['id']}/{channel}: {type(exc).__name__}");continue
                for box,text,score,*_ in result or []:
                    if float(score)<.5 or not str(text).strip():continue
                    quad=map_crop_quad(box,k,zoom.shape[1],zoom.shape[0],xa,ya,3.,3.)
                    lo,hi=quad.min(axis=0),quad.max(axis=0)
                    if (lo<[xa,ya]).any() or (hi>[xb,yb]).any():continue
                    item=dict(text=str(text),score=float(score),box=[*lo.tolist(),*hi.tolist()],quad=quad.tolist(),
                        center=((lo+hi)/2).tolist(),image_rotation=math.degrees(math.atan2(*(quad[1]-quad[0])[::-1])),
                        discovery='local_gap',region_id=region['id'],channel=channel)
                    if suspicious_line_box(item['box'],item['text']):continue
                    if any(_overlap(item['box'],e['box'])>.45 for e in existing):continue
                    # Different crop rotations are readings of the same region,
                    # not independent additional labels.
                    match=next((e for e in detections if _overlap(item['box'],e['box'])>.5),None)
                    if match:match.setdefault('crop_readings',[]).append(dict(text=item['text'],score=item['score']))
                    else:detections.append(item)
        if len(detections)==before and region['component_count']>=2 and max(xb-xa,yb-ya)<110 and hasattr(engine,'text_recognizer'):
            # Bypass only the detector for a program-measured small region.
            # Actual OCR transcription remains mandatory; no model-only text.
            for channel in ('original','clean'):
                raw=crop if channel=='original' else views['clean']
                if raw.ndim==2:raw=cv2.cvtColor(raw,cv2.COLOR_GRAY2BGR)
                if raw.shape[0]>raw.shape[1]*1.5:raw=np.ascontiguousarray(np.rot90(raw,3))
                try:
                    readings,_=engine.text_recognizer([raw])
                    text,score=readings[0]
                except Exception as exc:
                    warnings.append(f"{region['id']}/recognizer: {type(exc).__name__}");continue
                if float(score)<.5 or not str(text).strip():continue
                item=dict(text=str(text),score=float(score),box=[xa,ya,xb,yb],
                    center=[(xa+xb)/2,(ya+yb)/2],discovery='local_gap',region_id=region['id'],
                    direct_recognition=True,channel=channel)
                match=next((e for e in detections if _overlap(item['box'],e['box'])>.5),None)
                if match:match.setdefault('crop_readings',[]).append(dict(text=item['text'],score=item['score']))
                else:detections.append(item)
    return detections,warnings


def supported_discoveries(candidates,reviewed,existing):
    """A model reading must agree with a real OCR reading of that same region."""
    accepted=[]
    normalized=lambda text:''.join(text.split())
    for item in select_annotations(reviewed):
        if item.get('review_status')!='ai_reviewed':continue
        if '\ufffd' in item['text']:continue # undecodable OCR/model characters are not labels
        source=next((c for c in candidates if c['box']==item['box']),None)
        if not source:continue
        readings=[source]+source.get('crop_readings',[])
        if not any(r.get('score',0)>=.5 and normalized(r['text'])==normalized(item['text']) for r in readings):continue
        if any(_overlap(item['box'],e['box'])>.45 for e in existing):continue
        if any(_overlap(item['box'],e['box'])>.45 for e in accepted):continue
        accepted.append(item)
    return accepted


def confirm_local_labels(image,candidates,out,provider='dashscope'):
    """Two blind crop readings, no neighboring notes or example answers."""
    from engineering.ai_review import validate_reply
    from tools.vlm import ask_vision
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    accepted=[];records=[];warnings=[]
    for offset in range(0,len(candidates),6):
        batch=candidates[offset:offset+6];ids=[f'l{offset+j}' for j in range(len(batch))]
        readings=[]
        for pass_id,pad in enumerate((3,6)):
            tiles=np.full((math.ceil(len(batch)/3)*180,1050,3),255,np.uint8)
            for j,(ident,item) in enumerate(zip(ids,batch)):
                x0,y0,x1,y1=item['box'];crop=image[max(0,int(y0)-pad):min(image.shape[0],math.ceil(y1)+pad),
                    max(0,int(x0)-pad):min(image.shape[1],math.ceil(x1)+pad)]
                if y1-y0>(x1-x0)*1.5:crop=np.ascontiguousarray(np.rot90(crop,3))
                factor=min(330/crop.shape[1],140/crop.shape[0],4)
                crop=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
                x,y=j%3*350,j//3*180
                cv2.putText(tiles,ident,(x+8,y+23),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,80,0),1)
                tiles[y+32:y+32+crop.shape[0],x+8:x+8+crop.shape[1]]=crop
            sheet=out/f'local_{offset}_{pass_id}.png';imwrite(sheet,tiles)
            prompt=('逐格检查工程图原图裁剪，蓝色编号仅是程序ID。只转录该格中央真正印刷的文字；'
                '圆孔、圆弧、剖面线、箭头、虚线和表格边框不是文字。没有明确文字必须is_text=false，text为空字符串。'
                '不得凭机械常识猜数字或标注，不得借用附近区域的文字。原图模糊则降低confidence。'
                '每个ID返回一次，不能添加坐标、代码或其他ID。只输出JSON对象，顶层items，每项字段为id、text字符串、'
                'is_text布尔值、confidence有限0到1数字、reason字符串。必须回答的ID：'+json.dumps(ids))
            try:
                raw=ask_vision(sheet,prompt,provider=provider,max_tokens=1800,timeout=55)
                (out/f'local_{offset}_{pass_id}_response.txt').write_text(raw,encoding='utf8')
                payload=json.loads(raw[raw.find('{'):raw.rfind('}')+1])
                replies=validate_reply(payload,set(ids));readings.append({r['id']:r for r in replies})
            except Exception as exc:
                warnings.append(f'local batch {offset}/{pass_id}: {type(exc).__name__}');readings.append({})
        for ident,item in zip(ids,batch):
            first,second=(reading.get(ident,{}) for reading in readings)
            ok=(first.get('is_text') and second.get('is_text') and min(first.get('confidence',0),second.get('confidence',0))>=.97
                and ''.join(first.get('text','').split())==''.join(second.get('text','').split()))
            records.append(dict(id=ident,box=item['box'],first=first,second=second,agreed=bool(ok)))
            if ok:
                accepted.append(dict(item,text=first['text'].strip(),score=.99,review_status='ai_reviewed',
                    numeric_confirmation=dict(accepted=True,reply=second),ocr_original=item['text']))
    return accepted,dict(status='partial' if warnings else 'completed',records=records,warnings=warnings)


def trim_glyph_geometry(ir,glyph_mask):
    """Remove only measured glyph pixels; retain crossing/continued geometry."""
    result=ir.model_copy(deep=True);entities=[];colors=[];removed=0
    palette=ir.meta.get('stroke_colors_rgb',[None]*len(ir.entities))
    if len(palette)!=len(ir.entities):palette=[None]*len(ir.entities)
    mask=glyph_mask # no dilation back into the explicitly protected crossing line
    protected=set()
    for plan in ir.meta.get('native_dimensions',{}).get('converted',[]):
        protected.update(p['entity'] for p in plan['line_parts'])
        protected.update(p['entity'] for p in plan.get('owned_graphics',[]))
    h,w=mask.shape
    def glyph(points):
        xy=np.rint(points).astype(int)
        inside=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w)&(xy[:,1]<h)
        hits=np.zeros(len(xy),bool);p=xy[inside]
        hits[inside]=mask[p[:,1],p[:,0]]>0
        return hits
    for index,e in enumerate(ir.entities):
        color=palette[index]
        if index in protected or e.type=='text' or e.layer.startswith('centerline'):
            entities.append(e);colors.append(color);continue
        points=sample_entity(e,spacing=.5)
        if not len(points) or not glyph(points).any():
            entities.append(e);colors.append(color);continue
        if e.type not in ('line','polyline'):
            # A small traced 0 may be a circle, but a real circle crossing the
            # crop is protected unless essentially its entire ink is a glyph.
            if glyph(points).mean()<.98:
                entities.append(e);colors.append(color)
            else:removed+=1
            continue
        pts=[e.start,e.end] if e.type=='line' else e.points+[e.points[0]] if e.closed else e.points
        runs=[];current=[];changed=False
        for a,b in zip(pts,pts[1:]):
            a,b=np.array(a),np.array(b);length=float(np.linalg.norm(b-a))
            count=max(2,math.ceil(length/.5)+1);ts=np.linspace(0,1,count)
            line=a+(b-a)*ts[:,None];hits=glyph(line)
            if hits.any():changed=True
            for p,hit in zip(line,hits):
                if hit:
                    if len(current)>1:runs.append(current)
                    current=[]
                elif not current or np.linalg.norm(p-current[-1])>.05:current.append(p)
        if len(current)>1:runs.append(current)
        if not changed:
            entities.append(e);colors.append(color);continue
        removed+=1
        for run in runs:
            if np.linalg.norm(np.diff(run,axis=0),axis=1).sum()<.8:continue
            if e.type=='line':piece=Entity(type='line',start=tuple(run[0]),end=tuple(run[-1]),layer=e.layer)
            else:
                # Simplify straight sample runs without fitting across erased ink.
                approx=cv2.approxPolyDP(np.array(run,np.float32),.25,False).reshape(-1,2)
                if len(approx)<2:continue
                piece=Entity(type='polyline',points=[tuple(p) for p in approx],layer=e.layer)
            entities.append(piece);colors.append(color)
    result.entities=entities;result.meta['stroke_colors_rgb']=colors
    return result,removed


def apply_text_patch(ir,image,items):
    texts,mask,records=extract_annotations(ink_mask(cv2.GaussianBlur(image,(3,3),0)),items)
    revised,removed=trim_glyph_geometry(ir,mask)
    old_texts=[e for e in revised.entities if e.type=='text']
    old_records=revised.meta.get('annotations',[])
    new_records=[dict(a) for a in old_records];added=[]
    for text,record in zip(texts,[a for a in records if 'text_index' in a]):
        match=next((i for i,e in enumerate(old_texts) if e.content==text.content and np.linalg.norm(np.array(e.pos)-text.pos)<2),None)
        if match is not None:continue # Existing native text and placement stay fixed.
        if any(np.linalg.norm(np.array(e.pos)-text.pos)<2 for e in old_texts):continue
        record=dict(record,text_index=len(old_texts));new_records.append(record)
        revised.entities.append(text);revised.meta['stroke_colors_rgb'].append(None)
        old_texts.append(text);added.append(text.content)
    revised.meta['annotations']=new_records
    return revised,dict(added_labels=added,trimmed_glyph_entities=removed)


def repair_selected(image,ir,selected,out,use_ai_review=True,provider='dashscope',use_ocr=True):
    """One bounded source reread, export, objective gate, then final comparison."""
    from engineering.pipeline import write_ir,score_dxf,annotation_mask
    from engineering.cleanup import structure_report
    from engineering.association import review_dimension_associations
    source=imread(image);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    existing=select_annotations(ir.meta.get('ocr',[]))
    annotations=ir.meta.get('annotations',[])
    regions=missing_text_regions(source,annotations) if use_ocr else []
    report=dict(status='checked',regions=regions,ocr_candidates=[],accepted_new_labels=[],warnings=[],attempts=[])
    additions=[]
    if regions and use_ai_review:
        try:
            candidates,warnings=reread_regions(source,regions,existing,out/'ocr')
            report['ocr_candidates']=candidates;report['warnings'].extend(warnings)
            if candidates:
                reviewed,review=confirm_local_labels(source,candidates,out/'text_review',provider)
                report['text_review']=review
                additions=supported_discoveries(candidates,reviewed,existing)
        except Exception as exc:report['warnings'].append(f'local OCR: {type(exc).__name__}')
    elif regions:report['warnings'].append('New OCR regions require configured visual confirmation; not adopted offline')
    # Accepted labels are not rewritten. This pass removes remaining traced
    # glyphs at those coordinates and inserts confirmed missing native text.
    revised,patch=apply_text_patch(ir,source,existing+additions)
    revised.meta['ocr']=ir.meta.get('ocr',[])+additions
    report['patch']=patch
    if use_ai_review:
        try:revised.meta['dimension_association']=review_dimension_associations(image,revised,out/'association',provider)
        except Exception as exc:report['warnings'].append(f'association: {type(exc).__name__}')
    revised.meta['cad_structure']=structure_report(revised.entities)
    target=out/'repaired.dxf';dims=write_ir(revised,target)
    (out/'repaired.json').write_text(revised.to_json(),encoding='utf8')
    score=score_dxf(target,image,out/'repaired.png')
    ignore=annotation_mask(source.shape[:2],revised.meta['annotations'])
    baseline=compare(imread(selected['png']),source,ignore_mask=ignore)
    geometry=compare(imread(score['png']),source,ignore_mask=ignore)
    expected=Counter(e.content for e in revised.entities if e.type=='text')
    doc=ezdxf.readfile(target);native=Counter(e.dxf.text for e in doc.modelspace().query('TEXT DIMENSION'))
    from ezdxf.entities import MText
    for leader in doc.modelspace().query('MULTILEADER'):
        text=MText();text.text=leader.context.mtext.default_content;native[text.plain_text()]+=1
    old=ir.meta.get('native_dimensions',{})
    benefit=(bool(patch['added_labels']) or patch['trimmed_glyph_entities']>0 or
        dims['native_dimensions']+dims['native_leaders']>old.get('native_dimensions',0)+old.get('native_leaders',0))
    valid=(score['valid'] and not doc.audit().has_errors and native==expected and
        not dims['errors'] and not dims['leaders']['errors'] and
        geometry['recall']>=baseline['recall']-.002 and combined_score(geometry)>=combined_score(baseline)-.003 and benefit and
        dims['native_dimensions']+dims['native_leaders']>=old.get('native_dimensions',0)+old.get('native_leaders',0))
    report.update(status='adopted' if valid else 'retained_original',accepted_new_labels=patch['added_labels'] if valid else [],
        attempts=[dict(adopted=valid,geometry_before=baseline,geometry_after=geometry,
            labels_preserved=native==expected,native_dimensions=dims['native_dimensions'],native_leaders=dims['native_leaders'])])
    (out/'repair_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if not valid:return ir,selected,report
    score.update(geometry=geometry,native_dimensions=dims['native_dimensions'],native_leaders=dims['native_leaders'],
        ir=str(out/'repaired.json'),proposal=selected['proposal'],cad_structure=revised.meta['cad_structure'],
        geometry_review=revised.meta.get('geometry_review',{}),artifact_review=revised.meta.get('artifact_review',{}))
    return revised,score,report
