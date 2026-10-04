"""Bounded visual-model review of OCR regions; all replies are untrusted data."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from engineering.text import select_annotations, _overlap
from tools.image_io import imread, imwrite

PROMPT = '''你在复核图片转 CAD 的文字。输入是原图文字区域的裁剪拼图，每格左上角 t编号是程序ID，不是原图文字。竖排标签已旋转为便于阅读的方向。
独立逐字转录每格的原图文字，保留真实显示的数字、中文和符号。不得凭机械常识更改字符，不得因为尺寸不合理而改数字；不要把标注附近的引线当成字符笔画。
区分文字和圆孔/虚线/剖面线。确实不是文字则is_text=false。看不清则confidence<0.9并说明。
所有文字最终必须输出CAD TEXT/MTEXT或DIMENSION内的文字，不能用线段模拟字形。
每个ID必须返回一项，不增加坐标，不编造额外标签，不输出代码。
必须回答的ID: {items}
输出JSON: {{"items":[{{"id":"t0","text":"60","is_text":true,"confidence":0.99,"reason":"原图清晰可读"}}]}}
'''


def configured():
    from core.config import get
    return bool(get('DASHSCOPE_API_KEY'))


def validate_reply(payload, ids):
    if not isinstance(payload, dict) or not isinstance(payload.get('items'), list):
        raise ValueError('Review reply must contain items')
    seen = set()
    for item in payload['items']:
        ident = item.get('id')
        if ident not in ids or ident in seen: raise ValueError('Unknown or duplicate review ID')
        seen.add(ident)
        confidence = item.get('confidence')
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid review confidence')
        if not isinstance(item.get('is_text'), bool): raise ValueError('Invalid text classification')
        if not isinstance(item.get('text'), str) or len(item['text']) > 120 or any(ord(c) < 32 for c in item['text']):
            raise ValueError('Invalid reviewed text')
        if any(c in item['text'] for c in ('\\', '{', '}')) or '%%' in item['text']:
            raise ValueError('CAD control syntax is not a text label')
    if seen != set(ids): raise ValueError('Review omitted IDs')
    return payload['items']


def review_annotations(image, ocr, out, provider='dashscope'):
    from tools.vlm import ask_vision
    # Include low-confidence OCR regions for a second reading, but a model
    # cannot add regions where the detector found no text evidence.
    candidates = select_annotations(ocr)
    for item in sorted(ocr, key=lambda a: -a.get('score', 0)):
        if item.get('score', 0) < .5 or not item.get('text', '').strip(): continue
        if any(_overlap(item['box'], k['box']) > .45 for k in candidates): continue
        candidates.append(dict(item))
    bgr = imread(image)
    records, accepted, warnings = [], [], []
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(candidates), 16):
        batch = candidates[offset:offset+16]
        tiles = np.full((math.ceil(len(batch)/4)*190, 1200, 3), 255, np.uint8)
        ids = []
        for j, item in enumerate(batch):
            ident = f't{offset+j}'; ids.append(ident)
            x0,y0,x1,y1 = item['box']
            pad = max(5, int(min(x1-x0,y1-y0)*.2))
            xa,ya=max(0,int(x0)-pad),max(0,int(y0)-pad)
            xb,yb=min(bgr.shape[1],int(x1)+pad+1),min(bgr.shape[0],int(y1)+pad+1)
            crop=bgr[ya:yb,xa:xb]
            if not crop.size: raise ValueError('Empty OCR crop')
            if y1-y0 > (x1-x0)*1.15 and len(item['text'].strip()) >= 2:
                crop=np.ascontiguousarray(np.rot90(crop,3))
            scale=min(280/crop.shape[1],145/crop.shape[0],3)
            crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            ox,oy=(j%4)*300,(j//4)*190
            cv2.putText(tiles,ident,(ox+8,oy+24),cv2.FONT_HERSHEY_SIMPLEX,.65,(160,80,0),1)
            tiles[oy+35:oy+35+crop.shape[0],ox+10:ox+10+crop.shape[1]]=crop
        sheet=out/f'text_review_{offset//16:02d}.png'; imwrite(sheet,tiles)
        descriptions=ids  # blind transcription avoids anchoring on bad OCR
        try:
            raw=ask_vision(sheet,PROMPT.format(items=json.dumps(descriptions,ensure_ascii=False)),
                           provider=provider,max_tokens=3000,timeout=55)
            payload=json.loads(raw[raw.find('{'):raw.rfind('}')+1])
            replies=validate_reply(payload,set(ids))
            by_id={r['id']:r for r in replies}
            (out/f'text_review_{offset//16:02d}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        except Exception as exc:
            warnings.append(f'batch {offset//16}: {type(exc).__name__}')
            accepted.extend(batch)
            continue
        for ident,item in zip(ids,batch):
            reply=by_id[ident]
            record=dict(id=ident,original=item['text'],reply=reply,applied=False)
            if reply['confidence'] >= .95 and reply['is_text'] and reply['text'].strip():
                # A proposed numeric change receives an independent full-image
                # check below before replacing a high-confidence OCR label.
                record['applied']=True
                updated={**item,'text':reply['text'].strip(),'score':max(item['score'],.95),
                         'review_status':'ai_reviewed','ocr_original':item['text'],
                         'ocr_original_score':item['score'],'ai_review_id':ident,'ai_review':reply}
                accepted.append(updated)
            elif reply['confidence'] >= .97 and not reply['is_text'] and item['score'] < .90:
                record['applied']=True; record['action']='reject_non_text'
            else:
                accepted.append(item)
            records.append(record)
    # Review content disagreements in their full drawing context. Two model
    # readings are still not a manufacturing guarantee; keep both in the log.
    normalized = lambda text: re.sub(r'\s+', '', text).replace('x', '×')
    disagreements=[]
    for index,item in enumerate(accepted):
        if item.get('ocr_original') and normalized(item['text']) != normalized(item['ocr_original']):
            disagreements.append((index,item))
    if disagreements:
        ids=[f'n{i}' for i in range(len(disagreements))]
        locations=[dict(id=ident,box=item['box']) for ident,(_,item) in zip(ids,disagreements)]
        prompt=('请独立读取原工程图指定像素框内的标注。不要修改尺寸，不要猜。框为左上原点的[x0,y0,x1,y1]。'
                f'原图尺寸为{bgr.shape[1]}×{bgr.shape[0]}。'
                '每个ID必须返回且只返回一次。text必须为字符串，confidence必须为0到1数值，is_text必须为布尔值。'
                '只输出JSON对象，严格使用结构：'
                '{"items":[{"id":"n0","text":"3.2","is_text":true,"confidence":0.99,"reason":"清晰可读"}]}。'
                '示例内容不是答案。需读取的实际区域：'+json.dumps(locations,ensure_ascii=False))
        try:
            raw=ask_vision(image,prompt,provider=provider,max_tokens=1800,timeout=55)
            (out/'numeric_confirmation_response.txt').write_text(raw,encoding='utf8')
            replies=validate_reply(json.loads(raw[raw.find('{'):raw.rfind('}')+1]),set(ids))
            by_id={r['id']:r for r in replies}
        except Exception as exc:
            by_id={};warnings.append(f'numeric confirmation: {type(exc).__name__}')
        for ident,(index,item) in zip(ids,disagreements):
            confirmation=by_id.get(ident,{})
            ok=(confirmation.get('is_text') and confirmation.get('confidence',0)>=.95
                and normalized(confirmation.get('text',''))==normalized(item['text']))
            item['numeric_confirmation']=dict(accepted=bool(ok),reply=confirmation)
            if not ok:
                item['ai_suggested_text']=item['text']
                item['text']=item['ocr_original']
                item['score']=item['ocr_original_score']
                item['review_status']='numeric_conflict'
            for record in records:
                if record['id']==item['ai_review_id']:
                    record['numeric_confirmation']=item['numeric_confirmation']
                    record['applied']=bool(ok)
    return accepted,dict(status='partial' if warnings else 'completed',provider=provider,
                         records=records,warnings=warnings,crops=len(candidates),
                         coordinate_changes_allowed=False)


def review_final_drawing(image, preview, summary, out, provider='dashscope'):
    """Final visual audit reports issues; it cannot mutate the selected CAD."""
    from tools.vlm import ask_vision
    source,rendered=imread(image),imread(preview)
    panels=[]
    for label,picture in (('SOURCE',source),('CAD EXPORT',rendered)):
        scale=min(1.,1200/picture.shape[1])
        picture=cv2.resize(picture,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        panel=np.full((picture.shape[0]+40,picture.shape[1],3),255,np.uint8)
        panel[40:]=picture
        cv2.putText(panel,label,(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        panels.append(panel)
    sheet=Path(out)/'final_review.png';imwrite(sheet,np.concatenate(panels,axis=1))
    prompt=('复核图片转CAD结果，左是原图，右是实际DXF渲染。检查漏字、错字、方向、缺线、额外线和几何形状。'
            '只报告具体可见差异，不得称为合格制造图。CAD实体类型不能单凭图片判断，以下程序统计才是依据：'
            +json.dumps(summary,ensure_ascii=False)
            +'。返回JSON {"findings":[{"region":"位置","issue":"具体问题","severity":"minor或major"}],'
            '"needs_review":true}。只报告，不生成代码或坐标。')
    try:
        raw=ask_vision(sheet,prompt,provider=provider,max_tokens=2200,timeout=55)
        payload=json.loads(raw[raw.find('{'):raw.rfind('}')+1])
        if not isinstance(payload,dict) or not isinstance(payload.get('findings'),list):raise ValueError('Invalid audit')
        findings=[]
        for item in payload['findings'][:30]:
            if not isinstance(item,dict):raise ValueError('Invalid finding')
            findings.append({k:str(item.get(k,''))[:1000] for k in ('region','issue','severity')})
        return dict(status='completed',findings=findings,geometry_modified=False,
                    needs_review=True,comparison=str(sheet))
    except Exception as exc:
        return dict(status='failed',reason=type(exc).__name__,geometry_modified=False,needs_review=True)
