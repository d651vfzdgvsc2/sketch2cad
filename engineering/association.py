"""Visual association selects measured candidate IDs; no coordinates or code."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2

from engineering.dimensions import dimension_candidates
from tools.image_io import imread, imwrite


PROMPT = '''判断工程图文字与尺寸线的归属。图片保留原图，蓝色 t编号指文字，红色 d编号和细红线指程序找到的候选尺寸线。编号都不是原图内容。
只选JSON候选表中的ID；不更改文字、坐标、长度、比例，不输出代码。每条候选已有两端尺寸界线证据，但仍可能是普通轮廓或图框。结合原图箭头、界线、文字位置确认真正的尺寸线。不要把零件边缘、剖面线、表格边框当尺寸线。无法确定则candidate_id=null。
每个text_id必须返回一次；confidence为有限0到1数值；同一尺寸线不能给不同文字重复分配。不允许增加字段或坐标。
必须原样返回的text_id列表：{text_ids}。列表中的编号可能不连续，绝对不能重新从t0开始编号，也不能为表中没有的文字生成项。
候选表：{candidates}
仅返回JSON对象，顶层只有items数组。每项只包含text_id、candidate_id、confidence、reason四个字段，candidate_id必须是本项text_id下的候选id或null。
'''


def validate_associations(payload, candidates, allow_partial=False):
    if not isinstance(payload,dict) or set(payload)!={'items'} or not isinstance(payload['items'],list):
        raise ValueError('Invalid association schema')
    by_id={c['id']:c for c in candidates}
    expected={f"t{c['text_index']}" for c in candidates}
    seen=set();used=set();decisions={}
    for item in payload['items']:
        if not isinstance(item,dict) or set(item)!={'text_id','candidate_id','confidence','reason'}:
            raise ValueError('Unexpected association fields')
        text_id=item['text_id'];ident=item['candidate_id'];confidence=item['confidence']
        if not isinstance(text_id,str) or text_id not in expected or text_id in seen:
            raise ValueError('Invalid text ID')
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('Invalid association confidence')
        if not isinstance(item['reason'],str) or len(item['reason'])>1000:raise ValueError('Invalid reason')
        seen.add(text_id)
        if ident is not None:
            if not isinstance(ident,str) or ident not in by_id or f"t{by_id[ident]['text_index']}"!=text_id:
                raise ValueError('Candidate does not belong to text')
            candidate=by_id[ident];keys={(p['entity'],p['edge']) for p in candidate['line_parts']}
            if keys&used:raise ValueError('Duplicate dimension line')
            used.update(keys)
        # Uncertain model replies cannot remove a previously valid local plan.
        if confidence>=.95:decisions[text_id[1:]]=ident if ident is not None else 'rejected'
    if seen!=expected and not allow_partial:raise ValueError('Missing text IDs')
    return decisions


def review_dimension_associations(image, ir, out, provider='dashscope'):
    from tools.vlm import ask_vision
    candidates=dimension_candidates(ir)
    if not candidates:return dict(status='no_candidates',decisions={},records=[])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Batch whole labels, keeping all of a label's choices together.
    groups={}
    for c in candidates:groups.setdefault(c['text_index'],[]).append(c)
    groups=list(groups.values());decisions={};records=[];warnings=[]
    for offset in range(0,len(groups),12):
        batch=[c for group in groups[offset:offset+12] for c in group]
        picture=imread(image)
        for c in batch:
            a,b=tuple(round(v) for v in c['p1']),tuple(round(v) for v in c['p2'])
            cv2.line(picture,a,b,(60,60,220),1,cv2.LINE_AA)
            mid=tuple(round((x+y)/2) for x,y in zip(a,b))
            cv2.putText(picture,c['id'],mid,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,210),1)
        for group in groups[offset:offset+12]:
            c=group[0];pos=tuple(round(v) for v in c['text_position'])
            cv2.putText(picture,f"t{c['text_index']}",(pos[0]+5,pos[1]+12),cv2.FONT_HERSHEY_SIMPLEX,.4,(210,70,0),1)
        sheet=out/f'association_{offset//12}.png';imwrite(sheet,picture)
        table=[dict(id=c['id'],text_id=f"t{c['text_index']}",text=c['text'],
                    p1=c['p1'],p2=c['p2'],witnesses=c['witnesses']) for c in batch]
        try:
            raw=ask_vision(sheet,PROMPT.format(candidates=json.dumps(table,ensure_ascii=False),
                           text_ids=json.dumps(list(dict.fromkeys(c['text_id'] for c in table)))),
                           provider=provider,max_tokens=2500,timeout=55)
            (out/f'association_{offset//12}_response.txt').write_text(raw,encoding='utf8')
            cleaned=raw.strip()
            if cleaned.startswith('```'):cleaned='\n'.join(cleaned.splitlines()[1:-1]).strip()
            payload=json.loads(cleaned)
            if isinstance(payload,list):payload={'items':payload}
            selected=validate_associations(payload,batch,allow_partial=True)
            missing={c['text_id'] for c in table}-{r['text_id'] for r in payload['items']}
            if missing:warnings.append(f'batch {offset//12}: omitted labels {sorted(missing)} retained local evidence')
            decisions.update(selected);records.extend(payload['items'])
            (out/f'association_{offset//12}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        except Exception as exc:warnings.append(f'batch {offset//12}: {type(exc).__name__}')
    return dict(status='partial' if warnings else 'completed',decisions=decisions,
                records=records,warnings=warnings,coordinate_changes_allowed=False,
                candidate_count=len(candidates))
