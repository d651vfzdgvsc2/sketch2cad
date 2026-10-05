"""Engineering-only OCR and native CAD annotations; never trace known glyphs."""
from __future__ import annotations

import math
import re
from pathlib import Path
import cv2
import numpy as np
from emit.ir import Entity
from engineering.coordinates import annotation_anchor, image_to_cad
from engineering.ocr_preprocess import ocr_views, suspicious_line_box


def read_annotations(image, discover=True, diagnostic_dir=None):
    # Keep the quadrilateral and baseline direction lost by the shared OCR API.
    from tools.ocr import _engine, _map_box_back
    from tools.image_io import imread
    img = imread(image)
    h, w = img.shape[:2]
    items = []
    for k in (0, 1, 3):
        rotated = img if not k else np.ascontiguousarray(np.rot90(img, k))
        result, _ = _engine()(rotated)
        for box, text, score, *_ in result or []:
            quad = np.array(_map_box_back(box, k, w, h))
            lo, hi = quad.min(axis=0), quad.max(axis=0)
            direction = quad[1]-quad[0]
            angle = math.degrees(math.atan2(direction[1], direction[0]))
            items.append(dict(text=str(text), score=float(score), box=[*lo, *hi],
                              center=((lo+hi)/2).tolist(), quad=quad.tolist(),
                              image_rotation=angle, ocr_rotation=k))
    if discover:
        # Overlapping, enlarged quadrants expose small labels hidden by the
        # detector's full-image resize. Every box is mapped back in code.
        for tile_id, (xa, ya, xb, yb) in enumerate(ocr_tiles(w, h)):
            crop = img[ya:yb, xa:xb]
            scale = min(2., 1600/max(crop.shape[:2]))
            enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            sy, sx = enlarged.shape[0]/crop.shape[0], enlarged.shape[1]/crop.shape[1]
            for k in (0, 1):
                rotated = enlarged if not k else np.ascontiguousarray(np.rot90(enlarged, k))
                result, _ = _engine()(rotated)
                for box, text, score, *_ in result or []:
                    if float(score) < .5 or not str(text).strip(): continue
                    quad = map_crop_quad(box, k, enlarged.shape[1], enlarged.shape[0], xa, ya, sx, sy)
                    lo, hi = quad.min(axis=0), quad.max(axis=0)
                    candidate = dict(text=str(text), score=float(score), box=[*lo, *hi],
                                     center=((lo+hi)/2).tolist(), quad=quad.tolist(),
                                     image_rotation=math.degrees(math.atan2(*(quad[1]-quad[0])[::-1])),
                                     ocr_rotation=k, discovery='enlarged_tile', tile_id=tile_id)
                    # Existing coordinates have priority. Tile alternatives
                    # remain review evidence rather than relocating a label.
                    match = next((i for i in items if not i.get('discovery') and
                                  _overlap(i['box'], candidate['box']) > .55), None)
                    if match:
                        match.setdefault('crop_readings', []).append(dict(text=str(text), score=float(score)))
                    else:
                        items.append(candidate)
        views=ocr_views(img)
        if diagnostic_dir:
            from tools.image_io import imwrite
            folder=Path(diagnostic_dir);folder.mkdir(parents=True,exist_ok=True)
            for name,picture in views.items():imwrite(folder/f'{name}.png',picture)
        for channel in ('binary','clean'):
            for k in (0,1):
                picture=views[channel]
                rotated=picture if not k else np.ascontiguousarray(np.rot90(picture,k))
                try:result,_=_engine()(rotated)
                except Exception as exc:
                    for item in items:item.setdefault('discovery_warnings',[]).append(f'{channel}: {type(exc).__name__}')
                    continue
                for box,text,score,*_ in result or []:
                    if float(score)<.5 or not str(text).strip():continue
                    quad=np.array(_map_box_back(box,k,w,h));lo,hi=quad.min(axis=0),quad.max(axis=0)
                    candidate=dict(text=str(text),score=float(score),box=[*lo,*hi],center=((lo+hi)/2).tolist(),
                                   quad=quad.tolist(),image_rotation=math.degrees(math.atan2(*(quad[1]-quad[0])[::-1])),
                                   ocr_rotation=k,discovery=f'preprocessed_{channel}')
                    match=next((i for i in items if not i.get('discovery') and _overlap(i['box'],candidate['box'])>.55),None)
                    if match:match.setdefault('crop_readings',[]).append(dict(text=str(text),score=float(score),channel=channel))
                    else:items.append(candidate)
    for item in items:
        if suspicious_line_box(item['box'],item['text']):item['box_filter']='suspected_line_requires_review'
    return promote_consensus(items)


def promote_consensus(items):
    """Retain a literal numeral independently repeated across OCR channels."""
    result=[dict(item) for item in items]
    normalized=lambda text:re.sub(r'\s+','',text).replace('·','.')
    for item in result:
        label=normalized(item.get('text',''))
        if item.get('score',0)>=.8 or item.get('discovery') or item.get('box_filter'):continue
        if not re.fullmatch(r'\d{2,}(?:\.\d+)?|\d+\.\d+',label):continue
        reads=[dict(item,channel='original')]+item.get('crop_readings',[])
        channels={r.get('channel','enlarged_tile') for r in reads if r.get('score',0)>=.5 and normalized(r.get('text',''))==label}
        if len(channels)<3 or item.get('score',0)<.5:continue
        conflict=any(r.get('score',0)>=.75 and normalized(r.get('text',''))!=label for r in reads)
        conflict|=any(other.get('score',0)>=.8 and _overlap(item['box'],other['box'])>.55 and
            normalized(other.get('text',''))!=label for other in items)
        if conflict:continue
        item.update(ocr_consensus=dict(channels=sorted(channels),original_score=item['score'],literal_text=item['text']),
                    score=.9,review_status='ocr_consensus')
    return result


def ocr_tiles(width, height):
    if min(width, height) < 120: return []
    return [(x0, y0, x1, y1)
            for y0, y1 in ((0, math.ceil(height*.60)), (int(height*.40), height))
            for x0, x1 in ((0, math.ceil(width*.60)), (int(width*.40), width))]


def map_crop_quad(box, rotation, width, height, x, y, sx, sy):
    from tools.ocr import _map_box_back
    return np.array(_map_box_back(box, rotation, width, height))/[sx, sy]+[x, y]


def _overlap(a, b):
    a, b = np.array(a), np.array(b)
    area = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return area/max(1., min(np.prod(a[2:]-a[:2]), np.prod(b[2:]-b[:2])))


def select_annotations(items):
    """Prefer whole labels over high-confidence single characters inside them.

    Conflicting strings stay in alternatives for review/calibration; do not
    silently turn Q/omega into a diameter sign or infer a missing dimension.
    """
    candidates = [dict(i) for i in items if i.get('score', 0) >= .80
                  and (not i.get('discovery') or i.get('review_status') == 'ai_reviewed')
                  and (not i.get('box_filter') or i.get('review_status') == 'ai_reviewed')
                  and re.search(r'[\w\u4e00-\u9fff]', i.get('text', ''), re.UNICODE)]
    def rank(i):
        # Coverage wins over small confidence changes for contained substrings.
        n = len(i['text'].replace(' ', ''))
        orientation = abs((i.get('image_rotation', 0)+180) % 360-180)
        readable = .015 if orientation < 15 or abs(orientation-90) < 15 else 0
        return float(i['score']) + .045*min(n, 8) + readable
    selected = []
    for item in sorted(candidates, key=rank, reverse=True):
        match = next((s for s in selected if _overlap(s['box'], item['box']) > .55), None)
        if match is not None:
            if item['text'] != match['text']:
                match.setdefault('alternatives', []).append(
                    {k: item[k] for k in ('text', 'score', 'box', 'center')})
            continue
        selected.append(item)
    return sorted(selected, key=lambda i: (i['box'][1], i['box'][0]))


def crossing_strokes(ink,box):
    """Protect measured strokes continuing outside a label, including diagonals."""
    h,w=ink.shape;x0,y0,x1,y1=map(float,box)
    margin=max(18,int(min(x1-x0,y1-y0)*1.5))
    xa,ya=max(0,int(x0)-margin),max(0,int(y0)-margin)
    xb,yb=min(w,int(x1)+margin+1),min(h,int(y1)+margin+1)
    local=ink[ya:yb,xa:xb];protected=np.zeros_like(local)
    size=max(10,min(x1-x0,y1-y0))
    lines=cv2.HoughLinesP(local,1,np.pi/360,threshold=max(10,int(size*.65)),
                          minLineLength=max(15,int(size*1.5)),maxLineGap=2)
    for line in ([] if lines is None else np.asarray(lines).reshape(-1,4)):
        a,b=np.array(line[:2],float)+[xa,ya],np.array(line[2:],float)+[xa,ya]
        # A glyph stroke confined to the detector region is never protected.
        outside=lambda p:p[0]<x0-5 or p[0]>x1+5 or p[1]<y0-5 or p[1]>y1+5
        if not outside(a) and not outside(b):continue
        cv2.line(protected,tuple(line[:2]),tuple(line[2:]),255,2)
    result=np.zeros_like(ink);result[ya:yb,xa:xb]=protected
    return result


def extract_annotations(ink, items):
    """Remove contained glyph components, protecting lines crossing OCR boxes.

    A connected component that reaches outside a label box is never erased:
    it may be a dimension/leader/contour. Ambiguous labels remain reviewable.
    """
    h, w = ink.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    mask = np.zeros_like(ink)
    entities, records = [], []
    for item in select_annotations(items):
        x0, y0, x1, y1 = item['box']
        xa, ya = max(0, int(x0)-2), max(0, int(y0)-2)
        xb, yb = min(w, int(math.ceil(x1))+3), min(h, int(math.ceil(y1))+3)
        ids = np.unique(labels[ya:yb, xa:xb])
        contained = []
        for idx in ids:
            if idx == 0: continue
            x, y, cw, ch, _ = stats[idx]
            if x >= xa and y >= ya and x+cw <= xb and y+ch <= yb:
                contained.append(idx)
        glyph = np.isin(labels[ya:yb, xa:xb], contained)
        separation = 'isolated_components'
        if item.get('review_status') in ('ai_reviewed','ocr_consensus'):
            # A reviewed label can touch a dimension/table line. Preserve
            # strokes that extend beyond its box, remove the remaining glyphs.
            protected = crossing_strokes(ink,item['box'])
            touching = (ink[ya:yb, xa:xb] > 0) & (protected[ya:yb, xa:xb] == 0)
            glyph |= touching
            separation = 'reviewed_text_with_crossing_lines_preserved'
        gy, gx = np.where(glyph)
        if len(gx) < 3:
            if item.get('review_status') not in ('ai_reviewed','ocr_consensus') or np.count_nonzero(ink[ya:yb,xa:xb])<3:
                records.append({**item, 'status': 'review_no_isolated_glyphs'})
                continue
            # Confirmed text still deserves a native entity when every source
            # pixel touches a protected stroke. Do not erase ambiguous geometry.
            gx=np.array([int(x0),int(x1)])-xa;gy=np.array([int(y0),int(y1)])-ya
            separation='native_text_fallback_with_geometry_retained'
        # Measure actual ink bounds instead of OCR detector padding.
        gx, gy = gx+xa, gy+ya
        center = annotation_anchor(item)
        angle = item.get('image_rotation')
        if angle is None:
            angle = -90. if y1-y0 > (x1-x0)*1.15 else 0.
        # RapidOCR may internally rotate its recognition crop by 90 degrees.
        # The detector's top edge then is NOT the recognized text baseline.
        # Resolve orthogonal layout from actual glyph bounds (not that edge).
        ink_width, ink_height = gx.max()-gx.min()+1, gy.max()-gy.min()+1
        if len(item['text'].replace(' ', '')) >= 2:
            if ink_width > ink_height*1.12: angle = 0.
            elif ink_height > ink_width*1.12: angle = -90.
        # OCR box tilts under 2 degrees are detector noise on straight drawings.
        nearest = round(angle/90)*90
        if abs(angle-nearest) < 2: angle = float(nearest)
        theta = math.radians(angle)
        points = np.column_stack((gx-center[0], gy-center[1]))
        local = points @ np.array([[math.cos(theta), -math.sin(theta)],
                                  [math.sin(theta), math.cos(theta)]])
        width, height = np.ptp(local, axis=0)+1
        mask[ya:yb, xa:xb][glyph] = 255
        index = len(entities)
        entities.append(Entity(type='text', content=item['text'], pos=center,
                               height=float(height), rotation=-angle, layer='text'))
        uncertain = (item['score'] < .95 or bool(item.get('alternatives'))
                     or bool(re.search(r'[QΩ]', item['text']))
                     or item.get('review_status') == 'numeric_conflict')
        records.append({**item, 'status': 'review' if uncertain else 'recognized',
                        'separation': separation,
                        'text_index': index, 'width_px': float(width), 'height_px': float(height),
                        'position': list(center), 'anchor_source': 'ocr_box_center',
                        'cad_position_pixels': list(image_to_cad(center, h)), 'cad_rotation': -angle})
    return entities, mask, records


def configure_native_text(doc, ir):
    """Position TEXT by its centre with a real font and measured width."""
    from ezdxf.enums import TextEntityAlignment
    from ezdxf.fonts import fonts
    styles = {'ENG_LATIN': 'times.ttf', 'ENG_CJK': 'simhei.ttf'}
    for name, filename in styles.items():
        if name not in doc.styles: doc.styles.new(name, dxfattribs={'font': filename})
    records = {r['text_index']: r for r in ir.meta.get('annotations', []) if 'text_index' in r}
    text_ir = [e for e in ir.entities if e.type == 'text']
    for i, (entity, source) in enumerate(zip(doc.modelspace().query('TEXT'), text_ir)):
        style = 'ENG_CJK' if re.search(r'[\u4e00-\u9fff]', source.content) else 'ENG_LATIN'
        entity.dxf.style = style
        entity.dxf.rotation = source.rotation
        entity.dxf.lineweight = -1
        entity.set_placement(image_to_cad(source.pos, ir.height),
                             align=TextEntityAlignment.MIDDLE_CENTER)
        if i in records:
            font = fonts.make_font(styles[style], source.height or 2.5)
            nominal = font.text_width(source.content)
            if nominal > 0:
                entity.dxf.width = max(.2, min(5., records[i]['width_px']/nominal))
