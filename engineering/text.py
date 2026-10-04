"""Engineering-only OCR and native CAD annotations; never trace known glyphs."""
from __future__ import annotations

import math
import re
import cv2
import numpy as np
from emit.ir import Entity


def read_annotations(image):
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
    return items


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
        gy, gx = np.where(glyph)
        if len(gx) < 3:
            records.append({**item, 'status': 'review_no_isolated_glyphs'})
            continue
        # Measure actual ink bounds instead of OCR detector padding.
        gx, gy = gx+xa, gy+ya
        center = ((gx.min()+gx.max())/2, (gy.min()+gy.max())/2)
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
                     or bool(re.search(r'[QΩ]', item['text'])))
        records.append({**item, 'status': 'review' if uncertain else 'recognized',
                        'text_index': index, 'width_px': float(width), 'height_px': float(height),
                        'position': list(center), 'cad_rotation': -angle})
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
        entity.set_placement((source.pos[0], ir.height-source.pos[1]),
                             align=TextEntityAlignment.MIDDLE_CENTER)
        if i in records:
            font = fonts.make_font(styles[style], source.height or 2.5)
            nominal = font.text_width(source.content)
            if nominal > 0:
                entity.dxf.width = max(.2, min(5., records[i]['width_px']/nominal))
