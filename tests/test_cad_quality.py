"""CAD semantics regressions independent of rendered image similarity."""
import math
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR, Entity, LayerSpec
from engineering.cleanup import (continuous_paths, assemble_paths, heal_scan_strokes,
                                 assemble_centerlines, repair_color_crossings)
from engineering.pipeline import build_proposals, write_ir
from engineering.primitives import Feature, build
from engineering.text import extract_annotations, select_annotations
from tools.image_io import imwrite


def test_native_text_export_replaces_glyphs_and_preserves_crossing_line(tmp_path):
    image = np.full((140, 240, 3), 255, np.uint8)
    cv2.putText(image, '60', (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.line(image, (10, 76), (220, 76), (0, 0, 0), 2)
    cv2.circle(image, (170, 35), 20, (0, 0, 0), 2)
    ocr = [dict(text='60', score=.99, box=[56, 40, 104, 80], center=[80, 60])]
    path = tmp_path/'input.png'; imwrite(path, image)
    proposals, _, _ = build_proposals(path, ocr, False)
    ir = proposals['CLEAN']; target = tmp_path/'text.dxf'; write_ir(ir, target)
    doc = ezdxf.readfile(target)
    assert [e.dxf.text for e in doc.modelspace().query('TEXT')] == ['60']
    assert any(e.type == 'line' and abs(e.start[1]-76) < 1 and math.dist(e.start,e.end)>205 for e in ir.entities)
    assert not any(e.type != 'text' and e.center and 55<e.center[0]<110 and 38<e.center[1]<70 for e in ir.entities)
    text = doc.modelspace().query('TEXT')[0]
    assert text.dxf.halign == 1 and text.dxf.valign == 2
    assert text.dxf.style == 'ENG_LATIN' and not doc.audit().has_errors
    assert all(e.dxf.lineweight <= 25 for e in doc.modelspace())


def test_rotated_ocr_crop_does_not_rotate_horizontal_text():
    ink = np.zeros((100, 150), np.uint8)
    cv2.putText(ink, 'R10', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    texts, _, _ = extract_annotations(ink, [dict(text='R10', score=.99,
        box=[16, 30, 95, 68], center=[55, 49], image_rotation=90)])
    assert len(texts) == 1 and texts[0].rotation == 0


def test_ocr_whole_label_wins_and_conflict_is_retained():
    items = [dict(text='3.2', score=.96, box=[10, 10, 65, 40], center=[37,25]),
             dict(text='2', score=.9999, box=[45, 10, 65, 40], center=[55,25])]
    result = select_annotations(items)
    assert len(result) == 1 and result[0]['text'] == '3.2'
    assert result[0]['alternatives'][0]['text'] == '2'


def test_intersection_is_two_continuous_lines_without_diagonal_shortcut():
    mask = np.zeros((81,81), bool)
    mask[40,10:71] = True; mask[10:71,40] = True
    entities = assemble_paths(continuous_paths(mask), [])
    assert len(entities)==2 and all(e.type=='line' for e in entities)
    assert sorted(math.dist(e.start,e.end) for e in entities)==[60,60]


def test_template_replacement_keeps_leader_attached():
    points = [(x, 30) for x in range(30,71)]+[(70,y) for y in range(31,61)]
    points += [(x,60) for x in range(69,29,-1)]+[(30,y) for y in range(59,29,-1)]
    leader = [(x,45) for x in range(70,100)]
    params=dict(center=[50,45],width=40,height=30)
    feature=Feature('rectangle',params,build('rectangle',params))
    ents=assemble_paths([points,leader],[feature])
    assert len(ents)==2
    line=next(e for e in ents if e.type=='line')
    assert math.dist(line.start,(70,45))<.8
    assert line.end==(99,45)


def test_scan_cracks_heal_but_real_holes_and_open_gaps_survive():
    ink=np.zeros((100,200),np.uint8)
    cv2.rectangle(ink,(10,10),(100,30),255,-1)
    ink[20,15:96]=0
    cv2.circle(ink,(150,50),20,255,5)
    ink[78:82,10:50]=255; ink[78:82,55:90]=255
    healed=heal_scan_strokes(ink)
    assert healed[20,15:96].all()
    assert healed[50,150]==0
    assert not healed[78:82,50:55].any()


def test_native_dashes_do_not_connect_separate_views(tmp_path):
    entities=[Entity(type='line',start=(x,30),end=(x+10,30))
              for x in [10,30,50,70,90,300,320,340,360,380]]
    assembled,patterns=assemble_centerlines(entities)
    assert len(assembled)==2 and len(patterns)==2
    assert max(math.dist(e.start,e.end) for e in assembled)==90
    ir=DrawingIR(width=450,height=80,entities=assembled,
        layers=[LayerSpec(name=k,color=1) for k in patterns],meta={'linetype_patterns':patterns})
    path=tmp_path/'dashes.dxf'; write_ir(ir,path)
    doc=ezdxf.readfile(path)
    assert all(doc.layers.get(e.dxf.layer).dxf.linetype!='Continuous' for e in doc.modelspace())
    assert not doc.audit().has_errors


def test_color_crossing_repair_does_not_close_unrelated_dash_gap():
    ink=np.zeros((50,100),np.uint8); ink[25,10:90]=255; ink[25,48:52]=0
    ink[10,10:40]=255; ink[10,44:90]=255
    color=np.zeros_like(ink); color[15:35,48:52]=255
    repaired=repair_color_crossings(ink,color)
    assert repaired[25,10:90].all()
    assert not repaired[10,40:44].any()
