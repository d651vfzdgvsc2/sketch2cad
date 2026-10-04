import math
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR, Entity
from engineering.review import review_geometry
from engineering.pipeline import write_ir
from engineering.ai_review import validate_reply
from engineering.dimensions import plan_dimensions


def test_circle_polyline_becomes_native_circle_with_raster_evidence():
    image=np.full((300,300,3),255,np.uint8)
    cv2.circle(image,(150,150),80,(0,0,0),2)
    pts=[(150+80*math.cos(a),150+80*math.sin(a)) for a in np.linspace(0,math.tau,65)[:-1]]
    ir=DrawingIR(width=300,height=300,entities=[Entity(type='polyline',points=pts,closed=True)])
    revised=review_geometry(ir,image)
    assert revised.entities[0].type=='circle'
    assert revised.entities[0].radius==pytest.approx(80,abs=.2)
    assert ir.entities[0].type=='polyline'  # input preserved for rollback


def test_regular_octagon_is_not_mistaken_for_circle():
    pts=[(150+80*math.cos(a),150+80*math.sin(a)) for a in np.linspace(0,math.tau,9)[:-1]]
    image=np.full((300,300,3),255,np.uint8)
    cv2.polylines(image,[np.array(pts,dtype=np.int32)],True,(0,0,0),2)
    ir=DrawingIR(width=300,height=300,entities=[Entity(type='polyline',points=pts,closed=True)])
    assert review_geometry(ir,image).entities[0].type=='polyline'


def test_partial_arc_does_not_invent_full_circle():
    image=np.full((300,300,3),255,np.uint8)
    cv2.ellipse(image,(150,150),(80,80),0,10,290,(0,0,0),2)
    pts=[(150+80*math.cos(a),150+80*math.sin(a)) for a in np.linspace(math.radians(10),math.radians(290),60)]
    ir=DrawingIR(width=300,height=300,entities=[Entity(type='polyline',points=pts)])
    revised=review_geometry(ir,image)
    assert revised.entities[0].type=='arc'
    assert revised.entities[0].end_angle-revised.entities[0].start_angle==pytest.approx(280,abs=.2)


def dimension_fixture():
    return DrawingIR(width=260,height=150,entities=[
        Entity(type='line',start=(30,80),end=(230,80)),
        Entity(type='line',start=(30,60),end=(30,100)),
        Entity(type='line',start=(230,60),end=(230,100)),
        Entity(type='text',content='100',pos=(130,62),height=12),
    ],meta={'annotations':[dict(text_index=0,score=.99,width_px=23)]})


def test_actual_dxf_contains_dimension_and_native_block_text(tmp_path):
    ir=dimension_fixture();path=tmp_path/'dimension.dxf'
    report=write_ir(ir,path)
    assert report['native_dimensions']==1 and not report['errors']
    doc=ezdxf.readfile(path);msp=doc.modelspace()
    assert len(msp.query('DIMENSION'))==1
    assert len(msp.query('TEXT'))==0
    assert len(msp.query('LINE'))==2  # original witnesses remain
    dimension=msp.query('DIMENSION')[0]
    assert dimension.dxf.text=='100'
    assert dimension.get_measurement()==pytest.approx(200) # no invented scale
    texts=list(doc.blocks.get(dimension.dxf.geometry).query('MTEXT'))
    assert len(texts)==1 and texts[0].plain_text()=='100'
    assert dimension.has_xdata('ENG_DIMENSION')
    assert not doc.audit().has_errors


def test_unlinked_numeric_text_is_not_forced_into_dimension(tmp_path):
    ir=dimension_fixture();del ir.entities[1]
    assert not plan_dimensions(ir)
    path=tmp_path/'unlinked.dxf';report=write_ir(ir,path)
    assert report['native_dimensions']==0
    assert [e.dxf.text for e in ezdxf.readfile(path).modelspace().query('TEXT')]==['100']


@pytest.mark.parametrize('payload',[
    {'items':[dict(id='unknown',text='60',is_text=True,confidence=.99)]},
    {'items':[dict(id='t0',text='60',is_text=True,confidence=float('nan'))]},
    {'items':[dict(id='t0',text='\\H10;60',is_text=True,confidence=.99)]},
    {'items':[]},
])
def test_model_reply_cannot_add_ids_or_cad_formatting(payload):
    with pytest.raises(ValueError):validate_reply(payload,{'t0'})


def test_failed_model_batch_preserves_original_ocr(tmp_path,monkeypatch):
    import tools.vlm
    from engineering.ai_review import review_annotations
    from tools.image_io import imwrite
    image=tmp_path/'source.png';imwrite(image,np.full((80,160,3),255,np.uint8))
    ocr=[dict(text='60',score=.99,box=[20,20,70,50],center=[45,35])]
    def unavailable(*a,**k):raise TimeoutError('unavailable')
    monkeypatch.setattr(tools.vlm,'ask_vision',unavailable)
    result,report=review_annotations(image,ocr,tmp_path/'review')
    assert result[0]['text']=='60' and report['status']=='partial'
    assert report['warnings']


def test_numeric_disagreement_needs_second_confirmation(tmp_path,monkeypatch):
    import json
    import tools.vlm
    from engineering.ai_review import review_annotations
    from tools.image_io import imwrite
    image=tmp_path/'source.png';imwrite(image,np.full((80,160,3),255,np.uint8))
    ocr=[dict(text='3.2',score=.99,box=[20,20,70,50],center=[45,35])]
    replies=iter([
        {'items':[dict(id='t0',text='32',is_text=True,confidence=.99)]},
        {'items':[dict(id='n0',text='3.2',is_text=True,confidence=.99)]},
    ])
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:json.dumps(next(replies)))
    result,report=review_annotations(image,ocr,tmp_path/'review')
    assert result[0]['text']=='3.2'
    assert result[0]['review_status']=='numeric_conflict'
    assert not report['records'][0]['applied']


def test_vertical_dimension_can_have_horizontal_text(tmp_path):
    ir=dimension_fixture()
    for e in ir.entities:
        if e.type=='line':
            e.start=(e.start[1],e.start[0]); e.end=(e.end[1],e.end[0])
    ir.width,ir.height=150,260
    ir.entities[-1].pos=(55,130)
    path=tmp_path/'vertical.dxf';report=write_ir(ir,path)
    assert report['native_dimensions']==1
    assert ezdxf.readfile(path).modelspace().query('DIMENSION')[0].dxf.text=='100'


def test_dimension_scaling_keeps_original_label(tmp_path):
    from engineering.calibration import scale_selected_dxf
    source,target=tmp_path/'source.dxf',tmp_path/'scaled.dxf'
    write_ir(dimension_fixture(),source)
    scale_selected_dxf(source,target,.5)
    doc=ezdxf.readfile(target)
    dimension=doc.modelspace().query('DIMENSION')[0]
    assert dimension.get_measurement()==pytest.approx(100)
    assert dimension.dxf.text=='100' and not doc.audit().has_errors
