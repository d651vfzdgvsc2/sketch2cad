import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR,Entity
from engineering.pipeline import write_ir
from engineering.artifacts import review_artifacts
from engineering.dimension_graphics import remaining_intervals
from test_native_review import dimension_fixture


def test_whole_dimension_is_single_selectable_and_regenerates(tmp_path):
    path=tmp_path/'complete.dxf';report=write_ir(dimension_fixture(),path)
    doc=ezdxf.readfile(path);dim=doc.modelspace().query('DIMENSION')[0]
    assert len(doc.modelspace())==1 # no detached numeric text or witness lines
    assert report['owned_extension_segments']==2
    assert len(list(dim.virtual_entities()))>=4
    override=dim.override()
    assert override.get('dimse1')==0 and override.get('dimse2')==0
    override.render() # CAD regeneration preserves real extension definitions
    block=doc.blocks.get(dim.dxf.geometry)
    assert len(block.query('LINE'))>=3
    assert len(block.query('MTEXT'))==1
    assert not doc.audit().has_errors


def test_shared_witness_does_not_erase_rectangle_contour(tmp_path):
    ir=DrawingIR(width=200,height=150,entities=[
        Entity(type='polyline',points=[(30,30),(160,30),(160,110),(30,110)],closed=True),
        Entity(type='line',start=(10,30),end=(30,30)),
        Entity(type='line',start=(10,110),end=(30,110)),
        Entity(type='line',start=(15,30),end=(15,110)),
        Entity(type='text',content='80',pos=(5,70),height=6,rotation=90),
    ],meta={'annotations':[dict(text_index=0,score=.99,width_px=10)]})
    path=tmp_path/'shared.dxf';r=write_ir(ir,path)
    doc=ezdxf.readfile(path)
    assert r['native_dimensions']==1
    assert len(doc.modelspace().query('LWPOLYLINE'))==1
    assert len(doc.modelspace().query('LINE'))==0
    assert len(doc.modelspace().query('DIMENSION'))==1
    assert not doc.audit().has_errors


def test_shared_witness_continuation_is_retained(tmp_path):
    ir=dimension_fixture()
    # Witness crosses the part-edge junction at y=100 and continues to 140.
    ir.entities[1].end=(30,140);ir.entities[2].end=(230,140)
    ir.entities.insert(3,Entity(type='line',start=(30,100),end=(230,100)))
    path=tmp_path/'tail.dxf';r=write_ir(ir,path)
    doc=ezdxf.readfile(path)
    assert r['native_dimensions']==1
    lines=list(doc.modelspace().query('LINE'))
    assert len(lines)==3 # part base and both shared vertical tails
    assert sum(abs(e.dxf.start.x-e.dxf.end.x)<1e-6 for e in lines)==2
    assert any(g['preserves_shared_remainder'] for g in r['converted'][0]['owned_graphics'])


def test_overlapping_consumption_keeps_only_unclaimed_ranges():
    assert remaining_intervals([[.2,.5],[.4,.8]])==[(0.,.2),(.8,1.)]


def test_native_multileader_contains_note_and_leader_in_one_entity(tmp_path):
    ir=DrawingIR(width=240,height=150,entities=[
        Entity(type='circle',center=(50,90),radius=20),
        Entity(type='polyline',points=[(70,90),(100,70),(180,70)]),
        Entity(type='text',content='R20',pos=(140,58),height=12),
    ],meta={'annotations':[dict(text_index=0,score=.99,width_px=32)]})
    path=tmp_path/'leader.dxf';r=write_ir(ir,path)
    assert r['native_leaders']==1 and not r['leaders']['errors']
    doc=ezdxf.readfile(path);msp=doc.modelspace()
    assert len(msp.query('CIRCLE'))==1
    assert not msp.query('TEXT LWPOLYLINE LINE')
    leader=msp.query('MULTILEADER')[0]
    assert 'R20' in leader.context.mtext.default_content
    assert leader.context.leaders[0].lines[0].vertices[0].x==70
    assert leader.context.leaders[0].lines[0].vertices[-1].x==180
    assert leader.context.leaders[0].last_leader_point.x==180
    assert not doc.audit().has_errors


def test_artifact_review_preserves_hatching_arrows_small_holes_and_text():
    image=np.full((120,200,3),255,np.uint8)
    cv2.line(image,(20,30),(40,10),(0,0,0),1)
    cv2.circle(image,(60,40),3,(0,0,0),1)
    line=Entity(type='line',start=(20,30),end=(40,10))
    ir=DrawingIR(width=200,height=120,entities=[line,line.model_copy(),
        Entity(type='circle',center=(60,40),radius=3),
        Entity(type='line',start=(120,90),end=(130,90)),
        Entity(type='text',content='1',pos=(150,90),height=8)])
    result=review_artifacts(ir,image)
    assert [e.type for e in result.entities]==['line','circle','text']
    assert len(result.meta['artifact_review']['removed'])==2
    assert len(ir.entities)==5


def test_native_filled_arrows_are_in_dimension_block(tmp_path):
    from tools.image_io import imwrite
    image=np.full((150,260,3),255,np.uint8)
    cv2.line(image,(30,80),(230,80),(0,0,0),1)
    for pts in ([[30,80],[38,77],[38,83]],[[230,80],[222,77],[222,83]]):
        cv2.fillPoly(image,[np.array(pts,np.int32)],(0,0,0))
    source=tmp_path/'source.png';imwrite(source,image)
    ir=dimension_fixture();ir.meta['source']=str(source)
    path=tmp_path/'arrows.dxf';r=write_ir(ir,path)
    assert r['converted'][0]['native_arrow_size_px']>0
    doc=ezdxf.readfile(path);dim=doc.modelspace().query('DIMENSION')[0]
    assert len(doc.blocks.get(dim.dxf.geometry).query('INSERT'))==2


def test_revised_geometry_does_not_reuse_stale_numbered_model_choices():
    from engineering.dimensions import candidate_fingerprint,dimension_candidates,plan_dimensions
    ir=dimension_fixture();c=dimension_candidates(ir)
    ir.meta['dimension_association']=dict(decisions={'0':'d0'},candidate_fingerprint=candidate_fingerprint(c))
    assert plan_dimensions(ir)[0]['association']=='validated_model_selection'
    ir.entities[0].end=(228,80)
    # A still-valid local dimension may survive; the old model decision cannot.
    assert all(p['association']!='validated_model_selection' for p in plan_dimensions(ir))


def test_plain_dimension_stroke_does_not_invent_filled_arrows():
    from engineering.dimension_graphics import arrow_size
    from engineering.dimensions import plan_dimensions
    image=np.full((150,260,3),255,np.uint8)
    cv2.line(image,(30,80),(230,80),(0,0,0),2)
    plan=plan_dimensions(dimension_fixture())[0]
    assert arrow_size(plan,image)==0.
