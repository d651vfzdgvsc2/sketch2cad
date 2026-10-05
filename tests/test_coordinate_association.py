"""Engineering coordinate, missing-text and semantic-association regressions."""
import json
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR, Entity
from engineering.coordinates import image_to_cad
from engineering.text import extract_annotations, map_crop_quad, select_annotations
from engineering.association import validate_associations, review_dimension_associations
from engineering.dimensions import dimension_candidates, plan_dimensions
from engineering.pipeline import write_ir
from engineering.review import review_geometry
from test_native_review import dimension_fixture


def test_detector_anchor_is_hard_bound_despite_asymmetric_glyphs(tmp_path):
    ink=np.zeros((100,200),np.uint8)
    cv2.putText(ink,'60',(25,55),cv2.FONT_HERSHEY_SIMPLEX,.6,255,1)
    texts,_,records=extract_annotations(ink,[dict(text='60',score=.99,box=[20,30,80,62],center=[999,999])])
    assert texts[0].pos==(50,46) # center supplied by model/input is irrelevant
    ir=DrawingIR(width=200,height=100,entities=texts,meta={'annotations':records})
    path=tmp_path/'anchors.dxf';write_ir(ir,path)
    text=ezdxf.readfile(path).modelspace().query('TEXT')[0]
    assert tuple(text.dxf.align_point)[:2]==image_to_cad((50,46),100)==(50,54)


def test_enlarged_rotated_crop_maps_back_without_model_coordinates():
    # Original crop (100x60), enlarged to (200x120), rotated CCW.
    quad=map_crop_quad([(20,140),(20,100),(60,100),(60,140)],1,200,120,300,400,2,2)
    assert quad.tolist()==[[329.5,410],[349.5,410],[349.5,430],[329.5,430]]


def test_new_crop_detection_requires_confirmed_text_before_erasing_ink():
    candidate=dict(text='0',score=.999,box=[20,20,40,40],center=[30,30],discovery='enlarged_tile')
    assert not select_annotations([candidate]) # could be a hole
    assert select_annotations([{**candidate,'review_status':'ai_reviewed'}])


@pytest.mark.parametrize('update',[
    dict(candidate_id='unknown'),dict(text_id='t3'),dict(confidence=float('nan')),
    dict(p1=[999,999]),dict(confidence=True),
])
def test_model_cannot_change_coordinates_or_choose_foreign_ids(update):
    c=dimension_candidates(dimension_fixture())
    reply=dict(text_id='t0',candidate_id='d0',confidence=.99,reason='clear');reply.update(update)
    with pytest.raises(ValueError):validate_associations({'items':[reply]},c)


def test_semantic_selection_is_ids_only_and_rejects_stale_plan():
    ir=dimension_fixture();c=dimension_candidates(ir)
    ir.meta['dimension_association']={'decisions':validate_associations({'items':[
        dict(text_id='t0',candidate_id='d0',confidence=.99,reason='clear')]},c)}
    plan=plan_dimensions(ir)[0]
    assert plan['p1']==[30,80] and plan['association']=='validated_model_selection'
    ir.meta['dimension_association']['decisions']['0']='unknown'
    assert not plan_dimensions(ir)


def test_dimension_review_failure_preserves_rule_based_export(tmp_path,monkeypatch):
    import tools.vlm
    from tools.image_io import imwrite
    source=tmp_path/'source.png';imwrite(source,np.full((150,260,3),255,np.uint8))
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:'{"items": []}')
    ir=dimension_fixture()
    ir.meta['dimension_association']=review_dimension_associations(source,ir,tmp_path/'review')
    assert ir.meta['dimension_association']['status']=='partial'
    assert len(plan_dimensions(ir))==1


def test_collinear_fragments_merge_but_steps_and_dashes_stay():
    image=np.full((100,200,3),255,np.uint8)
    cv2.line(image,(10,20),(130,20),(0,0,0),1)
    cv2.line(image,(10,40),(50,40),(0,0,0),1);cv2.line(image,(55,40),(100,40),(0,0,0),1)
    step=[(10,60),(50,60),(50,63),(100,63)]
    cv2.polylines(image,[np.array(step,np.int32)],False,(0,0,0),1)
    ir=DrawingIR(width=200,height=100,entities=[
        Entity(type='polyline',points=[(10,20),(30,20),(50,20)]),
        Entity(type='line',start=(50,20),end=(130,20)),
        Entity(type='line',start=(10,40),end=(50,40)),
        Entity(type='line',start=(55,40),end=(100,40)),
        Entity(type='polyline',points=step)])
    result=review_geometry(ir,image)
    assert len(result.entities)==4
    assert result.entities[0].type=='line' and result.entities[0].start==(10,20) and result.entities[0].end==(130,20)
    assert result.entities[-1].type=='polyline' and result.entities[-1].points==step
    assert len(ir.entities)==5


def test_discovered_text_needs_two_agreeing_reads(tmp_path,monkeypatch):
    import tools.vlm
    from tools.image_io import imwrite
    from engineering.ai_review import review_annotations
    source=tmp_path/'source.png';imwrite(source,np.full((150,260,3),255,np.uint8))
    item=dict(text='60',score=.6,box=[20,20,60,40],center=[40,30],discovery='enlarged_tile')
    replies=iter([{'items':[dict(id='t0',text='60',confidence=.99,is_text=True)]},
                  {'items':[dict(id='n0',text='80',confidence=.99,is_text=True)]}])
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:json.dumps(next(replies)))
    result,_=review_annotations(source,[item],tmp_path/'review')
    assert not select_annotations(result)
    assert result[0]['review_status']=='numeric_conflict'


def test_dimension_line_split_around_text_exports_one_native_dimension(tmp_path):
    ir=dimension_fixture()
    ir.entities[0]=Entity(type='line',start=(30,80),end=(112,80))
    ir.entities.insert(1,Entity(type='line',start=(148,80),end=(230,80)))
    ir.entities[-1].pos=(130,80)
    path=tmp_path/'split.dxf';report=write_ir(ir,path)
    assert report['native_dimensions']==1
    doc=ezdxf.readfile(path)
    assert len(doc.modelspace().query('LINE'))==0 # entire annotation is owned
    assert doc.modelspace().query('DIMENSION')[0].get_measurement()==pytest.approx(200)
    assert not doc.audit().has_errors


def test_contextual_majority_can_recover_decimal_without_guessing(tmp_path,monkeypatch):
    import tools.vlm
    from tools.image_io import imwrite
    from engineering.ai_review import review_annotations
    source=tmp_path/'source.png';imwrite(source,np.full((150,260,3),255,np.uint8))
    item=dict(text='3.2 #',score=.65,box=[20,20,60,40],center=[40,30])
    replies=iter([{'items':[dict(id='t0',text='3-2',confidence=.99,is_text=True)]},
                  {'items':[dict(id='n0',text='3.2',confidence=.99,is_text=True)]},
                  {'items':[dict(id='n0',text='3.2',confidence=.99,is_text=True)]}])
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:json.dumps(next(replies)))
    result,_=review_annotations(source,[item],tmp_path/'review')
    assert select_annotations(result)[0]['text']=='3.2'
    assert result[0]['numeric_confirmation']['third_read']['text']=='3.2'


def test_partial_association_keeps_valid_rows_but_still_rejects_unknown_ids():
    ir=dimension_fixture()
    ir.entities.append(Entity(type='text',content='100',pos=(130,98),height=12))
    ir.meta['annotations'].append(dict(text_index=1,score=.99,width_px=23))
    c=dimension_candidates(ir)
    reply={'items':[dict(text_id='t0',candidate_id='d0',confidence=.99,reason='clear')]}
    assert validate_associations(reply,c,allow_partial=True)=={'0':'d0'}
    with pytest.raises(ValueError):validate_associations(reply,c)
    reply['items'][0]['candidate_id']='unknown'
    with pytest.raises(ValueError):validate_associations(reply,c,allow_partial=True)
