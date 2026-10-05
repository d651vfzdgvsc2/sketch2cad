import math
from pathlib import Path
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR,Entity
from engineering.text import extract_annotations,promote_consensus,select_annotations
from engineering.pipeline import write_ir,score_dxf
from engineering.local_repair import trim_glyph_geometry,apply_text_patch,reread_regions,repair_selected,supported_discoveries,table_text_regions,confirm_local_labels
from engineering.dimensions import dimension_candidates,candidate_fingerprint
from engineering.delivery import publish_result
from tools.image_io import imwrite


def choose(ir,predicate):
    candidates=dimension_candidates(ir)
    decisions={}
    for c in candidates:
        if predicate(c):decisions[str(c['text_index'])]=c['id']
    assert decisions
    ir.meta['dimension_association']=dict(decisions=decisions,candidate_fingerprint=candidate_fingerprint(candidates))


def test_adjacent_dimensions_share_a_source_line_without_duplicate_ownership(tmp_path):
    ir=DrawingIR(width=240,height=140,entities=[
        Entity(type='line',start=(20,80),end=(220,80)),
        *[Entity(type='line',start=(x,65),end=(x,110)) for x in (20,100,220)],
        Entity(type='text',content='20',pos=(60,60),height=10),
        Entity(type='text',content='30',pos=(160,60),height=10)],
        meta={'annotations':[dict(text_index=i,score=.99,width_px=18) for i in range(2)]})
    choose(ir,lambda c:c.get('intermediate_witness_partition'))
    path=tmp_path/'adjacent.dxf';r=write_ir(ir,path)
    assert r['native_dimensions']==2 and not r['errors']
    doc=ezdxf.readfile(path)
    assert [d.dxf.text for d in doc.modelspace().query('DIMENSION')]==['20','30']
    assert not doc.modelspace().query('TEXT LINE')
    assert not doc.audit().has_errors


@pytest.mark.parametrize('kind,label,expected_type,expected_measurement',[
    ('radius','R40',4,40),('diameter','Ø80',3,80),('angular','45°',2,45)])
def test_native_dimension_types_have_measured_geometry(tmp_path,kind,label,expected_type,expected_measurement):
    if kind=='angular':
        entities=[Entity(type='arc',center=(100,100),radius=40,start_angle=0,end_angle=45),
            Entity(type='line',start=(100,100),end=(170,100)),
            Entity(type='line',start=(100,100),end=(150,150))]
        position=(136,116)
    else:
        entities=[Entity(type='circle',center=(100,100),radius=40),
            Entity(type='line',start=(140,100) if kind=='radius' else (60,100),end=(195,100) if kind=='radius' else (140,100))]
        position=(173,86) if kind=='radius' else (100,85)
    entities.append(Entity(type='text',content=label,pos=position,height=10))
    ir=DrawingIR(width=260,height=200,entities=entities,meta={'annotations':[dict(text_index=0,score=.99,width_px=26)]})
    choose(ir,lambda c:c['kind']==kind)
    path=tmp_path/f'{kind}.dxf';r=write_ir(ir,path)
    assert r['native_dimensions']==1 and not r['errors']
    doc=ezdxf.readfile(path);dim=doc.modelspace().query('DIMENSION')[0]
    assert dim.dimtype==expected_type and dim.dxf.text==label
    assert dim.get_measurement()==pytest.approx(expected_measurement)
    if kind=='radius':assert tuple(dim.dxf.defpoint4)[:2]==(140,100)
    if kind=='diameter':assert {tuple(dim.dxf.defpoint)[:2],tuple(dim.dxf.defpoint4)[:2]}=={(60,100),(140,100)}
    assert not doc.modelspace().query('TEXT')
    assert not doc.audit().has_errors
    if kind!='angular':assert len(doc.modelspace().query('CIRCLE'))==1


def test_touching_glyph_can_be_native_without_erasing_diagonal_contour():
    ink=np.zeros((120,200),np.uint8)
    cv2.line(ink,(10,100),(180,15),255,1)
    cv2.putText(ink,'60',(70,65),cv2.FONT_HERSHEY_SIMPLEX,.6,255,1)
    texts,mask,_=extract_annotations(ink,[dict(text='60',score=.99,box=[66,47,97,68],review_status='ai_reviewed')])
    assert texts[0].content=='60' and mask.any()
    assert mask[60,90]==0 # measured diagonal remains protected


def test_glyph_removal_keeps_both_continuations_and_small_hole():
    mask=np.zeros((100,200),np.uint8);mask[45:56,80:101]=255
    ir=DrawingIR(width=200,height=100,entities=[
        Entity(type='line',start=(20,50),end=(170,50)),
        Entity(type='circle',center=(60,70),radius=3)])
    new,removed=trim_glyph_geometry(ir,mask)
    assert removed==1 and [e.type for e in new.entities]==['line','line','circle']
    assert new.entities[0].start==(20,50) and new.entities[1].end==(170,50)
    assert len(ir.entities)==2


def test_text_repair_cannot_erase_graphics_already_owned_by_native_dimension():
    mask=np.zeros((100,200),np.uint8);mask[45:56,80:101]=255
    ir=DrawingIR(width=200,height=100,entities=[Entity(type='line',start=(20,50),end=(170,50))],
        meta={'native_dimensions':{'converted':[{'line_parts':[{'entity':0,'edge':None}],'owned_graphics':[]}]}})
    repaired,removed=trim_glyph_geometry(ir,mask)
    assert removed==0 and repaired.entities==ir.entities


def test_new_local_label_is_added_once_existing_text_stays_fixed():
    image=np.full((100,200,3),255,np.uint8)
    cv2.putText(image,'60',(70,60),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
    item=dict(text='60',score=.99,box=[66,42,100,65],review_status='ai_reviewed')
    ir=DrawingIR(width=200,height=100,entities=[])
    a,report=apply_text_patch(ir,image,[item]);b,_=apply_text_patch(a,image,[item])
    assert report['added_labels']==['60']
    assert [e.content for e in b.entities if e.type=='text']==['60']
    assert a.entities[0].pos==b.entities[0].pos


def test_rotated_local_detection_maps_to_source_and_cannot_overwrite_existing(tmp_path):
    image=np.full((100,200,3),255,np.uint8)
    regions=[dict(id='g0',box=[100,20,140,60],component_count=1)]
    def engine(picture):
        return ([[[(30,30),(60,30),(60,60),(30,60)],'12',.99]],None)
    detections,_=reread_regions(image,regions,[],tmp_path,engine)
    assert detections[0]['box']==[110,30,120,40]
    assert not reread_regions(image,regions,[dict(box=[100,20,140,60])],tmp_path,engine)[0]


def test_result_folder_contains_only_final_calibrated_cad_and_refuses_overwrite(tmp_path):
    pixels=tmp_path/'pixels.dxf';mm=tmp_path/'mm.dxf'
    a=ezdxf.new();a.modelspace().add_circle((0,0),20);a.saveas(pixels)
    b=ezdxf.new();b.units=ezdxf.units.MM;b.modelspace().add_circle((0,0),5);b.saveas(mm)
    folder=tmp_path/'result';delivery=publish_result({'dxf':str(pixels)},{'dxf':str(mm)},folder,'drawing')
    assert list(folder.iterdir())==[Path(delivery['dxf'])]
    assert ezdxf.readfile(delivery['dxf']).modelspace().query('CIRCLE')[0].dxf.radius==5
    with pytest.raises(FileExistsError):publish_result({'dxf':str(pixels)},{},folder,'drawing')
    assert ezdxf.readfile(delivery['dxf']).modelspace().query('CIRCLE')[0].dxf.radius==5


def test_repair_failure_to_improve_retains_original_without_cloud(tmp_path,monkeypatch):
    import tools.vlm
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:pytest.fail('offline model call'))
    image=np.full((120,200,3),255,np.uint8);cv2.rectangle(image,(30,30),(160,90),(0,0,0),1)
    source=tmp_path/'source.png';imwrite(source,image)
    ir=DrawingIR(width=200,height=120,entities=[Entity(type='polyline',points=[(30,30),(160,30),(160,90),(30,90)],closed=True)])
    path=tmp_path/'before.dxf';write_ir(ir,path)
    score=score_dxf(path,source,tmp_path/'before.png');score['proposal']='CLEAN'
    revised,selected,report=repair_selected(source,ir,score,tmp_path/'repair',False,use_ocr=False)
    assert report['status']=='retained_original'
    assert revised is ir and selected is score


def test_split_measured_leader_becomes_one_native_object(tmp_path):
    ir=DrawingIR(width=240,height=160,entities=[
        Entity(type='circle',center=(50,100),radius=20),
        Entity(type='line',start=(70,100),end=(100,75)),
        Entity(type='line',start=(100,75),end=(180,75)),
        Entity(type='text',content='R20',pos=(140,60),height=12)],
        meta={'annotations':[dict(text_index=0,score=.99,width_px=32)]})
    path=tmp_path/'leader.dxf';r=write_ir(ir,path)
    assert r['native_leaders']==1
    msp=ezdxf.readfile(path).modelspace()
    assert not msp.query('LINE TEXT') and len(msp.query('MULTILEADER'))==1


def test_small_scan_tip_displacement_keeps_arrow_evidence():
    from engineering.dimension_graphics import arrow_size
    image=np.full((120,240,3),255,np.uint8)
    cv2.line(image,(20,61),(220,61),(0,0,0),1)
    for pts in ([[20,61],[35,57],[35,65]],[[220,61],[205,57],[205,65]]):
        cv2.fillPoly(image,[np.array(pts,np.int32)],(0,0,0))
    plan=dict(p1=[20,60],p2=[220,60],text_height=25)
    assert arrow_size(plan,image)>0


def test_two_model_guesses_cannot_replace_local_ocr_with_neighboring_note():
    candidate=dict(text='11',score=.6,box=[400,800,420,810],discovery='local_gap')
    reviewed=dict(candidate,text='M8',score=.99,review_status='ai_reviewed',numeric_confirmation={'accepted':True})
    assert not supported_discoveries([candidate],[reviewed],[])
    matching=dict(reviewed,text='11')
    assert supported_discoveries([candidate],[matching],[])==[matching]


def test_model_only_circle_region_is_never_added_as_native_text():
    reviewed=dict(text='Ø40',score=.99,box=[10,10,40,40],discovery='local_gap',review_status='ai_reviewed')
    assert not supported_discoveries([], [reviewed], [])


def test_table_cell_search_finds_faint_text_and_excludes_known_label():
    image=np.full((150,300,3),255,np.uint8)
    for y in (60,100,140):cv2.line(image,(20,y),(280,y),(0,0,0),1)
    for x in (20,100,180,280):cv2.line(image,(x,60),(x,140),(0,0,0),1)
    cv2.putText(image,'NAME',(30,86),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
    cv2.putText(image,'AB',(200,86),cv2.FONT_HERSHEY_SIMPLEX,.4,(170,170,170),1)
    records=[dict(text_index=0,box=[28,70,80,90])]
    regions=table_text_regions(image,records)
    assert regions and all(r['box'][0]>=180 for r in regions)
    assert any(r['box'][0]<220 and r['box'][1]<86 for r in regions)


def test_local_confirmation_needs_two_agreeing_blind_crop_reads(tmp_path,monkeypatch):
    import json,tools.vlm
    candidate=dict(text='AB',score=.8,box=[20,20,60,40],discovery='local_gap')
    replies=iter(['AB','AC'])
    monkeypatch.setattr(tools.vlm,'ask_vision',lambda *a,**k:json.dumps({'items':[
        dict(id='l0',text=next(replies),is_text=True,confidence=.99)]}))
    image=np.full((80,160,3),255,np.uint8)
    reviewed,report=confirm_local_labels(image,[candidate],tmp_path)
    assert not reviewed and not report['records'][0]['agreed']


def test_consistent_low_score_numerals_are_native_eligible_without_changing_text():
    item=dict(text='50',score=.6,box=[20,20,60,40],crop_readings=[
        dict(text='50',score=.6,channel='binary'),dict(text='50',score=.7,channel='clean')])
    promoted=promote_consensus([item])
    assert promoted[0]['text']=='50' and promoted[0]['score']==.9
    assert select_annotations(promoted) and item['score']==.6
    assert promoted[0]['ocr_consensus']['channels']==['binary','clean','original']


def test_repeated_same_crop_does_not_fake_consensus_and_conflicting_digits_block_it():
    item=dict(text='50',score=.6,box=[20,20,60,40],crop_readings=[dict(text='50',score=.7)]*5)
    assert promote_consensus([item])[0]['score']==.6
    item['crop_readings']=[dict(text='50',score=.6,channel='binary'),dict(text='50',score=.7,channel='clean'),dict(text='60',score=.9)]
    assert promote_consensus([item])[0]['score']==.6


def test_unconfirmed_model_change_keeps_literal_ocr_consensus_eligible(tmp_path,monkeypatch):
    import json,tools.vlm
    from engineering.ai_review import review_annotations
    source=tmp_path/'source.png';imwrite(source,np.full((80,160,3),255,np.uint8))
    item=dict(text='50',score=.9,box=[20,20,60,40],review_status='ocr_consensus',ocr_consensus={'channels':['original','binary','clean']})
    replies=iter(['60','50'])
    def response(*a,**k):
        text=next(replies);ident='t0' if text=='60' else 'n0'
        return json.dumps({'items':[dict(id=ident,text=text,is_text=True,confidence=.99)]})
    monkeypatch.setattr(tools.vlm,'ask_vision',response)
    reviewed,_=review_annotations(source,[item],tmp_path/'review')
    assert reviewed[0]['text']=='50' and reviewed[0]['review_status']=='ocr_consensus'
    assert select_annotations(reviewed)
