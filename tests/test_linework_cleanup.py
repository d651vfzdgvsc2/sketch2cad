"""Regression checks: remove known noise without damaging real CAD details."""
import cv2
import numpy as np
import ezdxf

from emit.ir import DrawingIR,Entity
from engineering.linework import review_linework
from engineering.dimensions import dimension_candidates,candidate_fingerprint,plan_dimensions
from engineering.pipeline import write_ir
from test_native_review import dimension_fixture


def canvas():return np.full((180,280,3),255,np.uint8)


def line(a,b,layer='outline'):return Entity(type='line',start=a,end=b,layer=layer)


def test_compressed_dark_red_dash_caps_are_not_black_geometry():
    image=canvas();cv2.line(image,(50,60),(72,60),(25,25,78),3)
    ir=DrawingIR(width=280,height=180,entities=[line((50,60),(72,60),'centerline'),line((72,59),(72,61))])
    result=review_linework(ir,image)
    assert len(result.entities)==1 and result.entities[0].layer=='centerline'
    assert result.meta['linework_review']['removed'][0]['reason']=='black_fragment_from_colored_stroke'
    assert len(ir.entities)==2


def test_colored_crossing_keeps_a_real_black_branch():
    image=canvas();cv2.line(image,(20,60),(100,60),(25,25,78),3)
    cv2.line(image,(70,55),(70,65),(0,0,0),1)
    ir=DrawingIR(width=280,height=180,entities=[line((20,60),(100,60),'centerline'),line((70,55),(70,65))])
    assert len(review_linework(ir,image).entities)==2


def test_false_short_arc_on_straight_red_dash_is_removed():
    image=canvas();cv2.line(image,(65,60),(90,60),(25,25,78),3)
    arc=Entity(type='arc',center=(80,64),radius=4,start_angle=225,end_angle=315)
    ir=DrawingIR(width=280,height=180,entities=[line((65,60),(90,60),'centerline'),arc])
    result=review_linework(ir,image)
    assert len(result.entities)==1 and result.entities[0].type=='line'


def test_real_colored_arc_is_not_discarded_as_a_dash_cap():
    image=canvas();cv2.ellipse(image,(90,90),(5,5),0,180,360,(25,25,78),1)
    arc=Entity(type='arc',center=(90,90),radius=5,start_angle=180,end_angle=360)
    ir=DrawingIR(width=280,height=180,entities=[arc])
    assert len(review_linework(ir,image).entities)==1


def test_short_unlabelled_clean_line_is_retained():
    image=canvas();cv2.line(image,(130,110),(135,110),(0,0,0),1)
    ir=DrawingIR(width=280,height=180,entities=[line((130,110),(135,110))])
    result=review_linework(ir,image)
    assert len(result.entities)==1
    assert result.meta['linework_review']['retained_uncertain']


def test_compact_isolated_speck_away_from_annotations_is_removed():
    image=canvas();cv2.line(image,(130,110),(131,111),(0,0,0),1)
    ir=DrawingIR(width=280,height=180,entities=[line((130,110),(131,111))])
    assert not review_linework(ir,image).entities


def test_tiny_connector_and_small_unlabelled_rectangle_are_preserved():
    image=canvas()
    edges=[((50,50),(55,50)),((55,50),(55,51)),((55,51),(50,51)),((50,51),(50,50))]
    for a,b in edges:cv2.line(image,a,b,(0,0,0),1)
    ir=DrawingIR(width=280,height=180,entities=[line(a,b) for a,b in edges])
    assert len(review_linework(ir,image).entities)==4


def test_black_dashes_and_arbitrary_angle_hatching_are_preserved():
    image=canvas();entities=[]
    for x in (30,40,50,60):
        entities.append(line((x,60),(x+3,60)));cv2.line(image,(x,60),(x+3,60),(0,0,0),1)
    for x in (100,106,112,118):
        entities.append(line((x,100),(x+3,103)));cv2.line(image,(x,100),(x+3,103),(0,0,0),1)
    ir=DrawingIR(width=280,height=180,entities=entities)
    result=review_linework(ir,image)
    assert len(result.entities)==8


def test_small_hole_closed_contour_and_ocr_region_are_protected():
    image=canvas();cv2.circle(image,(90,90),2,(0,0,0),1)
    cv2.line(image,(130,110),(131,111),(0,0,0),1)
    entities=[Entity(type='circle',center=(90,90),radius=2),
              Entity(type='polyline',points=[(40,40),(42,40),(42,42),(40,42)],closed=True),
              line((130,110),(131,111)),Entity(type='text',content='1',pos=(130,110),height=8)]
    ir=DrawingIR(width=280,height=180,entities=entities,meta={'ocr':[dict(box=[128,108,134,115],text='1',score=.4)]})
    assert len(review_linework(ir,image).entities)==4


def test_terminal_raster_twig_is_removed_without_shortcutting_parent():
    image=canvas();cv2.line(image,(30,60),(170,60),(0,0,0),3)
    ir=DrawingIR(width=280,height=180,entities=[line((30,60),(170,60)),line((90,60),(90,61.4))])
    result=review_linework(ir,image)
    assert len(result.entities)==1
    assert result.entities[0].start==(30,60) and result.entities[0].end==(170,60)


def test_nearly_straight_path_becomes_one_native_line_with_original_ends():
    image=canvas();cv2.line(image,(20,50),(160,50),(0,0,0),3)
    original=Entity(type='polyline',points=[(20,50),(55,50.8),(90,49.8),(125,50.7),(160,50)])
    result=review_linework(DrawingIR(width=280,height=180,entities=[original]),image)
    assert result.entities[0].type=='line'
    assert result.entities[0].start==original.points[0] and result.entities[0].end==original.points[-1]


def test_step_and_dashed_gaps_are_never_straightened_or_bridged():
    image=canvas()
    step=Entity(type='polyline',points=[(20,60),(60,60),(60,65),(120,65)])
    for a,b in zip(step.points,step.points[1:]):cv2.line(image,tuple(map(round,a)),tuple(map(round,b)),(0,0,0),1)
    entities=[step,line((30,110),(65,110)),line((75,110),(110,110))]
    for e in entities[1:]:cv2.line(image,tuple(map(round,e.start)),tuple(map(round,e.end)),(0,0,0),1)
    result=review_linework(DrawingIR(width=280,height=180,entities=entities),image)
    assert [e.model_dump() for e in result.entities]==[e.model_dump() for e in entities]


def test_annotation_binding_survives_entity_renumbering_and_native_export(tmp_path):
    ir=dimension_fixture();ir.entities.insert(0,line((250,20),(251,21)))
    image=np.full((150,280,3),255,np.uint8)
    for e in ir.entities:
        if e.type=='line':cv2.line(image,tuple(map(round,e.start)),tuple(map(round,e.end)),(0,0,0),1)
    cs=dimension_candidates(ir)
    ir.meta['dimension_association']=dict(decisions={'0':cs[0]['id']},candidate_fingerprint=candidate_fingerprint(cs))
    result=review_linework(ir,image)
    assert len(result.entities)==len(ir.entities)-1
    assert plan_dimensions(result)[0]['association']=='validated_model_selection'
    path=tmp_path/'clean.dxf';r=write_ir(result,path)
    doc=ezdxf.readfile(path)
    assert r['native_dimensions']==1 and not r['errors']
    assert len(doc.modelspace().query('DIMENSION'))==1 and not doc.audit().has_errors


def test_stale_dimension_ids_are_not_made_valid_by_cleanup():
    ir=dimension_fixture();cs=dimension_candidates(ir)
    ir.meta['dimension_association']=dict(decisions={'0':'d0'},candidate_fingerprint=candidate_fingerprint(cs))
    ir.entities[0].end=(228,80)
    ir.entities.insert(0,line((250,20),(251,21)))
    image=np.full((150,280,3),255,np.uint8);cv2.line(image,(250,20),(251,21),(0,0,0),1)
    result=review_linework(ir,image)
    assert all(p['association']!='validated_model_selection' for p in plan_dimensions(result))


def test_native_centerline_linetype_and_colors_survive_cleanup():
    image=canvas();cv2.line(image,(40,80),(41,81),(0,0,0),1)
    entities=[line((20,60),(220,60),'centerline_000'),line((40,80),(41,81))]
    ir=DrawingIR(width=280,height=180,entities=entities,
        meta={'linetype_patterns':{'centerline_000':[15,10,-5]},'stroke_colors_rgb':[[180,0,0],None]})
    result=review_linework(ir,image)
    assert len(result.entities)==1 and result.entities[0].layer=='centerline_000'
    assert result.meta['stroke_colors_rgb']==[[180,0,0]]
    assert result.meta['linetype_patterns']==ir.meta['linetype_patterns']


def test_raw_colored_dash_end_is_preserved_even_inside_native_dash_stroke():
    image=canvas();cv2.line(image,(30,60),(170,60),(25,25,78),3)
    entities=[line((30,60),(170,60),'centerline_000'),line((90,60),(90,61.4),'centerline')]
    ir=DrawingIR(width=280,height=180,entities=entities,meta={'linetype_patterns':{'centerline_000':[15,10,-5]}})
    result=review_linework(ir,image)
    assert len(result.entities)==2
