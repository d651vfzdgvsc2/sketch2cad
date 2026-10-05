from pathlib import Path
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR,Entity
from engineering.pipeline import write_ir,run_engineering
from engineering.hatching import finalize_hatches,plan_hatches
from tools.image_io import imwrite
from test_dash_hatching import hatch_fixture,line
from test_native_review import dimension_fixture


def test_source_closed_boundary_recovers_fragmented_cad_outline(tmp_path):
    ir,image=hatch_fixture(hole=False)
    border=ir.entities.pop(0)
    ir.entities.extend(line(a,b) for a,b in zip(border.points,border.points[1:]+border.points[:1]))
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    selected=tmp_path/'selected.dxf';write_ir(ir,selected)
    assert len(ezdxf.readfile(selected).modelspace().query('HATCH'))==0
    final=tmp_path/'final.dxf';report=finalize_hatches(ir,selected,final)
    assert report['native_hatches']==1 and report['converted'][0]['boundary_source']=='source_closed_thick_contour'
    assert not ezdxf.readfile(final).audit().has_errors
    # The selected candidate is retained as the unfilled rollback/evidence file.
    assert len(ezdxf.readfile(selected).modelspace().query('HATCH'))==0


def test_open_source_boundary_is_never_closed_for_fill():
    ir,image=hatch_fixture(hole=False);border=ir.entities.pop(0)
    ir.entities.extend(line(a,b) for a,b in zip(border.points,border.points[1:]))
    cv2.line(image,(30,75),(30,100),(255,255,255),9)
    assert plan_hatches(ir,image)==[]


def test_small_thin_section_is_native_fill_not_individual_diagonals(tmp_path):
    image=np.full((100,200,3),255,np.uint8);border=[(30,30),(150,30),(150,56),(30,56)]
    entities=[line(a,b) for a,b in zip(border,border[1:]+border[:1])]
    for total in range(70,205,10):
        points=[]
        for x,y in [(30,total-30),(150,total-150),(total-30,30),(total-56,56)]:
            if 30<=x<=150 and 30<=y<=56 and (x,y) not in points:points.append((x,y))
        if len(points)!=2:continue
        entities.append(line(*points));cv2.line(image,points[0],points[1],(0,0,0),1)
    cv2.polylines(image,[np.array(border)],True,(0,0,0),3)
    ir=DrawingIR(width=200,height=100,entities=entities)
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    selected=tmp_path/'selected.dxf';write_ir(ir,selected)
    report=finalize_hatches(ir,selected,tmp_path/'final.dxf')
    assert report['native_hatches']==1 and report['replaced_edges']>=8


def test_final_hatch_keeps_native_dimension_identity_and_draws_behind(tmp_path):
    ir,image=hatch_fixture(hole=False);dims=dimension_fixture()
    for e in dims.entities:
        if e.type=='line':e.start=(e.start[0],e.start[1]+200);e.end=(e.end[0],e.end[1]+200)
        else:e.pos=(e.pos[0],e.pos[1]+200)
    ir.entities.extend(dims.entities);ir.height=380;ir.meta.update(dims.meta)
    image=cv2.copyMakeBorder(image,0,150,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    selected=tmp_path/'selected.dxf';write_ir(ir,selected)
    before=ezdxf.readfile(selected);dim=before.modelspace().query('DIMENSION')[0]
    final=tmp_path/'final.dxf';finalize_hatches(ir,selected,final);after=ezdxf.readfile(final)
    unchanged=after.modelspace().query('DIMENSION')[0]
    assert unchanged.dxf.handle==dim.dxf.handle and unchanged.dxf.geometry==dim.dxf.geometry
    assert unchanged.dxfattribs()==dim.dxfattribs()
    order=dict(after.modelspace().get_redraw_order());hatch=after.modelspace().query('HATCH')[0]
    assert int(order[hatch.dxf.handle],16)<int(order[unchanged.dxf.handle],16)
    assert list(after.modelspace())[-1].dxftype()=='HATCH'
    with pytest.raises(ValueError):finalize_hatches(ir,final,tmp_path/'double.dxf')


def test_pipeline_fills_only_once_after_local_repair(tmp_path,monkeypatch):
    import engineering.pipeline as pipeline
    import engineering.local_repair as local
    calls=[];ir,image=hatch_fixture(hole=False)
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    monkeypatch.setattr(pipeline,'build_proposals',lambda *a,**k:({'CLEAN':ir},[],{'annotations':[],'ocr':[]}))
    for name in ('review_geometry','review_linework','review_artifacts','review_dash_axes'):
        monkeypatch.setattr(pipeline,name,lambda ir,*a,**k:ir)
    ir.meta['geometry_review']={}
    def repair(image,ir,selected,*a,**k):
        assert len(ezdxf.readfile(selected['dxf']).modelspace().query('HATCH'))==0
        calls.append('repair');return ir,selected,{'status':'checked'}
    monkeypatch.setattr(local,'repair_selected',repair)
    finalize=pipeline.finalize_hatches
    def counted(*a,**k):calls.append('fill');return finalize(*a,**k)
    monkeypatch.setattr(pipeline,'finalize_hatches',counted)
    result=run_engineering(source,tmp_path/'out',use_ocr=False,use_ai_review=False)
    assert calls==['repair','fill']
    assert len(ezdxf.readfile(result['delivery']['dxf']).modelspace().query('HATCH'))==1
    assert len(list(Path(result['delivery']['directory']).iterdir()))==1


def test_hatch_tracing_arc_is_consumed_but_real_round_feature_survives(tmp_path):
    ir,image=hatch_fixture(hole=False)
    # This shallow arc follows the same already-measured hatch stripe; it is
    # duplicate hatch geometry. A circular feature is an independent contour.
    a=np.array([75.,115.]);b=np.array([115.,75.]);mid=(a+b)/2
    normal=np.array([1.,1.])/np.sqrt(2);radius=1000.
    half=np.linalg.norm(b-a)/2;center=mid+normal*np.sqrt(radius**2-half**2)
    aa=np.degrees(np.arctan2(*(a-center)[::-1]));bb=np.degrees(np.arctan2(*(b-center)[::-1]))
    ir.entities.append(Entity(type='arc',center=tuple(center),radius=radius,start_angle=float(min(aa,bb)),end_angle=float(max(aa,bb))))
    ir.entities.append(Entity(type='circle',center=(110,110),radius=22))
    cv2.circle(image,(110,110),22,(0,0,0),2)
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    selected=tmp_path/'selected.dxf';write_ir(ir,selected)
    final=tmp_path/'final.dxf';report=finalize_hatches(ir,selected,final)
    assert report['unsupported_hatch_arcs_removed']==1
    assert len(ezdxf.readfile(final).modelspace().query('CIRCLE'))==1


def test_crossed_patterns_are_retained_until_both_families_supported():
    ir,image=hatch_fixture(hole=False)
    for shift in range(-140,150,14):
        points=[]
        for x,y in [(30,30+shift),(190,190+shift),(30-shift,30),(190-shift,190)]:
            if 30<=x<=190 and 30<=y<=190 and (x,y) not in points:points.append((x,y))
        if len(points)!=2:continue
        ir.entities.append(line(*points));cv2.line(image,points[0],points[1],(0,0,0),1)
    assert plan_hatches(ir,image)==[]


def test_one_mixed_polyline_can_supply_two_fills_without_losing_connector(tmp_path):
    ir,image=hatch_fixture(hole=False);other,second=hatch_fixture(hole=False)
    wide=np.full((230,520,3),255,np.uint8);wide[:,:300]=image;wide[:,220:]=np.minimum(wide[:,220:],second)
    for e in other.entities:
        if e.type=='line':e.start=(e.start[0]+220,e.start[1]);e.end=(e.end[0]+220,e.end[1])
        elif e.type=='polyline':e.points=[(p[0]+220,p[1]) for p in e.points]
    a=ir.entities.pop(8);b=other.entities.pop(8)
    ir.entities.extend(other.entities)
    ir.entities.append(Entity(type='polyline',points=[a.start,a.end,b.start,b.end]))
    ir.width=520;source=tmp_path/'source.png';imwrite(source,wide);ir.meta['source']=str(source)
    selected=tmp_path/'selected.dxf';write_ir(ir,selected)
    final=tmp_path/'final.dxf';report=finalize_hatches(ir,selected,final)
    assert report['native_hatches']==2
    doc=ezdxf.readfile(final)
    start=(a.end[0],ir.height-a.end[1],0);end=(b.start[0],ir.height-b.start[1],0)
    assert any(e.dxf.start.isclose(start) and e.dxf.end.isclose(end) for e in doc.modelspace().query('LINE'))
