import math
import cv2
import ezdxf
import numpy as np
import pytest

from emit.ir import DrawingIR,Entity
from engineering.dash_axes import review_dash_axes
from engineering.hatching import plan_hatches
from engineering.pipeline import write_ir
from tools.image_io import imwrite
from test_native_review import dimension_fixture


def line(a,b,layer='outline'): return Entity(type='line',start=tuple(a),end=tuple(b),layer=layer)


@pytest.mark.parametrize('angle',[0,45,90,27])
def test_jittered_dashes_share_axis_and_keep_blank_gaps(angle):
    image=np.full((260,260,3),255,np.uint8)
    u=np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))]);n=np.array([-u[1],u[0]])
    entities=[];origin=np.array([50.,50.])
    for i,jitter in enumerate([-.7,.7,-.5,.5,0]):
        a=origin+u*i*25+n*jitter;b=a+u*12+n*.4
        entities.append(line(a,b));cv2.line(image,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(0,0,0),2)
    ir=DrawingIR(width=260,height=260,entities=entities)
    result=review_dash_axes(ir,image)
    assert result.meta['dash_axis_review']['aligned_segments']==5
    points=np.array([p for e in result.entities for p in (e.start,e.end)])
    _,s,_=np.linalg.svd(points-points.mean(axis=0))
    assert s[1]<1e-6
    assert all(math.dist(a.end,b.start)>10 for a,b in zip(result.entities,result.entities[1:]))
    assert ir.entities[0].start!=result.entities[0].start


def test_separate_views_and_real_bend_are_not_joined():
    image=np.full((180,500,3),255,np.uint8);entities=[]
    for x in [20,40,60,80,300,320,340,360]:
        e=line((x,70),(x+8,70));entities.append(e);cv2.line(image,(x,70),(x+8,70),(0,0,0),1)
    bend=Entity(type='polyline',points=[(100,100),(120,100),(120,120)])
    entities.append(bend);cv2.polylines(image,[np.array(bend.points,dtype=np.int32)],False,(0,0,0),1)
    result=review_dash_axes(DrawingIR(width=500,height=180,entities=entities),image)
    assert len(result.meta['dash_axis_review']['trains'])==2
    assert result.entities[-1]==bend
    assert len(result.entities)==9


def test_missing_dash_at_crossing_does_not_leave_a_tilted_run():
    image=np.full((120,400,3),255,np.uint8);entities=[]
    for i in [0,1,2,3,5,6,7,8,9,10]:
        a=(20+i*25,60+(i%3-1)*.5);b=(a[0]+14,a[1]+1)
        entities.append(line(a,b,'centerline'))
        cv2.line(image,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(0,0,255),2)
    result=review_dash_axes(DrawingIR(width=400,height=120,entities=entities),image)
    assert result.meta['dash_axis_review']['aligned_segments']==10
    assert len({round(p[1],7) for e in result.entities for p in (e.start,e.end)})==1
    assert result.entities[4].start[0]-result.entities[3].end[0]>30


def hatch_fixture(hole=True,unknown_hole=False):
    image=np.full((230,300,3),255,np.uint8)
    border=[(30,30),(190,30),(190,190),(30,190)]
    entities=[Entity(type='polyline',points=border,closed=True)]
    for total in range(70,380,12):
        endpoints=[]
        for x,y in [(30,total-30),(190,total-190),(total-30,30),(total-190,190)]:
            if 30<=x<=190 and 30<=y<=190 and (x,y) not in endpoints:endpoints.append((x,y))
        if len(endpoints)!=2:continue
        a,b=np.array(endpoints[0],float),np.array(endpoints[1],float)
        # A mixed polyline keeps its non-hatch terminal edge after export.
        entities.append(line(a,b));cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),(0,0,0),1)
    if hole or unknown_hole:
        cv2.circle(image,(110,110),22,(255,255,255),-1);cv2.circle(image,(110,110),22,(0,0,0),2)
        if hole:entities.append(Entity(type='circle',center=(110,110),radius=22))
    cv2.polylines(image,[np.array(border)],True,(0,0,0),2)
    return DrawingIR(width=300,height=230,entities=entities),image


def test_native_hatch_measured_pattern_and_unfilled_circular_island(tmp_path):
    ir,image=hatch_fixture();source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    plans=plan_hatches(ir,image)
    assert len(plans)==1 and len(plans[0]['holes'])==1
    assert plans[0]['spacing']==pytest.approx(12/math.sqrt(2),abs=.3)
    path=tmp_path/'result.dxf';write_ir(ir,path);doc=ezdxf.readfile(path)
    assert not doc.audit().has_errors
    hatch=doc.modelspace().query('HATCH')[0]
    assert len(hatch.paths)==2 and hatch.dxf.solid_fill==0
    assert len(doc.modelspace().query('CIRCLE'))==1
    assert hatch.pattern.lines[0].angle==pytest.approx(-135,abs=1)
    assert len(doc.modelspace().query('LWPOLYLINE'))==1


def test_unknown_blank_hole_and_open_boundary_retain_original_lines():
    ir,image=hatch_fixture(hole=False,unknown_hole=True)
    assert plan_hatches(ir,image)==[]
    ir,image=hatch_fixture(hole=False);ir.entities[0].closed=False
    assert plan_hatches(ir,image)==[]


def test_sparse_feature_array_is_not_hatching():
    ir,image=hatch_fixture(hole=False)
    image[:]=255
    ir.entities=ir.entities[:5]
    for e in ir.entities[1:]:cv2.line(image,tuple(map(int,e.start)),tuple(map(int,e.end)),(0,0,0),1)
    assert plan_hatches(ir,image)==[]


def test_hatched_circle_is_not_mistaken_for_blank_island():
    ir,image=hatch_fixture(hole=False)
    ir.entities.append(Entity(type='circle',center=(110,110),radius=22))
    cv2.circle(image,(110,110),22,(0,0,0),2)
    plans=plan_hatches(ir,image)
    assert len(plans)==1 and plans[0]['holes']==[]


def test_native_dash_gaps_are_not_consumed_as_duplicate_strokes():
    image=np.full((100,200,3),255,np.uint8)
    for x in range(20,140,20):cv2.line(image,(x,50),(x+9,50),(0,0,255),2)
    ir=DrawingIR(width=200,height=100,entities=[line((20,50),(129,50),'centerline_000'),
        line((42,51),(47,51),'centerline'),line((53,51),(56,51),'centerline')],
        meta={'linetype_patterns':{'centerline_000':[20,9,-11]}})
    result=review_dash_axes(ir,image)
    assert result.meta['dash_axis_review']['native_axis_remnants_removed']==1
    assert len(result.entities)==2
    assert result.entities[1].start[0]==53


def test_hatch_and_dimensions_survive_same_export_and_repeat(tmp_path):
    ir,image=hatch_fixture(hole=False)
    dims=dimension_fixture()
    for e in dims.entities:
        if e.type=='line':e.start=(e.start[0],e.start[1]+200);e.end=(e.end[0],e.end[1]+200)
        else:e.pos=(e.pos[0],e.pos[1]+200)
    ir.entities.extend(dims.entities);ir.height=380;ir.meta.update(dims.meta)
    image=cv2.copyMakeBorder(image,0,150,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    for i in range(2):
        path=tmp_path/f'{i}.dxf';report=write_ir(ir,path);doc=ezdxf.readfile(path)
        assert report['native_dimensions']==1
        assert len(doc.modelspace().query('HATCH'))==1
        assert len(doc.modelspace().query('DIMENSION'))==1
        assert not doc.audit().has_errors


def test_mixed_polyline_keeps_non_hatch_edge(tmp_path):
    ir,image=hatch_fixture(hole=False)
    old=ir.entities[8];ir.entities[8]=Entity(type='polyline',points=[old.start,old.end,(old.end[0],old.end[1]+5)])
    cv2.line(image,tuple(map(int,old.end)),(int(old.end[0]),int(old.end[1]+5)),(0,0,0),1)
    source=tmp_path/'source.png';imwrite(source,image);ir.meta['source']=str(source)
    path=tmp_path/'mixed.dxf';write_ir(ir,path);doc=ezdxf.readfile(path)
    assert len(doc.modelspace().query('HATCH'))==1
    assert any(e.dxf.start.isclose((old.end[0],ir.height-old.end[1],0)) and
               e.dxf.end.isclose((old.end[0],ir.height-old.end[1]-5,0)) for e in doc.modelspace().query('LINE'))

