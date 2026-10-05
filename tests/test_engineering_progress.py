import json
import threading
from pathlib import Path
import cv2
import numpy as np

import engineering.dimensions as dimensions
from engineering.pipeline import run_engineering
from engineering.progress import ProgressReporter
from tools.image_io import imwrite
from test_native_review import dimension_fixture


def test_candidate_cache_is_scoped_and_returns_independent_plans(monkeypatch):
    ir=dimension_fixture();calls=[];compute=dimensions._compute_dimension_candidates
    def counted(ir):calls.append(1);return compute(ir)
    monkeypatch.setattr(dimensions,'_compute_dimension_candidates',counted)
    with dimensions.candidate_cache_scope():
        first=dimensions.dimension_candidates(ir);first[0]['text']='corrupted'
        assert dimensions.dimension_candidates(ir)[0]['text']=='100'
        assert len(calls)==1
        ir.entities[-1].content='200'
        assert dimensions.dimension_candidates(ir)[0]['text']=='200'
        assert len(calls)==2
        ir.meta['annotations'][0]['score']=.1
        assert dimensions.dimension_candidates(ir)==[]
        assert len(calls)==3
    ir.meta['annotations'][0]['score']=.99
    dimensions.dimension_candidates(ir)
    assert len(calls)==4


def test_candidate_cache_rechecks_source_contents_and_isolates_threads(tmp_path,monkeypatch):
    ir=dimension_fixture();source=tmp_path/'source.bin';source.write_bytes(b'first');ir.meta['source']=str(source)
    calls=[]
    monkeypatch.setattr(dimensions,'_compute_dimension_candidates',lambda ir:calls.append(1) or [])
    with dimensions.candidate_cache_scope():
        dimensions.dimension_candidates(ir);dimensions.dimension_candidates(ir)
        source.write_bytes(b'other');dimensions.dimension_candidates(ir)
        assert len(calls)==2
        thread=threading.Thread(target=lambda:dimensions.dimension_candidates(ir));thread.start();thread.join()
        assert len(calls)==3


def test_real_pipeline_reports_stages_and_ready_only_with_real_dxf(tmp_path):
    source=tmp_path/'source.png';image=np.full((160,200,3),255,np.uint8)
    cv2.rectangle(image,(30,30),(160,130),(0,0,0),2);imwrite(source,image)
    events=[]
    def callback(event):
        events.append(event)
        if event.get('cad_ready'):assert Path(event['delivery']['dxf']).is_file()
    result=run_engineering(source,tmp_path/'out',use_ocr=False,use_templates=False,use_ai_review=False,progress=callback)
    assert events[-1]['state']=='completed' and events[-1]['percent']==100
    assert all(e['percent']<100 for e in events[:-1])
    assert {e['step'] for e in events}==set(range(12))
    ready=next(e for e in events if e.get('cad_ready'))
    assert ready['step']==9 and ready['state']=='running'
    assert len(list(Path(result['delivery']['directory']).iterdir()))==1
    durable=json.loads((Path(result['out_dir'])/'progress.json').read_text(encoding='utf8'))
    assert durable['state']=='completed' and len(durable['timings'])==11


def test_pipeline_failure_persists_stage_and_traceback(tmp_path,monkeypatch):
    import engineering.pipeline as pipeline
    source=tmp_path/'source.png';imwrite(source,np.full((50,50,3),255,np.uint8))
    monkeypatch.setattr(pipeline,'build_proposals',lambda *a,**k:(_ for _ in ()).throw(ValueError('test geometry failure')))
    events=[]
    try:run_engineering(source,tmp_path/'out',use_ocr=False,use_ai_review=False,progress=events.append)
    except ValueError:pass
    else:raise AssertionError('Failure was swallowed')
    folder=next((tmp_path/'out'/'.work').iterdir())
    progress=json.loads((folder/'progress.json').read_text(encoding='utf8'))
    assert progress['state']=='failed' and progress['step']==3 and progress['percent']<100
    assert 'test geometry failure' in (folder/'failure.log').read_text(encoding='utf8')
    assert events[-1]['state']=='failed'


def test_progress_callback_error_does_not_break_conversion(tmp_path):
    def callback(event):raise RuntimeError('broken observer')
    reporter=ProgressReporter(callback);reporter.bind(tmp_path)
    reporter.phase(4,'测试步骤');reporter.complete()
    assert json.loads((tmp_path/'progress.json').read_text(encoding='utf8'))['state']=='completed'
