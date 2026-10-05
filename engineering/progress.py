"""Real stage events and durable diagnostics for long engineering conversions."""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime,timezone


class ProgressReporter:
    total = 11

    def __init__(self, callback=None):
        self.callback=callback;self.folder=None;self.started=time.perf_counter()
        self.stage_started=self.started;self.timings=[]
        self.event=dict(state='running',step=0,total=self.total,percent=0,message='准备工程图',elapsed_seconds=0.)

    def bind(self,folder):
        self.folder=folder
        self._emit()

    def _emit(self):
        now=time.perf_counter()
        self.event.update(elapsed_seconds=round(now-self.started,2),stage_seconds=round(now-self.stage_started,2),
                          updated_at=datetime.now(timezone.utc).isoformat(),timings=list(self.timings))
        if self.folder:
            try:
                target=self.folder/'progress.json';temporary=self.folder/'progress.tmp'
                temporary.write_text(json.dumps(self.event,ensure_ascii=False,indent=2),encoding='utf8')
                temporary.replace(target)
            except OSError:pass # an observer cannot destroy a valid drawing
        if self.callback:
            try:self.callback(dict(self.event))
            except Exception:pass

    def phase(self,step,message):
        now=time.perf_counter()
        if step!=self.event['step']:
            self.timings.append(dict(step=self.event['step'],message=self.event['message'],seconds=round(now-self.stage_started,2)))
            self.stage_started=now
        self.event.update(step=step,percent=min(99,round(step/self.total*100)),message=message)
        self._emit()

    def detail(self,message):
        self.event['message']=message;self._emit()

    def ready(self,delivery,preview):
        self.event.update(delivery=delivery,preview=str(preview),cad_ready=True)
        self._emit()

    def complete(self):
        self.phase(self.total,'处理完成，CAD 已生成')
        self.event.update(state='completed',percent=100)
        self._emit()

    def fail(self,exc):
        self.event.update(state='failed',error_type=type(exc).__name__,message='处理失败，请查看下方错误信息')
        if self.folder:
            try:(self.folder/'failure.log').write_text(traceback.format_exc(),encoding='utf8')
            except OSError:pass
        self._emit()
