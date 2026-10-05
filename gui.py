"""图纸矢量化系统 - 图形界面（tkinter，Python 自带，零依赖）

两种模式：
  ① 工程图（规则几何） → 多Agent集成流水线
  ② 假山（有机形状）   → Potrace 轮廓拟合
用法：python gui.py            （或双击 启动.bat）
自检：python gui.py --selftest （仅构建界面后立即退出）
"""
from __future__ import annotations

import sys
import threading
import queue
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "demo_out"
FONT = ("Microsoft YaHei UI", 10)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.image_path: str | None = None
        self.out_dir: Path | None = None
        self.progress_events=queue.SimpleQueue()
        self.progress_running=False
        self.progress_started=0.;self.stage_started=0.;self.last_step=-1

        root.title("图纸矢量化系统")
        root.geometry("720x540")
        root.configure(bg="#f5f5f7")

        tk.Label(root, text="图纸矢量化系统", font=("Microsoft YaHei UI", 16, "bold"),
                 bg="#f5f5f7").pack(pady=(16, 4))
        tk.Label(root, text="图片 → 可编辑 DXF", font=FONT, fg="#666", bg="#f5f5f7").pack()

        # 模式选择
        box = tk.LabelFrame(root, text=" 选择模式 ", font=FONT, bg="#f5f5f7", padx=12, pady=8)
        box.pack(fill="x", padx=20, pady=12)
        self.mode = tk.StringVar(value="cad")
        tk.Radiobutton(box, text="① 工程图（规则几何）— 方块 / 圆 / 槽 / 孔，模板拼装与原图校验",
                       variable=self.mode, value="cad", font=FONT, bg="#f5f5f7").pack(anchor="w")
        tk.Radiobutton(box, text="② 假山（有机形状）— 自由曲线轮廓，中心线提取",
                       variable=self.mode, value="rockery", font=FONT, bg="#f5f5f7").pack(anchor="w", pady=(4, 0))
        tk.Label(root, text="提示：简单/中等工程图用①；假山、植物等有机曲线用②。"
                            "复杂多视图装配图用①只能部分重建。",
                 font=("Microsoft YaHei UI", 8), fg="#a33", bg="#f5f5f7",
                 wraplength=660, justify="left").pack(anchor="w", padx=24)

        # 选图
        row = tk.Frame(root, bg="#f5f5f7")
        row.pack(fill="x", padx=20)
        tk.Button(row, text="选择图片…", font=FONT, command=self.choose, width=12).pack(side="left")
        self.path_label = tk.Label(row, text="（未选择）", font=FONT, fg="#888",
                                   bg="#f5f5f7", anchor="w")
        self.path_label.pack(side="left", padx=10, fill="x", expand=True)

        # 按钮
        btns = tk.Frame(root, bg="#f5f5f7")
        btns.pack(fill="x", padx=20, pady=10)
        self.run_btn = tk.Button(btns, text="开始处理", font=("Microsoft YaHei UI", 11, "bold"),
                                 bg="#2b7de9", fg="white", activebackground="#1c5fb8",
                                 command=self.start, width=14, height=1)
        self.run_btn.pack(side="left")
        self.open_btn = tk.Button(btns, text="打开结果文件夹", font=FONT,
                                  command=self.open_out, state="disabled", width=14)
        self.open_btn.pack(side="left", padx=10)

        # Engineering status is driven by real pipeline events on the UI thread.
        self.progress_panel=tk.LabelFrame(root,text=' 工程图处理进度 ',font=FONT,
                                         bg='#f5f5f7',padx=12,pady=7)
        self.progress_panel.pack(fill='x',padx=20,pady=(0,6))
        self.progress_message=tk.StringVar(value='选择图片后，点击“开始处理”')
        self.progress_label=tk.Label(self.progress_panel,textvariable=self.progress_message,
                                    font=FONT,bg='#f5f5f7',fg='#333',anchor='w',wraplength=650,justify='left')
        self.progress_label.pack(fill='x')
        self.progress_bar=ttk.Progressbar(self.progress_panel,maximum=100,mode='determinate')
        self.progress_bar.pack(fill='x',pady=(6,4))
        self.progress_time=tk.StringVar(value='步骤进度 0% · 已用 00:00')
        tk.Label(self.progress_panel,textvariable=self.progress_time,font=("Microsoft YaHei UI",9),
                 bg='#f5f5f7',fg='#666',anchor='w').pack(fill='x')
        self.mode.trace_add('write',self._show_progress)

        # 日志
        self.log_box = scrolledtext.ScrolledText(root, height=14, font=("Consolas", 9),
                                                 bg="white", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        root.after(100,self._poll_progress)

    def _show_progress(self,*_):
        if self.mode.get()=='cad' or self.progress_running:
            if not self.progress_panel.winfo_manager():
                self.progress_panel.pack(fill='x',padx=20,pady=(0,6),before=self.log_box)
        else:self.progress_panel.pack_forget()

    @staticmethod
    def _clock(seconds):
        seconds=max(0,int(seconds));return f'{seconds//60:02d}:{seconds%60:02d}'

    def _poll_progress(self):
        try:
            while True:
                event=self.progress_events.get_nowait()
                if 'percent' in event:self.progress_bar['value']=event['percent']
                self.progress_message.set(event.get('message','处理中…'))
                if event.get('step')!=self.last_step and 'step' in event:
                    self.last_step=event['step'];self.stage_started=time.monotonic()
                    self.log(f"[{event['step']}/{event.get('total',11)}] {event['message']}")
                if event.get('delivery'):
                    self.out_dir=Path(event['delivery']['directory'])
                    self.open_btn.configure(state='normal')
                if event.get('state') in ('completed','failed'):
                    self.progress_running=False
                    if event['state']=='failed':self.progress_label.configure(fg='#a33')
                    self._show_progress()
                elapsed=event.get('elapsed_seconds')
                if elapsed is not None:self.progress_started=time.monotonic()-elapsed
                self.progress_time.set(f"步骤进度 {int(float(self.progress_bar['value']))}% · 已用 {self._clock(time.monotonic()-self.progress_started)}")
        except queue.Empty:pass
        if self.progress_running:
            self.progress_time.set(f"步骤进度 {int(float(self.progress_bar['value']))}% · 已用 {self._clock(time.monotonic()-self.progress_started)} · 当前步骤 {self._clock(time.monotonic()-self.stage_started)}")
        self.root.after(100,self._poll_progress)

    def log(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def choose(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
            initialdir=str(ROOT / "data"),
        )
        if path:
            self.image_path = path
            self.path_label.configure(text=path, fg="#333")

    def start(self) -> None:
        if not self.image_path:
            messagebox.showwarning("提示", "请先选择一张图片")
            return
        self.run_btn.configure(state="disabled", text="处理中…")
        self.open_btn.configure(state="disabled")
        mode = self.mode.get()
        if mode=='cad':
            while not self.progress_events.empty():self.progress_events.get_nowait()
            self.out_dir=None;self.progress_running=True;self.last_step=-1
            self.progress_started=self.stage_started=time.monotonic()
            self.progress_bar['value']=0;self.progress_label.configure(fg='#333')
            self.progress_message.set('准备处理工程图…')
            self._show_progress()
        threading.Thread(target=self.worker, args=(self.image_path, mode), daemon=True).start()

    def worker(self, path: str, mode: str) -> None:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if mode == "cad":
                self.root.after(0, self.log, "[工程图模式] 像素测量 → 模板库拼装 → 原图校验…")
                from core.ensemble import run_ensemble

                res = run_ensemble(path, out_dir=OUT_DIR / "engineering",progress=self.progress_events.put)
                for name, p in res["proposals"].items():
                    mark = "  ← 选中" if name == res["picked"] else ""
                    self.root.after(0, self.log,
                                    f"  {name}  线条F1={p.get('f1', 0):.4f}  实体={p['n_entities']}{mark}")
                dst = Path(res['delivery']['dxf'])
                preview = res["best"]["png"]
                self.root.after(0, self.log,
                                f"选中 {res['picked']}，采用模板={res['selected_templates']}，"
                                f"线条F1={res['best']['f1']:.4f}（原图像素容差2px）")

                if res['delivery']['units']=='mm':
                    self.root.after(0, self.log, "最终 CAD 已按比例导出为毫米单位。")
                else:
                    self.root.after(0, self.log, "尺寸标注证据不足或冲突，保留像素单位；未生成未经确认的毫米版。")
                self.root.after(0, self.log, "结果文件夹只保留最终选出的 CAD 文件。")
            else:
                self.root.after(0, self.log, "[假山模式] Potrace 轮廓拟合启动…")
                from core.rockery import run_rockery

                res = run_rockery(path, OUT_DIR)
                dst = Path(res["dxf"])
                preview = res["png"]
                self.root.after(0, self.log,
                                f"  曲线={res['curves']}  图元={res['n_entities']}  耗时={res['secs']}s")

            self.out_dir = Path(res['delivery']['directory']) if mode == "cad" else OUT_DIR
            self.root.after(0, self.log, f"\n完成！DXF: {dst}")
            self.root.after(0, self.log, f"预览: {preview}")
            self.root.after(0, self.open_file, preview)
            self.root.after(0, self.open_btn.configure, {"state": "normal"})
        except Exception as e:  # noqa: BLE001
            if mode=='cad':self.progress_events.put(dict(state='failed',message='处理失败，请查看下方错误信息'))
            self.root.after(0, self.log, f"\n[错误] {type(e).__name__}: {e}")
        finally:
            self.root.after(0, self.run_btn.configure, {"state": "normal", "text": "开始处理"})

    def open_file(self, path: str) -> None:
        try:
            import os

            os.startfile(path)  # noqa: S606
        except Exception:  # noqa: BLE001
            pass

    def open_out(self) -> None:
        if self.out_dir:
            import os

            os.startfile(str(self.out_dir))  # noqa: S606


def main() -> None:
    root = tk.Tk()
    app = App(root)
    if "--selftest" in sys.argv:
        root.after(400, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
