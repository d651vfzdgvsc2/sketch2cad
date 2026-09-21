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
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "demo_out"
FONT = ("Microsoft YaHei UI", 10)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.image_path: str | None = None
        self.out_dir: Path | None = None

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
        tk.Radiobutton(box, text="① 工程图（规则几何）— 方块 / 圆 / 槽 / 孔，多Agent协同",
                       variable=self.mode, value="cad", font=FONT, bg="#f5f5f7").pack(anchor="w")
        tk.Radiobutton(box, text="② 假山（有机形状）— 自由曲线轮廓，Potrace拟合",
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

        # 日志
        self.log_box = scrolledtext.ScrolledText(root, height=14, font=("Consolas", 9),
                                                 bg="white", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(4, 16))

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
        threading.Thread(target=self.worker, args=(self.image_path, mode), daemon=True).start()

    def worker(self, path: str, mode: str) -> None:
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            if mode == "cad":
                self.root.after(0, self.log, "[工程图模式] 多Agent集成流水线启动…")
                from core.ensemble import run_ensemble

                res = run_ensemble(path)
                for name, p in res["proposals"].items():
                    mark = "  ← 选中" if name == res["picked"] else ""
                    self.root.after(0, self.log,
                                    f"  {name}臂  SSIM={p['ssim']:.4f}  实体={p['n_entities']}{mark}")
                src = Path(res["best"]["dxf"])
                dst = OUT_DIR / f"{Path(path).stem}_best.dxf"
                dst.write_bytes(src.read_bytes())
                preview = res["best"]["png"]
                self.root.after(0, self.log, f"选中 {res['picked']}，SSIM={res['best']['ssim']:.4f}")

                # 比例尺校准 -> 真实尺寸 DXF
                try:
                    from core.calibrate import estimate_scale, scale_ir
                    from emit.to_dxf import ir_to_dxf
                    from tools.ocr import run_ocr
                    from vectorize.vectorize import vectorize

                    ocr = run_ocr(path)
                    ir_cal = vectorize(path, params={"assemble": False,
                                                     "dimension_action": "layer"}, ocr=ocr)
                    est = estimate_scale(ocr, ir_cal)
                    if est["mm_per_px"]:
                        ir_s = scale_ir(vectorize(path, ocr=ocr), est["mm_per_px"])
                        rs = OUT_DIR / f"{Path(path).stem}_realsize.dxf"
                        ir_to_dxf(ir_s, rs)
                        self.root.after(0, self.log,
                                        f"比例尺≈{est['mm_per_px']} mm/px，已导出真实尺寸: {rs.name}")
                except Exception as e:  # noqa: BLE001
                    self.root.after(0, self.log, f"比例尺校准跳过: {type(e).__name__}")
            else:
                self.root.after(0, self.log, "[假山模式] Potrace 轮廓拟合启动…")
                from core.rockery import run_rockery

                res = run_rockery(path, OUT_DIR)
                dst = Path(res["dxf"])
                preview = res["png"]
                self.root.after(0, self.log,
                                f"  曲线={res['curves']}  图元={res['n_entities']}  耗时={res['secs']}s")

            self.out_dir = OUT_DIR
            self.root.after(0, self.log, f"\n完成！DXF: {dst}")
            self.root.after(0, self.log, f"预览: {preview}")
            self.root.after(0, self.open_file, preview)
            self.root.after(0, self.open_btn.configure, {"state": "normal"})
        except Exception as e:  # noqa: BLE001
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
