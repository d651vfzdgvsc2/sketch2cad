"""确定性基线流水线：线稿图 -> IR -> DXF -> 渲染回图。"""
from __future__ import annotations

from pathlib import Path

from emit.to_dxf import ir_to_dxf
from render.render_dxf import render_dxf_to_image
from vectorize.vectorize import vectorize


def run(image_path: str, out_dir: str = "out") -> tuple:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ir = vectorize(image_path)
    dxf_path = out / "pred.dxf"
    ir_to_dxf(ir, dxf_path)
    png_path = out / "pred.png"
    render_dxf_to_image(dxf_path, png_path, width=ir.width, height=ir.height)
    (out / "pred.json").write_text(ir.to_json(), encoding="utf-8")
    return ir, dxf_path, png_path
