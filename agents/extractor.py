"""提取 Agent：确定性矢量化 + 把 OCR 文字合成为 text 图元。"""
from __future__ import annotations

from emit.ir import DrawingIR, Entity, LayerSpec
from vectorize.vectorize import DEFAULTS, vectorize


def _add_ocr_text(ir: DrawingIR, ocr: list[dict]) -> DrawingIR:
    if not ocr:
        return ir
    if "text" not in [l.name for l in ir.layers]:
        ir.layers.append(LayerSpec(name="text", color=3))
    for o in ocr:
        if not o.get("text"):
            continue
        x, y = o["center"]
        box_h = max(o["box"][3] - o["box"][1], 6.0)
        ir.entities.append(Entity(type="text", content=o["text"], pos=(x, y),
                                  height=box_h * 0.8, layer="text"))
    return ir


def extract(state: dict) -> dict:
    params = {**DEFAULTS, **(state.get("params") or {})}
    ir = vectorize(state["image"], params)
    ir = _add_ocr_text(ir, state.get("ocr", []))
    return {"ir_json": ir.to_json(), "counts": ir.counts(), "used_params": params}
