"""校验 Agent：把 IR 渲染回图片，与原图客观比对，产出差异报告。

关键设计：维护「历史最优」。Leader 调参可能变差，但最终一定返回整个回环里最好的一版，
从而保证多Agent结果 **永不差于** 单Agent基线。
"""
from __future__ import annotations

from pathlib import Path

from emit.ir import DrawingIR
from emit.to_dxf import ir_to_dxf
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.image_io import imread

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp"


def objective(score: dict) -> float:
    """收敛判据的复合目标（SSIM 为主 + 几何距离）。"""
    chamfer_term = max(0.0, 1.0 - min(score.get("chamfer_px", 5.0), 5.0) / 5.0)
    return round(0.6 * score.get("ssim", 0.0) + 0.4 * chamfer_term, 6)


def _is_better(ssim: float, chamfer: float, best_ssim: float, best_chamfer: float) -> bool:
    """字典序比较：SSIM 优先，几何距离次之。"""
    if ssim > best_ssim + 1e-6:
        return True
    return ssim >= best_ssim - 1e-6 and chamfer < best_chamfer - 1e-6


def verify(state: dict) -> dict:
    ir = DrawingIR.from_json(state["ir_json"])
    TMP.mkdir(parents=True, exist_ok=True)
    tag = state.get("round", 0)
    dxf = TMP / f"agent_{tag}.dxf"
    png = TMP / f"agent_{tag}.png"

    ir_to_dxf(ir, dxf)
    render_dxf_to_image(dxf, png, width=ir.width, height=ir.height)

    score = compare(imread(png), imread(state["image"]), tol=2)
    obj = objective(score)

    vlm_counts = (state.get("vlm") or {}).get("counts", {}) or {}
    circles_pred = sum(1 for e in ir.entities if e.type == "circle")
    circles_vlm = int(vlm_counts.get("circle", 0) or 0)

    issues: list[str] = []
    if circles_vlm and abs(circles_pred - circles_vlm) >= 2:
        issues.append(f"圆数量不一致: 检测={circles_pred}, VLM={circles_vlm}")

    feedback = {
        "score": score,
        "circles_pred": circles_pred,
        "circles_vlm": circles_vlm,
        "n_entities": len(ir.entities),
        "issues": issues,
    }

    updates: dict = {"score": score, "feedback": feedback, "obj": obj,
                     "render_png": str(png), "dxf": str(dxf)}

    best_ssim = state.get("best_ssim", -1.0)
    best_chamfer = state.get("best_chamfer", 1e9)
    if _is_better(score["ssim"], score["chamfer_px"], best_ssim, best_chamfer):
        updates.update({
            "best_ssim": score["ssim"],
            "best_chamfer": score["chamfer_px"],
            "best_ir_json": state["ir_json"],
            "best_score": score,
            "best_render_png": str(png),
            "best_round": state.get("round", 0),
        })
    return updates
