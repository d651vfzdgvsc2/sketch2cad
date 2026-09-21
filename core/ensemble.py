"""集成流水线（正式路径）：多臂提案 + 渲染回图客观仲裁。

提案：
- A：确定性 CV 矢量化（免费、稳定）
- D：接地式代码生成（CV 精确坐标 + OCR + VLM 描述）
- B：纯语义代码生成（VLM 文字 -> 代码）
仲裁：把每个提案渲染回图、与原图比对，取 SSIM 最优者（保最优）。
"""
from __future__ import annotations

from pathlib import Path

import ezdxf

from agents.codegen_agent import codegen_proposal
from emit.ir import DrawingIR
from emit.to_dxf import ir_to_dxf
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.image_io import imread
from tools.ocr import run_ocr
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "ensemble"


def _score(dxf: Path, image: str, png: Path) -> dict:
    arr = imread(image)
    h, w = arr.shape[:2]
    render_dxf_to_image(dxf, png, width=w, height=h)
    sc = compare(imread(png), imread(image), tol=2)
    try:
        sc["n_entities"] = len(list(ezdxf.readfile(str(dxf)).modelspace()))
    except Exception:  # noqa: BLE001
        sc["n_entities"] = -1
    sc["png"] = str(png)
    sc["dxf"] = str(dxf)
    return sc


def deterministic_proposal(image: str) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = TMP / f"{stem}_A.dxf"
    ir = vectorize(image, ocr=run_ocr(image))
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, TMP / f"{stem}_A.png")
    sc["proposal"] = "A"
    return sc


def run_ensemble(image: str, rounds: int = 2, use_b: bool = True, use_d: bool = True,
                 use_vlm: bool = True) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    proposals: dict[str, dict] = {"A": deterministic_proposal(image)}

    if use_d:
        proposals["D"] = codegen_proposal(image, grounded=True, rounds=rounds,
                                           out_dir=TMP, tag=f"{stem}_D")
    if use_b:
        proposals["B"] = codegen_proposal(image, grounded=False, rounds=rounds,
                                          out_dir=TMP, tag=f"{stem}_B")
    if use_vlm:  # DeepSeek 直接看图写代码（强模型主画）
        proposals["DS"] = codegen_proposal(image, direct=True, rounds=3,
                                           out_dir=TMP, tag=f"{stem}_DS",
                                           vlm_model="deepseek-chat", provider="deepseek")
        proposals["DS"]["proposal"] = "DS"

    picked = max(proposals, key=lambda k: proposals[k]["ssim"])
    return {
        "image": image,
        "picked": picked,
        "best": proposals[picked],
        "proposals": {k: {"ssim": v["ssim"], "n_entities": v["n_entities"], "png": v.get("png", "")}
                      for k, v in proposals.items()},
    }
