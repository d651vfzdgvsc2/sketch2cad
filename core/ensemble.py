"""工程图入口：原生文字、连续线条与参数化模板重建。

默认提案：CLEAN 连续线条、LIBRARY 模板组合；结构与图像证据共同选优。
旧 CV / 云端生成提案可显式启用；自动输出仍需人工 CAD 核对。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import ezdxf

from agents.codegen_agent import codegen_proposal
from emit.ir import DrawingIR
from emit.to_dxf import ir_to_dxf
from engineering.metrics import compare, combined_score
from engineering.render import render_dxf_to_image
from tools.image_io import imread
from tools.ocr import run_ocr
from tools.vlm import ask_vision
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "engineering"


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


def deterministic_proposal(image: str, out_dir=None) -> dict:
    tmp = Path(out_dir) if out_dir else TMP / uuid.uuid4().hex[:12]
    tmp.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = tmp / f"{stem}_A.dxf"
    ir = vectorize(image, ocr=run_ocr(image))
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, tmp / f"{stem}_A.png")
    sc["proposal"] = "A"
    return sc


def multiview_proposal(image: str, out_dir=None) -> dict:
    from core.multiview import vectorize_multiview

    tmp = Path(out_dir) if out_dir else TMP / uuid.uuid4().hex[:12]
    tmp.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = tmp / f"{stem}_MV.dxf"
    ir, n_views = vectorize_multiview(image, out_dir=tmp / "views")
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, tmp / f"{stem}_MV.png")
    sc["proposal"] = "MV"
    sc["views"] = n_views
    return sc


def centerline_proposal(image: str, out_dir=None) -> dict:
    """中心线臂（工程图）：骨架化 → 追踪 → 剪毛刺 → 合并 → 拟合(直线/圆弧/圆) → 文字。"""
    from vectorize.lineart import vectorize_lineart

    tmp = Path(out_dir) if out_dir else TMP / uuid.uuid4().hex[:12]
    tmp.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = tmp / f"{stem}_CL.dxf"
    ir, _stats = vectorize_lineart(image, correct_text=False)
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, tmp / f"{stem}_CL.png")
    sc["proposal"] = "CL"
    sc["n_entities"] = len(ir.entities)
    return sc


SPEC_PROMPT = """根据工程图输出参数化模板JSON数组。画布 {W}×{H}，全图像素坐标，左上原点，y向下。
模板库：{catalog}
每项：{{"type":"模板名","params":{{参数字典}}}}。
保留全部视图和可确认的圆、圆弧、椭圆、小圆角、槽、孔、短线，不设置图元数量上限。
长圆槽用slot(center/length/diameter/rotation)，圆角矩形用rounded_rectangle。
标注冲突时保留像素形状，不凭常识补孔，不拉伸原图，不把曲线简化为尖角矩形。
只输出JSON数组。"""


def spec_proposal(image: str, provider: str = "deepseek", out_dir=None) -> dict:
    from engineering.primitives import build, CATALOG
    from engineering.pipeline import write_ir
    import uuid
    tmp = Path(out_dir) if out_dir else TMP / uuid.uuid4().hex[:12]
    tmp.mkdir(parents=True, exist_ok=True)
    h, w = imread(image).shape[:2]
    raw = ask_vision(image, SPEC_PROMPT.format(W=w, H=h, catalog=json.dumps(CATALOG)),
                     provider=provider, max_tokens=8000)
    spec = json.loads(raw[raw.find("["):raw.rfind("]")+1])
    if not isinstance(spec, list) or not spec:
        raise ValueError("No template specification returned")
    entities = []
    for item in spec:
        entities.extend(build(item["type"], item["params"]))
    ir = DrawingIR(width=w, height=h, entities=entities)
    dxf = tmp / "SPEC.dxf"
    write_ir(ir, dxf)
    result = _score(dxf, image, tmp / "SPEC.png")
    result["proposal"] = "SPEC"
    return result


def run_ensemble(image: str, rounds: int = 2, use_b: bool = False, use_d: bool = False,
                 use_vlm: bool = False, use_mv: bool = False, use_cl: bool = False,
                 use_spec: bool = False, use_cache: bool = False, *, out_dir=None,
                 use_templates=True, use_ocr=True, use_semantic=False, use_cv=False):
    """Local measured reconstruction by default. Cloud arms are explicit opt-ins.

    use_cache remains accepted for compatibility; old cached files are not read.
    Every run has its own directory and content/code fingerprint.
    """
    from engineering.pipeline import run_engineering
    proposals = {}
    if use_cv:
        proposals["A"] = deterministic_proposal
    if use_cl:
        proposals["CL"] = centerline_proposal
    if use_mv:
        proposals["MV"] = multiview_proposal
    if use_spec:
        proposals["SPEC"] = lambda image, out: spec_proposal(image, out_dir=out)
    if use_b:
        proposals["B"] = lambda image, out: codegen_proposal(image, grounded=False, rounds=rounds, out_dir=out, tag="B")
    if use_d:
        proposals["D"] = lambda image, out: codegen_proposal(image, grounded=True, rounds=rounds, out_dir=out, tag="D")
    if use_vlm:
        proposals["DS"] = lambda image, out: codegen_proposal(image, direct=True, rounds=rounds,
                                                             out_dir=out, tag="DS", provider="deepseek")
    return run_engineering(image, out_dir, use_ocr=use_ocr, use_templates=use_templates,
                           use_semantic=use_semantic, legacy_proposals=proposals)
