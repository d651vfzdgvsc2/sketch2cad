"""A/B/C 三臂对照实验。

A = CV 精确几何 + LLM 决策回环（现有方案）
B = VLM 图->文字 -> LLM 写 Python 脚本 -> 渲染自校验（用户方案）
C = CV 精确坐标 + OCR 喂给 LLM 写 Python 脚本（接地气代码生成）
三臂都用「渲染回原图比对」的同一套客观指标。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import ezdxf

from agents.codegen_agent import codegen_proposal
from agents.verifier import objective
from core.graph import run_agent
from emit.ir import DrawingIR
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.codegen import extract_code, run_script
from tools.image_io import imread
from tools.llm import chat
from tools.ocr import run_ocr
from tools.vlm import ask_image
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "abc"
ROUNDS = 2

DESCRIBE_PROMPT = """这是一张工程/CAD 图纸。请用文字尽可能详细地描述，供另一个程序据此重绘：
1) 图形元素及位置：用 0~1 归一化坐标描述（如"圆，圆心约(0.5,0.55)，半径约0.13"）
2) 所有文字/尺寸标注及其内容
3) 哪些是几何轮廓，哪些是标注（尺寸线、中心线）
只输出描述文字，不要客套。"""

CODEGEN_DOC = """
你是资深 CAD 绘图工程师。请写一个 Python 脚本（只用 ezdxf 和标准库）重绘这张图。
坐标约定：画布宽 {W} 高 {H}，图像坐标原点在左上、x 向右、y 向下；
ezdxf 的 y 轴向上，请对每个点做转换：cad_y = {H} - y。
脚本最后必须保存到：r"{OUT}"
只输出一个 ```python 代码块，不要任何解释。
"""


def _wh(image: str) -> tuple[int, int]:
    arr = imread(image)
    h, w = arr.shape[:2]
    return int(w), int(h)


def _score_dxf(dxf: Path, image: str, tag: str) -> dict:
    w, h = _wh(image)
    png = TMP / f"{tag}.png"
    render_dxf_to_image(dxf, png, width=w, height=h)
    sc = compare(imread(png), imread(image), tol=2)
    try:
        doc = ezdxf.readfile(str(dxf))
        n = len(list(doc.modelspace()))
    except Exception:  # noqa: BLE001
        n = -1
    sc["n_entities"] = n
    sc["png"] = str(png)
    return sc


# ---------------- Arm A ----------------
def arm_a(image: str) -> dict:
    t0 = time.time()
    res = run_agent(image)
    ir = DrawingIR.from_json(res.get("best_ir_json") or res["ir_json"])
    sc = dict(res.get("best_score") or res["score"])
    sc["n_entities"] = len(ir.entities)
    sc["png"] = res.get("best_render_png", "")
    return {"arm": "A", "score": sc, "secs": round(time.time() - t0, 1),
            "rounds": len(res.get("log", [])), "png": res.get("best_render_png", "")}


# ---------------- Arm B ----------------
def arm_b(image: str) -> dict:
    t0 = time.time()
    sc = codegen_proposal(image, grounded=False, rounds=ROUNDS, out_dir=TMP, tag=f"b_{Path(image).stem}")
    return {"arm": "B", "score": sc, "secs": round(time.time() - t0, 1),
            "rounds": ROUNDS, "png": sc.get("png", "")}


# ---------------- Arm C ----------------
def _geometry_json(ir: DrawingIR) -> str:
    ents = []
    for e in ir.entities:
        if e.type == "line":
            ents.append({"type": "line", "start": [round(e.start[0], 1), round(e.start[1], 1)],
                         "end": [round(e.end[0], 1), round(e.end[1], 1)]})
        elif e.type == "circle":
            ents.append({"type": "circle", "center": [round(e.center[0], 1), round(e.center[1], 1)],
                         "radius": round(e.radius, 1)})
        elif e.type == "arc":
            ents.append({"type": "arc", "center": [round(e.center[0], 1), round(e.center[1], 1)],
                         "radius": round(e.radius, 1),
                         "start_deg": round(e.start_angle, 1), "end_deg": round(e.end_angle, 1)})
    return json.dumps(ents[:120], ensure_ascii=False)


def arm_c(image: str) -> dict:
    t0 = time.time()
    TMP.mkdir(parents=True, exist_ok=True)
    w, h = _wh(image)
    ir = vectorize(image)
    geom = _geometry_json(ir)
    texts = [o["text"] for o in run_ocr(image)]

    best, feedback = None, ""
    for r in range(ROUNDS):
        out = TMP / f"c_{Path(image).stem}_{r}.dxf"
        prompt = CODEGEN_DOC.format(W=w, H=h, OUT=str(out))
        user = (f"{prompt}\n\n算法从图中精确测量出的图元(JSON，坐标精确)：\n{geom}\n\n"
                f"OCR 识别到的文字：{texts}\n\n"
                "请判断哪些是真正的几何（保留），哪些是标注（尺寸线/中心线，可丢弃或放注释图层），"
                "然后还原真正的几何图形。")
        if feedback:
            user += f"\n\n上一版渲染对比原图的结果：{feedback}\n请修正后重写完整脚本。"
        code = extract_code(chat([{"role": "user", "content": user}], max_tokens=4096))
        ok, err, dxf = run_script(code, out)
        if not ok:
            feedback = f"脚本执行失败，错误：{err[:300]}"
            continue
        sc = _score_dxf(dxf, image, f"c_{Path(image).stem}_{r}")
        if best is None or sc["ssim"] > best["score"]["ssim"]:
            best = {"score": sc}
        feedback = (f"SSIM={sc['ssim']}, 图元数={sc['n_entities']}, "
                    f"墨迹比 pred={sc['ink_ratio_pred']} vs gt={sc['ink_ratio_gt']}")
        if sc["ssim"] >= 0.9:
            break
    if best is None:
        best = {"score": {"ssim": 0.0, "n_entities": -1, "png": ""}}
    return {"arm": "C", "score": best["score"], "secs": round(time.time() - t0, 1),
            "rounds": ROUNDS, "png": best["score"].get("png", "")}


# ---------------- Arm D : 融合（CV精确坐标 + B式克制）----------------
GEOM_MIN_LEN = 45.0

D_PROMPT = """
你是资深 CAD 绘图工程师。下面有三份信息，请据此写一个 Python 脚本（只用 ezdxf 和标准库）还原这张图。

【1】算法精确测量出的图元(JSON，坐标精确)：
{geom}

【2】OCR 识别到的文字：{texts}

【3】图纸语义描述：{desc}

【必须遵守的原则】
- 只保留**真正的几何**：外形轮廓、孔、槽、圆、圆弧。
- 尺寸线、尺寸界线、箭头、点划线中心线都属于**标注**，一律**不要画**。
- 能合并成完整形状的（如矩形四边、圆），要合并；**不要照抄碎线段**。
- 坐标直接用上面的精确值。画布宽 {W} 高 {H}，图像坐标原点左上、y 向下；
  ezdxf 的 y 轴向上，请转换：cad_y = {H} - y。
- 脚本最后保存到：r"{OUT}"
只输出一个 ```python 代码块，不要解释。
"""


def _geom_selective(ir: DrawingIR) -> str:
    ents = []
    for e in ir.entities:
        if e.type == "line":
            dx, dy = e.end[0] - e.start[0], e.end[1] - e.start[1]
            if (dx * dx + dy * dy) ** 0.5 < GEOM_MIN_LEN:
                continue
            ents.append({"type": "line", "start": [round(e.start[0], 1), round(e.start[1], 1)],
                         "end": [round(e.end[0], 1), round(e.end[1], 1)]})
        elif e.type == "circle":
            ents.append({"type": "circle", "center": [round(e.center[0], 1), round(e.center[1], 1)],
                         "radius": round(e.radius, 1)})
        elif e.type == "arc":
            ents.append({"type": "arc", "center": [round(e.center[0], 1), round(e.center[1], 1)],
                         "radius": round(e.radius, 1),
                         "start_deg": round(e.start_angle, 1), "end_deg": round(e.end_angle, 1)})
    return json.dumps(ents[:100], ensure_ascii=False)


def arm_d(image: str) -> dict:
    t0 = time.time()
    sc = codegen_proposal(image, grounded=True, rounds=ROUNDS, out_dir=TMP, tag=f"d_{Path(image).stem}")
    return {"arm": "D", "score": sc, "secs": round(time.time() - t0, 1),
            "rounds": ROUNDS, "png": sc.get("png", "")}


# ---------------- Arm E : 集成仲裁（跑 B 和 D，渲染比优，自动选）----------------
def arm_e(image: str) -> dict:
    t0 = time.time()
    rb = arm_b(image)
    rd = arm_d(image)
    winner = rb if rb["score"]["ssim"] >= rd["score"]["ssim"] else rd
    return {
        "arm": "E",
        "score": winner["score"],
        "secs": round(time.time() - t0, 1),
        "rounds": rb.get("rounds", 0) + rd.get("rounds", 0),
        "picked": winner["arm"],
        "proposals": {"B": rb["score"]["ssim"], "D": rd["score"]["ssim"]},
        "png": winner["score"].get("png", ""),
    }


ARMS = {"A": arm_a, "B": arm_b, "C": arm_c, "D": arm_d, "E": arm_e}
