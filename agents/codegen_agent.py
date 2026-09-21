"""代码生成 Agent：让 LLM 写 ezdxf 脚本还原图纸。

两种模式：
- grounded=True  （D 臂）：喂 CV 精确坐标 + OCR + VLM 描述，克制地只画几何
- grounded=False （B 臂）：只喂 VLM 文字描述，靠语义"少画不乱画"
每次生成后都渲染回图、与原图客观比对，并保留历史最优。
"""
from __future__ import annotations

import json
from pathlib import Path

import ezdxf

from emit.ir import DrawingIR
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.codegen import extract_code, run_script
from tools.image_io import imread
from tools.llm import chat
from tools.ocr import run_ocr
from tools.vlm import ask_image, ask_vision
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "ensemble"
GEOM_MIN_LEN = 45.0

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

【ezdxf 正确用法，请严格照用，不要臆造 API】
```python
import ezdxf
doc = ezdxf.new("R2010")
msp = doc.modelspace()
msp.add_line((x1, y1), (x2, y2))
msp.add_lwpolyline([(x1, y1), (x2, y2), ...], close=True)   # 多段线
msp.add_circle((cx, cy), radius)
msp.add_arc((cx, cy), radius, start_angle_deg, end_angle_deg)  # 角度是"度"，逆时针
msp.add_ellipse((cx, cy), (major_x, major_y), ratio)           # ratio = 短轴/长轴
t = msp.add_text("文字", height=20)
t.set_placement((x, y))                                        # 文字用 set_placement，不要用 set_pos(align=...)
doc.saveas(r"{OUT}")
```
【禁止】不要 import ezdxf.math.Matrix4（不存在）；不要用 add_text(...).set_pos(align='MIDDLE_CENTER')；
不要用不确定的 add_hatch 参数。宁可少画，也不要写会报错的代码。
"""

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


def _wh(image: str) -> tuple[int, int]:
    arr = imread(image)
    h, w = arr.shape[:2]
    return int(w), int(h)


def _score_dxf(dxf: Path, image: str, png: Path) -> dict:
    w, h = _wh(image)
    render_dxf_to_image(dxf, png, width=w, height=h)
    sc = compare(imread(png), imread(image), tol=2)
    try:
        sc["n_entities"] = len(list(ezdxf.readfile(str(dxf)).modelspace()))
    except Exception:  # noqa: BLE001
        sc["n_entities"] = -1
    sc["png"] = str(png)
    sc["dxf"] = str(dxf)
    return sc


def geometry_json(ir: DrawingIR, selective: bool = True) -> str:
    ents = []
    for e in ir.entities:
        if e.type == "line":
            if selective:
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


def codegen_proposal(image: str, grounded: bool = True, rounds: int = 2,
                     out_dir: str | Path | None = None, tag: str | None = None,
                     vlm_model: str | None = None, direct: bool = False,
                     provider: str = "dashscope") -> dict:
    """生成一个代码提案。

    grounded=True 接地(D臂) / False 纯语义(B臂)；
    direct=True   让视觉模型直接看图写脚本（类 Claude 直画），带【错误反馈重试】。
    provider      dashscope / deepseek（deepseek 用 tools.llm.chat 视觉通道）
    """
    out = Path(out_dir) if out_dir else TMP
    out.mkdir(parents=True, exist_ok=True)
    tag = tag or Path(image).stem
    w, h = _wh(image)

    if direct:  # 视觉模型直接看图 -> 代码（带错误反馈重试 + 保最优）
        best = None
        feedback = ""
        for r in range(rounds):
            dxf_path = out / f"{tag}_direct_r{r}.dxf"
            prompt = (CODEGEN_DOC.format(W=w, H=h, OUT=str(dxf_path))
                      + "\n\n请直接看这张图，写出能还原它的完整 Python 脚本。")
            if feedback:
                prompt += f"\n\n上一版脚本执行报错，请修正后重写完整脚本：\n{feedback[:600]}"
            try:
                raw = ask_vision(image, prompt, provider=provider,
                                 model=vlm_model, max_tokens=4096)
            except Exception as e:  # noqa: BLE001
                feedback = f"（调用失败：{type(e).__name__}）"
                continue
            code = extract_code(raw)
            ok, err, dxf = run_script(code, dxf_path)
            if not ok:
                feedback = err
                continue
            sc = _score_dxf(dxf, image, out / f"{tag}_direct_r{r}.png")
            if best is None or sc["ssim"] > best["ssim"]:
                best = sc
            if sc["ssim"] >= 0.90:
                break
        if best is None:
            best = {"ssim": 0.0, "chamfer_px": float("inf"), "n_entities": -1, "png": "", "dxf": ""}
        best["proposal"] = "direct"
        return best

    if grounded:
        ocr = run_ocr(image)
        geom = geometry_json(vectorize(image, ocr=ocr), selective=True)
        texts = [o["text"] for o in ocr]
        desc = ask_image(image, DESCRIBE_PROMPT, model=vlm_model, max_tokens=800)
    else:
        desc = ask_image(image, DESCRIBE_PROMPT, model=vlm_model, max_tokens=1200)

    best = None
    feedback = ""
    for r in range(rounds):
        dxf_path = out / f"{tag}_r{r}.dxf"
        if grounded:
            prompt = D_PROMPT.format(geom=geom, texts=texts, desc=desc, W=w, H=h, OUT=str(dxf_path))
        else:
            prompt = CODEGEN_DOC.format(W=w, H=h, OUT=str(dxf_path)) + f"\n\n图纸的文字描述：\n{desc}"
        user = prompt + (f"\n\n上一版渲染对比原图：{feedback}\n请修正后重写完整脚本。" if feedback else "")

        try:
            code = extract_code(chat([{"role": "user", "content": user}], max_tokens=4096))
        except Exception as e:  # noqa: BLE001
            feedback = f"生成失败：{type(e).__name__}"
            continue

        ok, err, dxf = run_script(code, dxf_path)
        if not ok:
            feedback = f"脚本执行失败，错误：{err[:300]}"
            continue

        sc = _score_dxf(dxf, image, out / f"{tag}_r{r}.png")
        if best is None or sc["ssim"] > best["ssim"]:
            best = sc
        feedback = (f"SSIM={sc['ssim']}, 图元数={sc['n_entities']}, "
                    f"墨迹比 pred={sc['ink_ratio_pred']} vs gt={sc['ink_ratio_gt']}")
        if sc["ssim"] >= 0.9:
            break

    if best is None:
        best = {"ssim": 0.0, "chamfer_px": float("inf"), "n_entities": -1, "png": "", "dxf": ""}
    best["proposal"] = "D" if grounded else "B"
    return best
