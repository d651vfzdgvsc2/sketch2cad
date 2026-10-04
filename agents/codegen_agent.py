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

from core.config import get
from emit.ir import DrawingIR
from engineering.metrics import compare, combined_score
from engineering.render import render_dxf_to_image
from tools.codegen import extract_code, run_script
from tools.image_io import imread
from tools.llm import chat
from tools.ocr import run_ocr
from tools.vlm import ask_image, ask_vision
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "ensemble"

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
doc.units = 0  # 本阶段为像素单位，真实毫米缩放由程序另行处理
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
不要用不确定的 add_hatch 参数。必须保留可确认的完整结构和标注；不要为简化代码省略关键特征。
优先使用规则结构：圆角矩形、两端半圆的长圆槽、同心圆、实测孔阵列。
位置与尺寸以图片像素证据为准；尺寸标注冲突时不修改数字、不进行非等比拉伸。
"""

D_PROMPT = """
你是资深 CAD 绘图工程师。下面有三份信息，请据此写一个 Python 脚本（只用 ezdxf 和标准库）还原这张图。

【1】算法检测候选(JSON，像素坐标仍有拟合误差，包含全部支持的图元类型)：
{geom}

【2】OCR 识别到的文字：{texts}

【3】图纸语义描述：{desc}

【必须遵守的原则】
- 保留外形轮廓、孔、槽、圆弧、椭圆、闭合多段线和有像素证据的短线。
- 尺寸线、尺寸界线、箭头、中心线和文字分图层保留，与原图完整性目标一致。
- 按参数化结构组织几何：长圆槽=两直线+相切半圆；圆角矩形保留圆角；孔阵列只使用实际观察到的孔。
- 不跨视图合并，不把圆弧替成折角。存在歧义时保留候选，不靠常识编造尺寸。
- 优先使用候选的实测坐标；有像素证据才修正。画布宽 {W} 高 {H}，图像坐标原点左上、y 向下；
  ezdxf 的 y 轴向上，请转换：cad_y = {H} - y。
- 脚本最后保存到：r"{OUT}"
只输出一个 ```python 代码块，不要解释。
"""


def _describe(image: str, prompt: str, max_tokens: int) -> str:
    """按 .env 的 VLM_PROVIDER 选择视觉模型：dashscope(通义千问) 或 deepseek。"""
    provider = get("VLM_PROVIDER", "dashscope").strip().lower()
    if provider == "deepseek":
        return ask_vision(image, prompt, provider="deepseek", max_tokens=max_tokens)
    return ask_image(image, prompt, max_tokens=max_tokens)


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
    """Lossless geometry contract. selective is retained for caller compatibility."""
    return json.dumps([{**e.model_dump(exclude_none=True), "id": f"entity_{i:05d}"}
                       for i, e in enumerate(ir.entities)], ensure_ascii=False)


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
                prompt += f"\n\n上一版执行或原图校验反馈，请据此修正：\n{feedback[:600]}"
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
            if best is None or combined_score(sc) > combined_score(best):
                best = sc
            feedback = f"像素对照 precision={sc.get('precision')}, recall={sc.get('recall')}, p95_px={sc.get('p95_px')}"
            if sc.get("f1", 0) >= 0.985:
                break
        if best is None:
            best = {"ssim": 0.0, "chamfer_px": None, "valid": False, "score": 0., "n_entities": -1, "png": "", "dxf": ""}
        best["proposal"] = "direct"
        return best

    if grounded:
        ocr = run_ocr(image)
        geom = geometry_json(vectorize(image, ocr=ocr), selective=True)
        texts = [o["text"] for o in ocr]
        desc = _describe(image, DESCRIBE_PROMPT, 800)
    else:
        desc = _describe(image, DESCRIBE_PROMPT, 1200)

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
        if best is None or combined_score(sc) > combined_score(best):
            best = sc
        feedback = (f"precision={sc.get('precision')}, recall={sc.get('recall')}, p95_px={sc.get('p95_px')}, 图元数={sc['n_entities']}, "
                    f"墨迹比 pred={sc['ink_ratio_pred']} vs gt={sc['ink_ratio_gt']}")
        if sc.get("f1", 0) >= 0.985:
            break

    if best is None:
        best = {"ssim": 0.0, "chamfer_px": None, "valid": False, "score": 0., "n_entities": -1, "png": "", "dxf": ""}
    best["proposal"] = "D" if grounded else "B"
    return best
