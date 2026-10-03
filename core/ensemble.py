"""集成流水线（正式路径）：多臂提案 + 渲染回图客观仲裁。

提案：
- A：确定性 CV 矢量化（免费、稳定）
- D：接地式代码生成（CV 精确坐标 + OCR + VLM 描述）
- B：纯语义代码生成（VLM 文字 -> 代码）
仲裁：把每个提案渲染回图、与原图比对，取 SSIM 最优者（保最优）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import ezdxf

from agents.codegen_agent import codegen_proposal
from emit.ir import DrawingIR, Entity, LayerSpec
from emit.to_dxf import ir_to_dxf
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.image_io import imread
from tools.ocr import run_ocr
from tools.vlm import ask_vision
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


def multiview_proposal(image: str) -> dict:
    from core.multiview import vectorize_multiview

    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = TMP / f"{stem}_MV.dxf"
    ir, n_views = vectorize_multiview(image)
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, TMP / f"{stem}_MV.png")
    sc["proposal"] = "MV"
    sc["views"] = n_views
    return sc


def centerline_proposal(image: str) -> dict:
    """中心线臂（工程图）：骨架化 → 追踪 → 剪毛刺 → 合并 → 拟合(直线/圆弧/圆) → 文字。"""
    from vectorize.lineart import vectorize_lineart

    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    dxf = TMP / f"{stem}_CL.dxf"
    ir, _stats = vectorize_lineart(image)
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, TMP / f"{stem}_CL.png")
    sc["proposal"] = "CL"
    sc["n_entities"] = len(ir.entities)
    return sc


SPEC_PROMPT = """这是一张 CAD 工程图纸（黑白线稿）。画布尺寸 {W} x {H} 像素，原点在左上角，x 向右、y 向下。
图上 OCR 识别到的文字/尺寸标注有：{texts}

请把图里的几何图形写成一份 JSON 数组（只用像素坐标），元素类型：
- 矩形：{{"type":"rect","x":左,"y":上,"w":宽,"h":高}}
- 圆：  {{"type":"circle","cx":圆心x,"cy":圆心y,"r":半径}}
- 直线：{{"type":"line","x1":..,"y1":..,"x2":..,"y2":..}}
- 多段线：{{"type":"poly","points":[[x,y],...],"closed":true 或 false}}

要求：
1. 用尺寸标注辅助推算位置与大小。
2. 坐标尽量贴近图上真实位置。
3. 外形轮廓优先用一条 poly（闭合多段线），不要一段一段地列直线。
4. 图元总数不要超过 30 个，只描述主要形状，忽略细小纹理/剖面线。
5. 只输出一个 JSON 数组，不要解释、不要代码块。"""


def spec_proposal(image: str, provider: str = "deepseek") -> dict:
    """规格臂：视觉大模型看图输出 JSON 图元规格，再由确定性代码画成 DXF（干净、结构化）。"""
    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    bgr = imread(image)
    h, w = bgr.shape[:2]
    ocr = run_ocr(image)
    texts = [o["text"] for o in ocr]

    raw = ask_vision(image, SPEC_PROMPT.format(W=w, H=h, texts=json.dumps(texts, ensure_ascii=False)),
                     provider=provider, max_tokens=8000)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    spec = json.loads(m.group(0)) if m else []

    entities: list[Entity] = []
    for it in spec:
        t = it.get("type")
        try:
            if t == "rect":
                x, y, ww, hh = it["x"], it["y"], it["w"], it["h"]
                pts = [(x, y), (x + ww, y), (x + ww, y + hh), (x, y + hh)]
                entities.append(Entity(type="polyline", points=[(float(a), float(b)) for a, b in pts],
                                       closed=True))
            elif t == "circle":
                entities.append(Entity(type="circle", center=(float(it["cx"]), float(it["cy"])),
                                       radius=float(it["r"])))
            elif t == "line":
                entities.append(Entity(type="line", start=(float(it["x1"]), float(it["y1"])),
                                       end=(float(it["x2"]), float(it["y2"]))))
            elif t == "poly":
                pts = [(float(a), float(b)) for a, b in it["points"]]
                if len(pts) >= 2:
                    entities.append(Entity(type="polyline", points=pts, closed=bool(it.get("closed"))))
        except Exception:  # noqa: BLE001
            continue
    for o in ocr:  # 文字用真 TEXT
        if o.get("text"):
            cx, cy = o["center"]
            bh = max(o["box"][3] - o["box"][1], 6.0)
            entities.append(Entity(type="text", content=o["text"], pos=(float(cx), float(cy)),
                                   height=float(bh * 0.8), layer="text"))

    ir = DrawingIR(width=float(w), height=float(h), entities=entities,
                   layers=[LayerSpec(name="outline"), LayerSpec(name="text", color=3)])
    dxf = TMP / f"{stem}_SPEC.dxf"
    ir_to_dxf(ir, dxf)
    sc = _score(dxf, image, TMP / f"{stem}_SPEC.png")
    sc["proposal"] = "SPEC"
    sc["n_entities"] = len(entities)
    return sc


def combined_score(sc: dict) -> float:
    """综合仲裁分：SSIM + Dice + Chamfer。

    只看 SSIM 时，线稿图的大面积留白会让“画得少但巧”的臂拿到高分（实测会选错臂）；
    这里把线条重合度（Dice）与几何距离（Chamfer）也纳入，避免选到垃圾输出。
    """
    ssim = float(sc.get("ssim", 0.0))
    dice = float(sc.get("dice", 0.0))
    chamfer = float(sc.get("chamfer_px", 999.0))
    chamfer_term = max(0.0, 1.0 - min(chamfer, 40.0) / 40.0)
    return round(0.4 * ssim + 0.4 * dice + 0.2 * chamfer_term, 6)


# ---------------- 结果缓存（同一张图重复跑时秒出，零质量损失）----------------
CACHE_VERSION = "1"  # 改动流水线后请 +1，避免命中旧结果
CACHE_FILE = TMP / "_cache.json"


def _cache_key(image: str, opts: dict) -> str:
    p = Path(image)
    try:
        st = p.stat()
        sig = f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}"
    except OSError:
        sig = str(image)
    return hashlib.md5((CACHE_VERSION + "|" + sig + "|"
                        + json.dumps(opts, sort_keys=True)).encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _try_proposal(fn, *args, **kwargs):
    """单条臂失败不影响整体：捕获异常并返回 None（由调用方跳过）。"""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[ensemble] 提案失败已跳过：{type(e).__name__}: {e}")
        return None


def run_ensemble(image: str, rounds: int = 2, use_b: bool = True, use_d: bool = True,
                 use_vlm: bool = True, use_mv: bool = True, use_cl: bool = True,
                 use_spec: bool = True, use_cache: bool = True) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    opts = {"rounds": rounds, "b": use_b, "d": use_d, "vlm": use_vlm,
            "mv": use_mv, "cl": use_cl, "spec": use_spec}
    key = _cache_key(image, opts)
    if use_cache:
        hit = _load_cache().get(key)
        if hit and hit.get("best", {}).get("dxf") and Path(hit["best"]["dxf"]).exists():
            print("[ensemble] 命中缓存，直接返回（零耗时）")
            return hit
    proposals: dict[str, dict] = {}

    def _add(name: str, res) -> None:
        if res:
            proposals[name] = res

    _add("A", _try_proposal(deterministic_proposal, image))
    if use_cl:
        _add("CL", _try_proposal(centerline_proposal, image))
    if use_spec:
        _add("SPEC", _try_proposal(spec_proposal, image))
    if use_mv:
        _add("MV", _try_proposal(multiview_proposal, image))
    if use_d:
        _add("D", _try_proposal(codegen_proposal, image, grounded=True, rounds=rounds,
                                out_dir=TMP, tag=f"{stem}_D"))
    if use_b:
        _add("B", _try_proposal(codegen_proposal, image, grounded=False, rounds=rounds,
                                out_dir=TMP, tag=f"{stem}_B"))
    if use_vlm:  # DeepSeek 直接看图写代码（强模型主画）
        ds = _try_proposal(codegen_proposal, image, direct=True, rounds=3,
                           out_dir=TMP, tag=f"{stem}_DS",
                           vlm_model="deepseek-chat", provider="deepseek")
        if ds:
            ds["proposal"] = "DS"
            _add("DS", ds)

    if not proposals:
        raise RuntimeError("所有提案都失败了")

    picked = max(proposals, key=lambda k: combined_score(proposals[k]))
    result = {
        "image": image,
        "picked": picked,
        "best": proposals[picked],
        "proposals": {k: {"ssim": v["ssim"], "dice": v.get("dice"), "iou": v.get("iou"),
                          "chamfer_px": v.get("chamfer_px"), "score": combined_score(v),
                          "n_entities": v["n_entities"], "png": v.get("png", "")}
                      for k, v in proposals.items()},
    }
    if use_cache:
        cache = _load_cache()
        cache[key] = result
        _save_cache(cache)
    return result
