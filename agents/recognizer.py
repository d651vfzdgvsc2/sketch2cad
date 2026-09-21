"""识别 Agent：OCR（本地）+ VLM（通义千问）-> 图纸语义理解。"""
from __future__ import annotations

from tools.image_io import imread
from tools.llm import extract_json
from tools.ocr import run_ocr
from tools.vlm import ask_image

VLM_PROMPT = """这是一张黑白线稿图纸（工程/CAD 草图）。请只输出 JSON，不要任何解释、不要代码块：
{{"shape_summary":"一句话描述图纸内容","counts":{{"line":0,"circle":0,"rectangle":0,"polyline":0}},"texts":["图上的文字"]}}
我已用 OCR 识别到这些文字，可参考：{texts}"""


def recognize(state: dict) -> dict:
    img = state["image"]
    arr = imread(img)
    h, w = (arr.shape[:2] if arr is not None else (1000, 1000))

    ocr = run_ocr(img)
    texts = [o["text"] for o in ocr]

    try:
        vlm = extract_json(ask_image(img, VLM_PROMPT.format(texts=texts)))
    except Exception as e:  # noqa: BLE001
        vlm = {"error": str(e), "counts": {}, "texts": []}

    return {"width": float(w), "height": float(h), "ocr": ocr, "vlm": vlm}
