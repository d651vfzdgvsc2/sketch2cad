"""阶段2 自测：OCR（本地）+ VLM（通义千问）。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tools.ocr import run_ocr
from tools.vlm import ask_image

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp"


def make_text_image() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (600, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 40), "ROOM A 3500", fill="black")
    d.text((30, 110), "R=120  WALL-01", fill="black")
    img.save(TMP / "test_text.png")
    return TMP / "test_text.png"


def main() -> None:
    print("=== OCR（本地 RapidOCR）===")
    for item in run_ocr(make_text_image()):
        print("  ", item["text"], f"(score={item['score']})")

    print("\n=== VLM（通义千问 qwen-vl-max）看图 ===")
    sample = ROOT / "data" / "synth" / "000.png"
    prompt = (
        "这是一张黑白线稿图纸（工程/CAD 草图）。请只输出 JSON，不要解释："
        '{"shape_summary":"一句话描述","counts":{"line":0,"circle":0,"rectangle":0,"polyline":0},'
        '"texts":["图上的文字"]}'
    )
    print(ask_image(sample, prompt))


if __name__ == "__main__":
    main()
