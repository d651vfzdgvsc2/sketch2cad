"""阶段3 自测：跑一次完整多Agent协同。"""
from __future__ import annotations

import sys
from pathlib import Path

from core.graph import run_agent

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    img = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "synth" / "000.png")
    print(f"输入: {img}\n")
    res = run_agent(img)

    print("VLM 理解:", res.get("vlm"))
    print("OCR 文字:", [o["text"] for o in res.get("ocr", [])])
    print("\n--- Agent 决策日志 ---")
    for item in res.get("log", []):
        print(" ", item)
    print("\n最终指标:", res.get("score"))
    print("最终图元:", res.get("counts"))
    print("预览图 :", res.get("render_png"))
    print("DXF    :", res.get("dxf"))


if __name__ == "__main__":
    main()
