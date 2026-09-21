"""调试：让 VLM 直接看图写脚本，打印原始输出与执行错误。"""
from __future__ import annotations

from pathlib import Path

from agents.codegen_agent import CODEGEN_DOC, _wh
from tools.codegen import extract_code, run_script
from tools.vlm import ask_image

ROOT = Path(__file__).resolve().parent.parent


def run(model: str, img: str = "data/real/real03.jpg") -> None:
    w, h = _wh(img)
    out = ROOT / "data" / "tmp" / "vlmtest" / f"dbg_{model}.dxf"
    prompt = (CODEGEN_DOC.format(W=w, H=h, OUT=str(out))
              + "\n\n请直接看这张图，写出能还原它的完整 Python 脚本。")
    raw = ask_image(img, prompt, model=model, max_tokens=4096)
    print(f"=== {model} 原始输出（前600字）===")
    print(raw[:600])
    code = extract_code(raw)
    print(f"--- 抽取代码长度: {len(code)} ---")
    ok, err, dxf = run_script(code, out)
    print(f"执行: ok={ok} err={err[:400]}")


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else "qwen-vl-max")
