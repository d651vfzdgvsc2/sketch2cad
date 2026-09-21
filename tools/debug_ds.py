"""调试 DeepSeek 直画：打印原始输出/代码/执行错误。"""
from __future__ import annotations

import sys
from pathlib import Path

from agents.codegen_agent import CODEGEN_DOC, _wh
from tools.codegen import extract_code, run_script
from tools.vlm import ask_vision

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent


def main(img: str, model: str = "deepseek-chat") -> None:
    w, h = _wh(img)
    out = ROOT / "data" / "tmp" / "dsdraw" / "dbg.dxf"
    prompt = (CODEGEN_DOC.format(W=w, H=h, OUT=str(out))
              + "\n\n请直接看这张图，写出能还原它的完整 Python 脚本。")
    raw = ask_vision(img, prompt, provider="deepseek", model=model, max_tokens=4096)
    print("raw len:", len(raw))
    code = extract_code(raw)
    print("code len:", len(code))
    ok, err, dxf = run_script(code, out)
    print("ok:", ok)
    if not ok:
        print("err:", err[:600])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/real/real03.jpg")
