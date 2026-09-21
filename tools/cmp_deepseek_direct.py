"""测试 DeepSeek (deepseek-flash / deepseek-v4-pro) 能否看图并写代码画图。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from agents.codegen_agent import CODEGEN_DOC, _score_dxf, _wh
from tools.codegen import extract_code, run_script
from tools.llm import chat, encode_image_data_url

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tmp" / "dstest"
MODELS = ["deepseek-flash", "deepseek-v4-pro", "deepseek-chat"]


def run_one(model: str, img: str) -> None:
    w, h = _wh(img)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"ds_{model}"
    dxf = OUT / f"{tag}.dxf"
    prompt = (CODEGEN_DOC.format(W=w, H=h, OUT=str(dxf))
              + "\n\n请直接看这张图，写出能还原它的完整 Python 脚本。")
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": encode_image_data_url(img)}},
        {"type": "text", "text": prompt},
    ]}]
    t = time.time()
    try:
        raw = chat(messages, model=model, max_tokens=4096)
    except Exception as e:  # noqa: BLE001
        print(f"{model:18} chat失败: {type(e).__name__}: {e}")
        return
    print(f"{model:18} 原始输出长度={len(raw)}  前80字={raw[:80]!r}")
    code = extract_code(raw)
    ok, err, dxfp = run_script(code, dxf)
    if not ok:
        print(f"{model:18} 执行失败 len={len(code)}: {err[:200]}")
        return
    sc = _score_dxf(dxfp, img, OUT / f"{tag}.png")
    print(f"{model:18} SSIM={sc['ssim']:.4f} ent={sc['n_entities']:3} code={len(code)} {round(time.time()-t,1)}s")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/real/real03.jpg"
    print("image:", img)
    for m in MODELS:
        run_one(m, img)
