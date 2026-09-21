"""探测通义千问哪些视觉模型可用（哪个最强先看能不能调通）。"""
from __future__ import annotations

from pathlib import Path

import requests

from core.config import get
from tools.llm import encode_image_data_url, _SESSION

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "real" / "real03.jpg"

CANDIDATES = [
    "qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-plus", "qwen-vl-plus-latest",
    "qwen3-vl-plus", "qwen3-vl-max", "qwen3-vl-235b-a22b-instruct",
    "qwen2.5-vl-72b-instruct", "qwen2.5-vl-32b-instruct", "qwen2.5-vl-7b-instruct",
    "qwen-vl-max-2025-08-13", "qwen-vl-max-2025-04-08",
]


def main() -> None:
    base = get("DASHSCOPE_BASE_URL").rstrip("/")
    key = get("DASHSCOPE_API_KEY")
    url = base + "/chat/completions"
    data_url = encode_image_data_url(IMG)

    for model in CANDIDATES:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "一句话：这张图是什么？"},
            ]}],
            "max_tokens": 40,
        }
        try:
            r = _SESSION.post(url, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=payload, timeout=60)
            tag = "OK " if r.status_code == 200 else "!! "
            msg = r.text[:120].replace("\n", " ")
            print(f"{tag}{model:30} HTTP {r.status_code}  {msg}")
        except Exception as e:  # noqa: BLE001
            print(f"!! {model:30} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
