"""测试 DeepSeek 是否有可用的 vision 模型（同一个 key / 同一个 API）。"""
from __future__ import annotations

from pathlib import Path

import requests

from core.config import get
from tools.llm import encode_image_data_url

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "real" / "real01.jpg"

CANDIDATES = [
    "deepseek-chat", "deepseek-vl", "deepseek-vl2", "deepseek-vision",
    "deepseek-vl-7b-chat", "deepseek-vl2-tiny", "deepseek-reasoner",
]


def main() -> None:
    base = get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    key = get("DEEPSEEK_API_KEY")
    url = base + "/chat/completions"
    data_url = encode_image_data_url(IMG)

    for model in CANDIDATES:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "这张图里有什么？一句话。"},
                ],
            }],
            "max_tokens": 60,
        }
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=payload, timeout=60)
            tag = "OK " if r.status_code == 200 else "!! "
            print(f"{tag}{model:22} HTTP {r.status_code}  {r.text[:160].replace(chr(10),' ')}")
        except Exception as e:  # noqa: BLE001
            print(f"!! {model:22} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
