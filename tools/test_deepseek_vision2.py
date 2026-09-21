"""测试 deepseek-flash / deepseek-v4-pro 是否具备真实视觉能力。"""
from __future__ import annotations

from pathlib import Path

import requests

from core.config import get
from tools.llm import encode_image_data_url

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "real" / "real01.jpg"


def ask(model: str, img: Path) -> None:
    base = get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    key = get("DEEPSEEK_API_KEY")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(img)}},
                {"type": "text", "text": "这张工程图里：1)最外面的形状是什么？2)里面有圆吗？几个？"
                                        "3)左下角的数字是多少？请只回答这三点。"},
            ],
        }],
        "max_tokens": 200,
    }
    r = requests.post(base + "/chat/completions",
                      headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=90)
    print(f"\n=== {model} HTTP {r.status_code} ===")
    if r.status_code == 200:
        print(r.json()["choices"][0]["message"]["content"])
    else:
        print(r.text[:200])


def main() -> None:
    for m in ["deepseek-flash", "deepseek-v4-pro"]:
        try:
            ask(m, IMG)
        except Exception as e:  # noqa: BLE001
            print(f"{m} ERROR: {e}")


if __name__ == "__main__":
    main()
