"""诊断 DashScope 403：区分账号级 / 模型级 / 地域级问题。"""
from __future__ import annotations

from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENDPOINTS = {
    "北京 dashscope.aliyuncs.com": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "新加坡 dashscope-intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}
MODELS = ["qwen-turbo", "qwen-plus", "qwen-vl-max", "qwen-vl-plus"]


def main() -> None:
    env = load_env(ROOT / ".env")
    key = env["DASHSCOPE_API_KEY"]
    for name, base in ENDPOINTS.items():
        print(f"\n=== {name} ===")
        for model in MODELS:
            try:
                r = requests.post(
                    base + "/chat/completions",
                    timeout=30,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                )
                tag = "OK " if r.status_code == 200 else "!! "
                msg = r.text[:130].replace("\n", " ")
                print(f"  {tag}{model:14} HTTP {r.status_code}  {msg}")
            except Exception as e:  # noqa: BLE001
                print(f"  !! {model:14} ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
