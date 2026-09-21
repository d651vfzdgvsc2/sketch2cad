"""检查 .env 里的 API Key 是否可用。用法：python tools/check_keys.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        print(f"[!] 找不到 {path}")
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def test(name: str, url: str, key: str, model: str) -> None:
    if not key:
        print(f"--- {name}: 未配置 key，跳过")
        return
    try:
        r = requests.post(
            url,
            timeout=40,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
        )
        flag = "OK " if r.status_code == 200 else "!! "
        print(f"{flag}{name:10} [{model}] HTTP {r.status_code}")
        print("    ", r.text[:240].replace("\n", " "))
    except Exception as e:  # noqa: BLE001
        print(f"!! {name} 连接失败: {type(e).__name__}: {e}")


def main() -> None:
    env = load_env(ROOT / ".env")
    test("DeepSeek", env.get("DEEPSEEK_BASE_URL", "") + "/chat/completions",
         env.get("DEEPSEEK_API_KEY", ""), env.get("DEEPSEEK_MODEL", "deepseek-chat"))
    test("DashScope", env.get("DASHSCOPE_BASE_URL", "") + "/chat/completions",
         env.get("DASHSCOPE_API_KEY", ""), env.get("VLM_MODEL", "qwen-vl-max"))


if __name__ == "__main__":
    main()
