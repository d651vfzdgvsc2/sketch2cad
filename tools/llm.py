"""DeepSeek 文本大模型封装：生成 / 整理 JSON-IR。"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import requests

from core.config import get

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# 国内 API 直连：忽略系统代理环境变量，避免本地代理没开时报 ProxyError
_SESSION = requests.Session()
_SESSION.trust_env = False


def encode_image_data_url(path: str | Path) -> str:
    p = Path(path)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{_MIME.get(p.suffix.lower(), 'image/png')};base64,{b64}"


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.2,
         max_tokens: int = 2048, timeout: int = 120) -> str:
    base = get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    key = get("DEEPSEEK_API_KEY")
    model = model or get("DEEPSEEK_MODEL", "deepseek-chat")
    r = _SESSION.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"模型未返回 JSON: {text[:200]}")
    return json.loads(m.group(0))


def chat_json(messages: list[dict], **kw) -> dict:
    return extract_json(chat(messages, **kw))
