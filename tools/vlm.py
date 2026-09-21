"""视觉大模型（通义千问 Qwen-VL）封装：理解图纸语义。"""
from __future__ import annotations

from pathlib import Path

import requests

from core.config import get
from tools.llm import _SESSION, encode_image_data_url


def ask_image(image_path: str | Path, prompt: str, model: str | None = None,
              max_tokens: int = 1500, timeout: int = 180) -> str:
    base = get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    key = get("DASHSCOPE_API_KEY")
    model = model or get("VLM_MODEL", "qwen-vl-max")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
            {"type": "text", "text": prompt},
        ],
    }]
    r = _SESSION.post(
        base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_vision(image_path, prompt: str, provider: str = "dashscope",
               model: str | None = None, max_tokens: int = 4096, timeout: int = 180) -> str:
    """统一视觉入口：provider = 'dashscope' | 'deepseek'。"""
    if provider == "deepseek":
        from tools.llm import chat

        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
            {"type": "text", "text": prompt},
        ]}]
        return chat(msgs, model=model or "deepseek-chat", max_tokens=max_tokens, timeout=timeout)
    return ask_image(image_path, prompt, model=model, max_tokens=max_tokens, timeout=timeout)
