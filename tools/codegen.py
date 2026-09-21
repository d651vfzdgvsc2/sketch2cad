"""LLM 代码生成执行器：抽取 Python 代码 -> 子进程沙箱执行 -> 产出 DXF。

安全：在独立子进程执行，设置超时；仅实验用途。
"""
from __future__ import annotations

import re
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "codegen"


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    return code.strip()


def run_script(code: str, dxf: str | Path, timeout: int = 120) -> tuple[bool, str, Path]:
    """执行 LLM 生成的脚本，校验它是否写出了指定 DXF。返回 (成功?, 错误信息, dxf路径)。"""
    TMP.mkdir(parents=True, exist_ok=True)
    dxf = Path(dxf)
    tag = uuid.uuid4().hex[:8]
    script = TMP / f"script_{tag}.py"
    script.write_text(code, encoding="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(ROOT), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, "执行超时", dxf
    if p.returncode != 0:
        return False, (p.stderr or p.stdout or "未知错误")[-900:], dxf
    if not dxf.exists():
        return False, "脚本没有生成 DXF（请确认保存到了指定路径）", dxf
    return True, "", dxf
