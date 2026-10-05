"""从 docs/PRD.md 生成 Sketch2CAD 的 PRD Word 文档到桌面（单一数据源）。

以前这里手写整篇内容，容易和代码脱节；现在以 docs/PRD.md 为准，本脚本只负责渲染。
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "PRD.md"
OUT = Path.home() / "Desktop" / "Sketch2CAD：图像到CAD图纸的多Agent矢量化系统.docx"


def _add_runs(par, text: str) -> None:
    """把 **粗体** 转成加粗 run，其余按普通 run。"""
    for seg in re.split(r"(\*\*[^*]+\*\*)", text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            par.add_run(seg[2:-2]).bold = True
        else:
            par.add_run(seg)


def _mono(par, text: str) -> None:
    run = par.add_run(text)
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.size = Pt(9)


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "微软雅黑"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if line.startswith("```"):
            block, i = [], i + 1
            while i < n and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            _mono(doc.add_paragraph(), "\n".join(block))
            i += 1
            continue

        if line.strip().startswith("|"):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            if rows:
                width = max(len(r) for r in rows)
                table = doc.add_table(rows=len(rows), cols=width)
                table.style = "Light Grid Accent 1"
                for ri, row in enumerate(rows):
                    for ci in range(width):
                        cell = table.cell(ri, ci).paragraphs[0]
                        if ci < len(row):
                            _add_runs(cell, row[ci])
            continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            doc.add_heading(line.lstrip("#").strip(), level=min(level, 4))
            i += 1
            continue

        if re.match(r"^[-*] ", line):
            _add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
            i += 1
            continue

        if re.match(r"^\d+\. ", line):
            _add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\d+\. ", "", line))
            i += 1
            continue

        if line.startswith(">"):
            _add_runs(doc.add_paragraph(style="Intense Quote"), line.lstrip("> ").strip())
            i += 1
            continue

        if line.strip() in ("", "---"):
            i += 1
            continue

        _add_runs(doc.add_paragraph(), line)
        i += 1

    doc.save(str(OUT))
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
