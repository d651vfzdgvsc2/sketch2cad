"""把多张图横向拼在一起，便于对比。用法：python -m tools.montage 输出 图1 图2 ..."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw


def montage(paths: list[str], labels: list[str], out: str, gap: int = 12) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h + 30), "white")
    d = ImageDraw.Draw(canvas)
    x = 0
    for img, label in zip(imgs, labels):
        canvas.paste(img, (x, 30))
        d.text((x + 6, 8), label, fill="black")
        x += img.width + gap
    canvas.save(out)
    print(f"[OK] {out}")


if __name__ == "__main__":
    out = sys.argv[1]
    paths = sys.argv[2:]
    labels = [Path(p).stem for p in paths]
    montage(paths, labels, out)
