"""阶段0：合成数据生成器。

ezdxf 画图形(DXF=标准答案) -> 渲染成黑白线稿 -> 加噪声/加粗 -> 得到 (线稿图, 真值) 配对。
用法：
    python -m datagen.make_dataset --n 30
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from emit.ir import DrawingIR, Entity
from emit.to_dxf import ir_to_dxf
from render.render_dxf import render_dxf_to_image
from tools.image_io import imwrite

W, H = 1000, 1000
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "synth"


def _rect(x: float, y: float, w: float, h: float, layer: str = "outline") -> Entity:
    return Entity(type="polyline", layer=layer,
                  points=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True)


def _poly(rng: random.Random, cx: float, cy: float, r: float, n: int) -> Entity:
    pts = []
    for i in range(n):
        ang = 2 * 3.1415926 * i / n + rng.uniform(-0.1, 0.1)
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    return Entity(type="polyline", points=pts, closed=True)


def gen_drawing(rng: random.Random) -> DrawingIR:
    """随机生成一张"规则图元"图纸：矩形房间 + 直线 + 圆 + 多边形。"""
    ents: list[Entity] = []
    margin = 60.0

    for _ in range(rng.randint(2, 4)):  # 矩形房间
        w = rng.uniform(150, 320)
        h = rng.uniform(150, 320)
        x = rng.uniform(margin, W - margin - w)
        y = rng.uniform(margin, H - margin - h)
        ents.append(_rect(x, y, w, h))

    for _ in range(rng.randint(2, 5)):  # 直线
        x1 = rng.uniform(margin, W - margin)
        y1 = rng.uniform(margin, H - margin)
        if rng.random() < 0.5:
            x2 = rng.uniform(margin, W - margin)
            y2 = y1
        else:
            x2 = x1
            y2 = rng.uniform(margin, H - margin)
        ents.append(Entity(type="line", start=(x1, y1), end=(x2, y2)))

    for _ in range(rng.randint(1, 3)):  # 圆
        cx = rng.uniform(margin, W - margin)
        cy = rng.uniform(margin, H - margin)
        ents.append(Entity(type="circle", center=(cx, cy), radius=rng.uniform(30, 90)))

    for _ in range(rng.randint(0, 2)):  # 多边形
        ents.append(_poly(rng, rng.uniform(150, W - 150), rng.uniform(150, H - 150),
                          rng.uniform(60, 130), rng.randint(5, 8)))

    return DrawingIR(width=W, height=H, entities=ents, meta={"gen": "synthetic"})


def distort(bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """把干净的 1px 线稿变得像真实线稿：加粗 + 轻微模糊 + 噪声。"""
    k = rng.choice([1, 2, 2, 3])
    kernel = np.ones((k, k), np.uint8)
    out = cv2.dilate(bgr, kernel, iterations=1)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    noise = rng.gauss(0, 5)
    out = np.clip(out.astype(np.float32) + np.random.normal(noise, 6, out.shape), 0, 255).astype(np.uint8)
    return out


def make(n: int, out_dir: Path = DEFAULT_OUT, seed: int = 0, apply_distort: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rng = random.Random(seed + i)
        ir = gen_drawing(rng)
        gt_dxf = out_dir / f"{i:03d}_gt.dxf"
        ir_to_dxf(ir, gt_dxf)

        img = render_dxf_to_image(gt_dxf, width=W, height=H)
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if apply_distort:
            arr = distort(arr, rng)

        imwrite(out_dir / f"{i:03d}.png", arr)
        (out_dir / f"{i:03d}_gt.json").write_text(ir.to_json(), encoding="utf-8")
    print(f"[OK] 生成 {n} 条样本 -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--clean", action="store_true", help="不加噪声/加粗")
    args = ap.parse_args()
    make(args.n, Path(args.out), args.seed, apply_distort=not args.clean)
