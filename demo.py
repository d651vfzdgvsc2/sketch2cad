"""一键演示：跑集成流水线，输出 DXF + 各臂对比图。

用法：
    python demo.py data/real/real01.jpg
    python demo.py data/real/real04.jpg --rounds 3
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.ensemble import run_ensemble
from tools.montage import montage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    image = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[*] 集成流水线处理中: {image}")
    res = run_ensemble(str(image), rounds=args.rounds)

    for name, p in res["proposals"].items():
        mark = "  ← 选中" if name == res["picked"] else ""
        print(f"    {name}: SSIM={p['ssim']:.4f}  实体={p['n_entities']}{mark}")

    # 导出最优 DXF
    best_dxf = out / f"{image.stem}_best.dxf"
    shutil.copy(res["best"]["dxf"], best_dxf)

    # 拼对比图：原图 + 各臂
    panels = [str(image)]
    labels = ["origin"]
    for name, p in res["proposals"].items():
        if p.get("png") and Path(p["png"]).exists():
            panels.append(p["png"])
            labels.append(f"{name} {p['ssim']:.3f}" + (" *" if name == res["picked"] else ""))
    montage(panels, labels, str(out / f"{image.stem}_compare.png"))

    print(f"\n[OK] DXF : {best_dxf}")
    print(f"[OK] 对比图: {out / f'{image.stem}_compare.png'}")
    print(f"     仲裁选中 {res['picked']}，SSIM={res['best']['ssim']:.4f}")


if __name__ == "__main__":
    main()
