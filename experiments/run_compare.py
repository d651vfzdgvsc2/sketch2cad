"""跑 A/B/C 三臂对照实验。

用法：python -m experiments.run_compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.arms import ARMS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "real"))
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--only", default="", help="只跑文件名包含该字符串的图")
    args = ap.parse_args()

    data = Path(args.dir)
    imgs = sorted([*data.glob("*.jpg"), *data.glob("*.png")])
    if args.only:
        imgs = [p for p in imgs if args.only in p.name]
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    rows = []
    for img in imgs:
        for a in arms:
            print(f"[run] {img.name}  arm={a} ...", flush=True)
            try:
                r = ARMS[a](str(img))
            except Exception as e:  # noqa: BLE001
                r = {"arm": a, "score": {"ssim": 0.0, "chamfer_px": 0.0, "n_entities": -1},
                     "secs": 0, "error": f"{type(e).__name__}: {e}"}
            r["image"] = img.name
            rows.append(r)
            sc = r.get("score", {})
            print(f"      -> SSIM={sc.get('ssim')}  ents={sc.get('n_entities')}  "
                  f"{r.get('secs')}s  {r.get('error', '')}", flush=True)

    print("\n================ 汇总 ================")
    print(f"{'image':16} {'arm':4} {'SSIM':>7} {'Chamfer':>8} {'实体':>6} {'秒':>6}")
    print("-" * 52)
    for r in rows:
        sc = r.get("score", {})
        print(f"{r['image']:16} {r['arm']:4} {sc.get('ssim', 0):7.4f} "
              f"{sc.get('chamfer_px', 0):8.3f} {sc.get('n_entities', -1):6} {r.get('secs', 0):6.1f}")

    out = ROOT / "data" / "abc_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] {out}")


if __name__ == "__main__":
    main()
