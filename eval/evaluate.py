"""阶段1 评测：在合成数据集上跑基线流水线，输出指标表 + report。

用法：python -m eval.evaluate --data data/synth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.pipeline import run
from emit.ir import DrawingIR
from eval.metrics import compare
from tools.image_io import imread

ROOT = Path(__file__).resolve().parent.parent
METRIC_KEYS = ["iou", "dice", "ssim", "chamfer_px"]


def evaluate_sample(png: Path, out_root: Path) -> dict:
    stem = png.stem
    gt_json = png.with_name(f"{stem}_gt.json")
    out_dir = out_root / stem

    ir, _dxf, pred_png = run(str(png), str(out_dir))
    pred_img = imread(pred_png)
    src_img = imread(png)

    metrics = compare(pred_img, src_img, tol=2)

    gt = DrawingIR.from_json(gt_json.read_text(encoding="utf-8"))
    gt_counts = gt.counts()
    pred_counts = ir.counts()

    return {
        "sample": stem,
        "metrics": metrics,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "n_entities": {"gt": len(gt.entities), "pred": len(ir.entities)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    samples = sorted(p for p in data.glob("*.png") if not p.stem.endswith("_gt"))
    if args.limit:
        samples = samples[: args.limit]

    out_root = data.parent / "eval_out"
    rows = [evaluate_sample(p, out_root) for p in samples]

    print(f"\n{'sample':8} {'IoU':>7} {'Dice':>7} {'SSIM':>7} {'Chamfer':>8}  ents(gt->pred)")
    print("-" * 62)
    agg = {k: 0.0 for k in METRIC_KEYS}
    for r in rows:
        m = r["metrics"]
        for k in METRIC_KEYS:
            agg[k] += m[k]
        print(f"{r['sample']:8} {m['iou']:7.4f} {m['dice']:7.4f} {m['ssim']:7.4f} "
              f"{m['chamfer_px']:8.3f}  {r['n_entities']['gt']}->{r['n_entities']['pred']}")

    n = max(len(rows), 1)
    print("-" * 62)
    print(f"{'MEAN':8} {agg['iou']/n:7.4f} {agg['dice']/n:7.4f} {agg['ssim']/n:7.4f} {agg['chamfer_px']/n:8.3f}")

    report = {
        "n_samples": len(rows),
        "mean": {k: round(agg[k] / n, 4) for k in METRIC_KEYS},
        "samples": rows,
    }
    (data.parent / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] {data.parent / 'report.json'}")


if __name__ == "__main__":
    main()
