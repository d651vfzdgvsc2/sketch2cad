"""阶段4：批量对比「单Agent基线」 vs 「多Agent协同」。

用法：python -m eval.compare_agent --data data/synth --limit 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.graph import run_agent
from emit.ir import DrawingIR
from emit.to_dxf import ir_to_dxf
from eval.metrics import compare
from render.render_dxf import render_dxf_to_image
from tools.image_io import imread
from vectorize.vectorize import vectorize

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "tmp" / "compare"
KEYS = ["iou", "dice", "ssim", "chamfer_px"]


def score_ir(ir: DrawingIR, image_path: str, tag: str) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    dxf = TMP / f"{tag}.dxf"
    png = TMP / f"{tag}.png"
    ir_to_dxf(ir, dxf)
    render_dxf_to_image(dxf, png, width=ir.width, height=ir.height)
    return compare(imread(png), imread(image_path), tol=2)


def run_one(png: Path) -> dict:
    ir_base = vectorize(str(png))
    s_base = score_ir(ir_base, str(png), f"{png.stem}_base")

    res = run_agent(str(png))
    best_json = res.get("best_ir_json") or res["ir_json"]
    ir_agent = DrawingIR.from_json(best_json)
    agent_score = res.get("best_score") or res["score"]

    gt = DrawingIR.from_json(png.with_name(f"{png.stem}_gt.json").read_text(encoding="utf-8"))

    return {
        "sample": png.stem,
        "baseline": s_base,
        "agent": agent_score,
        "base_counts": ir_base.counts() | {"total": len(ir_base.entities)},
        "agent_counts": ir_agent.counts() | {"total": len(ir_agent.entities)},
        "gt_total": len(gt.entities),
        "rounds": len(res.get("log", [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    data = Path(args.data)
    samples = sorted(p for p in data.glob("*.png") if not p.stem.endswith("_gt"))
    if args.limit:
        samples = samples[: args.limit]

    rows = []
    print(f"\n{'sample':8} | {'base IoU':>12} {'SSIM':>7} | {'agent IoU':>9} {'SSIM':>7} | {'实体 gt/base/agent':>18} | 轮数")
    print("-" * 84)
    for p in samples:
        try:
            r = run_one(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p.stem:8} | ERROR {type(e).__name__}: {e}")
            continue
        rows.append(r)
        b, a = r["baseline"], r["agent"]
        print(f"{r['sample']:8} | {b['iou']:12.4f} {b['ssim']:7.4f} | "
              f"{a['iou']:9.4f} {a['ssim']:7.4f} | "
              f"{r['gt_total']:6} {r['base_counts']['total']:6} {r['agent_counts']['total']:6} | {r['rounds']:4}")

    if not rows:
        print("没有可用样本")
        return

    n = len(rows)
    base_mean = {k: sum(r["baseline"][k] for r in rows) / n for k in KEYS}
    agent_mean = {k: sum(r["agent"][k] for r in rows) / n for k in KEYS}
    print("-" * 66)
    print(f"{'MEAN':8} | {base_mean['iou']:12.4f} {base_mean['ssim']:7.4f} | "
          f"{agent_mean['iou']:9.4f} {agent_mean['ssim']:7.4f} | "
          f"{sum(r['rounds'] for r in rows)/n:4.1f}")

    report = {"n": n, "baseline_mean": base_mean, "agent_mean": agent_mean, "rows": rows}
    out = data.parent / "compare_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# 基线 vs 多Agent协同 对比报告", "",
          f"- 样本数：{n}",
          f"- 基线 平均：IoU {base_mean['iou']:.4f} / Dice {base_mean['dice']:.4f} / "
          f"SSIM {base_mean['ssim']:.4f} / Chamfer {base_mean['chamfer_px']:.3f}",
          f"- 多Agent 平均：IoU {agent_mean['iou']:.4f} / Dice {agent_mean['dice']:.4f} / "
          f"SSIM {agent_mean['ssim']:.4f} / Chamfer {agent_mean['chamfer_px']:.3f}",
          "",
          f"- SSIM 提升：{(agent_mean['ssim'] - base_mean['ssim']):+.4f}",
          f"- Chamfer 下降：{(agent_mean['chamfer_px'] - base_mean['chamfer_px']):+.3f}"]
    (data.parent / "compare_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[report] {out}")
    print(f"[report] {data.parent / 'compare_report.md'}")


if __name__ == "__main__":
    main()
