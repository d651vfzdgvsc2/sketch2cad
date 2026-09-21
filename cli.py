"""命令行入口。

    python cli.py run <图片> --out out/
    python cli.py gen --n 30
    python cli.py eval --limit 10
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="cad-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="单张图 -> DXF")
    p_run.add_argument("image")
    p_run.add_argument("--out", default="out")

    p_gen = sub.add_parser("gen", help="生成合成数据集")
    p_gen.add_argument("--n", type=int, default=30)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--clean", action="store_true")

    p_eval = sub.add_parser("eval", help="在数据集上评测")
    p_eval.add_argument("--data", default="data/synth")
    p_eval.add_argument("--limit", type=int, default=0)

    p_ens = sub.add_parser("ensemble", help="集成流水线：多臂提案 + 客观仲裁（推荐）")
    p_ens.add_argument("image")
    p_ens.add_argument("--rounds", type=int, default=2)

    p_cal = sub.add_parser("calibrate", help="用尺寸标注校准比例尺，导出真实尺寸 DXF")
    p_cal.add_argument("image")
    p_cal.add_argument("--out", default="out")

    args = ap.parse_args()

    if args.cmd == "run":
        from core.pipeline import run

        ir, dxf, png = run(args.image, args.out)
        print(f"图元统计: {ir.counts()}")
        print(f"DXF: {dxf}  预览: {png}")
    elif args.cmd == "gen":
        from datagen.make_dataset import make

        make(args.n, seed=args.seed, apply_distort=not args.clean)
    elif args.cmd == "eval":
        import sys

        from eval import evaluate

        sys.argv = ["evaluate", "--data", args.data, "--limit", str(args.limit)]
        evaluate.main()
    elif args.cmd == "ensemble":
        from core.ensemble import run_ensemble

        res = run_ensemble(args.image, rounds=args.rounds)
        print(f"各提案 SSIM: {res['proposals']}")
        print(f"仲裁选中: {res['picked']}  SSIM={res['best']['ssim']}  "
              f"实体={res['best']['n_entities']}")
        print(f"DXF : {res['best']['dxf']}")
        print(f"预览: {res['best']['png']}")
    elif args.cmd == "calibrate":
        from pathlib import Path

        from core.calibrate import estimate_scale, scale_ir
        from emit.to_dxf import ir_to_dxf
        from tools.ocr import run_ocr
        from vectorize.vectorize import vectorize

        ocr = run_ocr(args.image)
        ir_cal = vectorize(args.image, params={"assemble": False, "dimension_action": "layer"}, ocr=ocr)
        est = estimate_scale(ocr, ir_cal)
        print(f"比例尺估计：{est}")
        if est["mm_per_px"]:
            ir = vectorize(args.image, ocr=ocr)
            ir_scaled = scale_ir(ir, est["mm_per_px"])
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            dxf = out / f"{Path(args.image).stem}_realsize.dxf"
            ir_to_dxf(ir_scaled, dxf)
            print(f"真实尺寸 DXF：{dxf}  (1px = {est['mm_per_px']}mm)")
        else:
            print("未能估计出比例尺（图上可解析的数字标注不足）")


if __name__ == "__main__":
    main()
