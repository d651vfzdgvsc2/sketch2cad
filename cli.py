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
    p_ens.add_argument("--out", default=None, help="工程图运行结果目录（每次运行自动隔离）")
    p_ens.add_argument("--no-ocr", action="store_true", help="关闭本地OCR，保留原图文字笔画")
    p_ens.add_argument("--semantic", action="store_true", help="可选：调用云端模型复核模板语义")
    p_ens.add_argument("--cloud-proposals", action="store_true", help="可选：额外运行旧云端生成候选")

    p_cal = sub.add_parser("calibrate", help="用尺寸标注校准比例尺，导出真实尺寸 DXF")
    p_cal.add_argument("image")
    p_cal.add_argument("--out", default="out")

    args = ap.parse_args()

    if args.cmd == "run":
        from core.ensemble import run_ensemble
        result = run_ensemble(args.image, out_dir=args.out)
        print(f"工程图结果: {result['picked']}  实体={result['best']['n_entities']}")
        print(f"DXF: {result['best']['dxf']}  预览: {result['best']['png']}")
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

        res = run_ensemble(args.image, rounds=args.rounds, out_dir=args.out,
                           use_ocr=not args.no_ocr, use_semantic=args.semantic,
                           use_b=args.cloud_proposals, use_d=args.cloud_proposals,
                           use_vlm=args.cloud_proposals)
        print(f"各提案固定画布指标: {res['proposals']}")
        print(f"仲裁选中: {res['picked']}  线条F1={res['best']['f1']}  "
              f"实体={res['best']['n_entities']}")
        print(f"DXF : {res['best']['dxf']}")
        print(f"预览: {res['best']['png']}")
    elif args.cmd == "calibrate":
        from core.ensemble import run_ensemble
        result = run_ensemble(args.image, out_dir=args.out)
        print(f"最佳像素版: {result['best']['dxf']}")
        print(f"尺寸关联与校准: {result['calibration']}")


if __name__ == "__main__":
    main()
