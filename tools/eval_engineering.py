"""Offline acceptance runner. Explicit input list; never discovers rockery images.

python -m tools.eval_engineering image1.jpg image2.jpg --out out/engineering_review
Optional --baseline-dir points at existing historical DXFs, read-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.ensemble import run_ensemble
from engineering.pipeline import score_dxf


def evaluate(images, out, baseline_dir=None, use_ocr=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for image in images:
        image = Path(image)
        result = run_ensemble(str(image), out_dir=out, use_ocr=use_ocr)
        baseline_scores = []
        if baseline_dir:
            destination = Path(result['out_dir']) / 'historical_fixed_canvas'
            destination.mkdir()
            for source in sorted(Path(baseline_dir).glob(f'{image.stem}_*.dxf')):
                if 'rockery' in source.name.lower():
                    continue
                try:
                    score = score_dxf(source, str(image), destination / f'{source.stem}.png', legacy_style=True)
                    if score.get('valid'):
                        baseline_scores.append(score)
                except Exception as exc:
                    result.setdefault('warnings', []).append(f'Baseline {source.name}: {type(exc).__name__}')
            result['historical_candidates'] = baseline_scores
            result['historical_best'] = max(baseline_scores, key=lambda s: s['score']) if baseline_scores else None
        results.append(result)
        (Path(result['out_dir'])/'acceptance.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'image': image.name, 'picked': result['picked'],
                          'f1': result['best']['f1'], 'dice': result['best']['dice'],
                          'old_best_f1': (result.get('historical_best') or {}).get('f1'),
                          'entities': result['best']['n_entities'],
                          'templates_used': result['selected_templates'],
                          'seconds': result['seconds'], 'out_dir': result['out_dir']}, ensure_ascii=True), flush=True)
    (out/'acceptance_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--baseline-dir')
    parser.add_argument('--no-ocr', action='store_true')
    args = parser.parse_args()
    evaluate(args.images, args.out, args.baseline_dir, not args.no_ocr)


if __name__ == '__main__':
    main()
