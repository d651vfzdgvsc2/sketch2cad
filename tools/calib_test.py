"""测试比例尺估计。用法：python -m tools.calib_test"""
from __future__ import annotations

from core.calibrate import estimate_scale
from tools.ocr import run_ocr
from vectorize.vectorize import vectorize


def main() -> None:
    for name in ["real01", "real02", "real03", "real04"]:
        p = f"data/real/{name}.jpg"
        ocr = run_ocr(p)
        # 用"保留尺寸线、不装配"的版本做校准
        ir = vectorize(p, params={"assemble": False, "dimension_action": "layer"}, ocr=ocr)
        nums = [o["text"] for o in ocr]
        res = estimate_scale(ocr, ir)
        print(f"{name}: 数字={nums}")
        print(f"    => mm/px≈{res['mm_per_px']}  (样本{res['n']}, 内点{res['n_inliers']})  {res['estimates']}")


if __name__ == "__main__":
    main()
