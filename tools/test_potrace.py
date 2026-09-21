"""Test pure-Python Potrace (package 'potracer', import 'potrace')."""
from __future__ import annotations

import cv2
import numpy as np
import potrace

from tools.image_io import imread


def main() -> None:
    gray = imread("data/real/real01.jpg", cv2.IMREAD_GRAYSCALE)
    bw = gray < 128
    bmp = potrace.Bitmap(bw)
    path = bmp.trace()

    n_curves = len(path)
    n_segments = sum(len(c) for c in path)
    print(f"potrace OK: curves={n_curves}, total_segments={n_segments}")

    for i, curve in enumerate(path[:3]):
        print(f"  curve[{i}] start={tuple(round(v, 1) for v in curve.start_point)} "
              f"segments={len(curve)}")

    _ = np  # noqa: F841


if __name__ == "__main__":
    main()
