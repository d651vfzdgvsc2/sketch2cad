"""Test vtracer (raster -> vector) and potracer availability."""
from __future__ import annotations

import os
import time


def test_vtracer() -> None:
    import vtracer

    t = time.time()
    vtracer.convert_image_to_svg_py(
        "data/real/real01.jpg", "data/tmp/test_vtracer.svg",
        colormode="binary", mode="spline")
    size = os.path.getsize("data/tmp/test_vtracer.svg")
    txt = open("data/tmp/test_vtracer.svg", encoding="utf-8").read()
    print(f"vtracer OK: svg={size} bytes, paths={txt.count('<path')}, "
          f"secs={time.time() - t:.2f}")


def test_potracer() -> None:
    try:
        import potracer  # noqa: F401
        print("potracer OK")
    except Exception as e:  # noqa: BLE001
        print(f"potracer NOT available: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_vtracer()
    test_potracer()
