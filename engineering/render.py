"""Fixed-canvas DXF rendering for engineering only. Rockery renderer is untouched."""
from __future__ import annotations

import io
from pathlib import Path

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


def render_dxf_to_image(dxf_path, out_png=None, width=1000, height=1000, dpi=100,
                        legacy_style=False):
    width, height = int(width), int(height)
    if min(width, height) <= 0:
        raise ValueError("Positive pixel canvas required")
    doc = ezdxf.readfile(str(dxf_path))
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        cfg = Configuration(color_policy=ColorPolicy.BLACK if legacy_style else ColorPolicy.COLOR,
                            background_policy=BackgroundPolicy.WHITE,
                            lineweight_scaling=0 if legacy_style else 72/25.4,
                            min_lineweight=1 if legacy_style else 0.18)
        backend = MatplotlibBackend(ax, adjust_figure=False)
        Frontend(RenderContext(doc), backend, config=cfg).draw_layout(doc.modelspace(), finalize=True)
        # Backend.finalize autoscales. Set limits AFTER drawing and disable it.
        ax.set_autoscale_on(False)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.set_size_inches(width/dpi, height/dpi, forward=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches=None, pad_inches=0)
        buf.seek(0)
        result = Image.open(buf).convert("RGB")
        if result.size != (width, height):
            raise ValueError(f"Renderer changed canvas: {result.size} != {(width, height)}")
        if out_png is not None:
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)
            result.save(out_png)
        return result
    finally:
        plt.close(fig)
