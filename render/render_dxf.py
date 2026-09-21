"""DXF -> 图片 渲染（校验回环用）。输出与原始线稿同尺寸、同朝向。"""
from __future__ import annotations

from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ezdxf.addons.drawing import Frontend, RenderContext  # noqa: E402
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration  # noqa: E402
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  # noqa: E402


def render_dxf_to_image(
    dxf_path: str | Path,
    out_png: str | Path | None = None,
    width: float = 1000,
    height: float = 1000,
    dpi: int = 100,
):
    """渲染 DXF 为白底黑线 PNG。返回 PIL.Image。"""
    from PIL import Image

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cfg = Configuration(
        color_policy=ColorPolicy.BLACK,
        background_policy=BackgroundPolicy.WHITE,
        lineweight_scaling=0,
        min_lineweight=1,
    )
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=cfg).draw_layout(msp, finalize=True)

    if out_png is not None:
        fig.savefig(str(out_png), dpi=dpi, facecolor="white")
    else:
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf).convert("RGB")

    plt.close(fig)
    return Image.open(str(out_png)).convert("RGB")
