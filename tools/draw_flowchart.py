"""绘制 Sketch2CAD 流程图（白底黑线）到桌面。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path.home() / "Desktop" / "Sketch2CAD流程图.png"
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
BOLD = FontProperties(fname=r"C:\Windows\Fonts\msyhbd.ttc") if Path(
    r"C:\Windows\Fonts\msyhbd.ttc").exists() else FONT


def box(ax, x, y, w, h, text, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec="black", lw=1.6, zorder=2))
    ax.text(x, y, text, ha="center", va="center", zorder=3,
            fontproperties=(BOLD if bold else FONT), fontsize=fs)


def arrow(ax, p1, p2, label=None):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                                shrinkA=0, shrinkB=0), zorder=1)
    if label:
        ax.text((p1[0] + p2[0]) / 2 + 0.15, (p1[1] + p2[1]) / 2, label,
                ha="left", va="center", fontproperties=FONT, fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 13.5), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # 顶部
    box(ax, 5, 13.3, 3.8, 0.8, "输入：黑白线稿 / 图片", fs=12, bold=True)
    box(ax, 5, 12.1, 4.4, 0.8, "模式选择（手动）", fs=12, bold=True)
    arrow(ax, (5, 12.9), (5, 12.5))

    # 工程图列
    L = 2.7
    box(ax, L, 10.75, 4.6, 1.0,
        "① 图像清洗\n二值化 / 去噪 · 图框标题栏剔除 · 剖面线剔除", fs=9.5)
    box(ax, L, 9.25, 4.6, 1.0,
        "② 精确矢量化 + 图元装配\n直线 · 圆 · 圆弧 · 椭圆（+覆盖度验证）", fs=9.5)
    box(ax, L, 7.5, 5.4, 1.1,
        "③ 多臂提案\nA(CV) · MV(多视图) · D(接地) · B(语义) · DS(DeepSeek直画)", fs=9)
    box(ax, L, 5.9, 4.8, 0.9,
        "④ 校验：渲染回图 + 客观比对\nSSIM / IoU / Chamfer", fs=9.5)
    box(ax, L, 4.45, 4.8, 0.9,
        "⑤ 客观仲裁：取最优 + 保最优\n（输出永不劣于基线）", fs=9.5)
    box(ax, L, 3.0, 4.3, 0.9, "⑥ 比例尺校准\n标注反推 mm/px → 真实尺寸", fs=9.5)

    arrow(ax, (4.2, 11.7), (L, 11.25), "工程图")
    for y1, y2 in [(10.25, 9.75), (8.75, 8.05), (6.95, 6.35),
                   (5.45, 4.9), (4.0, 3.45)]:
        arrow(ax, (L, y1), (L, y2))

    # 假山列
    R = 7.9
    box(ax, R, 9.25, 3.7, 1.0, "② 假山通道\nPotrace 轮廓拟合（有机曲线）", fs=9.5)
    box(ax, R, 7.6, 3.7, 0.9, "曲线 → IR 多段线 → DXF", fs=9.5)
    arrow(ax, (5.9, 11.7), (R, 9.75), "假山")
    arrow(ax, (R, 8.75), (R, 8.05))

    # 汇合输出
    box(ax, 5, 1.3, 6.2, 0.9, "输出：可编辑 DXF + 预览图 + 指标报告", fs=11, bold=True)
    arrow(ax, (L, 2.55), (5, 1.75))
    arrow(ax, (R, 7.15), (5, 1.75))

    # 标题
    ax.text(5, 13.85, "Sketch2CAD 系统流程图", ha="center", va="center",
            fontproperties=BOLD, fontsize=15)

    fig.savefig(str(OUT), dpi=160, facecolor="white", bbox_inches="tight")
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
