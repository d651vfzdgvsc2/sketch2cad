"""图元装配层：把零碎的线段规整成结构化形状。

步骤：
1. 去掉过短的碎片
2. 角度吸附（对齐到 0/45/90/135…）
3. 共线合并（把断成一节节的同一直线接起来）
4. 端点吸附（把接近的端点合成一个点 → 接缝闭合）
5. 矩形装配（4 条边组成闭合四边形 → 合并成一条闭合多段线）
"""
from __future__ import annotations

import math

from emit.ir import DrawingIR, Entity

DEFAULT_PARAMS: dict[str, float] = {
    "asm_min_len": 25.0,       # 短于此的线段丢弃
    "asm_angle_step": 45.0,    # 吸附角度步长
    "asm_angle_tol": 8.0,      # 角度吸附容差(度)
    "asm_merge_angle": 3.0,    # 共线合并角度容差
    "asm_merge_dist": 6.0,     # 共线合并偏移容差
    "asm_merge_gap": 22.0,     # 共线合并间隙容差
    "asm_snap_tol": 14.0,      # 端点吸附容差
    "asm_rect_tol": 16.0,      # 矩形装配端点匹配容差
}


def _len(e: Entity) -> float:
    return math.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1])


def _params(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
    return ang, p1[0] * nx + p1[1] * ny, length


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def snap_angles(lines: list[Entity], step: float, tol: float) -> list[Entity]:
    """把接近 step 倍数的线，绕中点旋转到精确角度（长度不变）。"""
    out = []
    for e in lines:
        ang = math.degrees(math.atan2(e.end[1] - e.start[1], e.end[0] - e.start[0]))
        nearest = round(ang / step) * step
        if abs(((ang - nearest + 180) % 360) - 180) <= tol:
            mx, my = (e.start[0] + e.end[0]) / 2, (e.start[1] + e.end[1]) / 2
            L = _len(e)
            ux, uy = math.cos(math.radians(nearest)), math.sin(math.radians(nearest))
            out.append(Entity(type="line", layer=e.layer,
                              start=(mx - ux * L / 2, my - uy * L / 2),
                              end=(mx + ux * L / 2, my + uy * L / 2)))
        else:
            out.append(e)
    return out


def _merge_one_group(group: list[Entity], gap: float) -> list[Entity]:
    e0 = group[0]
    ang, dist, _ = _params(e0.start, e0.end)
    ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
    px, py = nx * dist, ny * dist
    iv = []
    for e in group:
        t1 = e.start[0] * ux + e.start[1] * uy
        t2 = e.end[0] * ux + e.end[1] * uy
        iv.append((min(t1, t2), max(t1, t2)))
    iv.sort()
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [Entity(type="line", layer=e0.layer,
                   start=(px + ux * a, py + uy * a), end=(px + ux * b, py + uy * b))
            for a, b in merged]


def merge_collinear(lines: list[Entity], angle_tol: float, dist_tol: float, gap: float) -> list[Entity]:
    groups: list[list[Entity]] = []
    for e in lines:
        ang, dist, _ = _params(e.start, e.end)
        placed = False
        for g in groups:
            ga, gd, _ = _params(g[0].start, g[0].end)
            if _angle_diff(ang, ga) <= angle_tol and abs(dist - gd) <= dist_tol:
                g.append(e)
                placed = True
                break
        if not placed:
            groups.append([e])
    out: list[Entity] = []
    for g in groups:
        out.extend(_merge_one_group(g, gap))
    return out


def snap_endpoints(lines: list[Entity], tol: float) -> list[Entity]:
    pts = []
    for i, e in enumerate(lines):
        pts.append((i, 0, e.start))
        pts.append((i, 1, e.end))
    used = [False] * len(pts)
    new_pts = [list(e.start) for e in lines], [list(e.end) for e in lines]
    starts, ends = new_pts
    for i in range(len(pts)):
        if used[i]:
            continue
        idx, which, pt = pts[i]
        grp = [i]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            q = pts[j][2]
            if math.hypot(pt[0] - q[0], pt[1] - q[1]) <= tol:
                grp.append(j)
                used[j] = True
        cx = sum(pts[k][2][0] for k in grp) / len(grp)
        cy = sum(pts[k][2][1] for k in grp) / len(grp)
        for k in grp:
            ii, w, _ = pts[k]
            if w == 0:
                starts[ii] = [cx, cy]
            else:
                ends[ii] = [cx, cy]
    return [Entity(type="line", layer=e.layer, start=tuple(starts[i]), end=tuple(ends[i]))
            for i, e in enumerate(lines)]


def detect_rectangles(lines: list[Entity], tol: float) -> tuple[list[Entity], list[Entity]]:
    """由 2 条水平线 + 2 条垂直线组成闭合四边形 → 合并为一条闭合多段线。"""
    H, V = [], []
    for e in lines:
        ang, _, _ = _params(e.start, e.end)
        if _angle_diff(ang, 0) <= 5:
            H.append(e)
        elif _angle_diff(ang, 90) <= 5:
            V.append(e)

    used: set[int] = set()
    rects: list[Entity] = []
    for a in range(len(H)):
        for b in range(a + 1, len(H)):
            ha, hb = H[a], H[b]
            ya, yb = (ha.start[1] + ha.end[1]) / 2, (hb.start[1] + hb.end[1]) / 2
            if abs(ya - yb) < tol:
                continue
            xa0, xa1 = sorted((ha.start[0], ha.end[0]))
            xb0, xb1 = sorted((hb.start[0], hb.end[0]))
            x0, x1 = max(xa0, xb0), min(xa1, xb1)
            if x1 - x0 < tol:
                continue
            for c in range(len(V)):
                for d in range(c + 1, len(V)):
                    va, vb = V[c], V[d]
                    xc = (va.start[0] + va.end[0]) / 2
                    xd = (vb.start[0] + vb.end[0]) / 2
                    if abs(xc - xd) < tol:
                        continue
                    if not (x0 - tol <= min(xc, xd) and max(xc, xd) <= x1 + tol):
                        continue
                    ya0, ya1 = sorted((va.start[1], va.end[1]))
                    yb0, yb1 = sorted((vb.start[1], vb.end[1]))
                    y0, y1 = max(ya0, yb0), min(ya1, yb1)
                    if abs(y0 - min(ya, yb)) > tol or abs(y1 - max(ya, yb)) > tol:
                        continue
                    rects.append(Entity(type="polyline", layer=ha.layer, closed=True, points=[
                        (min(xc, xd), min(ya, yb)), (max(xc, xd), min(ya, yb)),
                        (max(xc, xd), max(ya, yb)), (min(xc, xd), max(ya, yb))]))
                    used.update({id(ha), id(hb), id(va), id(vb)})
    remaining = [e for e in lines if id(e) not in used]
    return rects, remaining


def assemble_lines(lines: list[Entity], params: dict | None = None):
    """对线元做规整，返回 (lines, rects)。"""
    p = {**DEFAULT_PARAMS, **(params or {})}
    lines = [e for e in lines if _len(e) >= p["asm_min_len"]]
    lines = snap_angles(lines, p["asm_angle_step"], p["asm_angle_tol"])
    lines = merge_collinear(lines, p["asm_merge_angle"], p["asm_merge_dist"], p["asm_merge_gap"])
    lines = snap_endpoints(lines, p["asm_snap_tol"])
    rects, lines = detect_rectangles(lines, p["asm_rect_tol"])
    return lines, rects


def assemble_ir(ir: DrawingIR, params: dict | None = None) -> DrawingIR:
    others = [e for e in ir.entities if e.type != "line"]
    lines, rects = assemble_lines([e for e in ir.entities if e.type == "line"], params)
    return DrawingIR(width=ir.width, height=ir.height, layers=ir.layers,
                     entities=[*lines, *rects, *others],
                     meta={**ir.meta, "assembled": True,
                           "rects": len(rects), "lines_after": len(lines)})
