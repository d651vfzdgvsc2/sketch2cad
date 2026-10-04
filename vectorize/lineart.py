"""工程图线稿通道（独立模块，不影响假山）。

链路：骨架化 → 追踪 → 剪毛刺 → 端点吸附 → 合并碎段 → 拟合(直线/圆弧/圆) → 文字(OCR+可选大模型纠错)。
目标：把“折线拼出来的碎图”升级成“光滑且结构化的图元”。
"""
from __future__ import annotations

import json
import math
import re

import cv2
import numpy as np

from emit.ir import DrawingIR, Entity, LayerSpec
from tools.image_io import imread
from tools.ocr import run_ocr
from tools.vlm import ask_vision
from vectorize.arcs import arc_span, kasa_fit
from vectorize.cleaners import detect_hatching
from vectorize.preprocess import skeleton, to_ink


# ---------------- 骨架处理 ----------------
def _prune_skeleton(mask: np.ndarray, n: int = 10) -> np.ndarray:
    from engineering.trace import prune_spurs
    return prune_spurs(mask, max_length=n)


def _trace_skeleton(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    ys, xs = np.where(mask)
    pts = set(zip(xs.tolist(), ys.tolist()))

    def nbrs(p):
        x, y = p
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in pts:
                    out.append(q)
        return out

    deg = {p: len(nbrs(p)) for p in pts}
    used: set = set()

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    def walk(a, b):
        path = [a, b]
        used.add(edge(a, b))
        prev, cur = a, b
        while deg.get(cur, 0) == 2:
            nxt = [q for q in nbrs(cur) if q != prev]
            if not nxt:
                break
            q = nxt[0]
            if edge(cur, q) in used:
                break
            used.add(edge(cur, q))
            path.append(q)
            prev, cur = cur, q
        return path

    paths: list[list[tuple[int, int]]] = []
    for p in list(pts):
        if deg.get(p, 0) != 2:
            for q in nbrs(p):
                if edge(p, q) not in used:
                    paths.append(walk(p, q))
    for p in list(pts):
        for q in nbrs(p):
            if edge(p, q) not in used:
                paths.append(walk(p, q))
    return paths


def _simplify(path, eps: float = 1.5):
    if len(path) < 3:
        return path
    arr = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    ap = cv2.approxPolyDP(arr, eps, False)
    return [tuple(int(v) for v in q[0]) for q in ap]


def _dir_in(p, e, k=6):
    a = p[0] if e == 0 else p[-1]
    b = p[min(k, len(p) - 1)] if e == 0 else p[max(0, len(p) - 1 - k)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1.0
    return (dx / n, dy / n)


def _turn(di, dj):
    dot = max(-1.0, min(1.0, di[0] * dj[0] + di[1] * dj[1]))
    return 180.0 - math.degrees(math.acos(dot))


def _snap_and_cluster(paths, gap=8.0):
    paths = [list(p) for p in paths if len(p) >= 2]
    n = len(paths)
    keys = [(i, e) for i in range(n) for e in (0, 1)]
    par = {k: k for k in keys}

    def find(k):
        r = k
        while par[r] != r:
            r = par[r]
        while par[k] != r:
            par[k], k = r, par[k]
        return r

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    def ep(i, e):
        return paths[i][0] if e == 0 else paths[i][-1]

    cells = {}
    for k in keys:
        x, y = ep(*k)
        cells.setdefault((int(x // gap), int(y // gap)), []).append(k)
    for (cx, cy), ks in cells.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand += cells.get((cx + dx, cy + dy), [])
        for a in ks:
            for b in cand:
                if a >= b:
                    continue
                pa, pb = ep(*a), ep(*b)
                if (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 <= gap * gap:
                    uni(a, b)
    grp = {}
    for k in keys:
        grp.setdefault(find(k), []).append(k)
    centro = {}
    for mem in grp.values():
        xs = [ep(i, e)[0] for i, e in mem]
        ys = [ep(i, e)[1] for i, e in mem]
        c = (sum(xs) / len(xs), sum(ys) / len(ys))
        for k in mem:
            centro[k] = c
    return paths, grp, centro


def _merge(paths, grp, centro, turn_tol=18.0):
    P = {i: list(paths[i]) for i in range(len(paths))}
    endcluster = {k: r for r, mem in grp.items() for k in mem}
    for i in P:
        P[i][0] = centro[(i, 0)]
        P[i][-1] = centro[(i, 1)]
    while True:
        clusters = {}
        for i in list(P):
            for e in (0, 1):
                clusters.setdefault(endcluster.get((i, e)), []).append((i, e))
        merged_any = False
        for cid, ends in clusters.items():
            ends = [x for x in ends if x[0] in P]
            if len(ends) < 2:
                continue
            best, best_turn = None, 1e9
            for a in range(len(ends)):
                for b in range(a + 1, len(ends)):
                    i, ei = ends[a]
                    j, ej = ends[b]
                    if i == j:
                        continue
                    t = _turn(_dir_in(P[i], ei), _dir_in(P[j], ej))
                    if t < best_turn:
                        best_turn, best = t, (i, ei, j, ej)
            if best and best_turn <= turn_tol:
                i, ei, j, ej = best
                A = P[i] if ei == 1 else P[i][::-1]
                B = P[j] if ej == 0 else P[j][::-1]
                P[i] = A + B[1:]
                oi, oj = (i, 1 - ei), (j, 1 - ej)
                endcluster[(i, 0)] = endcluster.get(oi, cid)
                endcluster[(i, 1)] = endcluster.get(oj, cid)
                del P[j]
                merged_any = True
        if not merged_any:
            break
    return [P[i] for i in P]


def _fit_entities(chains, gap=8.0, rms_max=2.5, r_min=8.0, r_max=4000.0, straight=0.96):
    ents: list[Entity] = []
    for p in chains:
        xs = [q[0] for q in p]
        ys = [q[1] for q in p]
        if max(max(xs) - min(xs), max(ys) - min(ys)) < 5:
            continue
        a = np.array(p, dtype=float)
        a0, a1 = a[0], a[-1]
        closed = (a0[0] - a1[0]) ** 2 + (a0[1] - a1[1]) ** 2 <= gap * gap
        seglen = float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))
        chord = float(np.hypot(a1[0] - a0[0], a1[1] - a0[1]))
        # 1) 直线
        if not closed and chord > 8 and seglen > 1e-6 and chord / seglen > straight:
            ents.append(Entity(type="line", start=(float(a0[0]), float(a0[1])),
                               end=(float(a1[0]), float(a1[1]))))
            continue
        # 2) 圆 / 弧
        pts = a[:-1] if closed else a
        if len(pts) >= 8:
            cx, cy, r, rms = kasa_fit(pts)
            bbox = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
            s, e, span = arc_span(pts, cx, cy)
            if (r_min <= r <= r_max and rms <= rms_max and rms <= max(1.5, 0.08 * r)
                    and r <= 0.9 * bbox + 5):
                if closed and span > 5.6:
                    ents.append(Entity(type="circle", center=(float(cx), float(cy)), radius=float(r)))
                    continue
                if span >= math.radians(35):
                    ents.append(Entity(type="arc", center=(float(cx), float(cy)), radius=float(r),
                                       start_angle=math.degrees(s), end_angle=math.degrees(e)))
                    continue
        # 3) 折线
        sp = _simplify([(int(x), int(y)) for x, y in p], eps=1.5)
        if len(sp) >= 2:
            ents.append(Entity(type="polyline", points=[(float(x), float(y)) for x, y in sp],
                               closed=bool(closed)))
    return ents


# ---------------- 文字 ----------------
def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def _clean_ocr(items):
    out = []
    for it in sorted(items, key=lambda d: -d["score"]):
        t = (it.get("text") or "").strip()
        if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", t):
            continue
        if any(_iou(it["box"], k["box"]) > 0.3 for k in out):
            continue
        out.append(it)
    return out


def _correct_ocr(image, items):
    if not items:
        return items
    texts = [it["text"] for it in items]
    prompt = ("这是一张CAD图纸。下面是从图上 OCR 识别出的文字列表（顺序对应）：\n"
              + json.dumps(texts, ensure_ascii=False)
              + "\n请对照图片，纠正明显的识别错误（如符号、数字认错），"
                "保持数组长度和顺序完全不变，只输出一个 JSON 字符串数组，不要解释。")
    try:
        raw = ask_vision(image, prompt, provider="deepseek", max_tokens=1500)
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) == len(texts):
                for it, t in zip(items, arr):
                    it["text"] = str(t)
    except Exception:  # noqa: BLE001
        pass
    return items


# ---------------- 入口 ----------------
def vectorize_lineart(image: str, prune: int = 10, remove_hatch: bool = True,
                      correct_text: bool = True, ocr: list | None = None):
    """工程图线稿 → IR。返回 (DrawingIR, stats)。"""
    gray = imread(image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image)
    h0, w0 = gray.shape[:2]

    if ocr is None:
        ocr = run_ocr(image)
    ocr = _clean_ocr(ocr)
    if correct_text:
        ocr = _correct_ocr(image, ocr)

    ink = to_ink(gray)
    for o in ocr:  # 遮掉文字区域
        x0, y0, x1, y1 = [int(v) for v in o["box"]]
        ink[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = 0

    if remove_hatch:  # 剔除剖面线（密集斜线填充）
        try:
            hp = {"hatch_min_len": 6, "hatch_max_len": 90, "hatch_min_count": 40,
                  "hatch_cell": 40, "hatch_thr": 6, "hatch_min_region": 0.004}
            ink[detect_hatching(ink, hp) > 0] = 0
        except Exception:  # noqa: BLE001
            pass

    skel = skeleton(ink) > 0
    if prune > 0:
        skel = _prune_skeleton(skel, prune)
    paths = _trace_skeleton(skel)
    paths, grp, centro = _snap_and_cluster(paths, gap=8.0)
    chains = _merge(paths, grp, centro, turn_tol=18.0)
    entities = _fit_entities(chains, gap=8.0)

    for o in ocr:  # 真 TEXT
        if o.get("text"):
            cx, cy = o["center"]
            bh = max(o["box"][3] - o["box"][1], 6.0)
            entities.append(Entity(type="text", content=o["text"],
                                   pos=(float(cx), float(cy)), height=float(bh * 0.8), layer="text"))

    ir = DrawingIR(width=float(w0), height=float(h0), entities=entities,
                   layers=[LayerSpec(name="outline"), LayerSpec(name="text", color=3)],
                   meta={"source": image, "engine": "lineart"})
    stats = {"chunks": len(chains), "entities": len(entities),
             "texts": len([o for o in ocr if o.get("text")])}
    return ir, stats
