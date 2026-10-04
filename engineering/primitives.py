"""A parameterized CAD feature library, in image coordinates (y down).

Templates encode exact relationships; recognition supplies measured parameters.
All templates expand into the existing, unchanged DrawingIR entity contract.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from emit.ir import Entity


CATALOG = {
    "line": "two measured endpoints",
    "circle": "center, radius",
    "arc": "center, radius, start_angle, end_angle (clockwise in image coordinates)",
    "ellipse": "center, major/minor semi-axes, rotation in degrees",
    "rectangle": "center, width, height, rotation",
    "rounded_rectangle": "center, width, height, radius, rotation; tangent corners",
    "slot": "center, length, diameter, rotation; two tangent semicircles",
    "hole_array": "explicit measured centers, common radius (never invent holes)",
    "concentric_circles": "center, measured radii",
}


def _positive(*values):
    if any(not math.isfinite(float(v)) or v <= 0 for v in values):
        raise ValueError("Template dimensions must be finite and positive")


def _point(center, x, y, rotation):
    a = math.radians(rotation)
    return (center[0] + x * math.cos(a) - y * math.sin(a),
            center[1] + x * math.sin(a) + y * math.cos(a))


def build(kind: str, p: dict, layer: str = "outline") -> list[Entity]:
    """Instantiate a library feature. Invalid or unknown features fail explicitly."""
    if kind not in CATALOG:
        raise ValueError(f"Unknown engineering template: {kind}")
    c = tuple(p.get("center", (0, 0)))
    angle = float(p.get("rotation", 0))
    if not all(math.isfinite(float(v)) for v in (*c, angle)):
        raise ValueError("Non-finite template position")
    if kind == "line":
        points = (*p["start"], *p["end"])
        if not all(math.isfinite(float(v)) for v in points):
            raise ValueError("Non-finite line")
        return [Entity(type="line", start=p["start"], end=p["end"], layer=layer)]
    if kind in ("circle", "arc"):
        _positive(p["radius"])
        fields = {"center": c, "radius": p["radius"], "layer": layer}
        if kind == "arc":
            for key in ("start_angle", "end_angle"):
                fields[key] = float(p[key])
                if not math.isfinite(fields[key]):
                    raise ValueError("Non-finite arc angle")
        return [Entity(type=kind, **fields)]
    if kind == "ellipse":
        _positive(p["major"], p["minor"])
        if p["minor"] > p["major"]:
            raise ValueError("Ellipse minor axis exceeds major axis")
        return [Entity(type="ellipse", center=c, major=p["major"], minor=p["minor"],
                       rotation=angle, layer=layer)]
    if kind == "hole_array":
        _positive(p["radius"])
        if not p["centers"]:
            raise ValueError("A hole array needs measured centers")
        return [e for center in p["centers"]
                for e in build("circle", {"center": center, "radius": p["radius"]}, layer)]
    if kind == "concentric_circles":
        return [e for radius in p["radii"]
                for e in build("circle", {"center": c, "radius": radius}, layer)]

    w = float(p["length"] if kind == "slot" else p["width"])
    h = float(p["diameter"] if kind == "slot" else p["height"])
    _positive(w, h)
    r = h / 2 if kind == "slot" else float(p.get("radius", 0))
    if kind == "rectangle":
        r = 0
    if not math.isfinite(r) or r < 0 or r > min(w, h) / 2 + 1e-8:
        raise ValueError("Corner radius exceeds feature dimensions")
    point = lambda x, y: _point(c, x, y, angle)
    if r == 0:
        return [Entity(type="polyline", closed=True, layer=layer,
                       points=[point(-w/2, -h/2), point(w/2, -h/2),
                               point(w/2, h/2), point(-w/2, h/2)])]
    if kind == "slot":
        a = (w-h)/2
        if a < 1e-8:
            return build("circle", {"center": c, "radius": r}, layer)
        return [
            Entity(type="line", start=point(-a, -r), end=point(a, -r), layer=layer),
            Entity(type="arc", center=point(a, 0), radius=r,
                   start_angle=angle-90, end_angle=angle+90, layer=layer),
            Entity(type="line", start=point(a, r), end=point(-a, r), layer=layer),
            Entity(type="arc", center=point(-a, 0), radius=r,
                   start_angle=angle+90, end_angle=angle+270, layer=layer),
        ]
    x, y = w/2-r, h/2-r
    result = []
    for a, b in [((-x, -h/2), (x, -h/2)), ((w/2, -y), (w/2, y)),
                 ((x, h/2), (-x, h/2)), ((-w/2, y), (-w/2, -y))]:
        if math.dist(a, b) > 1e-8:
            result.append(Entity(type="line", start=point(*a), end=point(*b), layer=layer))
    for x0, y0, start in [(x, -y, -90), (x, y, 0), (-x, y, 90), (-x, -y, 180)]:
        result.append(Entity(type="arc", center=point(x0, y0), radius=r,
                             start_angle=start+angle, end_angle=start+angle+90, layer=layer))
    return result


def sample_entity(e: Entity, spacing: float = 0.7) -> np.ndarray:
    """Dense, ordered geometric samples used for evidence checking."""
    if e.type in ("line", "polyline"):
        pts = [e.start, e.end] if e.type == "line" else list(e.points)
        if e.type == "polyline" and e.closed:
            pts.append(pts[0])
        segments = [np.linspace(a, b, max(2, int(math.dist(a, b)/spacing)+1))
                    for a, b in zip(pts, pts[1:])]
        return np.concatenate(segments)
    if e.type in ("circle", "arc", "ellipse"):
        a, b = (0., 2*math.pi)
        r = e.major if e.type == "ellipse" else e.radius
        if e.type == "arc":
            a = math.radians(e.start_angle)
            b = a + math.radians((e.end_angle-e.start_angle) % 360)
        t = np.linspace(a, b, min(100000, max(12, int((b-a)*r/spacing)+1)))
        if e.type == "ellipse":
            x, y = e.major*np.cos(t), e.minor*np.sin(t)
            rot = math.radians(e.rotation)
            return np.column_stack((e.center[0]+x*np.cos(rot)-y*np.sin(rot),
                                    e.center[1]+x*np.sin(rot)+y*np.cos(rot)))
        return np.column_stack((e.center[0]+r*np.cos(t), e.center[1]+r*np.sin(t)))
    return np.empty((0, 2))


@dataclass
class Feature:
    kind: str
    params: dict
    entities: list[Entity]
    rms: float = 0.0
    support: float = 0.0
    id: str = ""
    evidence: dict = field(default_factory=dict)

    def samples(self):
        return np.concatenate([sample_entity(e) for e in self.entities])

    def record(self):
        return {"id": self.id, "template": self.kind, "params": self.params,
                "rms_px": round(self.rms, 3), "pixel_support": round(self.support, 4),
                "evidence": self.evidence}
