"""Conservative dimension linking and scaling of the ACTUAL selected drawing."""
from __future__ import annotations

import math
import re
from pathlib import Path

import ezdxf
import numpy as np

from emit.ir import DrawingIR, Entity


def parse_dimension(text):
    text = (text or "").strip().replace(" ", "")
    plain = re.fullmatch(r"(\d+(?:\.\d+)?)(mm|毫米)?", text, re.IGNORECASE)
    radius = re.fullmatch(r"[Rr](\d+(?:\.\d+)?)", text)
    diameter = re.fullmatch(r"[ØøΦφ⌀](\d+(?:\.\d+)?)", text)
    if plain:
        return {"kind": "linear", "value": float(plain[1]), "unit": "mm" if plain[2] else None}
    if radius:
        return {"kind": "radius", "value": float(radius[1]), "unit": None}
    if diameter:
        return {"kind": "diameter", "value": float(diameter[1]), "unit": None}
    return {"kind": "other", "value": None, "unit": None}


def _segments(ir):
    lines = []
    for entity in ir.entities:
        if entity.type == "line":
            lines.append((np.array(entity.start), np.array(entity.end)))
        elif entity.type == "polyline":
            points = list(entity.points)
            if entity.closed:
                points.append(points[0])
            lines.extend((np.array(a), np.array(b)) for a, b in zip(points, points[1:]))
    return lines


def link_dimensions(ocr, ir):
    """Require a centered label and perpendicular witnesses at BOTH endpoints.

    This intentionally abstains for split dimension lines, leaders, diameter and
    radius dimensions until their associations are explicitly supplied.
    """
    lines = _segments(ir)
    links = []
    for index, item in enumerate(ocr):
        parsed = parse_dimension(item.get("text"))
        if parsed["kind"] != "linear" or not parsed["value"] or item.get("score", 1) < .85:
            continue
        center = np.array(item["center"])
        box = item.get("box", [*center, *center])
        text_size = max(6., min(box[2]-box[0], box[3]-box[1]))
        candidates = []
        for line_id, (a, b) in enumerate(lines):
            length = float(np.linalg.norm(b-a))
            if length < max(25, 2*text_size):
                continue
            direction = (b-a)/length
            projection = float((center-a)@direction)
            offset = center-a
            normal_dist = abs(float(direction[0]*offset[1]-direction[1]*offset[0]))
            if not (.30*length <= projection <= .70*length and
                    text_size*.3 <= normal_dist <= max(40, text_size*2.5)):
                continue
            witnesses = []
            for endpoint in (a, b):
                matches = []
                for j, (c, d) in enumerate(lines):
                    other_len = float(np.linalg.norm(d-c))
                    if j == line_id or other_len < 5:
                        continue
                    unit = (d-c)/other_len
                    if abs(float(unit@direction)) > .15:
                        continue
                    along = float((endpoint-c)@unit)
                    offset = endpoint-c
                    across = abs(float(unit[0]*offset[1]-unit[1]*offset[0]))
                    if -2 <= along <= other_len+2 and across <= 2:
                        matches.append(j)
                witnesses.append(matches)
            if not witnesses[0] or not witnesses[1] or set(witnesses[0]) & set(witnesses[1]):
                continue
            candidates.append({"ocr_index": index, "line_id": line_id, "length_px": length,
                               "ocr_score": item.get("score", 1),
                               "value": parsed["value"], "mm_per_px": parsed["value"]/length,
                               "witness_ids": witnesses, "label_distance_px": normal_dist,
                               "kind": "linear", "evidence": "two_perpendicular_witnesses"})
        if candidates:
            candidates.sort(key=lambda c: c["label_distance_px"])
            best = candidates[0]
            # Ambiguous nearby alternatives are not silently accepted.
            if len(candidates) > 1 and candidates[1]["label_distance_px"]-best["label_distance_px"] < text_size*.7:
                continue
            links.append(best)
    return links


def estimate_scale(ocr, ir, tol=50., min_len=8.):
    del tol, min_len  # Legacy signature; never return to nearest-line heuristics.
    links = link_dimensions(ocr, ir)
    # Repeated OCR detections of the same dimension are not independent evidence.
    groups = {}
    for link in links:
        groups.setdefault(link["line_id"], []).append(link)
    ambiguous = [group for group in groups.values() if len({link["value"] for link in group}) > 1]
    links = [max(group, key=lambda link: link["ocr_score"]) for group in groups.values()
             if len({link["value"] for link in group}) == 1]
    estimates = [link["mm_per_px"] for link in links]
    result = {"mm_per_px": None, "n": len(links), "n_inliers": 0,
              "estimates": estimates, "links": links, "status": "needs_review",
              "reason": "Need at least two independently linked, consistent dimensions"}
    if ambiguous:
        result.update(ambiguous_ocr=ambiguous, reason="OCR alternatives disagree on a dimension; verify the label")
        return result
    if len(links) < 2:
        return result
    median = float(np.median(estimates))
    if any(abs(v/median-1) > .03 for v in estimates):
        result.update(status="conflict", reason="Dimension scales disagree; pixel geometry retained")
        return result
    result.update(mm_per_px=median, n_inliers=len(links), status="consistent",
                  unit_assumption="mm (drawing convention; verify source units)",
                  reason="Independent dimension links agree within 3%; verify before manufacture")
    return result


def scale_selected_dxf(source, target, factor):
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Positive finite mm/px required")
    from ezdxf.math import Matrix44
    from ezdxf.transform import inplace
    doc = ezdxf.readfile(str(source))
    before = [entity.dxftype() for entity in doc.modelspace()]
    errors = inplace(doc.modelspace(), Matrix44.scale(factor))
    if len(errors):
        raise ValueError("Selected DXF contains entities that could not be scaled")
    if [entity.dxftype() for entity in doc.modelspace()] != before:
        raise ValueError("Scaling changed selected entity types")
    doc.units = ezdxf.units.MM
    doc.saveas(str(target))
    return str(target)


def calibrate_selected(selected, ocr, out):
    ir_path = selected.get("ir")
    if not ir_path or not Path(ir_path).is_file():
        return {"status": "needs_review", "mm_per_px": None,
                "reason": "Selected candidate has no dimension evidence; no replacement geometry generated"}
    ir = DrawingIR.from_json(Path(ir_path).read_text(encoding="utf-8"))
    estimate = estimate_scale(ocr, ir)
    if estimate["mm_per_px"] is not None:
        estimate["dxf"] = scale_selected_dxf(selected["dxf"], Path(out)/"best_realsize.dxf",
                                             estimate["mm_per_px"])
    return estimate
