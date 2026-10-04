"""Engineering reconstruction with native annotations and continuous CAD geometry."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
import uuid
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from emit.ir import DrawingIR, LayerSpec
from emit.to_dxf import ir_to_doc
from engineering.metrics import compare, combined_score
from engineering.primitives import CATALOG, sample_entity
from engineering.prompts import PROMPT_VERSION, TEMPLATE_PROMPT, validate_semantics
from engineering.recognize import recognize_features, measured_patterns, assign_views
from engineering.render import render_dxf_to_image
from engineering.trace import ink_mask, skeleton
from engineering.cleanup import (continuous_paths, assemble_paths, close_fitted_junctions,
                                 structure_report, heal_scan_strokes, repair_color_crossings,
                                 assemble_centerlines, choose_assemblies)
from engineering.text import read_annotations, extract_annotations, configure_native_text
from engineering.review import review_geometry
from engineering.dimensions import add_native_dimensions
from tools.image_io import imread, imwrite

ROOT = Path(__file__).resolve().parent.parent
VERSION = "engineering-cad-v3-review"


def source_version():
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(p.name.encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()[:16]


def annotation_mask(shape, ocr):
    mask = np.zeros(shape, np.uint8)
    for item in ocr:
        if float(item.get("score", 0)) < .65:
            continue
        x0, y0, x1, y1 = map(int, item["box"])
        x0, y0 = max(0, x0-1), max(0, y0-1)
        x1, y1 = min(shape[1], x1+2), min(shape[0], y1+2)
        mask[y0:y1, x0:x1] = 255
    return mask


def safe_ocr(image):
    try:
        return read_annotations(image), None
    except Exception as exc:
        # OCR failure cannot destroy otherwise usable line geometry.
        return [], f"OCR unavailable: {type(exc).__name__}"


def write_ir(ir, target):
    doc = ir_to_doc(ir)
    doc.units = 0  # DXF $INSUNITS=0: pixel coordinates must not pretend to be mm.
    # CAD lineweights describe plotted pen widths, never source pixel thickness.
    doc.header['$LWDISPLAY'] = False
    for layer in doc.layers:
        layer.dxf.lineweight = 18 if layer.dxf.name == 'text' else 25
    for layer, pattern in ir.meta.get('linetype_patterns', {}).items():
        name = f'ENG_{layer.upper()}'
        doc.linetypes.new(name, dxfattribs={'description': 'Measured engineering dashes', 'pattern': pattern})
        doc.layers.get(layer).dxf.linetype = name
    configure_native_text(doc, ir)
    for entity, rgb in zip(doc.modelspace(), ir.meta.get("stroke_colors_rgb", [])):
        if rgb is not None:
            entity.rgb = tuple(rgb)
    dimension_report = add_native_dimensions(doc, ir)
    ir.meta['native_dimensions'] = dimension_report
    doc.saveas(str(target))
    return dimension_report


def score_dxf(dxf, image, png, *, legacy_style=False):
    source = imread(image)
    h, w = source.shape[:2]
    render_dxf_to_image(dxf, png, width=w, height=h, legacy_style=legacy_style)
    score = compare(imread(png), source)
    score.update(dxf=str(dxf), png=str(png),
                 n_entities=len(ezdxf.readfile(str(dxf)).modelspace()))
    return score


def build_proposals(image, ocr=None, templates=True):
    bgr = imread(image)
    if bgr is None:
        raise ValueError("Cannot read engineering image")
    h, w = bgr.shape[:2]
    ocr = ocr or []
    # Mild antialias smoothing suppresses isolated scan notches before thinning.
    ink = ink_mask(cv2.GaussianBlur(bgr, (3, 3), 0))
    texts, glyph_mask, annotations = extract_annotations(ink, ocr)
    clean_ink = ink.copy()
    clean_ink[glyph_mask > 0] = 0
    mask = annotation_mask((h, w), [a for a in annotations if 'text_index' in a])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colored_pixels = ((hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 80)
                      & (np.ptp(bgr.astype(np.int16), axis=2) > 45))
    geometry_ink = clean_ink.copy()
    geometry_ink[colored_pixels] = 0
    geometry_ink = heal_scan_strokes(repair_color_crossings(geometry_ink, colored_pixels))
    color_ink = clean_ink.copy()
    color_ink[~colored_pixels] = 0
    skel = skeleton(geometry_ink)
    if not skel.any():
        raise ValueError("Image contains no usable strokes")
    layers = [LayerSpec(name="outline"), LayerSpec(name="text"), LayerSpec(name="centerline", color=1)]
    paths = continuous_paths(skel)
    color_entities = assemble_paths(continuous_paths(skeleton(color_ink)), []) if color_ink.any() else []
    color_entities, patterns = assemble_centerlines(color_entities)
    layers.extend(LayerSpec(name=name, color=1) for name in patterns)
    base_meta = {"source": str(image), "engine": VERSION, "ocr": ocr, "annotations": annotations,
                 "linetype_patterns": patterns}
    clean = DrawingIR(width=w, height=h, layers=layers,
                      entities=close_fitted_junctions(assemble_paths(paths, []))+color_entities+texts,
                      meta=dict(base_meta))
    proposals = {"CLEAN": clean}
    # Colored centerlines remain in the output, but do not split template contours.
    features, rejected = recognize_features(geometry_ink, skeleton(geometry_ink), mask) if templates else ([], [])
    views = assign_views(geometry_ink, features, mask)
    if features:
        entities, adopted = choose_assemblies(paths, features)
        entities = close_fitted_junctions(entities)+color_entities+texts
        proposals["LIBRARY"] = DrawingIR(width=w, height=h, layers=layers, entities=entities,
                                        meta={**base_meta,
                                              "features": [f.record() for f in adopted]})
    for ir in proposals.values():
        colors = []
        for entity in ir.entities:
            samples = sample_entity(entity, spacing=2.)
            if not len(samples):
                colors.append(None)
                continue
            xy = np.rint(samples).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, w-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, h-1)
            colored = colored_pixels[xy[:, 1], xy[:, 0]] & (ink[xy[:, 1], xy[:, 0]] > 0)
            if colored.mean() > .5:
                rgb = np.median(bgr[xy[colored, 1], xy[colored, 0]], axis=0)[::-1]
                colors.append([int(v) for v in rgb])
            else:
                colors.append(None)
        ir.meta["stroke_colors_rgb"] = colors
        ir.meta['cad_structure'] = structure_report(ir.entities)
    return proposals, features, {"templates": [f.record() for f in features],
                                "views": views,
                                "patterns": measured_patterns(features), "rejected": rejected,
                                "ocr": ocr, "line_pixels": int(skel.sum()),
                                "annotations": annotations,
                                "annotation_policy": "native TEXT; only isolated recognized glyphs removed; uncertain labels require review"}


def semantic_review(image, features, ocr, provider="dashscope"):
    from tools.vlm import ask_vision
    bgr = imread(image)
    by_view = {}
    for feature in features:
        by_view.setdefault(feature.evidence.get("view_id", "unknown"), []).append(feature.record())
    batches = []
    candidate_batches = [records[i:i+24] for records in by_view.values() for i in range(0, len(records), 24)]
    for batch in candidate_batches:
        prompt = TEMPLATE_PROMPT.format(W=bgr.shape[1], H=bgr.shape[0],
                                        catalog=json.dumps(CATALOG, ensure_ascii=False),
                                        candidates=json.dumps(batch, ensure_ascii=False),
                                        texts=json.dumps(ocr, ensure_ascii=False))
        raw = ask_vision(image, prompt, provider=provider, max_tokens=4000)
        start, end = raw.find("{"), raw.rfind("}")
        payload = json.loads(raw[start:end+1])
        batches.append(validate_semantics(payload, [f["id"] for f in batch]))
    return {"prompt_version": PROMPT_VERSION, "provider": provider, "batches": batches,
            "geometry_modified": False}


def save_diagnostics(image, predicted, out):
    source = imread(image)
    from eval.metrics import to_ink_mask
    a, b = to_ink_mask(source), to_ink_mask(imread(predicted))
    overlay = np.full(source.shape, 255, np.uint8)
    overlay[a] = (210, 110, 20)  # source: blue; prediction: red; overlap: black
    overlay[b] = (30, 40, 225)
    overlay[a & b] = (30, 30, 30)
    imwrite(out / "overlay.png", overlay)
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    error = np.full(source.shape, 255, np.uint8)
    error[a & (db > 2)] = (210, 110, 20)
    error[b & (da > 2)] = (30, 40, 225)
    imwrite(out / "error.png", error)


def run_engineering(image, out_dir=None, *, use_ocr=True, use_templates=True,
                    use_semantic=False, provider="dashscope", legacy_proposals=None,
                    use_ai_review=None):
    start = time.perf_counter()
    image = str(Path(image).resolve())
    digest = hashlib.sha256(Path(image).read_bytes()).hexdigest()
    parent = Path(out_dir).resolve() if out_dir else ROOT / "data" / "engineering"
    out = parent / f"{Path(image).stem}_{digest[:8]}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    ocr, warning = safe_ocr(image) if use_ocr else ([], None)
    raw_ocr = ocr
    from engineering.ai_review import configured, review_annotations
    if use_ai_review is None:
        from core.config import get
        use_ai_review = configured() and get('ENGINEERING_AI_REVIEW', '1') != '0'
    ai_report = {'status': 'disabled'}
    if use_ai_review and use_ocr and ocr:
        try:
            ocr, ai_report = review_annotations(image, ocr, out/'review', provider)
        except Exception as exc:
            ai_report = {'status': 'failed', 'reason': type(exc).__name__}
    irs, features, report = build_proposals(image, ocr, use_templates)
    report['ai_text_review'] = ai_report
    report['ocr_before_review'] = raw_ocr
    source_image = imread(image)
    irs = {name: review_geometry(ir, source_image) for name, ir in irs.items()}
    for ir in irs.values():
        ir.meta['cad_structure'] = structure_report(ir.entities)
    scores = {}
    for name, ir in irs.items():
        dxf = out / f"{name}.dxf"
        dimensions = write_ir(ir, dxf)
        (out / f"{name}.json").write_text(ir.to_json(), encoding="utf-8")
        scores[name] = score_dxf(dxf, image, out / f"{name}.png")
        # Compare geometry independently of font glyph differences. The full
        # image score is retained as a diagnostic, not the acceptance criterion.
        source, rendered = imread(image), imread(out / f"{name}.png")
        text_regions = annotation_mask(source.shape[:2],
                                       [a for a in report['annotations'] if 'text_index' in a])
        source[text_regions > 0] = rendered[text_regions > 0] = 255
        scores[name]['geometry'] = compare(rendered, source)
        scores[name]['cad_structure'] = ir.meta['cad_structure']
        scores[name]['native_dimensions'] = dimensions['native_dimensions']
        scores[name]['geometry_review'] = ir.meta['geometry_review']
        scores[name].update(proposal=name, ir=str(out / f"{name}.json"))
    if legacy_proposals:
        for name, fn in legacy_proposals.items():
            try:
                scores[name] = fn(image, out)
            except Exception as exc:
                report.setdefault("warnings", []).append(f"{name} failed: {type(exc).__name__}")
    eligible = {k: v for k, v in scores.items()
                if v.get("valid", False) and Path(v.get("dxf", "")).is_file()}
    if not eligible:
        raise RuntimeError("No valid engineering candidate")
    def cad_score(candidate):
        structure = candidate.get('cad_structure', {})
        return (combined_score(candidate.get('geometry', candidate))
                - .20*structure.get('short_segment_fraction', 1.)
                - (.15 if ocr and not structure.get('native_texts') else 0.))
    picked = max(eligible, key=lambda k: cad_score(eligible[k]))
    # Templates must preserve geometry completeness and reduce fragmentation.
    lib = eligible.get("LIBRARY")
    best = eligible[picked]
    if lib and cad_score(lib) >= cad_score(best)-.005 and lib['geometry']['recall'] >= best.get('geometry', best)['recall']-.005:
        picked = "LIBRARY"
    selected = dict(scores[picked])
    shutil.copy2(selected["dxf"], out / "best.dxf")
    selected["dxf"] = str(out / "best.dxf")
    # All proposals are scored in pixels, including optional generated scripts.
    selected_doc = ezdxf.readfile(selected["dxf"])
    if selected_doc.units != 0:
        selected_doc.units = 0
        selected_doc.saveas(selected["dxf"])
    save_diagnostics(image, selected["png"], out)
    report.update(image=image, image_sha256=digest, version=VERSION, code_version=source_version(),
                  python=platform.python_version(), opencv=cv2.__version__, ezdxf=ezdxf.__version__,
                  picked=picked, best=selected, proposals=scores, out_dir=str(out),
                  template_count=len(features), selected_templates=(len(irs[picked].meta.get('features', [])) if picked in irs else 0),
                  units="pixels", semantic_enabled=use_semantic)
    report['native_dimensions'] = irs[picked].meta.get('native_dimensions', {}) if picked in irs else {}
    if warning:
        report.setdefault("warnings", []).append(warning)
    unresolved = [a for a in report['annotations'] if a['status'].startswith('review')]
    report['cad_review'] = {'status': 'needs_review',
                            'text_items_to_review': len(unresolved),
                            'ocr_enabled': use_ocr,
                            'ocr_available': warning is None if use_ocr else False,
                            'reason': 'Automatic reconstruction requires topology, text and dimension review; image score is not CAD acceptance',
                            'native_cad_application_verified': False,
                            'manufacturing_dimensions_verified': False}
    if use_semantic and features:
        try:
            report["semantic_review"] = semantic_review(image, features, ocr, provider)
        except Exception as exc:
            report.setdefault("warnings", []).append(f"Semantic review failed: {type(exc).__name__}")
    if use_ai_review:
        from engineering.ai_review import review_final_drawing
        report['ai_final_review'] = review_final_drawing(image, selected['png'],
            {'entity_count': selected['n_entities'],
             'native_dimensions': selected.get('native_dimensions', 0),
             'geometry_review': selected.get('geometry_review', {})}, out, provider)
    from engineering.calibration import calibrate_selected
    try:
        report["calibration"] = calibrate_selected(selected, ocr, out)
    except Exception as exc:
        report["calibration"] = {"status": "needs_review", "mm_per_px": None,
                                 "reason": f"Calibration unavailable: {type(exc).__name__}"}
    report["seconds"] = round(time.perf_counter()-start, 2)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
