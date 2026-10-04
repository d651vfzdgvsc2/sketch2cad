"""Measured image anchors and the engineering DXF coordinate contract."""
from __future__ import annotations

import math


def image_to_cad(point, image_height):
    """Pixel drawing: left-top image -> left-bottom CAD, no assumed mm scale."""
    x, y = map(float, point)
    if not all(math.isfinite(v) for v in (x, y, image_height)):
        raise ValueError('Non-finite coordinate')
    return (x, float(image_height)-y)


def annotation_anchor(item):
    """Lock the detector box center; transcription never supplies coordinates."""
    x0, y0, x1, y1 = map(float, item['box'])
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid annotation box')
    return ((x0+x1)/2, (y0+y1)/2)


CONTRACT = dict(image_origin='top_left', cad_origin='bottom_left',
                mapping='X=x; Y=image_height-y', units='pixels',
                anchor='measured_ocr_box_center', model_coordinates_allowed=False,
                physical_scale='only separate evidence-validated calibration')
