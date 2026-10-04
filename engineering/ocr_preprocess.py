"""Extra OCR views only: original color geometry is never modified."""
from __future__ import annotations

import re
import cv2
import numpy as np


def ocr_views(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink=255-binary
    h,w=gray.shape
    lines=np.zeros_like(ink)
    # Kernels exceed ordinary glyph strokes; round outlines are intentionally
    # retained so 0/O/diameter signs cannot be erased as presumed circles.
    for kernel in (np.ones((1,max(45,w//16)),np.uint8),
                   np.ones((max(45,h//16),1),np.uint8)):
        lines|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    lines=cv2.dilate(lines,np.ones((3,3),np.uint8))
    clean=binary.copy();clean[lines>0]=255
    return dict(gray=gray,binary=binary,line_mask=lines,clean=clean)


def suspicious_line_box(box, text):
    """Flag extreme aspect ratios relative to string length, including vertical."""
    x0,y0,x1,y1=map(float,box)
    width,height=x1-x0,y1-y0
    if width<=0 or height<=0:return True
    ratio=max(width,height)/min(width,height)
    chars=len(re.sub(r'\s+','',text))
    # Long real words and rows of digits are not discarded merely for being wide.
    return ratio>max(14,chars*3.5) or (min(width,height)<2 and ratio>6)
