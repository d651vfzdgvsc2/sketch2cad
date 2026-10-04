import cv2
import numpy as np

from engineering.ocr_preprocess import ocr_views,suspicious_line_box
from engineering.text import select_annotations


def test_ocr_views_remove_long_lines_keep_digits_and_original():
    image=np.full((160,320,3),255,np.uint8)
    cv2.line(image,(10,110),(300,110),(255,0,0),2)
    cv2.line(image,(250,10),(250,140),(0,0,255),2)
    cv2.putText(image,'101',(30,65),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.circle(image,(180,55),20,(0,0,0),2)
    original=image.copy();views=ocr_views(image)
    assert np.array_equal(image,original)
    assert set(np.unique(views['binary']))=={0,255}
    assert np.all(views['clean'][110,20:230]==255)
    assert np.all(views['clean'][20:100,250]==255)
    assert np.array_equal(views['clean'][30:80,25:100],views['binary'][30:80,25:100])
    assert np.array_equal(views['clean'][30:80,155:205],views['binary'][30:80,155:205])


def test_extreme_box_filter_preserves_long_words_vertical_digits():
    assert suspicious_line_box([0,0,180,4],'1')
    assert not suspicious_line_box([0,0,180,10],'工程图尺寸标注文字')
    assert not suspicious_line_box([0,0,10,40],'100')
    item=dict(text='1',score=.999,box=[0,0,180,4],center=[90,2],box_filter='suspected_line_requires_review')
    assert not select_annotations([item])
    assert select_annotations([{**item,'review_status':'ai_reviewed'}])
