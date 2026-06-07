import cv2
import numpy as np

from parkingai.autocalib import estimate
from parkingai.config import CalibrationConfig
from parkingai.distortion import Undistorter


def straight_grid(w=480, h=360, step=40):
    img = np.full((h, w, 3), 30, np.uint8)
    for x in range(0, w + 1, step):
        cv2.line(img, (x, 0), (x, h), (0, 200, 0), 1)
    for y in range(0, h + 1, step):
        cv2.line(img, (0, y), (w, y), (0, 200, 0), 1)
    return img


def test_straight_image_needs_little_correction():
    res = estimate(straight_grid())
    assert res["ok"] is True
    assert abs(res["k1"]) < 0.1  # already straight -> near-zero guess


def test_detects_and_reduces_distortion():
    # warp a straight grid to introduce curvature, then estimate
    warped = Undistorter(CalibrationConfig(enabled=True, model="pinhole", k1=0.25)).apply(
        straight_grid()
    )
    res = estimate(warped)
    assert res["ok"] is True
    assert abs(res["k1"]) > 0.05            # found a non-trivial correction
    assert res["line_len_after"] >= res["line_len_before"]  # straighter than before


def test_blank_image_reports_failure():
    blank = np.full((360, 480, 3), 30, np.uint8)
    res = estimate(blank)
    assert res["ok"] is False
