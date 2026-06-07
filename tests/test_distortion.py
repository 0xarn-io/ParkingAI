import numpy as np

from parkingai.config import CalibrationConfig
from parkingai.distortion import Undistorter


def _frame():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


def test_disabled_is_passthrough():
    u = Undistorter(CalibrationConfig(enabled=False, k1=-0.3))
    f = _frame()
    out = u.apply(f)
    assert out is f  # untouched, same object


def test_pinhole_keeps_shape():
    u = Undistorter(CalibrationConfig(enabled=True, model="pinhole", k1=-0.25))
    f = _frame()
    out = u.apply(f)
    assert out.shape == f.shape
    assert out.dtype == f.dtype


def test_fisheye_keeps_shape():
    u = Undistorter(CalibrationConfig(enabled=True, model="fisheye", k1=-0.05, focal_scale=0.6))
    f = _frame()
    out = u.apply(f)
    assert out.shape == f.shape


def test_maps_are_cached():
    u = Undistorter(CalibrationConfig(enabled=True, model="pinhole", k1=-0.2))
    u.apply(_frame())
    map1 = u._map1
    u.apply(_frame())
    assert u._map1 is map1  # same size -> reused remap tables
