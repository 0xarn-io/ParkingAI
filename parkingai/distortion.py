"""Lens distortion correction (barrel / pincushion / fisheye).

Most cheap CCTV/bullet cameras add radial (barrel) distortion - straight lines
bow outward near the edges. This straightens them so parking spots line up with
your polygons.

No checkerboard calibration needed: tune a couple of coefficients by eye.
``k1`` is the main knob - use a small *negative* value (e.g. -0.25) to pull in
barrel distortion. The remap is precomputed once, so it's cheap per frame.

IMPORTANT: enable this *before* drawing zones in the editor, so the polygons are
defined in the same (corrected) coordinate space the engine processes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class Undistorter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._size: Optional[Tuple[int, int]] = None
        self._map1 = None
        self._map2 = None

    def _camera_matrix(self, w: int, h: int) -> np.ndarray:
        f = self.cfg.focal_scale * w
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)

    def _build(self, w: int, h: int) -> None:
        c = self.cfg
        K = self._camera_matrix(w, h)
        if c.model == "fisheye":
            D = np.array([c.k1, c.k2, c.k3, 0.0], dtype=np.float64)
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=c.balance
            )
            self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_k, (w, h), cv2.CV_16SC2
            )
        else:  # pinhole / standard radial model
            D = np.array([c.k1, c.k2, c.p1, c.p2, c.k3], dtype=np.float64)
            new_k, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), c.balance, (w, h))
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                K, D, None, new_k, (w, h), cv2.CV_16SC2
            )
        self._size = (w, h)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return frame
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self._build(w, h)
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)
