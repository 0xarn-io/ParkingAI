"""Vehicle appearance fingerprinting and friendly name generation.

The fingerprint is an HSV colour histogram of the car crop - cheap, no model
download, and good enough to *re-match* a returning car most of the time. It is
fuzzy: two similar silver sedans can collide. The identity layer is deliberately
isolated here so a stronger signal (e.g. a make/model classifier embedding) can
be concatenated later without touching the engine or store.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

Box = Sequence[float]

_ANIMALS = [
    "Falcon", "Otter", "Lynx", "Heron", "Bison", "Marten", "Osprey", "Badger",
    "Raven", "Stoat", "Egret", "Ibex", "Puma", "Tern", "Vole", "Wren",
    "Gecko", "Koi", "Mako", "Civet", "Tapir", "Quokka", "Saola", "Dingo",
]


def _crop(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def embed(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """Return a normalised HSV histogram fingerprint, or None if the crop is tiny."""
    crop = _crop(frame, box)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya distance between two fingerprints (0 = identical)."""
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32),
                                 cv2.HISTCMP_BHATTACHARYYA))


def color_name(frame: np.ndarray, box: Box) -> str:
    """Coarse dominant-colour word for a friendly name."""
    crop = _crop(frame, box)
    if crop is None:
        return "Grey"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = (float(np.median(hsv[:, :, i])) for i in range(3))
    if v < 50:
        return "Black"
    if s < 40:
        return "White" if v > 200 else ("Silver" if v > 120 else "Grey")
    if h < 10 or h >= 160:
        return "Red"
    if h < 22:
        return "Orange"
    if h < 33:
        return "Yellow"
    if h < 85:
        return "Green"
    if h < 130:
        return "Blue"
    return "Purple"


def make_name(color: str) -> str:
    return f"{color} {random.choice(_ANIMALS)}"
