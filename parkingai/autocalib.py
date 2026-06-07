"""Automatic distortion estimation ("guess the correction from a picture").

In a typical CCTV/street scene most long edges (platform lines, kerbs,
railings, road markings) are really straight, so lens distortion is whatever
makes them *look* curved. We sweep the ``k1`` coefficient and keep the value
that maximises the total length of straight lines a line detector can find -
correcting distortion turns bowed edges back into long straight segments.

Undistortion is done with the borders cropped (``balance=0``) so the black
frame edge can't masquerade as a straight line and bias the result.

It's a best-effort guess, not a metric calibration - review it on the live
stream and nudge the slider if needed. Works best on scenes with several long,
genuinely straight lines.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .config import CalibrationConfig
from .distortion import Undistorter


def _straight_line_length(gray: np.ndarray, min_len: float) -> float:
    """Total length of straight segments a probabilistic Hough finds."""
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=70,
        minLineLength=int(min_len), maxLineGap=6,
    )
    if lines is None:
        return 0.0
    seg = lines.reshape(-1, 4).astype(np.float64)
    return float(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]).sum())


def estimate(
    frame: np.ndarray,
    model: str = "pinhole",
    focal_scale: float = 1.0,
    k1_range=(-0.6, 0.4),
    steps: int = 41,
    work_width: int = 640,
) -> Dict:
    """Guess the best ``k1`` for ``frame``. Returns a result dict."""
    h, w = frame.shape[:2]
    scale = work_width / w if w > work_width else 1.0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    sh, sw = frame.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    min_len = sw * 0.15  # only count clearly "long" lines

    baseline = _straight_line_length(gray, min_len)  # score at k1 = 0
    if baseline <= 0:
        return {"ok": False, "reason": "no straight lines found to estimate from"}

    best_k1, best_score = 0.0, baseline
    for k1 in np.linspace(k1_range[0], k1_range[1], steps):
        cfg = CalibrationConfig(enabled=True, model=model, k1=float(k1),
                                focal_scale=focal_scale, balance=0.0)
        warped = Undistorter(cfg).apply(gray)
        score = _straight_line_length(warped, min_len)
        if score > best_score:
            best_k1, best_score = float(k1), score

    improvement = (best_score - baseline) / baseline if baseline > 0 else 0.0
    return {
        "ok": True,
        "k1": round(best_k1, 3),
        "model": model,
        "focal_scale": focal_scale,
        "line_len_before": round(baseline, 1),
        "line_len_after": round(best_score, 1),
        "improvement": round(improvement, 3),  # fractional gain in straight-line length
    }


def _main() -> None:
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(description="Guess lens distortion from a picture")
    ap.add_argument("--image", help="image file to analyse")
    ap.add_argument("--source", help="camera source (RTSP/file/index)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="pinhole", choices=["pinhole", "fisheye"])
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        src = args.source or load_config(args.config).camera.source
        src = int(src) if str(src).isdigit() else src
        cap = cv2.VideoCapture(src)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("could not read a frame from the source")
    if frame is None:
        raise SystemExit("could not load the image")

    res = estimate(frame, model=args.model)
    print(res)
    if res.get("ok"):
        print(f"\nSuggested config.yaml:\n  calibration:\n    enabled: true\n"
              f"    model: {args.model}\n    k1: {res['k1']}")


if __name__ == "__main__":
    _main()
