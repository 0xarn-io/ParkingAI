"""Interactive parking-zone editor (desktop only - needs a display).

Grabs one frame from the configured camera (or a still image) and lets you
click out polygons for each parking spot, then saves them to the zones file.

Usage::

    python -m parkingai.editor                      # uses config.yaml
    python -m parkingai.editor --source video.mp4   # override source
    python -m parkingai.editor --image frame.jpg     # draw on a still

Controls:
    left click   add a point to the current polygon
    n            finish current polygon, start a new one
    u            undo last point
    s            save zones and quit
    q / Esc      quit without saving
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .config import load_config
from .zones import Zone, save_zones


def _grab_frame(source: str):
    import os

    src = int(source) if str(source).isdigit() else source
    if isinstance(src, str) and src.startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from the source")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="ParkingAI zone editor")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="override camera source (RTSP/file/index)")
    ap.add_argument("--image", help="draw on a still image instead of a camera")
    ap.add_argument("--output", help="zones file to write (default: from config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    output = args.output or cfg.zones_file

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
    else:
        frame = _grab_frame(args.source or cfg.camera.source)

    # Draw on the SAME corrected frame the engine processes, so zones line up.
    from .distortion import Undistorter
    frame = Undistorter(cfg.calibration).apply(frame)

    zones: list[Zone] = []
    current: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    win = "ParkingAI zone editor  (n=next  u=undo  s=save  q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def commit_current():
        if len(current) >= 3:
            zones.append(Zone(id=str(len(zones) + 1), points=list(current)))
        current.clear()

    while True:
        disp = frame.copy()
        for z in zones:
            pts = np.array(z.points, np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [pts], True, (0, 200, 0), 2)
            cv2.putText(disp, z.id, z.points[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 0), 2)
        for p in current:
            cv2.circle(disp, p, 3, (0, 165, 255), -1)
        if len(current) >= 2:
            cv2.polylines(disp, [np.array(current, np.int32).reshape(-1, 1, 2)],
                          False, (0, 165, 255), 1)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and current:
            current.pop()
        if key == ord("n"):
            commit_current()
        if key == ord("s"):
            commit_current()
            save_zones(output, zones)
            print(f"Saved {len(zones)} zones -> {output}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
