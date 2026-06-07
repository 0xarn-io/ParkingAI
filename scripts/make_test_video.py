"""Generate a synthetic parking-lot video + matching zones.json for testing.

Creates a short clip with a few parking spots; some get a parked "car"
rectangle. Useful to exercise the whole pipeline without a real RTSP camera::

    python scripts/make_test_video.py
    PARKINGAI_CAMERA__SOURCE=test_lot.mp4 python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from parkingai.zones import Zone, save_zones

W, H, FPS, SECONDS = 640, 360, 10, 6
SPOTS = [(60, 120, 120, 220), (200, 120, 260, 220),
         (340, 120, 400, 220), (480, 120, 540, 220)]
# Which spots have a parked car for the whole clip.
PARKED = {0, 2}


def main() -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("test_lot.mp4", fourcc, FPS, (W, H))
    for _ in range(FPS * SECONDS):
        frame = np.full((H, W, 3), 60, np.uint8)  # asphalt grey
        for (x1, y1, x2, y2) in SPOTS:             # painted spot lines
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)
        for i in PARKED:
            x1, y1, x2, y2 = SPOTS[i]
            cv2.rectangle(frame, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (40, 40, 160), -1)
        out.write(frame)
    out.release()

    zones = [Zone(id=str(i + 1), points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
             for i, (x1, y1, x2, y2) in enumerate(SPOTS)]
    save_zones("zones.json", zones)
    print("Wrote test_lot.mp4 and zones.json "
          f"({len(zones)} spots, {len(PARKED)} occupied)")


if __name__ == "__main__":
    main()
