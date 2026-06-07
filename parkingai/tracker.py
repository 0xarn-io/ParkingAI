"""Lightweight multi-object tracker (IoU association).

Parked cars barely move between detections, so simple greedy IoU matching is
robust and cheap - no Kalman filter or deep tracker needed, which keeps it
Pi-friendly even when detection is throttled to once a second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: Box
    conf: float = 1.0
    hits: int = 1
    misses: int = 0


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Sequence[float]]) -> List[Track]:
        """Associate detections (x1,y1,x2,y2[,conf]) to tracks; return visible ones."""
        dets: List[Box] = [tuple(d[:4]) for d in detections]  # type: ignore[misc]
        confs = [float(d[4]) if len(d) > 4 else 1.0 for d in detections]

        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, db in enumerate(dets):
                score = iou(t.box, db)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_t, matched_d = set(), set()
        for _, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            t = self._tracks[ti]
            t.box, t.conf, t.hits, t.misses = dets[di], confs[di], t.hits + 1, 0
            matched_t.add(ti)
            matched_d.add(di)

        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.misses += 1

        for di, db in enumerate(dets):
            if di not in matched_d:
                self._tracks.append(Track(self._next_id, db, confs[di]))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return [t for t in self._tracks if t.misses == 0]
