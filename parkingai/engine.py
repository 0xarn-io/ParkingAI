"""Processing engine: camera -> detector -> occupancy -> annotated frames.

Runs a background thread. Detection is throttled to ``detect_interval`` while
the annotated MJPEG frame is refreshed at ``stream_fps`` so the live view stays
smooth even when the (expensive) detector runs once a second.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import AppConfig
from .zones import Zone, zone_coverage


class ZoneState:
    """Tracks a single zone's debounced occupancy."""

    def __init__(self, zone: Zone) -> None:
        self.zone = zone
        self.occupied = False
        self.coverage = 0.0
        self._pending = 0  # consecutive detections disagreeing with `occupied`
        self.last_changed = time.time()

    def update(self, coverage: float, threshold: float, smoothing: int) -> None:
        self.coverage = coverage
        candidate = coverage >= threshold
        if candidate == self.occupied:
            self._pending = 0
            return
        self._pending += 1
        if self._pending >= smoothing:
            self.occupied = candidate
            self._pending = 0
            self.last_changed = time.time()


class Engine:
    def __init__(self, camera, detector, zones: List[Zone], cfg: AppConfig) -> None:
        self.camera = camera
        self.detector = detector
        self.cfg = cfg
        self.states: Dict[str, ZoneState] = {z.id: ZoneState(z) for z in zones}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._last_boxes: List = []
        self._last_detect = 0.0
        self.fps = 0.0
        self.started_at = time.time()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "Engine":
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.camera.stop()

    # -- main loop -------------------------------------------------------
    def _loop(self) -> None:
        target_dt = 1.0 / max(self.cfg.server.stream_fps, 1.0)
        last_t = time.time()
        while self._running:
            loop_start = time.time()
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - self._last_detect >= self.cfg.detect_interval:
                try:
                    self._last_boxes = self.detector.detect(frame)
                except Exception as exc:  # keep the stream alive on detector errors
                    print(f"[engine] detector error: {exc}")
                    self._last_boxes = []
                self._last_detect = now
                self._update_states()

            annotated = self._annotate(frame)
            ok, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.server.jpeg_quality]
            )
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()

            # fps (exponential moving average of the loop rate)
            dt = now - last_t
            last_t = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            sleep = target_dt - (time.time() - loop_start)
            if sleep > 0:
                time.sleep(sleep)

    def _update_states(self) -> None:
        occ = self.cfg.occupancy
        for state in self.states.values():
            cov = zone_coverage(state.zone, self._last_boxes)
            state.update(cov, occ.coverage_threshold, occ.smoothing_frames)

    # -- rendering -------------------------------------------------------
    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        # vehicle boxes (thin grey)
        for b in self._last_boxes:
            x1, y1, x2, y2 = (int(v) for v in b[:4])
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

        free = 0
        for state in self.states.values():
            pts = np.array(state.zone.points, dtype=np.int32).reshape(-1, 1, 2)
            color = (0, 0, 255) if state.occupied else (0, 200, 0)
            if not state.occupied:
                free += 1
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            cx = int(np.mean([p[0] for p in state.zone.points]))
            cy = int(np.mean([p[1] for p in state.zone.points]))
            cv2.putText(
                out, state.zone.id, (cx - 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        total = len(self.states)
        banner = f"Free: {free}/{total}   Occupied: {total - free}   {self.fps:.0f} fps"
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # -- accessors -------------------------------------------------------
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        zones = [
            {
                "id": s.zone.id,
                "occupied": s.occupied,
                "coverage": round(s.coverage, 3),
                "last_changed": s.last_changed,
            }
            for s in self.states.values()
        ]
        total = len(zones)
        occupied = sum(1 for z in zones if z["occupied"])
        return {
            "timestamp": time.time(),
            "camera_connected": getattr(self.camera, "connected", False),
            "fps": round(self.fps, 1),
            "total": total,
            "occupied": occupied,
            "free": total - occupied,
            "zones": zones,
        }
