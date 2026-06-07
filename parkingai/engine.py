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

from . import identity
from .config import AppConfig
from .distortion import Undistorter
from .tracker import Track, Tracker
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
    def __init__(self, camera, detector, zones: List[Zone], cfg: AppConfig,
                 store=None) -> None:
        self.camera = camera
        self.detector = detector
        self.cfg = cfg
        self.store = store
        self.undistorter = Undistorter(cfg.calibration)
        self.states: Dict[str, ZoneState] = {z.id: ZoneState(z) for z in zones}
        self.tracker = Tracker(cfg.recognition.iou_threshold, cfg.recognition.max_misses)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._last_boxes: List = []
        self._last_tracks: List[Track] = []
        # zone_id -> {"session": id, "start": ts, "name": str|None}
        self._open_sessions: Dict[str, dict] = {}
        self._track_names: Dict[int, str] = {}  # track id -> assigned vehicle name
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
        if self.store is not None:
            self.store.close()

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
            # Correct lens distortion first so detection + zones share one space.
            frame = self.undistorter.apply(frame)

            now = time.time()
            if now - self._last_detect >= self.cfg.detect_interval:
                try:
                    detections = self.detector.detect(frame)
                except Exception as exc:  # keep the stream alive on detector errors
                    print(f"[engine] detector error: {exc}")
                    detections = []
                self._last_tracks = self.tracker.update(detections)
                self._last_boxes = [t.box for t in self._last_tracks]
                self._last_detect = now
                self._update_states()
                self._update_sessions(frame, now)

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

    def _zone_occupant(self, zone: Zone) -> Optional[Track]:
        """The track whose box covers this zone the most (the parked car)."""
        best, best_cov = None, 0.0
        for t in self._last_tracks:
            cov = zone_coverage(zone, [t.box])
            if cov > best_cov:
                best, best_cov = t, cov
        return best if best_cov > 0.05 else None

    def _update_sessions(self, frame, now: float) -> None:
        """Open/close parking sessions on occupancy transitions; run re-ID."""
        if self.store is None or not self.cfg.recognition.enabled:
            return
        rec = self.cfg.recognition
        for zid, state in self.states.items():
            open_session = self._open_sessions.get(zid)
            if state.occupied and open_session is None:
                name, vehicle_id = None, None
                track = self._zone_occupant(state.zone)
                if track is not None:
                    emb = identity.embed(frame, track.box)
                    if emb is not None:
                        color = identity.color_name(frame, track.box)
                        vehicle_id, name, _ = self.store.match_or_create(
                            emb, color, rec.reid_threshold, now)
                        self._track_names[track.id] = name
                sid = self.store.open_session(zid, vehicle_id, now)
                self._open_sessions[zid] = {"session": sid, "start": now, "name": name}
            elif not state.occupied and open_session is not None:
                self.store.close_session(
                    open_session["session"], now, rec.min_session_seconds)
                self._open_sessions.pop(zid, None)

    # -- rendering -------------------------------------------------------
    @staticmethod
    def _label(img, text, org, color, scale):
        """Draw text with a solid background box so it's readable on any scene."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), base = cv2.getTextSize(text, font, scale, 2)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + base), color, -1)
        cv2.putText(img, text, (x, y - 2), font, scale, (255, 255, 255), 2, cv2.LINE_AA)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        draw = self.cfg.draw
        green, red = (0, 200, 0), (0, 0, 255)

        # 1) translucent fills (drawn on an overlay, then blended once)
        if draw.fill_alpha > 0:
            overlay = out.copy()
            for state in self.states.values():
                pts = np.array(state.zone.points, np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], red if state.occupied else green)
            cv2.addWeighted(overlay, draw.fill_alpha, out, 1 - draw.fill_alpha, 0, out)

        # 2) tracked vehicles (yellow box + name/track-id label)
        if draw.draw_boxes:
            for t in self._last_tracks:
                x1, y1, x2, y2 = (int(v) for v in t.box[:4])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                name = self._track_names.get(t.id, f"#{t.id}")
                self._label(out, name, (x1, max(y1, 18)), (40, 40, 40),
                            draw.font_scale * 0.8)

        # 3) zone outlines + labels
        free = 0
        for state in self.states.values():
            pts = np.array(state.zone.points, np.int32).reshape(-1, 1, 2)
            color = red if state.occupied else green
            if not state.occupied:
                free += 1
            cv2.polylines(out, [pts], True, color, draw.line_thickness, cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in state.zone.points]))
            cy = int(np.mean([p[1] for p in state.zone.points]))
            tag = f"{state.zone.id}:{'OCC' if state.occupied else 'FREE'}"
            self._label(out, tag, (cx - 24, cy), color, draw.font_scale)

        # 4) status banner
        total = len(self.states)
        h = int(34 * max(draw.font_scale, 0.7))
        cv2.rectangle(out, (0, 0), (out.shape[1], h), (0, 0, 0), -1)
        self._label(out, f"FREE {free}/{total}", (8, h - 8), green, draw.font_scale)
        self._label(out, f"OCCUPIED {total - free}", (190, h - 8), red, draw.font_scale)
        cv2.putText(out, f"{self.fps:.0f} fps", (out.shape[1] - 90, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, draw.font_scale * 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # -- accessors -------------------------------------------------------
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_calibration(self) -> dict:
        return self.cfg.calibration.model_dump()

    def set_calibration(self, data: dict) -> dict:
        """Apply distortion-correction params live (from the UI tuner)."""
        self.undistorter.update(**data)
        return self.get_calibration()

    def status(self) -> dict:
        now = time.time()
        zones = []
        for s in self.states.values():
            sess = self._open_sessions.get(s.zone.id)
            zones.append({
                "id": s.zone.id,
                "occupied": s.occupied,
                "coverage": round(s.coverage, 3),
                "last_changed": s.last_changed,
                "occupant": sess["name"] if sess else None,
                "dwell_seconds": round(now - sess["start"], 1) if sess else None,
            })
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
