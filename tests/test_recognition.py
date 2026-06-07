import time

import numpy as np
from fastapi.testclient import TestClient

from parkingai.api import create_app
from parkingai.config import AppConfig
from parkingai.detector import DummyDetector
from parkingai.engine import Engine
from parkingai.store import Store
from parkingai.zones import Zone


class FakeCamera:
    connected = True

    def __init__(self):
        # a non-uniform frame so the colour fingerprint isn't degenerate
        f = np.zeros((120, 160, 3), dtype=np.uint8)
        f[:, :80] = (60, 20, 200)   # reddish on one side
        self._frame = f

    def start(self):
        return self

    def read(self):
        return self._frame.copy()

    def stop(self):
        pass


def build(tmp_path, boxes):
    cfg = AppConfig()
    cfg.detect_interval = 0.0
    cfg.occupancy.smoothing_frames = 1
    cfg.server.stream_fps = 30.0
    cfg.recognition.min_session_seconds = 0.0
    zone = Zone(id="1", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
    store = Store(str(tmp_path / "rec.db"))
    return Engine(FakeCamera(), DummyDetector(boxes), [zone], cfg, store=store), store


def _wait(cond, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_car_parks_then_leaves_records_session(tmp_path):
    detector_boxes = [(0, 0, 100, 100, 0.9)]
    eng, store = build(tmp_path, detector_boxes)
    eng.start()
    try:
        assert _wait(lambda: store.summary()["currently_parked"] == 1)
        st = eng.status()
        zone = st["zones"][0]
        assert zone["occupant"] is not None        # got a made-up name
        assert zone["dwell_seconds"] is not None

        # car leaves
        eng.detector.boxes = []
        assert _wait(lambda: store.summary()["completed_sessions"] == 1)
        summ = store.summary()
        assert summ["vehicles_known"] == 1
        assert summ["currently_parked"] == 0
    finally:
        eng.stop()


def test_stats_endpoints(tmp_path):
    eng, _ = build(tmp_path, [(0, 0, 100, 100, 0.9)])
    app = create_app(engine=eng)
    with TestClient(app) as client:
        assert _wait(lambda: client.get("/api/stats").json()["currently_parked"] == 1)
        vehicles = client.get("/api/vehicles").json()["vehicles"]
        assert len(vehicles) == 1
        vid = vehicles[0]["id"]
        assert client.post(f"/api/vehicles/{vid}/rename",
                           json={"name": "Test Car"}).json()["renamed"] is True
        assert client.get("/api/sessions").json()["sessions"]  # at least one open session
