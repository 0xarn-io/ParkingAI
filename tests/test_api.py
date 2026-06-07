import time

import numpy as np
from fastapi.testclient import TestClient

from parkingai.api import create_app
from parkingai.config import AppConfig
from parkingai.detector import DummyDetector
from parkingai.engine import Engine
from parkingai.zones import Zone


class FakeCamera:
    """A camera that always returns the same synthetic frame."""

    connected = True

    def __init__(self):
        self._frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def start(self):
        return self

    def read(self):
        return self._frame.copy()

    def stop(self):
        pass


def build_engine(boxes):
    cfg = AppConfig()
    cfg.detect_interval = 0.0           # detect every loop
    cfg.occupancy.smoothing_frames = 1  # flip immediately
    cfg.server.stream_fps = 30.0
    zone = Zone(id="1", points=[(0, 0), (100, 0), (100, 100), (0, 100)])
    return Engine(FakeCamera(), DummyDetector(boxes), [zone], cfg)


def _wait_for_jpeg(client, tries=50):
    for _ in range(tries):
        r = client.get("/snapshot")
        if r.status_code == 200:
            return r
        time.sleep(0.05)
    return r


def test_status_reports_occupied():
    engine = build_engine([(0, 0, 100, 100, 0.9)])
    app = create_app(engine=engine)
    with TestClient(app) as client:
        _wait_for_jpeg(client)
        # give the engine a moment to run detection at least once
        deadline = time.time() + 3
        data = {}
        while time.time() < deadline:
            data = client.get("/api/status").json()
            if data["occupied"] == 1:
                break
            time.sleep(0.05)
        assert data["total"] == 1
        assert data["occupied"] == 1
        assert data["free"] == 0


def test_status_reports_free_when_no_cars():
    engine = build_engine([])
    app = create_app(engine=engine)
    with TestClient(app) as client:
        _wait_for_jpeg(client)
        time.sleep(0.3)
        data = client.get("/api/status").json()
        assert data["total"] == 1
        assert data["occupied"] == 0
        assert data["free"] == 1


def test_endpoints_available():
    engine = build_engine([])
    app = create_app(engine=engine)
    with TestClient(app) as client:
        _wait_for_jpeg(client)
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert client.get("/api/zones").json()["zones"][0]["id"] == "1"
        assert client.get("/snapshot").headers["content-type"] == "image/jpeg"
