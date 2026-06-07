# ParkingAI

Detect free/occupied parking spots from an **RTSP camera**.

You define each parking spot once as a polygon ("zone"); a **YOLO** vehicle
detector (PyTorch/Ultralytics) finds cars each frame; ParkingAI reports
per-zone occupancy over a small **FastAPI** service with a live MJPEG stream
and JSON status. Designed to run on a desktop and **port cleanly to a
Raspberry Pi**.

## How it works

```
RTSP camera ──▶ Camera (threaded, auto-reconnect)
                   │  newest frame
                   ▼
              Detector (YOLOv8, vehicle classes only)
                   │  car bounding boxes
                   ▼
              Zones (polygon coverage + debounce)
                   │  per-spot occupied/free
                   ▼
              FastAPI  ──▶ /api/status (JSON)
                       ──▶ /stream     (annotated MJPEG)
                       ──▶ /           (viewer page)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Point at your camera (keeps creds out of git):
export PARKINGAI_CAMERA__SOURCE="rtsp://user:pass@192.168.1.50:554/stream1"

# 2) Draw your parking spots (desktop with a display):
python -m parkingai.editor          # click polygons, press s to save

# 3) Run the server:
python main.py                      # http://localhost:8000
```

No camera handy? Generate a synthetic clip and run against it:

```bash
python scripts/make_test_video.py            # writes test_lot.mp4 + zones.json
PARKINGAI_CAMERA__SOURCE=test_lot.mp4 python main.py
```

## API

| Endpoint       | Description                                            |
| -------------- | ------------------------------------------------------ |
| `GET /`        | Live viewer page (stream + counts)                     |
| `GET /api/status` | JSON: per-zone occupancy, totals, fps, camera state |
| `GET /api/zones`  | The configured zone polygons                         |
| `GET /stream`  | Annotated MJPEG stream (`multipart/x-mixed-replace`)   |
| `GET /snapshot`| Single annotated JPEG                                  |
| `GET /healthz` | Health / camera-connected check                        |

Example `GET /api/status`:

```json
{
  "timestamp": 1733600000.0,
  "camera_connected": true,
  "fps": 12.0,
  "total": 4, "occupied": 2, "free": 2,
  "zones": [
    {"id": "1", "occupied": true,  "coverage": 0.81, "last_changed": 1733599990.0},
    {"id": "2", "occupied": false, "coverage": 0.02, "last_changed": 1733599980.0}
  ]
}
```

## Configuration

All settings live in [`config.yaml`](config.yaml). Any field can be overridden
with an environment variable using `__` between nested keys:

```bash
export PARKINGAI_SERVER__PORT=9000
export PARKINGAI_DETECT_INTERVAL=2.0
export PARKINGAI_OCCUPANCY__COVERAGE_THRESHOLD=0.2
```

Key knobs:

- `occupancy.coverage_threshold` — how much of a spot a car must cover to count
  as occupied (raise if angled cameras cause false positives).
- `occupancy.smoothing_frames` — consecutive detections before a spot flips
  state (debounces flicker).
- `detect_interval` — how often YOLO runs (seconds). Parking changes slowly, so
  1–2s keeps CPU low without missing anything.

## Defining zones

Zones are polygons in `zones.json`:

```json
{ "zones": [ { "id": "1", "points": [[60,120],[120,120],[120,220],[60,220]] } ] }
```

Use the interactive editor (`python -m parkingai.editor`) to draw them on a
real camera frame, or hand-edit the file.

## Raspberry Pi notes

- Install **`opencv-python-headless`** instead of `opencv-python` (no GUI deps).
  Draw zones once on a desktop, then copy `zones.json` to the Pi.
- Keep `model: yolov8n.pt` and **export to NCNN** for a big CPU speedup:
  ```bash
  yolo export model=yolov8n.pt format=ncnn
  # then set detector.model: "yolov8n_ncnn_model" in config.yaml
  ```
- Lower `detector.imgsz` (e.g. 416 or 320) and raise `detect_interval` (2–3s)
  to cut CPU further — fine for parking.

## Tests

```bash
pip install pytest
pytest -q
```

Geometry, occupancy debouncing, and the API (with a stub detector, no torch
required) are covered.
