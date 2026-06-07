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

## Vehicle recognition & parking statistics

ParkingAI tracks cars, gives each one a friendly made-up name, tries to
re-recognise returning cars, and records parking sessions to SQLite.

- **Tracking** — a light IoU tracker keeps an ID on each car while it's in view
  (robust for mostly-stationary parked cars, cheap on a Pi).
- **Naming + re-ID** — when a car parks, ParkingAI fingerprints it (HSV colour
  histogram) and either matches a previously-seen car or creates a new one with
  a name like *"Silver Lynx"*. Rename any car from the UI or the API.
- **Statistics** — every park is a session: arrival, departure, duration, plus
  per-vehicle totals (visits, total time parked) and per-spot turnover.

> ⚠️ **Re-ID is fuzzy.** Colour-histogram fingerprints will occasionally merge
> two similar cars or fail to recognise the same car in very different light.
> It's a convenience layer, not an identity system. Tune `reid_threshold` in
> `config.yaml` (lower = stricter), and rename cars you care about.

### Stats endpoints

| Endpoint | Description |
| --- | --- |
| `GET /api/stats` | Summary: vehicles known, parked now, sessions today, avg stay, per-zone turnover |
| `GET /api/vehicles` | All known vehicles with visits + total time parked |
| `GET /api/sessions?limit=N` | Recent parking sessions |
| `POST /api/vehicles/{id}/rename` | `{"name": "..."}` — give a car a real name |

The live status (`/api/status`) also shows the current occupant name and dwell
time per spot. Turn the whole feature off with `recognition.enabled: false`.

### Can it identify car make & model?

Yes — make/model recognition (MMR) models exist (e.g. classifiers trained on
the Stanford Cars dataset, Spectrico's ONNX make/model net, or commercial APIs
like Plate Recognizer's MMR). They'd give a far stronger fingerprint than a
colour histogram. Two caveats for this project:

1. **Camera angle** — most MMR models are trained on front/rear views. A
   top-down roof view (like this camera) is out of distribution, so accuracy
   drops a lot. A lower-angle camera on the entrance would work much better.
2. **Weight** — an extra CNN per car adds load, though it's only run when a car
   parks (not every frame), so a Pi can manage it.

The identity layer (`parkingai/identity.py`) is isolated specifically so an MMR
embedding can be concatenated onto the fingerprint later without touching the
engine or storage.

## Fixing lens distortion (barrel / fisheye)

Cheap CCTV lenses bow straight lines outward near the edges, which throws off
rectangular parking zones. Enable correction in `config.yaml`:

```yaml
calibration:
  enabled: true
  model: "pinhole"   # or "fisheye" for very wide lenses
  k1: -0.25          # main knob: more negative = more barrel correction
  focal_scale: 1.0   # lower widens the correction
  balance: 0.0       # 0 crops black borders, 1 keeps all pixels
```

Tune `k1` by eye (start at `-0.25`) until edge lines look straight, then
**re-draw your zones** — the editor applies the same correction so polygons land
in the corrected image. No checkerboard calibration required. (For a precise
calibration you can compute real coefficients with an OpenCV checkerboard and
drop them in here.)

### Tuning it live from the web UI

You don't have to edit YAML and restart. Open `http://localhost:8000` and use
the **Lens distortion** panel — toggle it on and drag the `k1` / `focal` /
`balance` sliders while watching the live stream straighten in real time. Hit
**Save** to persist the values (written to `calibration.json`, which is loaded
automatically on the next start and overrides `config.yaml`). The same numbers
are picked up by the zone editor, so the workflow is: *tune distortion → Save →
re-draw zones.*

### Auto-guessing the correction

Click **Auto-guess** in the UI (or run it offline) and ParkingAI estimates the
correction straight from the camera image: it sweeps `k1` and keeps the value
that turns the scene's bowed edges into the most straight-line length. Best on
scenes with several genuinely straight lines (platform edges, kerbs, railings).
Treat it as a starting point and fine-tune the slider.

```bash
python -m parkingai.autocalib --source "$PARKINGAI_CAMERA__SOURCE"
python -m parkingai.autocalib --image frame.jpg     # or from a saved still
```

It's a heuristic, not a metric calibration — for a precise result use an OpenCV
checkerboard and enter the computed coefficients by hand.

## Adjusting the overlay

The `draw` section controls the on-screen overlay — zone fill translucency,
outline thickness, label size, and whether vehicle detection boxes are shown:

```yaml
draw:
  fill_alpha: 0.28   # 0 = outline only, higher = more solid fill
  line_thickness: 3
  draw_boxes: true   # yellow boxes around detected vehicles
  font_scale: 0.7
```

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
