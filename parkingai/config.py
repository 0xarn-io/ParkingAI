"""Configuration models and loader.

Config is read from a YAML file (default: ``config.yaml``). Any field can also
be overridden with an environment variable using a double-underscore to
separate nested keys, e.g.::

    export PARKINGAI_CAMERA__SOURCE="rtsp://user:pass@host:554/stream1"
    export PARKINGAI_SERVER__PORT=9000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "PARKINGAI_"


class CameraConfig(BaseModel):
    # RTSP URL, a video file path, or a webcam index as a string ("0").
    source: str = "0"
    # Force TCP transport for RTSP (more reliable than UDP on flaky networks).
    rtsp_tcp: bool = True
    # Seconds to wait before trying to reconnect after a read failure.
    reconnect_delay: float = 3.0
    # Loop a video file source forever (handy for testing).
    loop_file: bool = True
    width: Optional[int] = None
    height: Optional[int] = None


class DetectorConfig(BaseModel):
    # A YOLO weights file/name. yolov8n is the smallest/fastest (good for a Pi).
    model: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    # "auto" -> cuda if available else cpu. Can also be "cpu", "cuda", "0".
    device: str = "auto"
    # COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7.
    classes: List[int] = Field(default_factory=lambda: [2, 3, 5, 7])
    imgsz: int = 640


class OccupancyConfig(BaseModel):
    # Fraction of a zone's area that must be covered by a vehicle box for the
    # zone to count as occupied.
    coverage_threshold: float = 0.15
    # A zone must report the same candidate state for this many consecutive
    # detections before its reported state flips (debounces flicker).
    smoothing_frames: int = 3


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    jpeg_quality: int = 80
    # Target frames/sec for the MJPEG stream + annotation encode.
    stream_fps: float = 12.0


class AppConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    occupancy: OccupancyConfig = Field(default_factory=OccupancyConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    # Path to the JSON file holding parking-spot polygons.
    zones_file: str = "zones.json"
    # Run the (expensive) detector at most once every this many seconds. The
    # video keeps streaming at stream_fps; only detection is throttled. Parking
    # changes slowly, so 1-2s is plenty and keeps a Pi's CPU happy.
    detect_interval: float = 1.0


def _apply_env_overrides(data: dict) -> dict:
    """Overlay PARKINGAI_* environment variables onto a config dict."""
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - defensive
                break
        else:
            node[path[-1]] = value
    return data


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load configuration from YAML (if present) plus environment overrides."""
    data: dict = {}
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    data = _apply_env_overrides(data)
    return AppConfig.model_validate(data)
