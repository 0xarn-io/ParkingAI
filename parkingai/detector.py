"""Vehicle detection.

A tiny abstraction over an object detector so the rest of the app does not
depend on ultralytics/torch directly. This makes the pipeline testable with a
``DummyDetector`` and keeps the heavy import lazy.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# A detection: (x1, y1, x2, y2, confidence) in pixel coordinates.
Detection = Tuple[float, float, float, float, float]


class YoloDetector:
    """Wraps an ultralytics YOLO model, filtered to vehicle classes."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou: float = 0.5,
        device: str = "auto",
        classes: Sequence[int] | None = None,
        imgsz: int = 640,
    ) -> None:
        # Imported lazily so the API can run with a different detector even if
        # torch is not installed.
        from ultralytics import YOLO  # noqa: PLC0415

        if device == "auto":
            try:
                import torch  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.model = YOLO(model)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        out: List[Detection] = []
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out


class DummyDetector:
    """Returns a fixed set of boxes. Used for tests and offline demos."""

    def __init__(self, boxes: Sequence[Detection] | None = None) -> None:
        self.boxes: List[Detection] = list(boxes or [])

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return list(self.boxes)


def build_detector(cfg) -> "YoloDetector":
    """Construct the configured YOLO detector from a DetectorConfig."""
    return YoloDetector(
        model=cfg.model,
        confidence=cfg.confidence,
        iou=cfg.iou,
        device=cfg.device,
        classes=cfg.classes,
        imgsz=cfg.imgsz,
    )
