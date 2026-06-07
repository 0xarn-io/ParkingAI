"""Threaded RTSP / video capture with automatic reconnect.

A background thread continuously grabs the newest frame so consumers always
read the most recent image without buffering latency (important for RTSP).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np


def _parse_source(source: str) -> Union[int, str]:
    """Webcam indices ("0") become ints; everything else stays a string."""
    s = str(source)
    return int(s) if s.isdigit() else s


class Camera:
    """Grabs frames from an RTSP URL, video file, or webcam in a thread."""

    def __init__(
        self,
        source: str,
        rtsp_tcp: bool = True,
        reconnect_delay: float = 3.0,
        loop_file: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.source = _parse_source(source)
        self.rtsp_tcp = rtsp_tcp
        self.reconnect_delay = reconnect_delay
        self.loop_file = loop_file
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.frames_read = 0

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and os.path.exists(self.source)

    def _open(self) -> cv2.VideoCapture:
        if self.rtsp_tcp and isinstance(self.source, str) and self.source.startswith("rtsp"):
            # Force TCP transport - far more reliable than the UDP default.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        if isinstance(self.source, str):
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Keep the internal buffer tiny so read() returns the freshest frame.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> "Camera":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.connected = False
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok:
                # End of a video file: rewind and keep looping if asked.
                if self.is_file and self.loop_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.connected = False
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            self.connected = True
            self.frames_read += 1
            with self._lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent frame, or None if not ready."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
