"""ParkingAI - RTSP parking-spot occupancy detection.

A small pipeline that reads an RTSP camera, runs a YOLO vehicle detector,
and reports per-zone occupancy over a FastAPI HTTP API (JSON status + MJPEG
stream). Parking spots ("zones") are defined manually as polygons.
"""

__version__ = "0.1.0"
