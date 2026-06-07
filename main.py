"""ParkingAI entrypoint - starts the FastAPI server.

    python main.py                 # use config.yaml
    python main.py --config x.yaml
    python main.py --host 0.0.0.0 --port 8000

The original quick RTSP-viewer script lives on in spirit: the camera module
sets the same ``rtsp_transport;tcp`` option for reliable RTSP capture.
"""

from __future__ import annotations

import argparse

import uvicorn

from parkingai.api import create_app
from parkingai.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="ParkingAI server")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    app = create_app(config_path=args.config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
