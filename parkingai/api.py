"""FastAPI application: JSON status, MJPEG stream, snapshot, and a viewer page."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .config import AppConfig, load_config
from .engine import Engine
from .zones import load_zones

_MJPEG_BOUNDARY = "frame"

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ParkingAI</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
  header{padding:10px 16px;background:#000;font-size:18px}
  main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
  img{max-width:100%;border:1px solid #333;border-radius:6px}
  #stats{font-size:15px;line-height:1.6}
  .pill{display:inline-block;padding:2px 8px;border-radius:10px;margin:2px}
  .free{background:#0a0}.occ{background:#a00}
</style></head>
<body>
  <header>ParkingAI &mdash; live occupancy</header>
  <main>
    <img src="/stream" alt="live stream"/>
    <div id="stats">loading&hellip;</div>
  </main>
  <script>
    async function tick(){
      try{
        const r = await fetch('/api/status'); const s = await r.json();
        let html = `<b>Free ${s.free} / ${s.total}</b> &nbsp; Occupied ${s.occupied}<br>`;
        html += `camera: ${s.camera_connected?'connected':'disconnected'} &nbsp; ${s.fps} fps<br><br>`;
        for(const z of s.zones){
          html += `<span class="pill ${z.occupied?'occ':'free'}">${z.id}: `
               +  `${z.occupied?'occupied':'free'} (${z.coverage})</span> `;
        }
        document.getElementById('stats').innerHTML = html;
      }catch(e){ /* keep last view */ }
    }
    setInterval(tick, 1000); tick();
  </script>
</body></html>
"""


def create_app(config_path: str = "config.yaml", engine: Optional[Engine] = None) -> FastAPI:
    """Build the FastAPI app. Pass a prebuilt ``engine`` to skip auto-setup
    (used by tests with a dummy detector)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            # Lazy import so tests that inject an engine don't need torch.
            from .camera import Camera
            from .detector import build_detector

            cfg: AppConfig = load_config(config_path)
            cam = Camera(
                source=cfg.camera.source,
                rtsp_tcp=cfg.camera.rtsp_tcp,
                reconnect_delay=cfg.camera.reconnect_delay,
                loop_file=cfg.camera.loop_file,
                width=cfg.camera.width,
                height=cfg.camera.height,
            )
            detector = build_detector(cfg.detector)
            zones = load_zones(cfg.zones_file)
            app.state.engine = Engine(cam, detector, zones, cfg).start()
        elif not getattr(app.state.engine, "_running", False):
            app.state.engine.start()
        yield
        app.state.engine.stop()

    app = FastAPI(title="ParkingAI", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.get("/healthz")
    def healthz() -> dict:
        eng = app.state.engine
        return {"status": "ok", "camera_connected": bool(eng and getattr(eng.camera, "connected", False))}

    @app.get("/api/status")
    def status() -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        return JSONResponse(eng.status())

    @app.get("/api/zones")
    def zones() -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        return JSONResponse(
            {"zones": [{"id": s.zone.id, "points": [list(p) for p in s.zone.points]}
                       for s in eng.states.values()]}
        )

    @app.get("/snapshot")
    def snapshot() -> Response:
        eng = app.state.engine
        jpeg = eng.get_jpeg() if eng else None
        if jpeg is None:
            return Response("no frame yet", status_code=503, media_type="text/plain")
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/stream")
    def stream() -> StreamingResponse:
        eng = app.state.engine

        def gen():
            fps = eng.cfg.server.stream_fps if eng else 12.0
            delay = 1.0 / max(fps, 1.0)
            while True:
                jpeg = eng.get_jpeg() if eng else None
                if jpeg is not None:
                    yield (b"--" + _MJPEG_BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(delay)

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}"
        )

    return app
