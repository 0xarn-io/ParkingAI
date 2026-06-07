"""FastAPI application: JSON status, MJPEG stream, snapshot, and a viewer page."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import AppConfig, load_config, save_calibration
from .engine import Engine
from .zones import load_zones


class CalibrationUpdate(BaseModel):
    """Partial distortion-correction update from the UI (only set fields apply)."""

    enabled: Optional[bool] = None
    model: Optional[str] = None
    k1: Optional[float] = None
    k2: Optional[float] = None
    k3: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    focal_scale: Optional[float] = None
    balance: Optional[float] = None

_MJPEG_BOUNDARY = "frame"

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ParkingAI</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
  header{padding:10px 16px;background:#000;font-size:18px}
  main{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
  img{max-width:100%;border:1px solid #333;border-radius:6px}
  #side{font-size:15px;line-height:1.6;min-width:280px}
  .pill{display:inline-block;padding:2px 8px;border-radius:10px;margin:2px}
  .free{background:#0a0}.occ{background:#a00}
  .panel{margin-top:18px;padding:12px;background:#1b1b1b;border:1px solid #333;border-radius:8px}
  .panel h3{margin:0 0 8px;font-size:15px}
  .row{display:flex;align-items:center;gap:8px;margin:6px 0}
  .row label{width:90px;font-size:13px}
  .row input[type=range]{flex:1}
  .row .val{width:48px;text-align:right;font-variant-numeric:tabular-nums}
  button{background:#2a6;color:#fff;border:0;padding:7px 12px;border-radius:6px;cursor:pointer}
  select{background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:3px}
  .hint{font-size:12px;color:#999;margin-top:6px}
</style></head>
<body>
  <header>ParkingAI &mdash; live occupancy</header>
  <main>
    <img src="/stream" alt="live stream"/>
    <div id="side">
      <div id="stats">loading&hellip;</div>

      <div class="panel">
        <h3>Lens distortion</h3>
        <div class="row">
          <label><input type="checkbox" id="enabled"> enabled</label>
          <select id="model">
            <option value="pinhole">pinhole (barrel)</option>
            <option value="fisheye">fisheye</option>
          </select>
        </div>
        <div class="row"><label>k1</label><input type="range" id="k1" min="-0.8" max="0.4" step="0.01"><span class="val" id="k1_v"></span></div>
        <div class="row"><label>k2</label><input type="range" id="k2" min="-0.5" max="0.5" step="0.01"><span class="val" id="k2_v"></span></div>
        <div class="row"><label>k3</label><input type="range" id="k3" min="-0.3" max="0.3" step="0.01"><span class="val" id="k3_v"></span></div>
        <div class="row"><label>focal</label><input type="range" id="focal_scale" min="0.3" max="2.0" step="0.05"><span class="val" id="focal_scale_v"></span></div>
        <div class="row"><label>balance</label><input type="range" id="balance" min="0" max="1" step="0.05"><span class="val" id="balance_v"></span></div>
        <div class="row">
          <button onclick="autoCal()">Auto-guess</button>
          <button onclick="saveCal()">Save</button>
          <span id="saveMsg" style="color:#6c6"></span>
        </div>
        <div class="hint">Auto-guess estimates the correction from the current
          frame's straight edges. Then tune k1 by hand if needed, Save, and
          re-draw your zones (the editor uses these same values).</div>
      </div>
    </div>
  </main>
  <script>
    const FIELDS = ['k1','k2','k3','focal_scale','balance'];
    let debounce;

    async function loadCal(){
      const c = await (await fetch('/api/calibration')).json();
      document.getElementById('enabled').checked = !!c.enabled;
      document.getElementById('model').value = c.model;
      for(const f of FIELDS){
        document.getElementById(f).value = c[f];
        document.getElementById(f+'_v').textContent = (+c[f]).toFixed(2);
      }
    }
    function pushCal(){
      const body = {enabled: document.getElementById('enabled').checked,
                    model: document.getElementById('model').value};
      for(const f of FIELDS){
        const v = parseFloat(document.getElementById(f).value);
        body[f] = v;
        document.getElementById(f+'_v').textContent = v.toFixed(2);
      }
      clearTimeout(debounce);
      debounce = setTimeout(() => fetch('/api/calibration',
        {method:'POST', headers:{'Content-Type':'application/json'},
         body: JSON.stringify(body)}), 60);
    }
    async function saveCal(){
      const r = await fetch('/api/calibration/save', {method:'POST'});
      document.getElementById('saveMsg').textContent = (await r.json()).saved ? 'saved ✓' : 'error';
      setTimeout(() => document.getElementById('saveMsg').textContent='', 2000);
    }
    async function autoCal(){
      const msg = document.getElementById('saveMsg');
      msg.textContent = 'estimating…';
      const res = await (await fetch('/api/calibration/auto', {method:'POST'})).json();
      if(res.ok){
        await loadCal();
        msg.textContent = `k1=${res.k1} (-${Math.round(res.improvement*100)}% bend)`;
      } else {
        msg.textContent = res.reason || 'failed';
      }
      setTimeout(() => msg.textContent='', 4000);
    }
    ['enabled','model',...FIELDS].forEach(id =>
      document.getElementById(id).addEventListener('input', pushCal));

    async function tick(){
      try{
        const s = await (await fetch('/api/status')).json();
        let html = `<b>Free ${s.free} / ${s.total}</b> &nbsp; Occupied ${s.occupied}<br>`;
        html += `camera: ${s.camera_connected?'connected':'disconnected'} &nbsp; ${s.fps} fps<br><br>`;
        for(const z of s.zones){
          html += `<span class="pill ${z.occupied?'occ':'free'}">${z.id}: `
               +  `${z.occupied?'occupied':'free'} (${z.coverage})</span> `;
        }
        document.getElementById('stats').innerHTML = html;
      }catch(e){ /* keep last view */ }
    }
    loadCal();
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

    @app.get("/api/calibration")
    def get_calibration() -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        return JSONResponse(eng.get_calibration())

    @app.post("/api/calibration")
    def update_calibration(body: CalibrationUpdate) -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        changes = {k: v for k, v in body.model_dump().items() if v is not None}
        return JSONResponse(eng.set_calibration(changes))

    @app.post("/api/calibration/save")
    def persist_calibration() -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        data = eng.get_calibration()
        save_calibration(data)
        return JSONResponse({"saved": True, "calibration": data})

    @app.post("/api/calibration/auto")
    def auto_calibration() -> JSONResponse:
        eng = app.state.engine
        if eng is None:
            return JSONResponse({"error": "engine not ready"}, status_code=503)
        frame = eng.camera.read()  # raw frame, before correction
        if frame is None:
            return JSONResponse({"ok": False, "reason": "no frame yet"}, status_code=503)
        from .autocalib import estimate

        cur = eng.get_calibration()
        res = estimate(frame, model=cur["model"], focal_scale=cur["focal_scale"])
        if res.get("ok"):
            # apply the guess live so the user sees it immediately
            res["calibration"] = eng.set_calibration({"enabled": True, "k1": res["k1"]})
        return JSONResponse(res)

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
