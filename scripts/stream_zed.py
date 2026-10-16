"""Local ZED 2i UVC color preview. Does not import or command robot interfaces.

Run with .venv-camera/bin/python scripts/stream_zed.py.
This provides raw stereo color, not SDK rectification or depth.
"""

from __future__ import annotations
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlparse, parse_qs

import cv2

HTML = b"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ZED 2i live</title><style>
body{margin:0;background:#101820;color:#edf4fa;font:16px system-ui}main{max-width:1600px;margin:auto;padding:24px}header{display:flex;align-items:center;gap:24px;flex-wrap:wrap}h1{font-size:24px}button{background:#243442;color:inherit;border:1px solid #566574;border-radius:8px;padding:10px 20px;cursor:pointer}button.active{background:#057a8d;border-color:#47d9e7}#status{color:#8ddbc3}img{display:block;max-width:100%;max-height:80vh;margin:20px auto;background:#000}small{color:#a6b8c6}</style></head><body><main>
<header><h1>ZED 2i / Live</h1><nav><button data-view="left" class="active">Left</button> <button data-view="right">Right</button> <button data-view="stereo">Stereo</button></nav><span id="status">Connecting...</span></header>
<img id="video" src="/stream.mjpg?view=left" alt="Live camera feed"><small>Color preview &middot; 1280 x 720 per eye &middot; raw, unrectified image</small>
<script>let view='left';const video=document.querySelector('#video');document.querySelectorAll('button').forEach(b=>b.onclick=()=>{view=b.dataset.view;document.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));video.src='/stream.mjpg?view='+view});video.onerror=()=>setTimeout(()=>video.src='/stream.mjpg?view='+view+'&t='+Date.now(),1500);async function status(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();document.querySelector('#status').textContent=s.healthy?`${s.fps.toFixed(1)} fps | live`:'Camera unavailable: '+(s.error||'no fresh frame')}catch(e){document.querySelector('#status').textContent='Stream offline'}setTimeout(status,1000)}status();</script></main></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        help="V4L2 device path; defaults to the only ZED video-index0 device",
    )
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.device is None:
        devices = sorted(Path("/dev/v4l/by-id").glob("*ZED*video-index0"))
        if len(devices) != 1:
            parser.error(
                f"Found {len(devices)} ZED capture devices; specify --device PATH. Candidates: {devices}"
            )
        args.device = str(devices[0])
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    lock = threading.Condition()
    stop = threading.Event()
    state = {
        "sequence": 0,
        "last_frame": 0.0,
        "read_started": 0.0,
        "frames": {},
        "error": None,
        "times": deque(maxlen=90),
        "width": 0,
        "height": 0,
    }

    def capture():
        try:
            while not stop.is_set():
                read_started = time.monotonic()
                ok, frame = cap.read()
                read_finished = time.monotonic()
                if not ok:
                    raise RuntimeError("Camera stopped delivering frames")
                h, w = frame.shape[:2]
                if (w, h) != (2560, 720):
                    raise RuntimeError(f"Unexpected stereo frame size: {w}x{h}")
                views = {
                    "left": frame[:, : w // 2],
                    "right": frame[:, w // 2 :],
                    "stereo": frame,
                }
                encoded = {}
                for name, view in views.items():
                    success, jpeg = cv2.imencode(
                        ".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if not success:
                        raise RuntimeError("JPEG encoding failed")
                    encoded[name] = jpeg.tobytes()
                with lock:
                    state.update(
                        sequence=state["sequence"] + 1,
                        last_frame=read_finished,
                        read_started=read_started,
                        frames=encoded,
                        width=w,
                        height=h,
                    )
                    state["times"].append(state["last_frame"])
                    lock.notify_all()
        except Exception as exc:
            with lock:
                state["error"] = str(exc)
                state["frames"] = {}
                lock.notify_all()
            print(f"Capture error: {exc}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send(self, data, kind, status=200, headers=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(HTML, "text/html; charset=utf-8")
            if url.path == "/status":
                with lock:
                    times = list(state["times"])
                    age = time.monotonic() - state["last_frame"]
                    result = {
                        "healthy": age < 2 and state["error"] is None,
                        "sequence": state["sequence"],
                        "frame_age_seconds": round(age, 3),
                        "fps": (len(times) - 1) / (times[-1] - times[0])
                        if len(times) > 1
                        else 0,
                        "stereo_resolution": [state["width"], state["height"]],
                        "device": args.device,
                        "error": state["error"],
                    }
                return self.send(json.dumps(result).encode(), "application/json")
            view = parse_qs(url.query).get("view", ["left"])[0]
            if view not in ("left", "right", "stereo"):
                return self.send(b"Unknown view", "text/plain", 400)
            if url.path not in ("/stream.mjpg", "/snapshot.jpg"):
                return self.send(b"Not found", "text/plain", 404)
            with lock:
                jpeg = state["frames"].get(view)
                fresh = time.monotonic() - state["last_frame"] < 2
                frame_headers = {
                    "X-Frame-Sequence": state["sequence"],
                    "X-Frame-Monotonic": state["last_frame"],
                    "X-Frame-Read-Started": state["read_started"],
                    "X-Frame-Timestamp-Source": "host_read_completion_not_exposure",
                    "X-Camera-View": view,
                    "X-Image-Space": "raw",
                    "X-Camera-Device": args.device,
                }
            if not jpeg or not fresh:
                return self.send(b"No fresh camera frame", "text/plain", 503)
            if url.path == "/snapshot.jpg":
                return self.send(jpeg, "image/jpeg", headers=frame_headers)
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.connection.settimeout(5)
            last = 0
            try:
                while not stop.is_set():
                    with lock:
                        lock.wait_for(
                            lambda: state["sequence"] != last
                            or state["error"]
                            or stop.is_set(),
                            timeout=2,
                        )
                        if (
                            state["error"]
                            or time.monotonic() - state["last_frame"] >= 2
                        ):
                            break
                        if last == state["sequence"]:
                            continue
                        last = state["sequence"]
                        jpeg = state["frames"][view]
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=capture, name="zed-capture", daemon=True)
    worker.start()

    def terminate(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    print(f"ZED color stream: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        with lock:
            lock.notify_all()
        server.server_close()
        worker.join(timeout=2)
        if not worker.is_alive():
            cap.release()


if __name__ == "__main__":
    main()
