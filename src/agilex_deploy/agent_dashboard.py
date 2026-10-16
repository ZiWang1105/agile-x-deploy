"""Local browser monitor. Render copies of plant state without driving the robot."""

import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import mujoco
import numpy as np

from .agent_execution import AgentStopped


class AgentDashboard:
    def __init__(self, port, backend, instruction, model):
        self.lock = threading.RLock()
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.exited = threading.Event()
        self.render_stop = threading.Event()
        self.render_thread = None
        self.halt = None
        self.images = {}
        self.view = [135.0, -25.0, 1.4]
        self.state = {
            "backend": backend,
            "instruction": instruction,
            "model": model,
            "phase": "initializing",
            "message": "Loading robot and camera…",
            "turn": 0,
            "api_calls": 0,
            "api_cost_usd": 0,
            "events": [],
            "can_start": False,
        }
        self.token = secrets.token_urlsafe(32)
        dashboard = self
        template = Path(__file__).with_name("agent_dashboard.html").read_text()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, payload, content_type="application/json", status=200):
                if not isinstance(payload, bytes):
                    payload = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def valid_host(self):
                return self.headers.get("Host") in {
                    f"127.0.0.1:{dashboard.port}",
                    f"localhost:{dashboard.port}",
                }

            def do_GET(self):
                if not self.valid_host():
                    return self.send({"error": "Localhost only"}, status=403)
                path = urlsplit(self.path).path
                if path == "/":
                    return self.send(
                        template.replace("__TOKEN__", dashboard.token).encode(),
                        "text/html; charset=utf-8",
                    )
                with dashboard.lock:
                    if path == "/state":
                        state = dict(dashboard.state)
                        state["server_time"] = time.time()
                        return self.send(state)
                    image = dashboard.images.get(path)
                if image is not None:
                    return self.send(image, "image/jpeg")
                self.send({"error": "Frame not available"}, status=404)

            def do_POST(self):
                if (
                    not self.valid_host()
                    or self.headers.get("X-Agent-Token") != dashboard.token
                ):
                    return self.send({"error": "Invalid session token"}, status=403)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 16000:
                        raise ValueError("Invalid request size")
                    payload = json.loads(self.rfile.read(length))
                    if self.path == "/start":
                        instruction = payload.get("instruction", "").strip()
                        if not instruction or len(instruction) > 8000:
                            raise ValueError(
                                "Enter a task prompt (up to 8000 characters)"
                            )
                        with dashboard.lock:
                            if (
                                not dashboard.state["can_start"]
                                or dashboard.cancelled.is_set()
                            ):
                                raise ValueError("This session cannot be started")
                            dashboard.state.update(
                                instruction=instruction, can_start=False
                            )
                            dashboard.started.set()
                    elif self.path in ("/stop", "/exit"):
                        dashboard.cancelled.set()
                        with dashboard.lock:
                            halt = dashboard.halt
                        dashboard.update(
                            can_start=False,
                            phase="stopping",
                            message="Requesting HOLD…",
                        )
                        if halt is not None:
                            halt()
                        dashboard.update(
                            phase="stopped",
                            message="Policy stopped. HOLD requested; motors are not disabled.",
                        )
                        if self.path == "/exit":
                            dashboard.exited.set()
                    elif self.path == "/view":
                        values = [
                            float(payload[k]) for k in ("yaw", "pitch", "distance")
                        ]
                        if not all(np.isfinite(values)) or not (
                            -180 <= values[0] <= 180
                            and -85 <= values[1] <= 0
                            and 0.5 <= values[2] <= 3
                        ):
                            raise ValueError("Invalid viewing angle")
                        with dashboard.lock:
                            dashboard.view = values
                    else:
                        return self.send({"error": "Unknown operation"}, status=404)
                    self.send({"ok": True})
                except Exception as error:  # noqa: BLE001 - report HTTP/owner failures to the operator.
                    dashboard.update(message=str(error))
                    self.send({"error": str(error)}, status=400)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="agent-dashboard", daemon=False
        )
        self.thread.start()
        print(f"\nLive dashboard: http://127.0.0.1:{self.port}\n", flush=True)

    def update(self, **values):
        with self.lock:
            if values.get("phase", self.state["phase"]) != self.state["phase"]:
                self.state["phase_updated"] = time.time()
            self.state.update(values)

    def event(self, message, **values):
        with self.lock:
            events = self.state["events"]
            events.append(
                dict(time=time.strftime("%H:%M:%S"), message=message, **values)
            )
            del events[:-60]

    def image(self, name, rgb):
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if ok:
            with self.lock:
                self.images[f"/{name}.jpg"] = encoded.tobytes()
                self.state[f"{name}_updated"] = time.time()

    def observation(self, observed, frame):
        self.image("policy", frame.rgb)
        self.update(robot=observed.to_dict())

    def check_stop(self):
        if self.cancelled.is_set():
            error = self.state.get("viewer_error")
            raise AgentStopped(
                "Live input failed: " + error if error else "Policy stopped by operator"
            )

    def wait_start(self):
        physical = self.state["backend"] == "real"
        self.update(
            phase="ready",
            can_start=True,
            message=(
                "Live hardware connected, arms not enabled by this session. "
                "Start real policy enables the selected arm and runs your task."
                if physical
                else "Edit the task and click Start simulation."
            ),
        )
        while not self.started.wait(0.1):
            self.check_stop()
        self.check_stop()
        return self.state["instruction"]

    def watch_real(self, executor, camera):
        def capture():
            try:
                while not self.render_stop.is_set():
                    stamp, rgb = camera.preview()
                    observed = executor.observe()
                    if observed.mode == "fault":
                        raise RuntimeError(observed.reason)
                    self.image("head", rgb)
                    self.update(
                        robot=observed.to_dict(), camera_age_s=time.monotonic() - stamp
                    )
                    self.render_stop.wait(0.15)
            except Exception as error:  # noqa: BLE001 - missing live sensors must inhibit later commands.
                self.cancelled.set()
                self.update(
                    viewer_error=str(error),
                    can_start=False,
                    message="Live sensor/feedback failure: " + str(error),
                )
                with self.lock:
                    halt = self.halt
                if halt is not None:
                    try:
                        halt()
                    except Exception as hold_error:  # noqa: BLE001 - report unconfirmed HOLD.
                        self.update(
                            viewer_error=f"{error}; HOLD unconfirmed: {hold_error}"
                        )

        self.render_thread = threading.Thread(
            target=capture, name="agent-live-zed", daemon=False
        )
        self.render_thread.start()

    def watch_sim(self, robot, plant):
        def render():
            renderer = None
            try:
                renderer = mujoco.Renderer(plant.model, height=360, width=640)
                data = mujoco.MjData(plant.model)
                view = mujoco.MjvCamera()
                view.lookat[:] = [0.28, 0, 0.25]
                while not self.render_stop.is_set():
                    # Only snapshot under the controller lock; all rendering is outside it.
                    with robot.lock:
                        data.qpos[:] = plant.data.qpos
                        data.qvel[:] = plant.data.qvel
                        data.time = plant.data.time
                        observed = robot.observe().to_dict()
                    mujoco.mj_forward(plant.model, data)
                    with self.lock:
                        view.azimuth, view.elevation, view.distance = self.view
                    renderer.update_scene(data, camera=view)
                    self.image("scene", renderer.render())
                    renderer.update_scene(data, camera="head_rgbd")
                    self.image("head", renderer.render())
                    self.update(robot=observed, sim_time_s=float(data.time))
                    self.render_stop.wait(0.15)
            except Exception as error:  # noqa: BLE001 - surface renderer failures without touching control.
                self.update(viewer_error=f"{type(error).__name__}: {error}")
            finally:
                if renderer is not None:
                    renderer.close()

        self.render_thread = threading.Thread(
            target=render, name="agent-render", daemon=False
        )
        self.render_thread.start()

    def freeze(self):
        self.render_stop.set()
        if self.render_thread is not None:
            self.render_thread.join(timeout=5)
            if self.render_thread.is_alive():
                raise RuntimeError("Dashboard render thread failed to stop")
        with self.lock:
            self.halt = None
            self.state["can_start"] = False
            self.state["frozen"] = True

    def close(self):
        try:
            self.freeze()
        finally:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=3)
