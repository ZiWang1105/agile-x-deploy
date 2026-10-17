"""Local workshop API and operator dashboard. No arbitrary code execution endpoint."""

import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit


from .workshop_service import WorkshopError, arm_name


def dispatch(service, method, params, client_id, lease_token=None):
    if method == "session_claim":
        if params:
            raise ValueError("session_claim takes no parameters")
        return service.claim(client_id)
    if method == "robot_capabilities":
        if params:
            raise ValueError("robot_capabilities takes no parameters")
        return service.capabilities()
    with service.lock:
        if (
            not lease_token
            or not service.lease_token
            or not hmac.compare_digest(lease_token, service.lease_token)
        ):
            raise WorkshopError(
                "NOT_OWNER", "Missing or invalid participant lease token"
            )
        service._authorize(client_id)

    def wait_operation(operation_id, timeout_s=20):
        from .workshop_service import number, TERMINAL

        timeout_s = number(timeout_s, "timeout_s", 0, 20)
        deadline = time.monotonic() + timeout_s
        while True:
            result = service.operation(operation_id)
            if result["state"] in TERMINAL or time.monotonic() >= deadline:
                return result
            time.sleep(0.05)

    methods = {
        "robot_observe": service.observe,
        "camera_capture": service.capture,
        "camera_deproject": service.deproject,
        "kinematics_ik": service.ik_solve,
        "kinematics_fk": service.fk,
        "operation_get": service.operation,
        "operation_wait": wait_operation,
        "operation_cancel": lambda **kw: service.cancel(client_id, **kw),
        "robot_hold": lambda: service.hold(client_id),
        "session_release": lambda: service.release(client_id),
    }

    def move_tcp(arm, target, request_id, tool_frame="tcp", timeout_s=10):
        return service.submit(
            client_id,
            request_id,
            [dict(poses={arm_name(arm): target}, tool_frame=tool_frame)],
            timeout_s,
        )

    def move_both(targets, request_id, tool_frame="tcp", timeout_s=10):
        if set(targets) != {"left", "right"}:
            raise ValueError("Supply both left and right targets")
        return service.submit(
            client_id,
            request_id,
            [dict(poses=targets, tool_frame=tool_frame)],
            timeout_s,
        )

    def move_joints(arm, joints_rad, request_id, timeout_s=10):
        return service.submit(
            client_id, request_id, [dict(joints={arm_name(arm): joints_rad})], timeout_s
        )

    def waypoints(arm, targets, request_id, tool_frame="tcp", timeout_s=30):
        arm_name(arm)
        return service.submit(
            client_id,
            request_id,
            [dict(poses={arm: t}, tool_frame=tool_frame) for t in targets],
            timeout_s,
        )

    def gripper(arm, request_id, timeout_s=5, value=0):
        return service.submit(
            client_id, request_id, [dict(grippers={arm_name(arm): value})], timeout_s
        )

    methods.update(
        motion_move_tcp=move_tcp,
        motion_move_both=move_both,
        motion_move_joints=move_joints,
        motion_follow_waypoints=waypoints,
        gripper_open=lambda arm, request_id, timeout_s=5: gripper(
            arm, request_id, timeout_s, 1
        ),
        gripper_close=lambda arm, request_id, timeout_s=5: gripper(
            arm, request_id, timeout_s, 0
        ),
    )
    if method not in methods:
        raise WorkshopError("UNKNOWN_METHOD", "Unknown participant API method")
    return methods[method](**params)


class WorkshopHTTP:
    def __init__(self, service, port=8780, config=None):
        self.service = service
        self.token = secrets.token_urlsafe(32)
        self.operator_token = secrets.token_urlsafe(32)
        owner = self
        template = Path(__file__).with_name("workshop.html").read_text()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, value, status=200, mime="application/json"):
                body = (
                    value
                    if isinstance(value, bytes)
                    else json.dumps(value, allow_nan=False).encode()
                )
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def local(self):
                return self.headers.get("Host") in {
                    f"127.0.0.1:{owner.port}",
                    f"localhost:{owner.port}",
                }

            def do_GET(self):
                if not self.local():
                    return self.send({"error": "Localhost only"}, 403)
                path = urlsplit(self.path).path
                if path == "/":
                    return self.send(
                        template.encode(),
                        mime="text/html; charset=utf-8",
                    )
                try:
                    if path == "/state":
                        with service.lock:
                            return self.send(
                                dict(
                                    backend=service.backend,
                                    task=service.task,
                                    instruction=service.instruction,
                                    enabled=service.enabled,
                                    stopped=service.stopped,
                                    owner=service.owner,
                                    fault=service.fault,
                                    robot=service.observe(),
                                    operations=[
                                        service._public_operation(op)
                                        for op in list(service.operations.values())[
                                            -20:
                                        ]
                                    ],
                                    events=service.events[-15:],
                                )
                            )
                    if path == "/camera.png":
                        return self.send(
                            base64.b64decode(service.capture()["rgb_png_base64"]),
                            mime="image/png",
                        )
                    if (
                        path == "/scene.jpg"
                        and service.backend == "sim"
                        and hasattr(service, "scene_jpeg")
                    ):
                        return self.send(service.scene_jpeg, mime="image/jpeg")
                    return self.send({"error": "Not found"}, 404)
                except Exception as error:
                    return self.send({"error": str(error)}, 503)

            def do_POST(self):
                if not self.local():
                    return self.send({"error": "Localhost only"}, 403)
                try:
                    self.connection.settimeout(5)
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 100000:
                        raise ValueError("Invalid request size")
                    body = json.loads(
                        self.rfile.read(size),
                        parse_constant=lambda _: (_ for _ in ()).throw(
                            ValueError("Nonfinite JSON")
                        ),
                    )
                    if self.path.startswith("/operator/"):
                        if not hmac.compare_digest(
                            self.headers.get("X-Operator-Token", ""),
                            owner.operator_token,
                        ):
                            return self.send({"error": "Invalid operator token"}, 403)
                        if self.path == "/operator/enable":
                            service.enable()
                        elif self.path == "/operator/stop":
                            service.operator_stop()
                        else:
                            raise ValueError("Unknown operator action")
                        return self.send({"ok": True})
                    if self.path != "/api":
                        raise ValueError("Unknown path")
                    if not hmac.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + owner.token
                    ):
                        return self.send(
                            {
                                "error": {
                                    "code": "UNAUTHORIZED",
                                    "message": "Invalid API token",
                                }
                            },
                            403,
                        )
                    if set(body) != {"method", "params", "client_id"} or not isinstance(
                        body["params"], dict
                    ):
                        raise ValueError(
                            "Expected method, params object, and client_id"
                        )
                    result = dispatch(
                        service,
                        body["method"],
                        body["params"],
                        body["client_id"],
                        self.headers.get("X-Workshop-Lease"),
                    )
                    self.send({"result": result, "api_version": "1.0"})
                except Exception as error:
                    self.send(
                        {
                            "error": {
                                "code": getattr(error, "code", "INVALID_REQUEST"),
                                "message": str(error),
                            }
                        },
                        400,
                    )

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="workshop-http", daemon=False
        )
        self.thread.start()
        if config:
            path = Path(config)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(dict(endpoint=self.endpoint, token=self.token)) + "\n"
            )
            path.chmod(0o600)
            operator_path = path.with_name("operator_url.txt")
            operator_path.write_text(self.endpoint + "/#" + self.operator_token + "\n")
            operator_path.chmod(0o600)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
