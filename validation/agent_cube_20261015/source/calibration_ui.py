"""Local annotation UI. This server has no motion or enable endpoints."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import json
import re
import secrets
import time
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np

from .camera_calibration import load_dataset, save_json, solve_dataset
from .calibration_view import ModelView


def serve(dataset, port=8772):
    dataset = Path(dataset).resolve()
    view = ModelView()
    token = secrets.token_urlsafe(24)
    template = Path(__file__).with_name("calibration_ui.html").read_text()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, content_type="application/json", status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def pose(self, doc, pose_id):
            return next(p for p in doc["poses"] if p["id"] == pose_id)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path == "/":
                    return self.send(
                        template.replace("__TOKEN__", token).encode(),
                        "text/html; charset=utf-8",
                    )
                doc = load_dataset(dataset)
                if url.path == "/state":
                    return self.send(doc)
                pose = self.pose(doc, query.get("pose", [""])[0])
                if url.path == "/image":
                    image_path = (dataset.parent / pose["image"]).resolve()
                    if not image_path.is_relative_to(dataset.parent):
                        raise ValueError("Image path outside dataset")
                    return self.send(image_path.read_bytes(), "image/jpeg")
                if url.path == "/model":
                    settings = {
                        k: float(query[k][0])
                        for k in ("yaw", "pitch", "distance")
                        if k in query
                    }
                    image = view.image(pose["q_measured_rad"], settings, doc["anchors"])
                    return self.send(
                        cv2.imencode(".jpg", image)[1].tobytes(), "image/jpeg"
                    )
                self.send({"error": "not found"}, status=404)
            except Exception as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            try:
                if self.headers.get("X-Calibration-Token") != token:
                    return self.send({"error": "Invalid session token"}, status=403)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 100000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                doc = load_dataset(dataset)
                if self.path == "/solve":
                    output = (
                        dataset.parent / "results" / f"candidate_{time.time_ns()}.json"
                    )
                    result = solve_dataset(
                        dataset, output, allow_synthetic=doc["kind"] == "synthetic"
                    )
                    return self.send({"path": str(output), "result": result})
                pose = self.pose(doc, data["pose_id"])
                name = data["anchor_id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name):
                    raise ValueError(
                        "Use a short letters/digits/underscore feature name"
                    )
                if self.path == "/anchor":
                    if name in doc["anchors"]:
                        raise ValueError(
                            "Anchor already exists. Reuse it across poses; choose a new name to define a different point."
                        )
                    doc["anchors"][name] = view.pick(
                        pose["q_measured_rad"],
                        float(data["u"]),
                        float(data["v"]),
                        data.get("view"),
                    )
                elif self.path in ("/match", "/remove"):
                    if name not in doc["anchors"]:
                        raise ValueError("Define the 3D anchor first")
                    doc["observations"] = [
                        r
                        for r in doc["observations"]
                        if (r["pose_id"], r["anchor_id"]) != (pose["id"], name)
                    ]
                    if self.path == "/match":
                        uv = np.asarray([data["u"], data["v"]], float)
                        if (
                            not np.isfinite(uv).all()
                            or np.any(uv < 0)
                            or np.any(uv >= doc["camera"]["size_wh"])
                        ):
                            raise ValueError("Pixel outside original image")
                        doc["observations"].append(
                            dict(pose_id=pose["id"], anchor_id=name, uv=uv.tolist())
                        )
                else:
                    raise ValueError("Unknown annotation action")
                save_json(dataset, doc)
                self.send({"saved": True})
            except Exception as exc:
                self.send({"error": str(exc)}, status=400)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(
        f"Annotation UI: http://127.0.0.1:{port} | no robot motion endpoints",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        view.close()
