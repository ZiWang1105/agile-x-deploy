"""Stationary image / encoder acquisition. Standalone mode only receives CAN.

UVC supplies host read times, not qualified exposure timestamps. The full
stationary interval is retained as evidence; results remain provisional.
"""

from pathlib import Path
import time
import uuid
from urllib.request import urlopen
from urllib.parse import urlparse

import cv2
import numpy as np

from .camera_calibration import load_dataset, save_json, sha256
from .contract import JOINT_NAMES


def stationary_evidence(samples, image_time, max_span_rad=0.003):
    if (
        not np.isfinite(image_time)
        or not np.isfinite(max_span_rad)
        or max_span_rad <= 0
    ):
        raise ValueError("Invalid image timestamp or stationarity threshold")
    if len(samples) < 20:
        raise ValueError("Insufficient joint samples for a stationary capture")
    t = np.asarray([s["timestamp_s"] for s in samples], float)
    q = np.asarray([s["q"] for s in samples], float)
    if (
        q.shape != (len(samples), 14)
        or not np.isfinite(q).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Invalid encoder sample")
    if np.any(np.diff(t) < 0) or np.max(np.diff(t)) > 0.2:
        raise ValueError("Feedback has a backwards clock or a gap over 200 ms")
    if image_time - t[0] < 0.5 or t[-1] - image_time < 0.5:
        raise ValueError(
            "Image must have 0.5 s of measured stability before and after host read time"
        )
    span = np.ptp(q, axis=0)
    if np.max(span) > max_span_rad:
        raise ValueError(f"Arms moved during capture: {np.max(span):.5f} rad span")
    index = int(np.argmin(abs(t - image_time)))
    skew = float(abs(t[index] - image_time))
    if skew > 0.05:
        raise ValueError("No encoder sample within 50 ms of image host read time")
    return dict(
        accepted=True,
        joint_span_rad=span.tolist(),
        max_allowed_span_rad=max_span_rad,
        interval_monotonic_s=[float(t[0]), float(t[-1])],
        paired_skew_s=skew,
        selected_index=index,
        timing_qualified=False,
        timestamp_source="host_read_completion_not_exposure",
        samples=samples,
    )


def capture_pose(dataset, stream_url, read_sample, max_span_rad=0.003):
    """read_sample must return fresh measured q/times; never target positions."""
    dataset = Path(dataset)
    doc = load_dataset(dataset)
    if (
        doc["kind"] != "physical"
        or doc["camera"]["image_space"] != "raw"
        or doc["camera"]["size_wh"] != [1280, 720]
    ):
        raise ValueError("UVC capture requires a physical raw LEFT 1280x720 dataset")
    # Monotonic values are comparable only on this host; do not accept remote URLs.
    parsed = urlparse(stream_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError(
            "Capture requires the local camera server; remote clocks are not interchangeable"
        )
    samples, image, headers = [], None, None
    began = time.monotonic()
    while time.monotonic() - began < 2.0:
        sample = read_sample()
        if sample["joint_names"] != list(JOINT_NAMES):
            raise ValueError("Encoder joint order mismatch")
        age = time.monotonic() - sample["timestamp_s"]
        if not -0.01 <= age <= 0.2:
            raise ValueError("Stale measured feedback")
        samples.append(sample)
        if image is None and time.monotonic() - began > 0.85:
            # Short timeout also preserves the existing hardware-owner heartbeat
            # when read_sample is HardwareSession.observe.
            with urlopen(
                stream_url.rstrip("/") + "/snapshot.jpg?view=left", timeout=0.08
            ) as response:
                headers = dict(response.headers)
                image = response.read()
            if (
                headers.get("X-Frame-Timestamp-Source")
                != "host_read_completion_not_exposure"
            ):
                raise ValueError(
                    "Restart the updated ZED stream to expose acquisition metadata"
                )
            if (
                headers.get("X-Camera-View") != "left"
                or headers.get("X-Image-Space") != "raw"
            ):
                raise ValueError("Camera view/image-space mismatch")
            frame_time = float(headers["X-Frame-Monotonic"])
            if not 0 <= time.monotonic() - frame_time <= 0.2:
                raise ValueError("Image is stale or camera clock is incompatible")
        time.sleep(0.01)
    evidence = stationary_evidence(samples, frame_time, max_span_rad)
    decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or list(decoded.shape[1::-1]) != doc["camera"]["size_wh"]:
        raise ValueError("Image dimensions do not match intrinsics")
    pose_id = "pose_" + uuid.uuid4().hex[:12]
    image_path = dataset.parent / "images" / f"{pose_id}.jpg"
    image_path.parent.mkdir(exist_ok=True)
    image_path.write_bytes(image)
    pose = dict(
        id=pose_id,
        image=str(image_path.relative_to(dataset.parent)),
        image_sha256=sha256(image_path),
        q_measured_rad=samples[evidence["selected_index"]]["q"],
        stationary_capture=evidence,
        camera_headers=headers,
        camera_serial_verified=False,
    )
    # Keep an immutable per-capture record as well as the editable annotation index.
    save_json(image_path.with_suffix(".json"), pose)
    doc["poses"].append(pose)
    save_json(dataset, doc)
    return pose


def passive_capture(dataset, stream_url, mapping, max_span_rad=0.003):
    from .can_feedback import PassiveCanFeedback

    with PassiveCanFeedback(mapping=mapping) as feedback:
        feedback.read(timeout=3)

        def sample():
            observed = feedback.read(timeout=0.05)
            return dict(
                timestamp_s=observed.timestamp_s,
                q=list(observed.positions),
                joint_names=list(observed.joint_names),
                source="physical_CAN_receive_only",
            )

        return capture_pose(dataset, stream_url, sample, max_span_rad)
