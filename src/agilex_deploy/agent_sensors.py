"""RGB-D observation contract shared by rendered and physical camera drivers."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def transform(value):
    t = np.asarray(value, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1])
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1)
    ):
        raise ValueError("Invalid optical-to-base rigid transform")
    return t


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_base_from_optical: np.ndarray
    timestamp_s: float
    calibration: dict

    def validate(self):
        h, w = self.depth_m.shape
        if self.rgb.shape != (h, w, 3) or self.rgb.dtype != np.uint8:
            raise ValueError("Require registered uint8 RGB and metric optical-Z depth")
        if not np.isfinite(self.depth_m).all() or np.any(self.depth_m < 0):
            raise ValueError("Invalid depth must be encoded as zero")
        k = np.asarray(self.K)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1])
        ):
            raise ValueError("Invalid pinhole intrinsics")
        transform(self.T_base_from_optical)
        if not -0.01 <= time.monotonic() - self.timestamp_s <= 0.5:
            raise ValueError("RGB-D acquisition is stale")

    def metadata(self):
        return {
            "K": self.K.tolist(),
            "T_base_from_optical": self.T_base_from_optical.tolist(),
            "timestamp_s": self.timestamp_s,
            "size_wh": list(self.rgb.shape[1::-1]),
            "depth_units": "meters",
            "depth_type": "optical_z",
            "invalid_depth": 0,
            "optical_axes": "X right, Y down, Z forward",
            "calibration": self.calibration,
        }


def quantize_depth(depth):
    valid = np.isfinite(depth) & (depth > 0) & (depth < 65.535)
    return np.where(valid, np.round(np.where(valid, depth, 0) * 1000), 0).astype(
        np.uint16
    )


def ground_pixel(frame, uv, offset):
    if len(uv) != 2 or any(type(x) is not int for x in uv):
        raise ValueError("Pixel must be integer [u,v]")
    u, v = uv
    h, w = frame.depth_m.shape
    if not (1 <= u < w - 1 and 1 <= v < h - 1):
        raise ValueError("Pixel must have an interior 3x3 depth neighborhood")
    z = frame.depth_m[v, u]
    neighborhood = frame.depth_m[v - 1 : v + 2, u - 1 : u + 2]
    valid = neighborhood[neighborhood > 0]
    if z <= 0 or len(valid) < 5 or np.ptp(valid) > 0.03:
        raise ValueError("Selected pixel has invalid/discontinuous depth")
    offset = np.asarray(offset, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Invalid model-selected base-frame offset")
    surface = (
        frame.T_base_from_optical @ np.r_[np.linalg.solve(frame.K, [u, v, 1]) * z, 1]
    )[:3]
    return surface + offset, surface


def save_frame(frame, directory, index):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(directory / f"rgb_{index:03}.png"),
        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(str(directory / f"depth_{index:03}.png"), quantize_depth(frame.depth_m))
    (directory / f"camera_{index:03}.json").write_text(
        json.dumps(frame.metadata(), indent=2) + "\n"
    )


def load_calibration(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    return transform(value["T_base_from_camera"]), {
        "status": "candidate_not_physically_validated",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "camera_serial": value.get("camera", {}).get("serial"),
    }
