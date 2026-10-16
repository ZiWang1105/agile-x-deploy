"""ZED UVC stereo RGB-D without an SDK dependency.

Uses the manufacturer's factory calibration and OpenCV stereoRectify/SGBM,
following stereolabs/zed-opencv-native and zed-open-capture. Depth quality is
not equivalent to ZED SDK neural depth. No synthetic depth or RGB-only fallback.
"""

import configparser
import hashlib
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .agent_sensors import RGBDFrame, load_calibration, quantize_depth
from .model import ROOT


def stereo_geometry(factory_path, raw_to_base, width=640, height=360):
    config = configparser.ConfigParser()
    config.read(factory_path)
    matrices, distortions = [], []
    for side in ("LEFT", "RIGHT"):
        cam, distortion = config[f"{side}_CAM_HD"], config[f"{side}_DISTO"]
        k = np.array(
            [
                [float(cam["fx"]), 0, float(cam["cx"])],
                [0, float(cam["fy"]), float(cam["cy"])],
                [0, 0, 1],
            ]
        )
        k[:2] *= width / 1280
        # OpenCV pixel-center convention under resize.
        k[0, 2] += (width / 1280 - 1) / 2
        k[1, 2] += (height / 720 - 1) / 2
        matrices.append(k)
        distortions.append(
            np.array(
                [
                    float(distortion[n])
                    for n in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
                ]
            )
        )
    stereo = config["STEREO"]
    rotation = cv2.Rodrigues(
        np.array([float(stereo[f"{axis}_HD"]) for axis in ("RX", "CV", "RZ")])
    )[0]
    translation = (
        np.array(
            [
                -float(stereo["Baseline"]),
                float(stereo.get("TY_HD", stereo.get("TY", 0))),
                float(stereo.get("TZ_HD", stereo.get("TZ", 0))),
            ]
        )
        / 1000
    )
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        matrices[0],
        distortions[0],
        matrices[1],
        distortions[1],
        (width, height),
        rotation,
        translation.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    maps = [
        cv2.initUndistortRectifyMap(k, d, r, p, (width, height), cv2.CV_32FC1)
        for k, d, r, p in zip(matrices, distortions, (r1, r2), (p1, p2))
    ]
    rectified_to_raw = np.eye(4)
    rectified_to_raw[:3, :3] = r1.T
    return maps, p1[:3, :3], raw_to_base @ rectified_to_raw, abs(p2[0, 3] / p2[0, 0])


class ZedStereoCamera:
    def __init__(self, robot, calibration, factory_path=None, device=None):
        self.robot = robot
        factory_path = (
            factory_path
            or ROOT / "calibration/2026-10-15_zed35330610/physical/SN35330610.conf"
        )
        raw_to_base, self.calibration = load_calibration(calibration)
        self.calibration.update(
            factory_sha256=hashlib.sha256(Path(factory_path).read_bytes()).hexdigest(),
            depth_source="opencv_stereo_sgbm",
            timing="host_read_completion_not_exposure",
            serial="35330610",
            serial_binding="single_ZED_UVC_device_not_independently_verified",
        )
        self.maps, self.K, self.T, self.baseline = stereo_geometry(
            factory_path, raw_to_base
        )
        if device is None:
            devices = sorted(Path("/dev/v4l/by-id").glob("*ZED*video-index0"))
            if len(devices) != 1:
                raise RuntimeError(
                    "Require exactly one ZED UVC device or an explicit device path"
                )
            device = str(devices[0])
        self.capture_device = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        if not self.capture_device.isOpened():
            raise RuntimeError("Cannot open ZED UVC stream")
        self.capture_device.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.capture_device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture_device.set(cv2.CAP_PROP_FPS, 30)
        self.capture_device.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=192,
            blockSize=5,
            P1=8 * 25,
            P2=32 * 25,
            disp12MaxDiff=1,
            uniquenessRatio=12,
            speckleWindowSize=80,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.lock = threading.Lock()
        self.latest = self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._capture, name="zed-rgbd-capture", daemon=False
        )
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop_event.is_set():
                okay, image = self.capture_device.read()
                stamp = time.monotonic()
                if not okay or image.shape != (720, 2560, 3):
                    raise RuntimeError(
                        "ZED UVC acquisition failed or resolution changed"
                    )
                with self.lock:
                    self.latest = stamp, image
        except Exception as error:
            self.error = str(error)

    def capture(self):
        deadline = time.monotonic() + 2
        while (
            self.latest is None and self.error is None and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if self.error or self.latest is None:
            raise RuntimeError(self.error or "ZED acquisition timed out")
        with self.lock:
            stamp, image = self.latest
        observation = self.robot.observe()
        rectified = [
            cv2.remap(cv2.resize(eye, (640, 360)), *maps, cv2.INTER_LINEAR)
            for eye, maps in zip((image[:, :1280], image[:, 1280:]), self.maps)
        ]
        gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in rectified]
        disparity = self.matcher.compute(*gray).astype(float) / 16
        depth = np.divide(
            self.K[0, 0] * self.baseline,
            disparity,
            out=np.zeros_like(disparity),
            where=disparity > 0,
        )
        depth[(depth < 0.15) | (depth > 3)] = 0
        frame = RGBDFrame(
            cv2.cvtColor(rectified[0], cv2.COLOR_BGR2RGB),
            quantize_depth(depth).astype(float) / 1000,
            self.K,
            self.T,
            stamp,
            self.calibration,
        )
        frame.validate()
        if abs(observation.timestamp_s - stamp) > 0.1:
            raise RuntimeError("No robot feedback matched to ZED capture within 100 ms")
        return observation, frame

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("ZED capture did not stop; camera read remains active")
        self.capture_device.release()
