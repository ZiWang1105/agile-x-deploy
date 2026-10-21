"""Camera/feedback ordering with a simulated blocking enable; no devices opened."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from agilex_deploy.agent_zed import ZedStereoCamera


class ZedCaptureTimingTests(unittest.TestCase):
    def camera(self, *, image_age=0, observation_age=0, processing_delay=0):
        clock = [100.0]
        camera = ZedStereoCamera.__new__(ZedStereoCamera)

        def observe():
            # Models the service lock being held during controller enable.
            clock[0] += 1.0
            return SimpleNamespace(timestamp_s=clock[0] - observation_age)

        camera.robot = SimpleNamespace(observe=observe)
        camera._latest_stereo = Mock(side_effect=lambda: (
            clock[0] - image_age, np.zeros((720, 2560, 3), dtype=np.uint8)))
        camera.maps = [(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32))] * 2
        camera.K = np.array([[100., 0, 4], [0, 100., 4], [0, 0, 1]])
        camera.T = np.eye(4)
        camera.baseline = .1
        camera.calibration = {}

        def disparity(*_):
            clock[0] += processing_delay
            return np.full((8, 8), 160, dtype=np.int16)

        camera.matcher = SimpleNamespace(compute=disparity)
        return camera, clock

    def test_image_is_selected_after_blocking_robot_observation(self):
        camera, clock = self.camera()
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            observation, frame = camera.capture()
            frame.validate()
        self.assertEqual(frame.timestamp_s, observation.timestamp_s)
        self.assertEqual(frame.timestamp_s, 101.)

    def test_stream_capture_never_calls_robot(self):
        camera, clock = self.camera()
        camera.robot.observe = Mock(side_effect=AssertionError("Streaming must not call robot RPC"))
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            frame = camera.capture_frame()
        self.assertEqual(frame.timestamp_s, 100.)
        camera.robot.observe.assert_not_called()

    def test_actually_stale_image_still_fails(self):
        camera, clock = self.camera(image_age=1.)
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, "RGB-D acquisition is stale"):
                camera.capture()

    def test_feedback_image_skew_still_fails(self):
        camera, clock = self.camera(observation_age=.2)
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, "within 100 ms"):
                camera.capture()

    def test_slow_depth_processing_still_fails(self):
        camera, clock = self.camera(processing_delay=.6)
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, "RGB-D acquisition is stale"):
                camera.capture()
