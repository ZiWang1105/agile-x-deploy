"""Requested RGB-D frames survive live-camera churn without unbounded storage."""

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np

from agilex_deploy.agent_sensors import RGBDFrame
from agilex_deploy.workshop_service import CAPTURED_FRAME_LIMIT, WorkshopError, WorkshopService


class FrameRetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = WorkshopService.__new__(WorkshopService)
        self.service.backend = "sim"
        self.service.lock = threading.RLock()
        self.service.frames_lock = threading.RLock()
        self.service.frame_available = threading.Condition(self.service.frames_lock)
        self.service.frames = OrderedDict()
        self.service.captured_frames = OrderedDict()
        self.service.output = Path(self.directory.name)

    def acquire(self, frame_id, depth=1):
        timestamp = time.monotonic()
        frame = RGBDFrame(np.zeros((3, 3, 3), np.uint8), np.full((3, 3), depth, float),
                          np.eye(3), np.eye(4), timestamp, {})
        observed = SimpleNamespace(observation_id=frame_id, timestamp_s=timestamp, to_dict=lambda: {})
        self.service.frames[str(frame_id)] = (observed, frame)
        while len(self.service.frames) > 64:
            self.service.frames.popitem(last=False)

    def test_deprojection_keeps_original_depth_after_live_preview_eviction(self):
        self.acquire(1, depth=1)
        original = self.service.capture()
        for frame_id in range(2, 90):
            self.acquire(frame_id, depth=2)
            self.service.capture(retain=False)
        self.assertNotIn("1", self.service.frames)
        self.assertEqual(list(self.service.captured_frames), ["1"])
        result = self.service.deproject("1", [1, 1])
        self.assertEqual(result["position_m"], [1, 1, 1])
        self.assertEqual(result["acquisition_timestamp_s"], original["timestamp_s"])
        self.assertEqual(len(list((self.service.output / "policy_frames").glob("*.json"))), 1)

    def test_real_capture_pulls_latest_frame_after_robot_wait(self):
        self.service.backend = "real"
        self.acquire(1)

        def observe():
            self.acquire(2, depth=2)
            return self.service.frames["2"][0]

        self.service._raw_observe = Mock(side_effect=observe)
        result = self.service.capture()
        self.assertEqual(result["frame_id"], "2")
        self.assertEqual(result["observation_id"], 2)
        self.assertEqual(result["robot_camera_skew_s"], 0.)

    def test_real_preview_never_waits_for_robot_feedback(self):
        self.service.backend = "real"
        self.acquire(1)
        self.service.frames["1"] = (None, self.service.frames["1"][1])
        self.service._raw_observe = Mock(side_effect=AssertionError("Preview requested robot feedback"))
        result = self.service.capture(retain=False)
        self.assertEqual(result["frame_id"], "1")
        self.assertIsNone(result["robot_timestamp_s"])
        self.service._raw_observe.assert_not_called()

    def test_retention_is_bounded_by_policy_captures_not_stream_updates(self):
        for frame_id in range(CAPTURED_FRAME_LIMIT + 1):
            self.acquire(frame_id)
            self.service.capture()
        self.service.frames.clear()
        self.assertEqual(len(self.service.captured_frames), CAPTURED_FRAME_LIMIT)
        with self.assertRaises(WorkshopError):
            self.service.deproject("0", [1, 1])
        self.assertEqual(self.service.deproject("1", [1, 1])["position_m"], [1, 1, 1])
