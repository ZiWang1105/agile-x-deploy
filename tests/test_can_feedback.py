"""Receive-only commissioning tests; no physical CAN adapter is required."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from agilex_deploy.backends import SimBackend
from agilex_deploy.can_feedback import ANGLE_IDS, DRIVER_IDS, FeedbackAssembler, mapped_channels
from agilex_deploy.contract import SIDES
from agilex_deploy.interactive import make_display, update_mirror
from agilex_deploy.model import initial_positions


def feed_arm(assembler, side, stamp, value=1000, drivers=True):
    if drivers:
        for cid in DRIVER_IDS:
            assembler.feed(side, cid, bytes([0, 0, 0, 0, 0, 0x40, 0, 0]), stamp)
    for cid in ANGLE_IDS:
        assembler.feed(side, cid, struct.pack(">ii", value, -value), stamp)


class FeedbackTests(unittest.TestCase):
    def test_signed_units_joint_order_and_missing_measurements(self):
        a = FeedbackAssembler()
        feed_arm(a, "left", 10, value=90000)
        feed_arm(a, "right", 10.005, value=45000)
        s = a.snapshot(10.01)
        np.testing.assert_allclose(s.positions[:7], np.radians([90, -90, 90, -90, 90, -90, 90]))
        np.testing.assert_allclose(s.positions[7:], np.radians([45, -45, 45, -45, 45, -45, 45]))
        self.assertEqual(s.timestamp_s, 10)
        self.assertAlmostEqual(s.skew_s, .005)
        self.assertIsNone(s.velocities)
        self.assertIsNone(s.grippers)
        self.assertEqual(s.driver_status, (0x40,) * 14)

    def test_partial_repeated_and_stale_frames_cannot_refresh_pair(self):
        a = FeedbackAssembler()
        feed_arm(a, "left", 10)
        with self.assertRaisesRegex(RuntimeError, "complete"):
            a.snapshot(10)
        feed_arm(a, "right", 10)
        previous = a.snapshot(10)
        for _ in range(20):
            a.feed("left", 0x2A5, struct.pack(">ii", 1000, 2000), 10.1)
        self.assertEqual(a.snapshot(10.1), previous)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            a.snapshot(10.21)

    def test_skew_stale_driver_and_backwards_time_rejected(self):
        a = FeedbackAssembler()
        feed_arm(a, "left", 10)
        feed_arm(a, "right", 10.04)
        with self.assertRaisesRegex(RuntimeError, "skew"):
            a.snapshot(10.05)
        for side in SIDES:
            feed_arm(a, side, 10.3, drivers=False)
        with self.assertRaisesRegex(RuntimeError, "driver feedback stale"):
            a.snapshot(10.31)
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            a.feed("left", 0x2A5, bytes(8), 10.2)

    def test_malformed_or_flagged_frames_do_not_complete_snapshot(self):
        a = FeedbackAssembler()
        for side in SIDES:
            for cid in ANGLE_IDS:
                a.feed(side, cid, bytes(7), 10)
                a.feed(side, cid | 0x80000000, bytes(8), 10)
        self.assertFalse(a.angles)

    def test_usb_serial_resolves_renamed_interfaces_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            net = root / "net"
            net.mkdir()
            config = {"schema_version": 1, "automatic_hardware_enable": False, "arms": {}}
            for side, channel in (("left", "can7"), ("right", "can3")):
                usb = root / side
                usb.mkdir()
                identity = {"serial": side, "idVendor": "1d50", "idProduct": "606f"}
                for key, value in identity.items():
                    (usb / key).write_text(value)
                (net / channel).mkdir()
                (net / channel / "device").symlink_to(usb)
                config["arms"][side] = {"usb_identity": identity}
            path = root / "mapping.json"
            path.write_text(json.dumps(config))
            self.assertEqual(mapped_channels(path, net), {"left": "can7", "right": "can3"})
            (net / "can9").mkdir()
            (net / "can9" / "device").symlink_to(root / "left")
            with self.assertRaisesRegex(RuntimeError, "expected one"):
                mapped_channels(path, net)

    def test_sim_seed_and_measured_overlay_remain_independent(self):
        seed = initial_positions()
        seed[0] += .1
        plant = SimBackend(joint_positions=seed)
        try:
            np.testing.assert_allclose(plant.read().positions, seed)
            display = make_display(mirror=True)
            display.update(seed, (.36, .36))
            a = FeedbackAssembler()
            for side in SIDES:
                feed_arm(a, side, 10)
            sample = a.snapshot(10)
            update_mirror(display, sample)
            np.testing.assert_allclose(display.data.qpos[display.qids], seed)
            np.testing.assert_allclose(plant.read().positions, seed)
            self.assertAlmostEqual(display.data.joint("measured_left_joint1").qpos[0], np.radians(1))
        finally:
            plant.close()
        seed[0] = 10
        with self.assertRaisesRegex(ValueError, "outside model limits"):
            SimBackend(joint_positions=seed)


if __name__ == "__main__":
    unittest.main()
