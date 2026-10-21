"""Exercise the physical diagnostic path with simulated firmware; no live CAN."""

import struct
import sys
import time
import unittest
import json
import tempfile
from pathlib import Path
from contextlib import closing
from unittest.mock import Mock, patch

import numpy as np

from agilex_deploy.contract import SIDES
from agilex_deploy.gravity_comp_test import GravityCompTest, main, official_test_frames
from agilex_deploy.model import Kinematics
from nero_firmware_sim import FirmwareSim


class DiagnosticFirmware(FirmwareSim):
    """Adds SDK all-seven enable and stop delivery to the existing firmware model.

Stop effectiveness is intentionally not modeled or claimed. A sent stop is
recorded, and physical stop confirmation remains unknown in the production tool.
"""
    jump_on_enable = False
    interrupt_in_leader = False
    cache_after_enable_s = 0.
    cached_until = 0.
    last_sample = None

    def socket(self, *args):
        writer = super().socket(*args)
        original = writer.send
        def send(packet):
            cid, _, data = struct.unpack("=IB3x8s", packet)
            if cid == 0x471 and data[:2] == bytes([8, 2]):
                self.frames.append((writer.channel, cid, data))
                index = SIDES.index(writer.channel)*7
                if not self.ignore_enable:
                    self.enabled[index:index+7] = [0x40]*7
                if self.jump_on_enable:
                    self.plant.data.qpos[self.plant.qids[index]] += .1
                self.cached_until = time.monotonic() + self.cache_after_enable_s
                return 16
            return original(packet)
        writer.send = send
        return writer

    def read(self, *args, **kwargs):
        if self.interrupt_in_leader and "leader" in self.device_modes.values():
            raise KeyboardInterrupt
        if self.last_sample is not None and time.monotonic() < self.cached_until:
            return self.last_sample
        self.last_sample = super().read(*args, **kwargs)
        return self.last_sample


class GravityCompTests(unittest.TestCase):
    def setUp(self):
        self.bus = DiagnosticFirmware()
        self.addCleanup(self.bus.plant.close)
        self.limits = Kinematics().limits
        self.events = []
        # Any accidental use of live sockets fails, including during codec
        # extraction. The simulator writer below has no network connection.
        self.socket_patch = patch("socket.socket", side_effect=AssertionError("Live socket forbidden in offline validation"))
        self.socket_patch.start()
        self.addCleanup(self.socket_patch.stop)
        self.settle_patch = patch("agilex_deploy.gravity_comp_test.SETTLE_S", .05)
        self.settle_patch.start()
        self.addCleanup(self.settle_patch.stop)

    def tool(self, side="left"):
        writer = self.bus.socket()
        writer.bind((side,))
        return GravityCompTest(self.bus, writer, side, self.limits, self.events.append, Mock())

    def assert_only_diagnostic_frames(self, side):
        self.assertTrue(all(s == side for s, _, _ in self.bus.frames))
        self.assertTrue(all(cid in (0x471, 0x151, 0x470, 0x150) for _, cid, _ in self.bus.frames))
        self.assertFalse(any(cid == 0x471 and payload[1] != 2 for _, cid, payload in self.bus.frames))

    def test_official_all_joint_enable_and_damped_stop_encoding(self):
        enable, stop = official_test_frames()
        self.assertEqual(enable, ((0x471, bytes([8, 2])+bytes(6)),))
        self.assertEqual(stop, ((0x150, bytes([1])+bytes(7)),))

    def test_cached_fresh_sample_after_enable_waits_for_next_feedback(self):
        self.bus.cache_after_enable_s = .02
        test = self.tool()
        result = test.run(1)
        self.assertEqual(result["result"], "protocol_test_completed")
        self.assertTrue(test.leader_confirmed)
        self.assert_only_diagnostic_frames("left")

    def test_frozen_sequence_after_enable_still_stops_within_deadline(self):
        self.bus.cache_after_enable_s = 1.
        test = self.tool()
        with self.assertRaisesRegex(RuntimeError, "feedback sequence"):
            test.run(1)
        self.assertTrue(test.stop_requested)
        self.assertFalse(test.leader_confirmed)
        self.assertEqual([cid for _, cid, _ in self.bus.frames], [0x471, 0x150, 0x150, 0x150])

    def test_left_and_right_independently_enter_leader_without_targets(self):
        for side in SIDES:
            with self.subTest(side=side):
                if side == "right":
                    self.bus.plant.close()
                    self.bus = DiagnosticFirmware()
                test = self.tool(side)
                result = test.run(1)
                self.assertEqual(result["result"], "protocol_test_completed")
                self.assertTrue(test.leader_confirmed)
                self.assertTrue(test.stop_requested)
                self.assertFalse(result["physical_gravity_compensation_verified"])
                self.assert_only_diagnostic_frames(side)
                self.assertEqual(self.bus.device_modes[side], "leader")
                other = 1-SIDES.index(side)
                self.assertEqual(self.bus.enabled[other*7:other*7+7], [0]*7)
                self.assertEqual([cid for _, cid, _ in self.bus.frames], [0x471, 0x151, 0x470, 0x150, 0x150, 0x150])
        self.bus.plant.close()

    def test_unexpected_startup_movement_stops_without_position_replay(self):
        self.bus.jump_on_enable = True
        with self.assertRaisesRegex(RuntimeError, "excursion"):
            self.tool().run(1)
        self.assertEqual([cid for _, cid, _ in self.bus.frames], [0x471, 0x150, 0x150, 0x150])

    def test_retained_target_servo_motion_is_detected_before_leader_entry(self):
        self.bus.targets[0] += .6
        original = self.bus.read
        def read(**kwargs):
            if any(self.bus.enabled) and self.bus.device_modes["left"] == "follower":
                # Simulate a firmware control tick before enabled feedback is
                # delivered, with a latent position target from a prior owner.
                self.bus.last_step = time.monotonic() - .02
            return original(**kwargs)
        self.bus.read = read
        test = self.tool()
        with self.assertRaisesRegex(RuntimeError, "(speed|excursion)"):
            test.run(1)
        self.assertTrue(test.stop_requested)
        self.assertFalse(test.leader_confirmed)
        self.assertFalse(any(cid in (0x155, 0x156, 0x157, 0x170) for _, cid, _ in self.bus.frames))

    def test_missing_leader_packets_requests_stop_and_never_recovers_to_position_mode(self):
        self.bus.drop_leader.add("left")
        test = self.tool()
        with self.assertRaises(RuntimeError):
            test.run(1)
        self.assertTrue(test.stop_requested)
        self.assert_only_diagnostic_frames("left")
        self.assertFalse(any(cid == 0x470 and data[0] == 0xFC for _, cid, data in self.bus.frames))

    def test_keyboard_interrupt_in_transition_requests_stop(self):
        self.bus.interrupt_in_leader = True
        test = self.tool()
        with self.assertRaises(KeyboardInterrupt):
            test.run(1)
        self.assertTrue(test.stop_requested)
        self.assertEqual([cid for _, cid, _ in self.bus.frames][-3:], [0x150]*3)

    def test_enabled_arm_or_missing_driver_feedback_fails_without_commands(self):
        self.bus.enabled[7] = 0x40
        with self.assertRaisesRegex(RuntimeError, "BOTH arms disabled"):
            self.tool().run(1)
        self.assertEqual(self.bus.frames, [])
        self.bus.enabled[7] = 0
        self.bus.drop_driver.add("right")
        self.bus.assembler.drivers.clear()
        self.bus.assembler.latest = None
        with self.assertRaises(RuntimeError):
            self.tool().run(1)
        self.assertEqual(self.bus.frames, [])

    def test_preflight_only_sends_nothing(self):
        self.tool().preflight()
        self.assertEqual(self.bus.frames, [])

    def test_emergency_stop_does_not_depend_on_logging_or_stdout(self):
        test = self.tool()
        test.possibly_enabled = True
        test.record = Mock(side_effect=OSError("disk full"))
        test.announce = Mock(side_effect=BrokenPipeError())
        test.stop()
        self.assertTrue(test.stop_requested)
        self.assertEqual([cid for _, cid, _ in self.bus.frames], [0x150]*3)

    def test_stop_transport_failure_is_not_reported_as_sent(self):
        test = self.tool()
        test.possibly_enabled = True
        test.writer.send = Mock(side_effect=OSError("bus down"))
        with self.assertRaisesRegex(RuntimeError, "COULD NOT BE SENT"):
            test.stop()
        self.assertFalse(test.stop_requested)

    def test_pre_stop_cache_is_not_recorded_as_post_stop_measurement(self):
        test = self.tool()
        snapshot = test.sample()
        test.possibly_enabled = True
        self.bus.read = Mock(return_value=snapshot)
        test.stop()
        self.assertTrue(test.stop_requested)
        self.assertFalse(any(row["event"] == "post_stop_feedback" for row in self.events))
        stop = next(row for row in self.events if row["event"] == "stop_requested")
        self.assertGreater(stop["first_attempt_monotonic_s"], snapshot.timestamp_s)

    def test_cli_requires_explicit_support_before_any_hardware_access(self):
        with patch.object(sys, "argv", ["gravity_comp_test", "--side", "left"]), \
             patch("agilex_deploy.gravity_comp_test.query_firmware") as probe:
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            probe.assert_not_called()

    def test_stale_snapshot_is_rejected(self):
        from dataclasses import replace
        test = self.tool()
        sample = self.bus.read(timeout=.1)
        test.reader = Mock()
        test.reader.read.return_value = replace(sample, timestamp_s=time.monotonic()-1)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            test.run(1)
        self.assertEqual(self.bus.frames, [])

    def test_cli_end_to_end_logs_and_stops_with_all_hardware_replaced(self):
        for check_only in (True, False):
            with self.subTest(check_only=check_only), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "configs").mkdir()
                (root / "configs/commissioning.json").write_text(json.dumps({"controller_mounting_configuration_verified": True}))
                (root / "configs/hardware_mapping.json").write_text(json.dumps({
                    "arms": {s: {"usb_identity": {"serial": s}} for s in SIDES}}))
                args = ["gravity_comp_test", "--side", "left", "--seconds", "1"]
                args += ["--check-only"] if check_only else ["--arms-supported"]
                original_read = self.bus.read
                last_read = [None]
                max_gap = [0.]
                def queued_read(**kwargs):
                    now = time.monotonic()
                    gap = now - last_read[0]
                    max_gap[0] = max(max_gap[0], gap)
                    if gap > .25:
                        raise RuntimeError("CAN receive queue overflow: simulated continuous producer was not drained")
                    last_read[0] = now
                    return original_read(**kwargs)
                def open_reader(*args, **kwargs):
                    # Hardware sends continuously, unlike the original
                    # on-demand firmware model. Catch a countdown while a
                    # subscribed receive socket is left unread.
                    last_read[0] = time.monotonic()
                    self.bus.read = queued_read
                    return closing(self.bus)
                with patch("agilex_deploy.gravity_comp_test.ROOT", root), \
                     patch("agilex_deploy.gravity_comp_test.query_firmware", return_value={
                         "missing": [], "firmware": {s: {"software_version": "1.21"} for s in SIDES}}), \
                     patch("agilex_deploy.gravity_comp_test.PassiveCanFeedback", side_effect=open_reader), \
                     patch("agilex_deploy.gravity_comp_test.socket.socket", side_effect=lambda *a: closing(self.bus.socket())), \
                     patch.object(sys, "argv", args), patch("builtins.print"):
                    with self.assertRaises(SystemExit) as result:
                        main()
                self.bus.read = original_read
                self.assertEqual(result.exception.code, 0)
                self.assertLess(max_gap[0], .25)
                report = json.loads(next((root / "logs/gravity_comp").glob("*/report.json")).read_text())
                self.assertEqual(report["enable_attempted"], not check_only)
                self.assertEqual(report["stop_requested"], not check_only)
                self.assertFalse(report["physical_gravity_compensation_verified"])
                self.assertFalse(report["motor_disable_command_sent"])
                if check_only:
                    self.assertEqual(self.bus.frames, [])
                else:
                    self.assert_only_diagnostic_frames("left")


if __name__ == "__main__":
    unittest.main()
