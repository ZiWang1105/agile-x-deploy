"""Regression for the observed physical left-only transition failure, offline."""
import unittest
import time
from contextlib import closing
from unittest.mock import Mock, patch

import numpy as np

from agilex_deploy.hardware_power import _disable_and_verify
from agilex_deploy.backends import RealBackend
from agilex_deploy.runtime import Robot
from agilex_deploy.nero_transport import position_frames
from agilex_deploy.contract import SIDES
from agilex_deploy.hardware_session import control_worker

from nero_firmware_sim import FirmwareSim
from test_dual_control import fake_hardware


class FirmwareHandGuidingTests(unittest.TestCase):
    def test_worker_enable_idle_resume_and_close_using_firmware_streams(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                connection = Mock()
                connection.poll.return_value = True
                connection.recv.side_effect = [(1, "arm", None), (2, "observe", None),
                                               (3, "begin_control", None), (4, "enter_hand_guiding", None),
                                               (5, "close", None)]
                with patch("agilex_deploy.hardware_session.DualNeroTransport", return_value=transport), \
                     patch("agilex_deploy.hardware_session.signal.signal"):
                    control_worker(connection, "both", "unused", "unused", False, None, True,
                                   gravity_comp=True)
                replies = [c.args[0] for c in connection.send.call_args_list]
                self.assertEqual([r[0] for r in replies], [0, 1, 2, 3, 4, 5])
                self.assertTrue(all(r[1] for r in replies), replies)
                observed = next(r[2] for r in replies if r[0] == 2)
                self.assertEqual(observed.mode, "hand_guiding")
                self.assertEqual(observed.gripper_position, (None, None))
                self.assertEqual(bus.device_modes, {"left": "leader", "right": "leader"})
        finally:
            bus.plant.close()

    def test_pair_enters_leader_when_normal_status_stream_disappears(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                transport.enter_hand_guiding()
                self.assertEqual(transport.control_mode, "hand_guiding")
                self.assertEqual(bus.device_modes, {"left": "leader", "right": "leader"})
                sample = transport.read_encoders()
                self.assertEqual(sample.grippers, (None, None))
                self.assertEqual(sample.gripper_feedback, (None, None))
                self.assertTrue(all(r["motor_enabled"] is None for r in sample.feedback_status))
                self.assertTrue(all(r["velocity_source"] == "angle_difference" for r in sample.feedback_status))
                self.assertFalse(any(bus.push.values()))
                # Both mode writes precede the first reader confirmation.
                mixed = [r for r in bus.timeline if len(set(r["modes"].values())) > 1]
                self.assertEqual(mixed, [])
        finally:
            bus.plant.close()

    def test_exit_restores_status_and_reseeds_after_manual_repositioning(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                transport.enter_hand_guiding()
                bus.plant.data.qpos[bus.plant.qids[0]] += .1
                bus.plant.data.qpos[bus.plant.qids[7]] -= .1
                time.sleep(.02)
                transport.read_encoders()
                measured = np.asarray(transport.read_encoders().positions)
                bus.frames.clear()
                transport.begin_control()
                self.assertEqual(transport.control_mode, "follower")
                self.assertTrue(all(bus.push.values()))
                sample = transport.read_encoders()
                self.assertTrue(all(r["motor_enabled"] is True for r in sample.feedback_status))
                self.assertTrue(all(g is not None for g in sample.grippers))
                for i, side in enumerate(SIDES):
                    rows = [(cid, data) for s, cid, data in bus.frames if s == side]
                    self.assertEqual(rows[:4], position_frames(measured[i*7:i*7+7]))
                np.testing.assert_allclose(sample.positions, measured, atol=.001)
                self.assertFalse(any(cid in (0x471, 0x159) for _, cid, _ in bus.frames))
        finally:
            bus.plant.close()

    def test_supported_disable_handles_leader_without_normal_status(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                transport.enter_hand_guiding()
                transport.close()
                bus.frames.clear()
                with patch("agilex_deploy.hardware_power.socket.socket",
                           side_effect=lambda *args: closing(bus.socket(*args))):
                    result = _disable_and_verify(bus)
                self.assertEqual(result["left_enabled"], 0)
                self.assertEqual(result["right_enabled"], 0)
                self.assertEqual(bus.enabled, [0]*14)
                self.assertFalse(any(cid == 0x159 or (cid == 0x471 and data[1] == 2)
                                     for _, cid, data in bus.frames))
        finally:
            bus.plant.close()

    def test_dropped_leader_angles_fault_without_auto_recovery_or_disable(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                bus.drop_leader.add("left")
                with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                    transport.enter_hand_guiding()
                self.assertEqual(bus.device_modes, {"left": "leader", "right": "leader"})
                bus.frames.clear()
                with self.assertRaises(RuntimeError):
                    transport.begin_control()
                transport.close()
                self.assertEqual(bus.frames, [])
        finally:
            bus.plant.close()

    def test_runtime_preserves_grasp_targets_and_reports_unknown_feedback(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                robot = Robot(RealBackend(transport))
                transport.arm()
                robot.hold()
                robot.controller.grippers[:] = [0., 1.]
                robot.enter_hand_guiding()
                for _ in range(10):
                    robot.tick()
                    time.sleep(.01)
                observed = robot.observe()
                self.assertEqual(observed.mode, "hand_guiding")
                self.assertEqual(observed.gripper_position, (None, None))
                self.assertEqual([r["motor_enabled"] for r in observed.feedback_status], [None, None])
                robot.begin_control()
                np.testing.assert_allclose(robot.controller.grippers, [0., 1.])
                robot.close()
        finally:
            bus.plant.close()

    def test_follower_missing_driver_feedback_is_still_rejected(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                transport.enter_hand_guiding()
                bus.drop_driver.add("right")
                with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
                    transport.begin_control()
                self.assertIsNotNone(transport.fault)
                bus.frames.clear()
                transport.close()
                self.assertFalse(any(cid in (0x470, 0x471, 0x159) for _, cid, _ in bus.frames))
        finally:
            bus.plant.close()

    def test_shutdown_from_mixed_leader_follower_state(self):
        bus = FirmwareSim()
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                with patch.object(transport.endpoints["right"], "request_hand_guiding",
                                  side_effect=OSError("injected right mode write failure")):
                    with self.assertRaisesRegex(RuntimeError, "write failure"):
                        transport.enter_hand_guiding()
                self.assertEqual(bus.device_modes, {"left": "leader", "right": "follower"})
                transport.close()
                bus.frames.clear()
                with patch("agilex_deploy.hardware_power.socket.socket",
                           side_effect=lambda *args: closing(bus.socket(*args))):
                    result = _disable_and_verify(bus)
                self.assertEqual(result["left_enabled"] + result["right_enabled"], 0)
                self.assertEqual(bus.enabled, [0]*14)
        finally:
            bus.plant.close()


if __name__ == "__main__":
    unittest.main()
