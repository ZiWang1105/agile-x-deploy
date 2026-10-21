"""Hand-guiding lifecycle and official wire transitions, with no physical I/O."""

from dataclasses import replace
from contextlib import closing
import struct
import socket
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np

from agilex_deploy.backends import SimBackend
from agilex_deploy.can_feedback import ANGLE_IDS, LEADER_ANGLE_IDS, DRIVER_IDS, FeedbackAssembler, PassiveCanFeedback
from agilex_deploy.contract import ArmTarget, JointAction, SIDES
from agilex_deploy.hand_guiding import official_mode_frames
from agilex_deploy.hardware_session import HardwareSession, control_worker
from agilex_deploy.hardware_power import _disable_and_verify
from agilex_deploy.dual_transport import DualNeroTransport
from agilex_deploy.nero_transport import position_frames
from agilex_deploy.runtime import Robot
from agilex_deploy.workshop_service import WorkshopService
from test_dual_control import fake_hardware


class HandGuidingTests(unittest.TestCase):
    def test_default_worker_never_enters_leader_even_when_close_requests_it(self):
        with fake_hardware() as (transport, bus):
            connection = Mock()
            connection.poll.return_value = True
            connection.recv.side_effect = [(1, "arm", None), (2, "observe", None),
                                           (3, "close", True)]
            with patch("agilex_deploy.hardware_session.DualNeroTransport", return_value=transport), \
                 patch("agilex_deploy.hardware_session.signal.signal"), \
                 patch.object(transport, "enter_hand_guiding", side_effect=AssertionError("Unexpected leader mode")):
                control_worker(connection, "both", "unused", "unused", False, None, True)
            replies = {call.args[0][0]: call.args[0] for call in connection.send.call_args_list}
            self.assertTrue(replies[1][1], replies[1])
            self.assertEqual(replies[2][2].mode, "hold")
            self.assertTrue(replies[3][1], replies[3])
            self.assertFalse(any(cid == 0x470 and data[0] == 0xFA for _, cid, data in bus.frames))
            self.assertFalse(any(cid == 0x471 and data[1] == 1 for _, cid, data in bus.frames))

    def test_default_service_holds_on_enable_release_stop_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            service = WorkshopService(output=directory)
            try:
                with patch.object(service.robot, "enter_hand_guiding",
                                  side_effect=AssertionError("Unexpected hand-guiding")):
                    self.assertFalse(service.capabilities()["gravity_comp"])
                    self.assertEqual(service.capabilities()["normal_policy_exit"], "hold")
                    service.claim("default")
                    service.enable()
                    self.assertEqual(service.observe()["mode"], "hold")
                    service.release("default")
                    self.assertEqual(service.observe()["mode"], "hold")
                    service.operator_stop()
                    self.assertTrue(service.stopped)
                    self.assertEqual(service.observe()["mode"], "hold")
                    service.close()
            finally:
                service.close()

    def test_explicit_supported_power_off_leaves_leader_before_disable(self):
        with fake_hardware() as (transport, bus):
            transport.arm()
            transport.enter_hand_guiding()
            transport.close()
            bus.frames.clear()
            with patch("agilex_deploy.hardware_power.socket.socket",
                       side_effect=lambda *args: closing(bus.socket(*args))):
                result = _disable_and_verify(bus)
            self.assertEqual(result["left_enabled"], 0)
            self.assertEqual(result["right_enabled"], 0)
            for side in SIDES:
                rows = [(cid, data) for s, cid, data in bus.frames if s == side]
                switch = next(i for i, (cid, data) in enumerate(rows) if cid == 0x470 and data[0] == 0xFC)
                disable = next(i for i, (cid, _) in enumerate(rows) if cid == 0x471)
                self.assertLess(switch, disable)
                self.assertFalse(any(cid == 0x471 and data[1] == 2 for cid, data in rows))

    def test_locally_echoed_position_frames_never_become_leader_feedback(self):
        class EchoSocket:
            def recvmsg(self, *_):
                return struct.pack("=IB3x8s", 0x155, 8, bytes(8)), [], socket.MSG_DONTROUTE, None

        sock = EchoSocket()
        reader = PassiveCanFeedback.__new__(PassiveCanFeedback)
        reader.sockets = {sock: "left"}
        reader.assembler = FeedbackAssembler(discover_leaders=True)
        reader.clock_offset = time.monotonic() - time.time()
        with patch("agilex_deploy.can_feedback.select.select",
                   side_effect=[([sock], [], []), ([], [], [])]):
            with self.assertRaisesRegex(RuntimeError, "complete"):
                reader.read()
        self.assertFalse(reader.assembler.angles)
        self.assertEqual(reader.assembler.angle_sources["left"], "follower")

    def test_official_methods_preserve_mount_and_speed_without_enable_disable(self):
        for mode, value in (("leader", 0xFA), ("follower", 0xFC)):
            frames = official_mode_frames(mode)
            self.assertEqual(frames, (
                (0x151, bytes([1, 255, 100, 0, 0, 0, 2, 0])),
                (0x470, bytes([value]) + bytes(7)),
            ))
            self.assertNotIn(0x471, [cid for cid, _ in frames])

    def test_feedback_switch_rejects_old_normal_and_partial_leader_packets(self):
        a = FeedbackAssembler()
        for side in SIDES:
            for cid in DRIVER_IDS:
                a.feed(side, cid, bytes([0, 0, 0, 0, 0, 0x40, 0, 0]), 10)
            for cid in ANGLE_IDS:
                a.feed(side, cid, bytes(8), 10)
        previous = a.snapshot(10)
        a.set_angle_source("left", "leader", 10.01)
        for cid in ANGLE_IDS:
            a.feed("left", cid, bytes(8), 10.02)
        for cid in LEADER_ANGLE_IDS:
            a.feed("left", cid, bytes(8), 10)  # Queued pre-switch frames.
        with self.assertRaisesRegex(RuntimeError, "complete"):
            a.snapshot(10.02)
        for cid in LEADER_ANGLE_IDS[:-1]:
            a.feed("left", cid, struct.pack(">ii", 90000, -45000), 10.02)
        for cid in ANGLE_IDS:
            a.feed("right", cid, bytes(8), 10.02)
        with self.assertRaisesRegex(RuntimeError, "complete"):
            a.snapshot(10.02)
        a.feed("left", LEADER_ANGLE_IDS[-1], struct.pack(">ii", 30000, 0), 10.02)
        observed = a.snapshot(10.02)
        self.assertGreater(observed.sequence, previous.sequence)
        self.assertEqual(observed.angle_sources, ("leader", "follower"))
        np.testing.assert_allclose(observed.positions[:7], np.radians([90, -45, 90, -45, 90, -45, 30]))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            a.snapshot(10.3)

    def test_pair_mode_switch_has_no_position_or_gripper_writes_while_guiding(self):
        with fake_hardware() as (transport, bus):
            transport.arm()
            transport.enter_hand_guiding()
            self.assertEqual(transport.control_mode, "hand_guiding")
            self.assertTrue(all(v == 0x40 for v in bus.enabled))
            bus.frames.clear()
            transport.hold()
            transport.read_encoders()
            self.assertEqual(bus.frames, [])
            bus.q[0] += .15
            bus.q[7] -= .1
            measured = bus.q.copy()
            transport.begin_control()
            self.assertEqual(transport.control_mode, "follower")
            for i, side in enumerate(SIDES):
                rows = [(cid, data) for s, cid, data in bus.frames if s == side]
                modes = [n for n, (cid, _) in enumerate(rows) if cid == 0x470]
                self.assertEqual(len(modes), 1)
                self.assertEqual(rows[:4], position_frames(measured[i*7:i*7+7]))
                for cid, data in rows:
                    if cid in LEADER_ANGLE_IDS:
                        self.assertIn((cid, data), position_frames(measured[i*7:i*7+7]))
                self.assertFalse(any(cid in (0x471, 0x159) for cid, _ in rows))
            np.testing.assert_allclose(transport.guard.guards["left"].previous, measured)

    def test_partial_transition_fault_cannot_replay_old_setpoints_or_disable(self):
        with fake_hardware() as (transport, bus):
            transport.arm()
            with patch.object(transport.endpoints["right"], "request_hand_guiding",
                              side_effect=OSError("injected bus failure")):
                with self.assertRaisesRegex(RuntimeError, "bus failure"):
                    transport.enter_hand_guiding()
            self.assertIsNotNone(transport.fault)
            self.assertEqual(transport.endpoints["left"].control_mode, "transition")
            bus.frames.clear()
            bus.fail_channel = None
            with self.assertRaises(RuntimeError):
                transport.begin_control()
            transport.close()
            self.assertFalse(any(s == "left" and cid in LEADER_ANGLE_IDS for s, cid, _ in bus.frames))
            self.assertFalse(any(cid == 0x471 for _, cid, _ in bus.frames))

    def test_mode_request_without_new_leader_feedback_fails(self):
        with fake_hardware() as (transport, bus):
            transport.arm()
            bus.ignore_mode = True
            with self.assertRaisesRegex(RuntimeError, "encoder stream"):
                transport.enter_hand_guiding()
            endpoint = transport.endpoints["left"]
            self.assertEqual(endpoint.control_mode, "transition")
            bus.frames.clear()
            endpoint.hold()
            endpoint.close()
            self.assertEqual(bus.frames, [])

    def test_reconnect_adopts_only_measured_enabled_leader_and_never_reenables(self):
        with fake_hardware() as (transport, bus):
            transport.arm()
            transport.enter_hand_guiding()
            transport.close()
            bus.q[0] += .1
            bus.q[7] -= .1
            bus.frames.clear()
            with_transport = DualNeroTransport(gripper_control=True)
            try:
                self.assertEqual(bus.frames, [])
                with_transport.arm()
                self.assertEqual(with_transport.control_mode, "hand_guiding")
                self.assertEqual(bus.frames, [])
                with_transport.begin_control()
                np.testing.assert_allclose(with_transport.guard.guards["left"].previous, bus.q)
                self.assertFalse(any(cid == 0x471 for _, cid, _ in bus.frames))
            finally:
                with_transport.close()
            bus.device_modes = dict.fromkeys(SIDES, "follower")
            with self.assertRaisesRegex(RuntimeError, "START_REQUIRES_DISABLED_ARMS"):
                DualNeroTransport(gripper_control=True)

    def test_sim_hand_guiding_stops_writes_preserves_jaws_and_reseeds(self):
        with Robot(SimBackend()) as robot:
            backend = robot.backend
            jaw_ids = [backend.model.actuator(s + "_gripper").id for s in SIDES]
            jaw_targets = backend.data.ctrl[jaw_ids].copy()
            robot.enter_hand_guiding()
            self.assertEqual(robot.observe().mode, "hand_guiding")
            self.assertTrue(np.all(backend.model.actuator_gainprm[backend.arm_actuators] == 0))
            with patch.object(backend, "write", wraps=backend.write) as writer:
                for _ in range(10):
                    robot.tick()
                robot.hold()
                writer.assert_not_called()
            np.testing.assert_equal(backend.data.ctrl[jaw_ids], jaw_targets)
            # Model a human changing the posture while the servos are released.
            backend.data.qpos[backend.qids[0]] += .1
            robot.tick()
            measured = robot.observe().joint_positions_rad
            robot.begin_control()
            np.testing.assert_allclose(robot.controller.reference, measured)
            self.assertTrue(np.any(backend.model.actuator_gainprm[backend.arm_actuators] != 0))
            np.testing.assert_equal(backend.data.ctrl[jaw_ids], jaw_targets)

    def test_normal_pause_retains_loaded_gripper_targets_across_mode_changes(self):
        with Robot(SimBackend()) as robot:
            backend = robot.backend
            jaw_ids = [backend.model.actuator(s + "_gripper").id for s in SIDES]
            # An object can block the measured jaws before the closing target.
            targets = (0.0, 0.8)
            robot.controller.grippers = list(targets)
            backend.write(robot.last_sample.positions, targets)
            commanded = backend.data.ctrl[jaw_ids].copy()
            self.assertNotAlmostEqual(robot.last_sample.grippers[0], targets[0])
            robot.hold(preserve_grippers=True)
            np.testing.assert_allclose(backend.data.ctrl[jaw_ids], commanded)
            np.testing.assert_allclose(robot.controller.grippers, targets)
            robot.enter_hand_guiding()
            robot.begin_control()
            np.testing.assert_allclose(backend.data.ctrl[jaw_ids], commanded)
            np.testing.assert_allclose(robot.controller.grippers, targets)
            # Explicit stopping semantics still freeze at measured jaw widths.
            robot.hold()
            np.testing.assert_allclose(robot.controller.grippers, robot.last_sample.grippers)
            self.assertGreater(backend.data.ctrl[jaw_ids[0]], commanded[0])

    def test_process_enable_idle_resume_and_exit_without_extra_command(self):
        with HardwareSession("both", simulation=True, gripper_control=True, gravity_comp=True) as session:
            session.arm()
            before = session.observe()
            self.assertEqual(before.mode, "hand_guiding")
            time.sleep(.3)
            self.assertEqual(session.observe().mode, "hand_guiding")
            observed = session.observe()
            action = JointAction(*(ArmTarget(getattr(observed, s + "_tcp"), None) for s in SIDES),
                                 joint_positions_rad=observed.joint_positions_rad)
            session.submit_target(action, observed.observation_id)
            session.enter_hand_guiding()
            self.assertEqual(session.observe().mode, "hand_guiding")
        self.assertFalse(session.process.is_alive())

    def test_workshop_policy_release_resumes_and_stop_latches_in_hand_guiding(self):
        with tempfile.TemporaryDirectory() as directory:
            service = WorkshopService(output=directory, gravity_comp=True)
            try:
                service.claim("first")
                service.enable()
                self.assertEqual(service.observe()["mode"], "hand_guiding")
                service.release("first")
                self.assertEqual(service.observe()["mode"], "hand_guiding")
                service.claim("second")
                with service.lock:
                    service.robot.begin_control()
                self.assertEqual(service.observe()["mode"], "hold")
                service.operator_stop()
                self.assertTrue(service.stopped)
                self.assertEqual(service.observe()["mode"], "hand_guiding")
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
