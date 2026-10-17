"""Official gripper protocol, explicit command routing, and failure regressions.

All command tests use in-memory writers or a simulated plant; none opens CAN.
"""

import os
import struct
import time
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from agilex_deploy.backends import RealBackend
from agilex_deploy.can_feedback import (
    ANGLE_IDS,
    DRIVER_IDS,
    FeedbackAssembler,
    JointSnapshot,
)
from agilex_deploy.contract import JOINT_NAMES, Action, ArmTarget, SIDES
from agilex_deploy.gripper import decode_feedback, query_range, width_frame
from agilex_deploy.hardware_session import (
    HardwareSession,
    SimulatedCommissioningTransport,
)
from agilex_deploy.interactive import (
    constrain_commissioning_drag,
    run_viewer,
    toggle_gripper,
    viewer_targets,
)
from agilex_deploy.model import Kinematics, initial_positions
from agilex_deploy.nero_transport import MotionGuard, NeroTransport
from agilex_deploy.runtime import Robot


def feedback(width=0.04, status=0xC0, stamp=None):
    return decode_feedback(
        struct.pack(">ihBB", round(width * 1e6), -250, status, 0),
        time.monotonic() if stamp is None else stamp,
    )


def wire_transport(side="left"):
    t = NeroTransport.__new__(NeroTransport)
    t.side = side
    t.side_index = 0 if side == "left" else 1
    t.indices = slice(t.side_index * 7, t.side_index * 7 + 7)
    t.gripper_control = True
    t.gripper_active = False
    t.gripper_started = None
    t.gripper_force_n = 1.0
    t.gripper_range_m = 0.1
    t.armed = True
    t.fault = None
    t.last_send = time.monotonic()
    t.kin = Kinematics()
    q = initial_positions()
    t.guard = MotionGuard(side, q, t.kin.limits, gripper_control=True)
    states = [None, None]
    states[t.side_index] = feedback()
    now = time.monotonic()
    t.last = JointSnapshot(
        1,
        now,
        0,
        JOINT_NAMES,
        tuple(q),
        (0x40,) * 14,
        now,
        velocities=(0.0,) * 14,
        arm_status=(bytes([1, 0, 1]) + bytes(5),) * 2,
        status_timestamp_s=now,
        gripper_feedback=tuple(states),
    )
    t.last_command = q[t.indices].copy()
    t.writer = Mock()
    t.writer.send.return_value = 16
    t.reader = Mock()
    t.reader.read.return_value = t.last
    t.last_send = time.monotonic()
    return t, q


def frames(t):
    return [
        struct.unpack("=IB3x8s", call.args[0]) for call in t.writer.send.call_args_list
    ]


class GripperTests(unittest.TestCase):
    def test_official_width_scaling_sign_status_and_no_zeroing(self):
        cid, data = width_frame(0.5, 0.1, 1.0)
        self.assertEqual(cid, 0x159)
        self.assertEqual(data, bytes.fromhex("0000c35003e80100"))
        state = feedback()
        self.assertAlmostEqual(state.width_m, 0.04)
        self.assertAlmostEqual(state.force_n, -0.25)
        self.assertTrue(state.enabled)
        self.assertTrue(state.zeroed)
        self.assertAlmostEqual(state.normalized(0.1), 0.4)

    def test_angle_unknown_modes_faults_and_staleness_are_not_width(self):
        for mode in [1, 2, 255]:
            with self.assertRaisesRegex(ValueError, "WIDTH"):
                decode_feedback(bytes(7) + bytes([mode]), time.monotonic())
        for bit in range(6):
            with self.assertRaisesRegex(RuntimeError, "driver fault"):
                feedback(status=1 << bit).validate(time.monotonic())
        with self.assertRaisesRegex(RuntimeError, "stale"):
            feedback(stamp=time.monotonic() - 1).validate(time.monotonic())
        with self.assertRaisesRegex(RuntimeError, "outside"):
            feedback(width=0.12).normalized(0.1)

    def test_invalid_commands_never_get_encoded(self):
        for value in [float("nan"), -1, 1.1]:
            with self.assertRaises(ValueError):
                width_frame(value, 0.1)
        for force in [-1, 3.1, float("inf")]:
            with self.assertRaises(ValueError):
                width_frame(0.5, 0.1, force)
        for width in [0, 0.08, float("nan")]:
            with self.assertRaises(ValueError):
                width_frame(0.5, width)

    def test_range_query_is_read_only_and_has_no_guessed_fallback(self):
        sock = Mock()
        sock.__enter__ = Mock(return_value=sock)
        sock.__exit__ = Mock()
        sock.recv.return_value = struct.pack(
            "=IB3x8s", 0x47E, 8, bytes([100, 100, 1]) + bytes(5)
        )
        with (
            patch("agilex_deploy.gripper.socket.socket", return_value=sock),
            patch("agilex_deploy.gripper.select.select", return_value=([sock], [], [])),
        ):
            self.assertEqual(query_range("selected"), 0.1)
        self.assertEqual(
            struct.unpack("=IB3x8s", sock.send.call_args.args[0]),
            (0x477, 8, bytes([4]) + bytes(7)),
        )
        sock.send.reset_mock()
        with (
            patch("agilex_deploy.gripper.socket.socket", return_value=sock),
            patch("agilex_deploy.gripper.select.select", return_value=([], [], [])),
            self.assertRaisesRegex(RuntimeError, "No gripper range"),
        ):
            query_range("selected", 0.001)
        self.assertEqual(sock.send.call_count, 1)

    def test_paired_feedback_requires_new_gripper_frame_and_rejects_stale(self):
        assembler = FeedbackAssembler(gripper_feedback=("left",))
        now = time.monotonic()

        def joints(stamp):
            for side in ("left", "right"):
                for cid in DRIVER_IDS:
                    assembler.feed(side, cid, bytes(8), stamp)
                for cid in ANGLE_IDS:
                    assembler.feed(side, cid, bytes(8), stamp)

        joints(now)
        with self.assertRaisesRegex(RuntimeError, "Waiting"):
            assembler.snapshot(now)
        assembler.feed("left", 0x2A8, struct.pack(">ihBB", 40000, 0, 0, 0), now)
        first = assembler.snapshot(now)
        self.assertIsNone(first.gripper_feedback[1])
        self.assertAlmostEqual(first.gripper_feedback[0].width_m, 0.04)
        joints(now + 0.01)
        self.assertEqual(assembler.snapshot(now + 0.01).sequence, first.sequence)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            assembler.snapshot(now + 0.3)

    def test_arm_pose_commands_do_not_implicitly_enable_gripper(self):
        for side in ("left", "right"):
            t, q = wire_transport(side)
            g = [None, None]
            g[t.side_index] = 0.4
            t.send_positions(q, g)
            self.assertNotIn(0x159, [cid for cid, _, _ in frames(t)])
            action = t.guard.lift_target()
            t.activate_gripper(action)
            self.assertFalse(t.gripper_active)

    def test_selected_command_and_hold_use_only_official_gripper_id(self):
        for side in ("left", "right"):
            t, q = wire_transport(side)
            action = t.guard.lift_target()
            arms = {s: getattr(action, s) for s in ("left", "right")}
            arms[side] = ArmTarget(arms[side].pose, 1)
            action = Action(**arms)
            t.guard.action(action)
            t.activate_gripper(action)
            g = [None, None]
            g[t.side_index] = 1.0
            t.send_positions(q, g)
            self.assertEqual(
                frames(t)[-1], (0x159, 8, bytes.fromhex("000186a003e80100"))
            )
            t.writer.send.reset_mock()
            t.hold()
            # HOLD is measured 40 mm, not the last 100 mm goal or disable.
            self.assertEqual(
                frames(t)[-1], (0x159, 8, bytes.fromhex("00009c4003e80100"))
            )
            self.assertNotIn(0x471, [cid for cid, _, _ in frames(t)])

    def test_stale_gripper_prevents_all_new_motion_writes(self):
        t, q = wire_transport()
        t.gripper_active = True
        t.last = replace(
            t.last, gripper_feedback=(feedback(stamp=time.monotonic() - 1), None)
        )
        with self.assertRaisesRegex(RuntimeError, "stale"):
            t.send_positions(q, [1, None])
        t.writer.send.assert_not_called()

    def test_unselected_gripper_and_lost_enable_are_rejected(self):
        t, q = wire_transport()
        with self.assertRaisesRegex(ValueError, "not commissioned"):
            t.send_positions(q, [0.4, 0.5])
        t.writer.send.assert_not_called()
        t.gripper_active = True
        t.gripper_started = time.monotonic() - 1
        t.reader.read.return_value = replace(
            t.last, gripper_feedback=(feedback(status=0), None)
        )
        with self.assertRaisesRegex(RuntimeError, "lost enable"):
            t.read_encoders()

    def test_transport_exposes_measured_force_without_actuation(self):
        for side in ("left", "right"):
            t, _ = wire_transport(side)
            sample = t.read_encoders()
            sample.validate()
            measured = sample.gripper_feedback[t.side_index]
            self.assertAlmostEqual(measured.force_n, -.25)
            self.assertAlmostEqual(measured.width_m, .04)
            self.assertIsNone(sample.gripper_feedback[1 - t.side_index])
            t.writer.send.assert_not_called()

    def test_gripper_hold_error_does_not_prevent_arm_hold(self):
        t, _ = wire_transport()
        t.gripper_active = True
        t.last = replace(t.last, gripper_feedback=(feedback(width=0.2), None))
        with self.assertRaisesRegex(RuntimeError, "Arm HOLD sent"):
            t.hold()
        self.assertEqual([cid for cid, _, _ in frames(t)], [0x155, 0x156, 0x157, 0x170])

    def test_hold_does_not_enable_disabled_gripper(self):
        t, _ = wire_transport()
        t.gripper_active = True
        t.last = replace(t.last, gripper_feedback=(feedback(status=0), None))
        t.hold()
        self.assertNotIn(0x159, [cid for cid, _, _ in frames(t)])

    def test_viewer_reset_and_pose_drag_leave_gripper_unchanged(self):
        transport = SimulatedCommissioningTransport("left", gripper_control=True)
        with Robot(RealBackend(transport)) as robot:
            obs = robot.observe()
            target = viewer_targets(obs, True)
            self.assertIsNone(target.left.gripper_open)
            self.assertIsNone(target.right.gripper_open)
            with self.assertRaisesRegex(ValueError, "selected"):
                toggle_gripper(target, obs, "right", "left")
            opened = toggle_gripper(target, obs, "left", "left")
            self.assertEqual(opened.left.gripper_open, 1)
            self.assertEqual(
                toggle_gripper(opened, obs, "left", "left").left.gripper_open, 0
            )
            transport.arm()
            bounded, _ = constrain_commissioning_drag(
                opened, transport.guard.lift_target(), "left", 0.03
            )
            self.assertEqual(bounded.left.gripper_open, 1)

    def test_standalone_sim_first_key_does_not_move_untouched_jaw(self):
        from agilex_deploy.backends import SimBackend
        for side in SIDES:
            with self.subTest(side=side), Robot(SimBackend()) as robot:
                before = robot.observe()
                targets = viewer_targets(before)
                self.assertIsNone(targets.left.gripper_open)
                self.assertIsNone(targets.right.gripper_open)
                action = toggle_gripper(targets, before, side)
                robot.submit_target(action, before.observation_id)
                other = 1 - SIDES.index(side)
                for _ in range(180):
                    robot.target_deadline = time.monotonic() + .5
                    robot.tick()
                    self.assertAlmostEqual(robot.observe().gripper_position[other],
                                           before.gripper_position[other], delta=.001)
                self.assertGreater(robot.observe().gripper_position[SIDES.index(side)], .99)

    def test_shared_api_gripper_motion_keeps_both_arm_poses_and_other_gripper(self):
        transport = SimulatedCommissioningTransport("left", gripper_control=True)
        with Robot(RealBackend(transport)) as robot:
            transport.arm()
            robot.hold()
            robot.controller.gripper_rate = None
            obs = robot.observe()
            before = np.array(obs.joint_positions_rad)
            target = toggle_gripper(viewer_targets(obs, True), obs, "left", "left")
            transport.guard.action(target)
            robot.submit_target(target, obs.observation_id)
            transport.activate_gripper(target)
            for _ in range(120):
                robot.target_deadline = time.monotonic() + 0.5
                robot.tick()
            after = robot.observe()
            self.assertGreater(after.gripper_position[0], 0.99)
            self.assertIsNone(after.gripper_position[1])
            self.assertAlmostEqual(transport.plant.read().grippers[1], 0.36, places=4)
            np.testing.assert_allclose(after.joint_positions_rad, before, atol=1e-4)

    def test_process_watchdog_holds_gripper_and_latches(self):
        with HardwareSession("left", simulation=True, gripper_control=True) as robot:
            robot.arm()
            obs = robot.observe()
            target = toggle_gripper(viewer_targets(obs, True), obs, "left", "left")
            robot.submit_target(target, obs.observation_id)
            time.sleep(0.3)
            after = robot.observe()
            self.assertEqual(after.mode, "fault")
            self.assertIn("heartbeat", after.reason)
            with self.assertRaisesRegex(RuntimeError, "heartbeat"):
                robot.submit_target(target, after.observation_id)

    def test_viewer_keys_route_through_real_ipc_for_each_selected_gripper(self):
        for side, key in [("left", "1"), ("right", "2"), ("both", "1")]:
            submissions = []

            @contextmanager
            def renderer(model, data, key_callback, gripper_key=key, **_):
                class Viewer:
                    frame = 0
                    cam = SimpleNamespace(lookat=np.zeros(3))
                    opt = SimpleNamespace(geomgroup=np.zeros(6))
                    perturb = mujoco.MjvPerturb()

                    def lock(self):
                        return nullcontext()

                    def is_running(self):
                        self.frame += 1
                        if self.frame == 1:
                            key_callback(ord("E"))
                        if self.frame == 2:
                            key_callback(ord(gripper_key))
                        if self.frame == 3 and side == "both":
                            key_callback(ord("2"))
                        if self.frame == 6:
                            key_callback(32)
                        return self.frame <= 7

                    def sync(self, **_):
                        pass

                    def set_texts(self, *_):
                        pass

                yield Viewer()

            with (
                self.subTest(side=side),
                HardwareSession(side, simulation=True, gripper_control=True) as session,
            ):
                submit = session.submit_target

                def capture(action, observation_id, send=submit, recorded=submissions):
                    result = send(action, observation_id)
                    recorded.append(action)
                    return result

                with (
                    patch.dict(os.environ, DISPLAY="test"),
                    patch("agilex_deploy.interactive.managed_viewer", renderer),
                    patch.object(session, "submit_target", side_effect=capture),
                ):
                    run_viewer(session, hardware_control=True)
                self.assertEqual(session.observe().mode, "hold")
            self.assertTrue(submissions)
            if side == "both":
                self.assertEqual(submissions[-1].left.gripper_open, 1)
                self.assertEqual(submissions[-1].right.gripper_open, 1)
                continue
            for action in submissions:
                self.assertEqual(getattr(action, side).gripper_open, 1)
                self.assertIsNone(
                    getattr(action, "right" if side == "left" else "left").gripper_open
                )


if __name__ == "__main__":
    unittest.main()
