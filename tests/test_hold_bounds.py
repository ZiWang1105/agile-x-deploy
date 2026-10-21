"""Offline HOLD boundary regressions with memory-only I/O and deterministic IK."""

from dataclasses import replace
import re
import struct
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy.backends import EncoderSample
from agilex_deploy.can_feedback import JointSnapshot
from agilex_deploy.contract import Action, ArmTarget, JOINT_NAMES, Pose, SIDES
from agilex_deploy.controller import Controller
from agilex_deploy.dual_transport import DualMotionGuard, DualNeroTransport
from agilex_deploy.motion_limits import JointLimits
from agilex_deploy.nero_transport import MotionGuard, NeroTransport, position_frames
from agilex_deploy.runtime import Robot


NOW = 1234.0
# Fractional millidegrees deliberately make nearest-integer rounding escape
# these intervals at their edges. The recorded J2/J7 limits are preserved.
LIMITS = np.radians(np.column_stack((-30.0006 - np.arange(14), 40.0006 + np.arange(14))))
LIMITS[[1, 8], 1] = 1.74
LIMITS[[6, 13], 1] = 1.5707963


class MemoryKinematics:
    def __init__(self, limits=LIMITS):
        self.limits = np.array(limits, dtype=float, copy=True)
        self.q = self.limits.mean(axis=1)

    def update(self, q, grippers):
        self.q = np.array(q, dtype=float, copy=True)

    def pose(self, side):
        start = 7 * SIDES.index(side)
        local = self.q[start:start + 7]
        position = .01 * local[:3] + .001 * local.sum()
        quaternion = Rotation.from_rotvec([0, 0, .01 * local[6]]).as_quat()
        return Pose(position, quaternion)

    def jacobian(self, side):
        return np.zeros((6, 14))


class FixedIK:
    def __init__(self, goal):
        self.goal = np.asarray(goal).copy()
        self.calls = 0

    def solve(self, side, target, seed):
        self.calls += 1
        start = 7 * SIDES.index(side)
        return self.goal[start:start + 7].copy()


class MemoryBackend:
    def __init__(self, q):
        self.q = np.asarray(q).copy()
        self.sequence = 0
        self.writes = []
        self.holds = 0

    def read(self):
        self.sequence += 1
        return EncoderSample(self.sequence, NOW, JOINT_NAMES, tuple(self.q),
                             (0.,) * 14, (None, None))

    def write(self, positions, grippers):
        self.writes.append(np.asarray(positions).copy())

    def advance(self):
        pass

    def hold(self, **kwargs):
        self.holds += 1

    def close(self):
        pass


def target_from(kinematics, offset=0.):
    targets = []
    for side in SIDES:
        pose = kinematics.pose(side)
        targets.append(ArmTarget(Pose(np.asarray(pose.position_m) + [0, 0, offset],
                                     pose.quaternion_xyzw), None))
    return Action(*targets)


def snapshot(q):
    return JointSnapshot(1, NOW, 0., JOINT_NAMES, tuple(q), (0x40,) * 14, NOW,
                         velocities=(0.,) * 14,
                         arm_status=(bytes([1, 0, 1]) + bytes(5),) * 2,
                         status_timestamp_s=NOW)


def decode_positions(frames):
    values = [value for _, data in frames for value in struct.unpack(">ii", data)]
    return np.radians(np.asarray(values[:7]) * .001)


def memory_transport(side, limits=LIMITS):
    """Bypass construction entirely: no configuration reads or CAN sockets."""
    transport = NeroTransport.__new__(NeroTransport)
    transport.side = side
    transport.side_index = SIDES.index(side)
    transport.indices = slice(7 * transport.side_index, 7 * transport.side_index + 7)
    transport.control_mode = "follower"
    transport.armed = True
    transport.fault = None
    transport.writer = object()
    transport.gripper_control = transport.gripper_active = False
    transport.last_send = NOW
    transport.kin = MemoryKinematics(limits)
    seed = np.asarray(limits).mean(axis=1)
    # A wider model distinguishes live effective limits from model limits.
    model_limits = np.asarray(limits) + [-.1, .1]
    transport.guard = MotionGuard(
        side, seed, model_limits, kinematics=transport.kin,
        motion_limits=JointLimits.simulation(np.asarray(limits)[transport.indices]))
    transport.last = snapshot(seed)
    transport.last_command = seed[transport.indices].copy()
    transport.sent = []
    transport._send = Mock(side_effect=lambda frames: transport.sent.extend(frames))
    return transport


class OfflineBoundsTestCase(unittest.TestCase):
    def setUp(self):
        self.clock = patch("time.monotonic", return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        no_socket = patch("agilex_deploy.nero_transport.socket.socket",
                          side_effect=AssertionError("HOLD regression tests must not open sockets"))
        no_socket.start()
        self.addCleanup(no_socket.stop)

    def assert_bounded(self, q, limits):
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.all(np.asarray(q) >= limits[:, 0]))
        self.assertTrue(np.all(np.asarray(q) <= limits[:, 1]))


class ControllerHoldBoundsTests(OfflineBoundsTestCase):
    def test_active_controller_holds_every_joint_boundary_without_restarting_ik(self):
        seed = LIMITS.mean(axis=1)
        for joint in range(14):
            for edge, sign in ((0, -1), (1, 1)):
                with self.subTest(joint=JOINT_NAMES[joint], edge=edge):
                    kin = MemoryKinematics()
                    ik = FixedIK(seed + .02)
                    controller = Controller(kin, seed, (None, None), ik=ik)
                    sample = MemoryBackend(seed).read()
                    controller.step(sample, target_from(kin, .03))
                    self.assertTrue(np.any(controller.velocity != 0))
                    self.assertTrue(controller.planner.paths)
                    measured = seed.copy()
                    measured[joint] = LIMITS[joint, edge] + sign * .005
                    expected = np.clip(measured, *LIMITS.T)

                    controller.hold(measured, (None, None))

                    np.testing.assert_array_equal(controller.reference, expected)
                    np.testing.assert_array_equal(controller.hold_reference, expected)
                    np.testing.assert_array_equal(controller.velocity, np.zeros(14))
                    np.testing.assert_array_equal(kin.q, measured)
                    self.assertEqual(controller.planner.paths, {})
                    self.assertEqual(controller.hold_poses,
                                     {side: kin.pose(side) for side in SIDES})
                    calls = ik.calls
                    held, _, _ = controller.step(
                        replace(sample, positions=tuple(measured)), target_from(kin))
                    np.testing.assert_array_equal(held, expected)
                    self.assertEqual(ik.calls, calls)
                    np.testing.assert_array_equal(controller.velocity, np.zeros(14))

    def test_tracking_to_hold_keeps_next_runtime_write_bounded(self):
        seed = LIMITS.mean(axis=1)
        for joint in range(14):
            for edge, sign in ((0, -1), (1, 1)):
                with self.subTest(joint=JOINT_NAMES[joint], edge=edge):
                    backend = MemoryBackend(seed)
                    with patch("agilex_deploy.runtime.Kinematics", MemoryKinematics):
                        with Robot(backend, ik=FixedIK(seed + .02)) as robot:
                            robot.submit_target(target_from(robot.kinematics, .03),
                                                robot.observe().observation_id)
                            robot.tick()
                            self.assertEqual(robot.mode, "tracking")
                            self.assertTrue(np.any(robot.controller.velocity != 0))
                            backend.q[joint] = LIMITS[joint, edge] + sign * .005

                            robot.hold()
                            robot.tick()

                            self.assertEqual(robot.mode, "hold")
                            self.assertEqual(backend.holds, 1)
                            self.assertIsNone(robot.target)
                            np.testing.assert_array_equal(
                                backend.writes[-1], np.clip(backend.q, *LIMITS.T))
                            self.assert_bounded(backend.writes[-1], LIMITS)
                            self.assertEqual(robot.observe().joint_positions_rad[joint],
                                             backend.q[joint])
                            np.testing.assert_array_equal(robot.controller.velocity, np.zeros(14))

    def test_recorded_j2_j7_feedback_overshoots_hold_at_exact_source_bounds(self):
        measured = LIMITS.mean(axis=1)
        for offset in (0, 7):
            measured[offset + 1] = 1.7400059977757467
            measured[offset + 6] = 1.5707963267948966
        kin = MemoryKinematics()
        controller = Controller(kin, LIMITS.mean(axis=1), (None, None),
                                ik=FixedIK(LIMITS.mean(axis=1)))

        controller.hold(measured, (None, None))

        np.testing.assert_array_equal(controller.reference, np.clip(measured, *LIMITS.T))
        for offset in (0, 7):
            self.assertEqual(controller.reference[offset + 1], 1.74)
            self.assertEqual(controller.reference[offset + 6], 1.5707963)
        np.testing.assert_array_equal(kin.q, measured)


class GuardHoldBoundsTests(OfflineBoundsTestCase):
    def test_hold_clips_every_joint_and_resets_command_continuity(self):
        for side in SIDES:
            guard = memory_transport(side).guard
            seed = guard.seed.copy()
            for joint in range(14):
                for edge, sign in ((0, -1), (1, 1)):
                    with self.subTest(side=side, joint=JOINT_NAMES[joint], edge=edge):
                        measured = seed.copy()
                        measured[joint] = guard.limits[joint, edge] + sign * .005
                        expected = np.clip(measured, *guard.limits.T)
                        guard.velocity[:] = .1

                        guard.hold(measured)

                        np.testing.assert_array_equal(guard.previous, expected)
                        np.testing.assert_array_equal(guard.velocity, np.zeros(14))
                        selected = guard.command(expected, measured)
                        np.testing.assert_array_equal(selected, expected[guard.indices])
                        np.testing.assert_array_equal(guard.velocity, np.zeros(14))
            np.testing.assert_array_equal(guard.seed, seed)

    def test_feedback_at_tolerance_is_accepted_beyond_tolerance_is_atomic(self):
        for side in SIDES:
            guard = memory_transport(side).guard
            for joint in range(14):
                for edge, sign in ((0, -1), (1, 1)):
                    with self.subTest(side=side, joint=JOINT_NAMES[joint], edge=edge):
                        measured = guard.seed.copy()
                        measured[joint] = guard.limits[joint, edge] + sign * .02
                        guard.hold(measured)
                        self.assert_bounded(guard.previous, guard.limits)
                        previous = guard.previous.copy()
                        guard.velocity[:] = .125
                        velocity = guard.velocity.copy()
                        measured[joint] = guard.limits[joint, edge] + sign * .020001

                        with self.assertRaises(RuntimeError):
                            guard.hold(measured)

                        np.testing.assert_array_equal(guard.previous, previous)
                        np.testing.assert_array_equal(guard.velocity, velocity)

    def test_malformed_feedback_does_not_change_guard_state(self):
        for side in SIDES:
            guard = memory_transport(side).guard
            for malformed in ([np.nan] * 14, [np.inf] * 14, [0.] * 13, np.zeros((14, 1))):
                with self.subTest(side=side, shape=np.shape(malformed)):
                    previous = guard.previous.copy()
                    guard.velocity[:] = .125
                    with self.assertRaises(RuntimeError):
                        guard.hold(malformed)
                    np.testing.assert_array_equal(guard.previous, previous)
                    np.testing.assert_array_equal(guard.velocity, np.full(14, .125))

    def test_command_bound_error_identifies_joint_value_bound_and_delta(self):
        for side in SIDES:
            guard = memory_transport(side).guard
            for joint in range(14):
                for edge, sign in ((0, -1), (1, 1)):
                    with self.subTest(side=side, joint=JOINT_NAMES[joint], edge=edge):
                        command = guard.seed.copy()
                        bound = guard.limits[joint, edge]
                        command[joint] = bound + sign * .000006

                        with self.assertRaisesRegex(RuntimeError, "model-limit bounds") as caught:
                            guard.command(command, guard.seed)

                        message = str(caught.exception)
                        self.assertIn(JOINT_NAMES[joint], message)
                        fields = dict(re.findall(r"\b(value|bound|delta)=([+\-\d.eE]+) rad", message))
                        self.assertEqual(set(fields), {"value", "bound", "delta"})
                        self.assertAlmostEqual(float(fields["value"]), command[joint], places=10)
                        self.assertAlmostEqual(float(fields["bound"]), bound, places=10)
                        self.assertAlmostEqual(float(fields["delta"]), command[joint] - bound, places=12)
                        np.testing.assert_array_equal(guard.previous, guard.seed)


class WireHoldBoundsTests(OfflineBoundsTestCase):
    def test_default_wire_codec_preserves_sign_order_and_rounding(self):
        q = np.radians([-1.0004, 2.0006, -3.0006, 4.0004, -5., 6., -7.])
        frames = position_frames(q)
        self.assertEqual([cid for cid, _ in frames], [0x155, 0x156, 0x157, 0x170])
        values = [value for _, data in frames for value in struct.unpack(">ii", data)]
        self.assertEqual(values, [-1000, 2001, -3001, 4000, -5000, 6000, -7000, 0])

    def test_wire_quantization_stays_inside_every_effective_boundary(self):
        for side_index, side in enumerate(SIDES):
            limits = LIMITS[side_index * 7:side_index * 7 + 7]
            for joint in range(7):
                for edge in (0, 1):
                    with self.subTest(side=side, joint=joint + 1, edge=edge):
                        q = limits.mean(axis=1)
                        q[joint] = limits[joint, edge]
                        frames = position_frames(q, limits)
                        decoded = decode_positions(frames)
                        self.assert_bounded(decoded, limits)
                        self.assertLessEqual(abs(decoded[joint] - q[joint]), np.radians(.001))
                        self.assertEqual(struct.unpack(">ii", frames[-1][1])[1], 0)

    def test_wire_roundtrip_respects_bounds_one_float_inside_can_angles(self):
        integers = np.array([9, 13, 18, 26, 33, 36, 41])
        limits = np.column_stack((
            np.nextafter(np.radians(-integers * .001), np.inf),
            np.nextafter(np.radians(integers * .001), -np.inf),
        ))
        for edge, sign in ((0, -1), (1, 1)):
            with self.subTest(edge=edge):
                frames = position_frames(limits[:, edge], limits)
                decoded = decode_positions(frames)

                self.assert_bounded(decoded, limits)
                values = [value for _, data in frames for value in struct.unpack(">ii", data)]
                np.testing.assert_array_equal(values[:7], sign * (integers - 1))

    def test_single_arm_hold_and_next_command_encode_bounded_effective_limits(self):
        for side in SIDES:
            for joint in range(7):
                for edge, sign in ((0, -1), (1, 1)):
                    with self.subTest(side=side, joint=joint + 1, edge=edge):
                        transport = memory_transport(side)
                        global_joint = 7 * SIDES.index(side) + joint
                        measured = transport.guard.seed.copy()
                        measured[global_joint] = LIMITS[global_joint, edge] + sign * .005
                        transport.last = snapshot(measured)
                        transport.guard.velocity[:] = .1

                        transport.hold()

                        expected = np.clip(measured, *transport.guard.limits.T)
                        np.testing.assert_array_equal(transport.guard.previous, expected)
                        np.testing.assert_array_equal(transport.guard.velocity, np.zeros(14))
                        np.testing.assert_array_equal(transport.last_command, expected[transport.indices])
                        self.assert_bounded(decode_positions(transport.sent),
                                            transport.guard.motion_limits.position)
                        transport.sent.clear()
                        transport.send_positions(expected, (None, None))
                        self.assert_bounded(decode_positions(transport.sent),
                                            transport.guard.motion_limits.position)
                        np.testing.assert_array_equal(transport.guard.velocity, np.zeros(14))

    def test_invalid_feedback_retains_last_bounded_setpoint_and_guard_state(self):
        for side in SIDES:
            for kind in ("stale", "nan", "shape", "below_tolerance", "above_tolerance"):
                with self.subTest(side=side, kind=kind):
                    transport = memory_transport(side)
                    previous = transport.guard.seed.copy()
                    previous[transport.indices] = transport.guard.motion_limits.position[:, 1]
                    transport.guard.hold(previous)
                    transport.guard.velocity[:] = .125
                    transport.last_command = previous[transport.indices].copy()
                    measured = previous.copy()
                    if kind == "nan":
                        measured[transport.indices.start] = np.nan
                    elif kind == "shape":
                        measured = measured[:-1]
                    elif kind.endswith("tolerance"):
                        edge, sign = (0, -1) if kind.startswith("below") else (1, 1)
                        joint = transport.indices.start
                        measured[joint] = transport.guard.limits[joint, edge] + sign * .020001
                    transport.last = snapshot(measured)
                    if kind == "stale":
                        transport.last = replace(transport.last, timestamp_s=NOW - 1.)

                    transport.hold()

                    np.testing.assert_array_equal(transport.last_command, previous[transport.indices])
                    np.testing.assert_array_equal(transport.guard.previous, previous)
                    np.testing.assert_array_equal(transport.guard.velocity, np.full(14, .125))
                    self.assertEqual(transport.sent, position_frames(
                        previous[transport.indices], transport.guard.motion_limits.position))
                    self.assert_bounded(decode_positions(transport.sent),
                                        transport.guard.motion_limits.position)

    def test_paired_hold_encodes_both_arms_with_their_effective_limits(self):
        pair = DualNeroTransport.__new__(DualNeroTransport)
        pair.endpoints = {side: memory_transport(side) for side in SIDES}
        pair.guard = DualMotionGuard({side: endpoint.guard
                                     for side, endpoint in pair.endpoints.items()})
        measured = LIMITS[:, 1].copy() + .005
        measured[::2] = LIMITS[::2, 0] - .005
        for endpoint in pair.endpoints.values():
            endpoint.last = snapshot(measured)

        pair.hold()

        expected = np.clip(measured, *LIMITS.T)
        for side, endpoint in pair.endpoints.items():
            with self.subTest(side=side):
                np.testing.assert_array_equal(endpoint.guard.previous, expected)
                np.testing.assert_array_equal(endpoint.last_command, expected[endpoint.indices])
                np.testing.assert_array_equal(endpoint.guard.velocity, np.zeros(14))
                self.assert_bounded(decode_positions(endpoint.sent),
                                    endpoint.guard.motion_limits.position)


if __name__ == "__main__":
    unittest.main()
