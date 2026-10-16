"""Commissioning admission and process watchdog tests; no hardware writes."""
from dataclasses import replace
import copy
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np

from agilex_deploy.backends import EncoderSample, SimBackend
from agilex_deploy.can_feedback import ANGLE_IDS, DRIVER_IDS, MOTOR_IDS, FeedbackAssembler, JointSnapshot
from agilex_deploy.contract import Action, ArmTarget, JOINT_NAMES, Pose, SIDES
from agilex_deploy.controller import Controller, pose_error
from agilex_deploy.hardware_session import HardwareSession
from agilex_deploy.model import Kinematics, initial_positions
from agilex_deploy.nero_transport import MotionGuard, NeroTransport, position_frames, MAX_SPEED, MAX_ACCELERATION
from agilex_deploy.runtime import Robot
from agilex_deploy.motion_limits import JointLimits


def upward(observation, side="left", dz=.005):
    arms = {s: ArmTarget(getattr(observation, s + "_tcp"), None) for s in SIDES}
    pose = arms[side].pose
    arms[side] = ArmTarget(Pose(np.asarray(pose.position_m) + [0, 0, dz], pose.quaternion_xyzw), None)
    return Action(**arms)


class HardwareTests(unittest.TestCase):
    def test_full_workspace_uses_per_joint_readback_without_old_tracking_cap(self):
        kin = Kinematics()
        seed = initial_positions()
        guard = MotionGuard("left", seed, kin.limits)
        self.assertIsNone(guard.workspace_radius_m)
        for change, message in ((.04, "speed"), (.001, "acceleration")):
            q = seed.copy()
            q[0] += change
            with self.assertRaisesRegex(RuntimeError, message):
                guard.command(q, seed)
        q = seed.copy()
        q[0] = kin.limits[0, 1] + .01
        with self.assertRaisesRegex(RuntimeError, "model-limit"):
            guard.command(q, seed)
        q = seed.copy()
        speed = 0.
        for _ in range(100):
            speed = min(MAX_SPEED[0], speed + MAX_ACCELERATION[0] * .01)
            q[0] += speed * .01
            # Deliberately fixed feedback: no invented tracking-error threshold.
            guard.command(q, seed)
        self.assertGreater(q[0] - seed[0], .6)
        with self.assertRaisesRegex(RuntimeError, "joint excursion"):
            MotionGuard("left", seed, kin.limits, workspace_radius_m=.03).command(q, q)

    def test_full_workspace_lift_passes_old_joint_excursion_limit_in_simulation(self):
        seed = np.array([.1483355331, 1.4546446651, -1.3873273158, .8295026336,
                         .0134390352, -.2115339053, .5728519671, .1937490003,
                         1.4897083297, 1.3339027874, .2881189529, .964905277,
                         .0620290016, 1.1849563891])
        with Robot(SimBackend(seed)) as robot:
            plant = robot.backend
            before = robot.observe()
            action = upward(before, dz=.1)
            guard = MotionGuard("left", seed, plant.kinematics.limits, workspace_radius_m=None)
            guard.action(action)
            ctl = Controller(plant.kinematics, seed, (None, None), MAX_SPEED, MAX_ACCELERATION)
            for _ in range(4000):
                sample = plant.read()
                q, _, _ = ctl.step(sample, action)
                guard.command(q, sample.positions)
                command = seed.copy()
                command[:7] = q[:7]
                plant.write(command, [.36, .36])
                plant.advance()
            after = plant.read()
            plant.kinematics.update(after.positions, (None, None))
            error = pose_error(plant.kinematics.pose("left"), action.left.pose)
            # Official soft pose objective retains a measurable residual.
            self.assertLess(np.linalg.norm(error[:3]), .05)
            self.assertLess(np.linalg.norm(error[3:]), .005)
            self.assertGreater(np.max(abs(np.array(after.positions[:7]) - seed[:7])), .45)
            np.testing.assert_allclose(after.positions[7:], seed[7:], atol=1e-5)

    def test_visible_lift_from_recorded_hardware_pose_with_all_guards(self):
        # Recorded encoder posture, not a live hardware connection. Exercise
        # both sides separately at physical reference rates in the sim plant.
        seed = np.radians([-6.043, 84.556, -78.144, 31.865, 2.543, -11.349, 39.621,
                           11.229, 82.496, 76.438, 16.608, 56.629, .057, 65.461])
        for side in SIDES:
            with self.subTest(side=side), Robot(SimBackend(seed)) as robot:
                plant = robot.backend
                before = robot.observe()
                guard = MotionGuard(side, seed, plant.kinematics.limits)
                action = guard.lift_target()
                guard.action(action)
                ctl = Controller(plant.kinematics, seed, (None, None), MAX_SPEED, MAX_ACCELERATION)
                for _ in range(2000):
                    sample = plant.read()
                    command, _, _ = ctl.step(sample, action)
                    guard.command(command, sample.positions)
                    selected = seed.copy()
                    selected[guard.indices] = command[guard.indices]
                    plant.write(selected, [.36, .36])
                    plant.advance()
                    plant.kinematics.update(plant.read().positions)
                    delta = pose_error(getattr(before, side + "_tcp"), plant.kinematics.pose(side))
                final = plant.read()
                # Check movement toward the lift without claiming that official
                # regularized IK exactly reaches its Cartesian target.
                self.assertGreater(delta[2], .015)
                self.assertLess(abs(delta[2] - .03), .015)
                self.assertLess(np.linalg.norm(delta[3:]), .005)
                inactive = slice(7, 14) if side == "left" else slice(0, 7)
                np.testing.assert_allclose(np.array(final.positions)[inactive], seed[inactive], atol=1e-5)
                # The G target is anchored to arming, never to current feedback.
                guard.hold(final.positions)
                self.assertEqual(action, guard.lift_target())
                self.assertEqual(tuple(guard.seed), tuple(seed))
                guard.action(upward(before, side, .031))

    def test_arm_seeds_measured_angles_before_enable_and_never_touches_gripper(self):
        class Writer:
            def __init__(self):
                self.frames = []
            def setsockopt(self, *_): pass
            def bind(self, address): self.address = address
            def setblocking(self, *_): pass
            def send(self, packet):
                self.frames.append(struct.unpack("=IB3x8s", packet))
                return 16
            def close(self): pass

        writer = Writer()
        q = initial_positions()
        class Reader:
            channels = {"left": "test-left", "right": "test-right"}
            def read(self, **_):
                enabled = {data[0] for cid, _, data in writer.frames if cid == 0x471}
                states = tuple(0x40 if j in enabled else 0 for j in range(1, 8)) + (0,) * 7
                now = time.monotonic()
                return JointSnapshot(1, now, 0, JOINT_NAMES, tuple(q), states, now,
                    velocities=(0.,)*14, arm_status=(bytes([1, 0, 1])+bytes(5), bytes([3, 6])+bytes(6)), status_timestamp_s=now)
            def close(self): pass

        t = NeroTransport.__new__(NeroTransport)
        t.gripper_control = t.gripper_active = False
        t.gripper_started = None
        t.side, t.side_index, t.indices = "left", 0, slice(0, 7)
        t.workspace_radius_m = .03
        t.kin, t.reader = Kinematics(), Reader()
        t.writer = t.lock_file = t.guard = t.last_command = t.fault = None
        t.armed = t.closed = False
        t.last_send = 0
        # Missing limits must never reach a position command or motor enable.
        failed = copy.copy(t)
        with patch("agilex_deploy.nero_transport.socket.socket", return_value=writer), \
             patch("agilex_deploy.nero_transport.query_motion_limits",
                   side_effect=RuntimeError("incomplete readback")):
            with self.assertRaisesRegex(RuntimeError, "incomplete readback"):
                failed.arm()
        failed.close()
        self.assertEqual([cid for cid, _, _ in writer.frames], [0x151])
        writer.frames.clear()
        with patch("agilex_deploy.nero_transport.socket.socket", return_value=writer), \
             patch("agilex_deploy.nero_transport.query_motion_limits",
                   return_value=JointLimits.simulation(t.kin.limits[:7])) as read_limits:
            t.arm()
        read_limits.assert_called_once()
        self.assertTrue(t.armed)
        self.assertEqual(writer.address, ("test-left",))
        first_enable = next(i for i, (cid, _, _) in enumerate(writer.frames) if cid == 0x471)
        self.assertEqual(writer.frames[0], (0x151, 8, bytes([1, 1, 100, 0, 0, 0, 0, 0])))
        self.assertEqual([(cid, data) for cid, _, data in writer.frames[1:first_enable]], position_frames(q[:7]))
        enables = [data[:2] for cid, _, data in writer.frames if cid == 0x471]
        self.assertEqual(enables, [bytes([j, 2]) for j in range(1, 8)])
        t.last_send = time.monotonic() - .3
        with self.assertRaisesRegex(RuntimeError, "watchdog expired"):
            t.send_positions(q, (None, None))
        t.close()
        self.assertFalse(any(cid == 0x471 and data[1] == 1 for cid, _, data in writer.frames))

    def test_joint_wire_encoding_sign_order_and_no_gripper_frames(self):
        q = np.radians([-1, 2, -3, 4, -5, 6, -7])
        frames = position_frames(q)
        self.assertEqual([cid for cid, _ in frames], [0x155, 0x156, 0x157, 0x170])
        values = [value for _, data in frames for value in struct.unpack(">ii", data)]
        self.assertEqual(values, [-1000, 2000, -3000, 4000, -5000, 6000, -7000, 0])
        with self.assertRaises(ValueError):
            position_frames([np.nan] * 7)

    def test_motor_feedback_is_required_not_synthesized(self):
        a = FeedbackAssembler(motor_feedback=True)
        for side in SIDES:
            for cid in DRIVER_IDS:
                a.feed(side, cid, bytes(8), 10)
            for cid in ANGLE_IDS:
                a.feed(side, cid, bytes(8), 10)
            a.feed(side, 0x2A1, bytes([3, 6]) + bytes(6), 10)
        with self.assertRaisesRegex(RuntimeError, "complete"):
            a.snapshot(10)
        for side in SIDES:
            for cid in MOTOR_IDS:
                a.feed(side, cid, struct.pack(">hhi", -123, 0, 0), 10.001)
        self.assertEqual(a.snapshot(10.01).velocities, (-.123,) * 14)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            a.snapshot(10.3)

    def test_none_grippers_survive_observation_action_round_trip(self):
        pose = Pose([0, 0, 0], [0, 0, 0, 1])
        action = Action(ArmTarget(pose, None), ArmTarget(pose, None))
        self.assertEqual(Action.from_dict(action.to_dict()), action)
        sample = EncoderSample(1, time.monotonic(), JOINT_NAMES, (0.,) * 14, (0.,) * 14, (None, None))
        sample.validate()
        with self.assertRaises(ValueError):
            replace(sample, grippers=(np.nan, None)).validate()

    def test_motion_guard_rejects_jump_speed_acceleration_and_downward_goal(self):
        kin = Kinematics()
        q = initial_positions()
        guard = MotionGuard("left", q, kin.limits, workspace_radius_m=.03)
        jump = q.copy()
        jump[0] += .04
        with self.assertRaisesRegex(RuntimeError, "speed"):
            guard.command(jump, q)
        jump[0] = q[0] + .001
        with self.assertRaisesRegex(RuntimeError, "acceleration"):
            guard.command(jump, q)
        jump[0] = q[0] + .000005
        guard.command(jump, q)
        kin.update(q)
        arms = {s: ArmTarget(kin.pose(s), None) for s in SIDES}
        p = arms["left"].pose
        arms["left"] = ArmTarget(Pose(np.array(p.position_m) + [0, 0, -.001], p.quaternion_xyzw), None)
        with self.assertRaisesRegex(ValueError, "no downward"):
            guard.action(Action(**arms))

    def test_unarmed_rejects_motion_and_process_watchdog_latches(self):
        with HardwareSession("left", simulation=True) as session:
            observed = session.observe()
            self.assertEqual(observed.mode, "unarmed")
            self.assertEqual(observed.gripper_position, (None, None))
            with self.assertRaisesRegex(RuntimeError, "unarmed"):
                session.submit_target(upward(observed), observed.observation_id)
            lift = session.arm()
            self.assertAlmostEqual(lift.left.pose.position_m[2] - observed.left_tcp.position_m[2], .03)
            self.assertNotEqual(session.observe().mode, "unarmed")
            # The parent does not send heartbeat or targets; child must HOLD
            # without relying on parent callbacks or its GIL.
            time.sleep(.3)
            stopped = session.observe()
            self.assertEqual(stopped.mode, "fault")
            self.assertIn("heartbeat expired", stopped.reason)
            with self.assertRaisesRegex(RuntimeError, "heartbeat expired"):
                session.submit_target(upward(stopped), stopped.observation_id)
        self.assertFalse(session.process.is_alive())

    def test_selected_arm_tracks_small_target_while_other_arm_stays_still(self):
        with HardwareSession("left", simulation=True) as session:
            session.arm()
            before = session.observe()
            action = upward(before, dz=.03)
            until = time.monotonic() + 1.5
            while time.monotonic() < until:
                observation = session.observe()
                session.submit_target(action, observation.observation_id)
                time.sleep(.04)
            after = session.observe()
            self.assertNotEqual(after.mode, "fault", after.reason)
            np.testing.assert_allclose(after.joint_positions_rad[7:], before.joint_positions_rad[7:], atol=1e-5)
            self.assertGreater(after.left_tcp.position_m[2] - before.left_tcp.position_m[2], .002)
            session.hold()


if __name__ == "__main__":
    unittest.main()
