"""Dual-arm protocol and actual IPC/official-IK checks; no physical CAN access."""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import json
from pathlib import Path
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np

from agilex_deploy.can_feedback import JointSnapshot
from agilex_deploy.contract import Action, ArmTarget, JOINT_NAMES, Pose, SIDES
from agilex_deploy.dual_transport import DualNeroTransport
from agilex_deploy.gripper import decode_feedback
from agilex_deploy.hardware_session import HardwareSession
from agilex_deploy.interactive import (
    constrain_commissioning_drag,
    toggle_gripper,
    viewer_targets,
)
from agilex_deploy.model import ROOT, Kinematics, initial_positions
from agilex_deploy.motion_limits import JointLimits


class BusPlant:
    def __init__(self):
        self.q = initial_positions()
        self.enabled = [0] * 14
        self.frames = []
        self.readers = 0
        self.closed = False
        self.sequence = 0
        self.fail_channel = None
        self.ignore_enable = False
        self.gripper_fault = False
        self.channels = {s: s for s in SIDES}
        self.angle_sources = dict.fromkeys(SIDES, "follower")
        self.device_modes = dict.fromkeys(SIDES, "follower")
        self.ctrl_modes = dict.fromkeys(SIDES, 1)
        self.ignore_mode = False

    def snapshot(self):
        now = time.monotonic()
        self.sequence += 1
        grippers = tuple(
            decode_feedback(struct.pack(">ihBB", width, 0, 0x40, 0), now)
            for width in (20000, 40000)
        )
        if self.gripper_fault:
            grippers = (grippers[0], replace(grippers[1], timestamp_s=now - 1))
        return JointSnapshot(
            self.sequence,
            now,
            0,
            JOINT_NAMES,
            tuple(self.q),
            tuple(self.enabled),
            now,
            velocities=(0.0,) * 14,
            arm_status=tuple(bytes([self.ctrl_modes[s], 0, 1]) + bytes(5) for s in SIDES),
            status_timestamp_s=now,
            gripper_feedback=grippers,
            angle_sources=tuple(self.device_modes[s] for s in SIDES),
        )

    def set_angle_source(self, side, source, since):
        self.angle_sources[side] = source

    def reader(self, *args, **kwargs):
        self.readers += 1
        self.assert_feedback = kwargs.get("gripper_feedback")
        return self

    def read(self, **kwargs):
        return self.snapshot()

    def close(self):
        self.closed = True

    def socket(self, *args):
        plant = self

        class Writer:
            def setsockopt(self, *args):
                pass

            def setblocking(self, *args):
                pass

            def settimeout(self, *args):
                pass

            def bind(self, address):
                self.channel = address[0]

            def close(self):
                pass

            def send(self, packet):
                cid, dlc, data = struct.unpack("=IB3x8s", packet)
                if plant.fail_channel == self.channel:
                    raise OSError("injected bus failure")
                plant.frames.append((self.channel, cid, data))
                if cid == 0x151 and not plant.ignore_mode:
                    plant.ctrl_modes[self.channel] = data[0]
                if cid == 0x470 and not plant.ignore_mode:
                    plant.device_modes[self.channel] = "leader" if data[0] == 0xFA else "follower"
                if cid == 0x471 and not plant.ignore_enable:
                    plant.enabled[SIDES.index(self.channel) * 7 + data[0] - 1] = (
                        0x40 if data[1] == 2 else 0
                    )
                return 16

        return Writer()


@contextmanager
def fake_hardware(plant=None, *, allow_simulated_enable=True):
    plant = plant if plant is not None else BusPlant()

    def limits(channel, model):
        if hasattr(plant, "motion_limits"):
            return plant.motion_limits(channel, model)
        return JointLimits(
            model, np.full(7, 1.0 if channel == "left" else 2.0), np.full(7, 2.5)
        )

    with (
        patch(
            "agilex_deploy.nero_transport.PassiveCanFeedback", side_effect=plant.reader
        ),
        patch(
            "agilex_deploy.nero_transport.query_firmware",
            return_value={
                "missing": [],
                "firmware": {s: {"software_version": "1.21"} for s in SIDES},
            },
        ),
        patch("agilex_deploy.nero_transport.query_motion_limits", side_effect=limits),
        patch(
            "agilex_deploy.gripper.query_range",
            side_effect=lambda c: 0.07 if c == "left" else 0.1,
        ),
        patch("agilex_deploy.nero_transport.socket.socket", side_effect=plant.socket),
    ):
        with ExitStack() as stack:
            # Offline firmware tests have their own commissioning evidence.
            # Never depend on, or modify, this desktop's physical qualification.
            read_text = Path.read_text
            def fixture_read_text(path, *args, **kwargs):
                if path == ROOT / "configs/commissioning.json":
                    return json.dumps({
                        "mounting_and_support_confirmed_by_user": True,
                        "controller_mounting_configuration_verified": True,
                        "static_model_posture": {"confirmed_by_user": True},
                    })
                return read_text(path, *args, **kwargs)
            stack.enter_context(patch.object(Path, "read_text", fixture_read_text))
            # This bypass exists only inside a fixture with all CAN I/O mocked.
            if allow_simulated_enable:
                for module in ("nero_transport", "dual_transport"):
                    stack.enter_context(patch("agilex_deploy." + module + ".require_qualified_enable"))
                    stack.enter_context(patch("agilex_deploy." + module + ".require_qualified_hand_guiding"))
                stack.enter_context(patch("agilex_deploy.hardware_session.require_qualified_hand_guiding"))
            transport = DualNeroTransport(gripper_control=True)
            try:
                yield transport, plant
            finally:
                plant.fail_channel = None
                plant.gripper_fault = False
                transport.close()


class DualProtocolTests(unittest.TestCase):
    def test_constructor_is_read_only_and_owns_one_paired_reader(self):
        with fake_hardware() as (transport, plant):
            self.assertEqual(plant.readers, 1)
            self.assertEqual(plant.assert_feedback, SIDES)
            self.assertEqual(plant.frames, [])
            sample = transport.read_encoders()
            np.testing.assert_allclose(sample.grippers, [0.02 / 0.07, 0.4])

    def test_both_limits_and_seeds_precede_enable_and_grippers_route_independently(
        self,
    ):
        with fake_hardware() as (transport, plant):
            transport.arm()
            self.assertTrue(transport.armed)
            enabled_at = next(
                i for i, row in enumerate(plant.frames) if row[1] == 0x471
            )
            for side in SIDES:
                self.assertTrue(
                    any(
                        s == side and cid == 0x170
                        for s, cid, _ in plant.frames[:enabled_at]
                    )
                )
                joints = [
                    data[0]
                    for s, cid, data in plant.frames
                    if s == side and cid == 0x471
                ]
                self.assertEqual(joints, list(range(1, 8)))
            self.assertNotIn(0x159, [cid for _, cid, _ in plant.frames])
            np.testing.assert_allclose(
                transport.guard.motion_limits.velocity, [1.0] * 7 + [2.0] * 7
            )
            target = transport.guard.lift_target()
            action = Action(
                ArmTarget(target.left.pose, 1), ArmTarget(target.right.pose, 0)
            )
            transport.guard.action(action)
            transport.activate_gripper(action)
            plant.frames.clear()
            transport.read_encoders()
            transport.send_positions(plant.q, [1, 0])
            widths = {
                side: struct.unpack(">i", data[:4])[0]
                for side, cid, data in plant.frames
                if cid == 0x159
            }
            self.assertEqual(widths, {"left": 70000, "right": 0})
            plant.frames.clear()
            transport.hold()
            widths = {
                side: struct.unpack(">i", data[:4])[0]
                for side, cid, data in plant.frames
                if cid == 0x159
            }
            self.assertEqual(widths, {"left": 20000, "right": 40000})

    def test_invalid_right_target_sends_no_left_motion_and_latches_pair(self):
        with fake_hardware() as (transport, plant):
            transport.arm()
            plant.frames.clear()
            q = plant.q.copy()
            q[7] += 0.1
            with self.assertRaisesRegex(RuntimeError, "speed bound"):
                transport.send_positions(q, [None, None])
            # Only measured HOLD references may be emitted after rejection.
            for side, cid, data in plant.frames:
                if cid == 0x155:
                    expected = np.rint(
                        np.degrees(
                            plant.q[SIDES.index(side) * 7 : SIDES.index(side) * 7 + 2]
                        )
                        * 1000
                    )
                    np.testing.assert_array_equal(struct.unpack(">ii", data), expected)
            self.assertIsNotNone(transport.fault)
            before = len(plant.frames)
            with self.assertRaises(RuntimeError):
                transport.send_positions(plant.q, [None, None])
            self.assertEqual(len(plant.frames), before)

    def test_partial_bus_failure_attempts_hold_on_healthy_side(self):
        with fake_hardware() as (transport, plant):
            transport.arm()
            plant.frames.clear()
            plant.fail_channel = "left"
            with self.assertRaisesRegex(RuntimeError, "paired HOLD unconfirmed"):
                transport.send_positions(plant.q, [None, None])
            self.assertTrue(
                any(side == "right" and cid == 0x170 for side, cid, _ in plant.frames)
            )

    def test_second_limit_readback_failure_never_enables_either_arm(self):
        with fake_hardware() as (transport, plant):
            with patch(
                "agilex_deploy.nero_transport.query_motion_limits",
                side_effect=[
                    JointLimits.simulation(Kinematics().limits[:7]),
                    RuntimeError("readback failed"),
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "readback failed"):
                    transport.arm()
            self.assertNotIn(0x471, [cid for _, cid, _ in plant.frames])
            self.assertFalse(transport.armed)

    def test_lost_right_enable_is_detected_from_same_snapshot(self):
        with fake_hardware() as (transport, plant):
            transport.arm()
            plant.enabled[8] = 0
            with self.assertRaisesRegex(RuntimeError, "lost motor enable"):
                transport.read_encoders()

    def test_one_gripper_command_does_not_enable_or_command_the_other(self):
        for selected in SIDES:
            with self.subTest(selected=selected), fake_hardware() as (transport, plant):
                transport.arm()
                pose = transport.guard.lift_target()
                action = Action(*(ArmTarget(getattr(pose, side).pose,
                                           1 if side == selected else None) for side in SIDES))
                transport.activate_gripper(action)
                plant.frames.clear()
                transport.send_positions(plant.q, [1 if side == selected else None for side in SIDES])
                jaw_channels = {side for side, cid, _ in plant.frames if cid == 0x159}
                self.assertEqual(jaw_channels, {selected})
                other = 'right' if selected == 'left' else 'left'
                self.assertFalse(transport.endpoints[other].gripper_active)

    def test_second_arm_enable_failure_holds_pair_and_never_reports_armed(self):
        with fake_hardware() as (transport, plant):
            original_socket = plant.socket

            def socket(*args):
                writer = original_socket(*args)
                original_send = writer.send

                def send(packet):
                    cid, _, data = struct.unpack("=IB3x8s", packet)
                    result = original_send(packet)
                    if writer.channel == "right" and cid == 0x471:
                        plant.enabled[7:14] = [0] * 7
                    return result

                writer.send = send
                return writer

            with patch(
                "agilex_deploy.nero_transport.socket.socket", side_effect=socket
            ):
                with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
                    transport.arm()
            self.assertFalse(transport.armed)
            self.assertIsNotNone(transport.fault)
            self.assertEqual(
                {side for side, cid, _ in plant.frames[-8:] if cid == 0x170}, set(SIDES)
            )

    def test_stale_right_gripper_never_sends_either_open_target(self):
        with fake_hardware() as (transport, plant):
            transport.arm()
            target = transport.guard.lift_target()
            transport.activate_gripper(
                Action(ArmTarget(target.left.pose, 1), ArmTarget(target.right.pose, 1))
            )
            plant.gripper_fault = True
            snapshot = plant.snapshot()
            for endpoint in transport.endpoints.values():
                endpoint.last = snapshot
            plant.frames.clear()
            with self.assertRaisesRegex(RuntimeError, "stale"):
                transport.send_positions(plant.q, [1, 1])
            widths = {
                side: struct.unpack(">i", data[:4])[0]
                for side, cid, data in plant.frames
                if cid == 0x159
            }
            self.assertEqual(widths, {"left": 20000, "right": 40000})


class DualSessionTests(unittest.TestCase):
    def test_both_arms_track_one_action_grippers_hold_and_watchdog(self):
        with HardwareSession("both", simulation=True, gripper_control=True) as robot:
            self.assertEqual(robot.observe().mode, "unarmed")
            robot.arm()
            initial = robot.observe()
            goals = []
            for side in SIDES:
                pose = getattr(initial, side + "_tcp")
                goals.append(
                    ArmTarget(
                        Pose(
                            np.asarray(pose.position_m) + [0, 0, 0.03],
                            pose.quaternion_xyzw,
                        ),
                        1 if side == "left" else 0,
                    )
                )
            action = Action(*goals)
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                observed = robot.observe()
                self.assertNotEqual(observed.mode, "fault", observed.reason)
                robot.submit_target(action, observed.observation_id)
                time.sleep(0.025)
            observed = robot.observe()
            for side in SIDES:
                np.testing.assert_allclose(
                    getattr(observed, side + "_tcp").position_m,
                    getattr(action, side).pose.position_m,
                    atol=0.003,
                )
            self.assertGreater(observed.gripper_position[0], 0.99)
            self.assertLess(observed.gripper_position[1], 0.01)
            robot.hold()
            held = robot.observe()
            for _ in range(5):
                time.sleep(0.02)
                observed = robot.observe()
            np.testing.assert_allclose(
                observed.joint_positions_rad, held.joint_positions_rad, atol=0.001
            )
            time.sleep(0.3)
            self.assertIn("heartbeat", robot.observe().reason)

    def test_sequential_gripper_commands_preserve_the_other_jaw_and_both_arms(self):
        with HardwareSession('both', simulation=True, gripper_control=True) as robot:
            robot.arm()
            initial = robot.observe()
            targets = viewer_targets(initial, True)
            for side in ('left', 'right', 'left', 'right'):
                before = robot.observe()
                other_index = 1 - SIDES.index(side)
                targets = toggle_gripper(targets, before, side, 'both')
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    observed = robot.observe()
                    robot.submit_target(targets, observed.observation_id)
                    self.assertAlmostEqual(observed.gripper_position[other_index],
                                           before.gripper_position[other_index], delta=.001)
                    time.sleep(.025)
                after = robot.observe()
                self.assertAlmostEqual(after.gripper_position[SIDES.index(side)],
                                       getattr(targets, side).gripper_open, delta=.001)
                np.testing.assert_allclose(after.joint_positions_rad,
                                           initial.joint_positions_rad, atol=.0001)

    def test_viewer_both_handles_and_gripper_targets_survive_admission(self):
        with HardwareSession("both", simulation=True, gripper_control=True) as robot:
            anchor = robot.arm()
            obs = robot.observe()
            target = viewer_targets(obs, True)
            for side in SIDES:
                target = toggle_gripper(target, obs, side, "both")
            updated, constrained = constrain_commissioning_drag(
                target, anchor, "both", None
            )
            self.assertEqual(updated, target)
            self.assertFalse(constrained)
            robot.submit_target(updated, obs.observation_id)
            self.assertNotEqual(robot.observe().mode, "fault")


if __name__ == "__main__":
    unittest.main()
