"""Reproduce commissioning mouse rejection without opening CAN sockets."""
import os
import signal
import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from agilex_deploy.contract import Action, ArmTarget, Pose, SIDES
from agilex_deploy.hardware_session import HardwareSession
from agilex_deploy.interactive import (
    configure_handles, constrain_commissioning_drag, make_display, read_drag, write_handles,
    run_viewer,
)
from agilex_deploy.model import initial_positions
from agilex_deploy.nero_transport import MotionGuard, validate_workspace_radius


class CommissioningDragTests(unittest.TestCase):
    def test_full_workspace_removes_mouse_radius_floor_and_rotation_caps(self):
        kin = make_display()
        guard = MotionGuard("left", initial_positions(), kin.limits, workspace_radius_m=None)
        lift = guard.lift_target()
        candidate = Action(ArmTarget(Pose(np.asarray(lift.left.pose.position_m) + [.3, 0, -.2], [0, 0, 0, 1]), None), lift.right)
        projected, limited = constrain_commissioning_drag(candidate, lift, "left", None)
        self.assertEqual(projected, candidate)
        self.assertFalse(limited)
        guard.action(projected)
        moved_right = Action(candidate.left, ArmTarget(Pose(
            np.asarray(lift.right.pose.position_m) + [.01, 0, 0], lift.right.pose.quaternion_xyzw), None))
        with self.assertRaisesRegex(ValueError, "unselected arm"):
            guard.action(moved_right)

    def test_configured_workspace_matches_viewer_and_control_owner(self):
        kin = make_display()
        seed = initial_positions()
        guard = MotionGuard("left", seed, kin.limits, workspace_radius_m=.1)
        lift = guard.lift_target()
        pose = lift.left.pose
        candidate = Action(ArmTarget(Pose(np.asarray(pose.position_m) + [.08, 0, 0], pose.quaternion_xyzw), None), lift.right)
        projected, limited = constrain_commissioning_drag(candidate, lift, "left", .1)
        self.assertFalse(limited)
        guard.action(projected)
        with self.assertRaisesRegex(ValueError, "30 mm"):
            MotionGuard("left", seed, kin.limits, workspace_radius_m=.03).action(projected)
        with HardwareSession("left", simulation=True, workspace_radius_m=.1) as session:
            self.assertEqual(session.workspace_radius_m, .1)
            session.arm()
            session.submit_target(projected, session.observe().observation_id)
            session.hold()

    def test_invalid_workspace_is_rejected_before_starting_controller(self):
        for value in (float("nan"), float("inf"), -.1, .001, .21):
            with self.subTest(radius=value), self.assertRaises(ValueError):
                validate_workspace_radius(value)
        with patch("agilex_deploy.hardware_session.mp.get_context") as context:
            with self.assertRaises(ValueError):
                HardwareSession("left", workspace_radius_m=.5)
            context.assert_not_called()

    def test_viewer_arm_lift_then_sideways_drag_submits_a_valid_target(self):
        # Exercise the viewer event loop and real IPC owner with a simulated
        # plant. The fake renderer injects mouse state, never desktop events.
        texts = []
        submissions = []

        @contextmanager
        def renderer(model, data, key_callback, **_):
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
                    elif self.frame == 2:
                        key_callback(ord("G"))
                    return self.frame <= 6

                def sync(self, **_):
                    if self.frame == 3:
                        body = model.body("left_target")
                        self.perturb.select = body.id
                        self.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                        data.mocap_pos[body.mocapid[0], 0] += .005
                    else:
                        self.perturb.active = 0

                def set_texts(self, values):
                    texts.extend(str(v) for v in values)

            yield Viewer()

        for radius in (.03, None):
            texts.clear()
            submissions.clear()
            with self.subTest(radius=radius), HardwareSession("left", simulation=True, workspace_radius_m=radius) as session:
                submit = session.submit_target

                def capture(action, observation_id):
                    result = submit(action, observation_id)
                    submissions.append(action)
                    return result

                with patch.dict(os.environ, DISPLAY="test"), patch(
                        "agilex_deploy.interactive.managed_viewer", renderer), patch.object(
                        session, "submit_target", side_effect=capture):
                    run_viewer(session, hardware_control=True)
                self.assertEqual(session.observe().mode, "hold")
            self.assertGreaterEqual(len(submissions), 2)
            self.assertGreater(submissions[1].left.pose.position_m[0], submissions[0].left.pose.position_m[0] + .004)
            self.assertEqual(any("Target constrained" in text for text in texts), radius is not None)
            self.assertFalse(any("TARGET REJECTED" in text for text in texts))
            if radius is None:
                self.assertTrue(any("FULL WORKSPACE" in text for text in texts))

    def test_native_mocap_drag_after_lift_is_constrained_and_admitted(self):
        kin = make_display()
        seed = initial_positions()
        guard = MotionGuard("left", seed, kin.limits, workspace_radius_m=.03)
        lift = guard.lift_target()
        kin.update(seed, (None, None))
        write_handles(kin, lift)
        mujoco.mj_forward(kin.model, kin.data)
        target = kin.model.body("left_target")
        perturb = mujoco.MjvPerturb()
        perturb.select = target.id
        perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
        perturb.refpos[:] = lift.left.pose.position_m
        perturb.refpos[0] += .005
        perturb.refquat[:] = np.array(lift.left.pose.quaternion_xyzw)[[3, 0, 1, 2]]
        # This is the same native operation used by passive viewer.sync().
        mujoco.mjv_applyPerturbPose(kin.model, kin.data, perturb, 0)
        dragged = read_drag(kin, lift, target.id, perturb.active, sides=("left",))
        self.assertNotEqual(dragged, lift)
        with self.assertRaisesRegex(ValueError, "30 mm"):
            guard.action(dragged)
        accepted, limited = constrain_commissioning_drag(dragged, lift, "left")
        self.assertTrue(limited)
        guard.action(accepted)
        self.assertGreater(accepted.left.pose.position_m[0], lift.left.pose.position_m[0] + .004)
        self.assertLess(accepted.left.pose.position_m[2], lift.left.pose.position_m[2])
        self.assertEqual(accepted.right, lift.right)

    def test_inactive_handle_is_gray_and_cannot_change_target(self):
        kin = make_display()
        guard = MotionGuard("left", initial_positions(), kin.limits)
        lift = guard.lift_target()
        allowed = configure_handles(kin, ("left",))
        right = kin.model.body("right_target")
        self.assertNotIn(right.id, allowed)
        colors = kin.model.geom_rgba[kin.model.geom_bodyid == right.id]
        np.testing.assert_allclose(colors[:, 0], colors[:, 1])
        write_handles(kin, lift)
        kin.data.mocap_pos[right.mocapid[0], 2] += .02
        self.assertEqual(read_drag(kin, lift, right.id, True, sides=("left",)), lift)

    def test_drag_floor_and_orientation_are_constrained_for_both_sides(self):
        kin = make_display()
        for side in SIDES:
            guard = MotionGuard(side, initial_positions(), kin.limits)
            lift = guard.lift_target()
            arms = {s: getattr(lift, s) for s in SIDES}
            p = arms[side].pose
            arms[side] = ArmTarget(Pose(np.asarray(p.position_m) + [.1, 0, -.1], [0, 0, 0, 1]), None)
            accepted, limited = constrain_commissioning_drag(Action(**arms), lift, side)
            self.assertTrue(limited)
            guard.action(accepted)
            self.assertEqual(getattr(accepted, side).pose.quaternion_xyzw, p.quaternion_xyzw)

    def test_rejected_target_holds_previous_motion_and_ctrl_c_does_not_kill_owner(self):
        with HardwareSession("left", simulation=True, workspace_radius_m=.03) as session:
            lift = session.arm()
            observation = session.observe()
            session.submit_target(lift, observation.observation_id)
            bad_pose = Pose(np.asarray(lift.left.pose.position_m) + [.01, 0, 0], lift.left.pose.quaternion_xyzw)
            bad = Action(ArmTarget(bad_pose, None), lift.right)
            with self.assertRaisesRegex(RuntimeError, "30 mm"):
                session.submit_target(bad, session.observe().observation_id)
            self.assertEqual(session.observe().mode, "hold")
            os.kill(session.process.pid, signal.SIGINT)
            self.assertEqual(session.observe().mode, "hold")
            self.assertTrue(session.process.is_alive())
        self.assertFalse(session.process.is_alive())


if __name__ == "__main__":
    unittest.main()
