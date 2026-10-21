"""Workshop arm selection survives encoder jitter and the control-process boundary."""
from dataclasses import asdict, replace
import pickle
import time
from types import SimpleNamespace
import unittest

import numpy as np

from agilex_deploy.contract import JointAction, MixedAction, Pose, SIDES
from agilex_deploy.controller import Controller, pose_error
from agilex_deploy.hardware_session import HardwareSession
from agilex_deploy.model import Kinematics, initial_positions
from agilex_deploy.workshop_service import WorkshopService


class RegularizedIK:
    """Expose unwanted IK calls: even a satisfied pose gets a nonzero joint change."""

    def __init__(self):
        self.calls = []

    def solve(self, side, target, measured):
        self.calls.append((side, target))
        return np.asarray(measured) * .99


class ArmIsolationTests(unittest.TestCase):
    def setUp(self):
        self.q = initial_positions()
        self.kin = Kinematics()
        self.ik = RegularizedIK()
        self.controller = Controller(self.kin, self.q, (.36, .36), ik=self.ik)
        self.service = WorkshopService.__new__(WorkshopService)
        self.service.grippers = [None, None]
        self.service.kin = Kinematics()
        self.service.action_kin = Kinematics()

    def observation(self, q):
        self.kin.update(q, (.36, .36))
        return SimpleNamespace(joint_positions_rad=tuple(q),
                               left_tcp=self.kin.pose("left"),
                               right_tcp=self.kin.pose("right"))

    @staticmethod
    def pose_step(observed, sides, grippers=None):
        return dict(poses={side: asdict(Pose(
            np.asarray(getattr(observed, side + "_tcp").position_m) + [0, 0, .02],
            getattr(observed, side + "_tcp").quaternion_xyzw)) for side in sides},
            grippers=grippers or {})

    def test_single_arm_action_keeps_other_arm_at_captured_joints_under_jitter(self):
        for selected in SIDES:
            with self.subTest(selected=selected):
                self.controller.hold(self.q, (.36, .36))
                self.ik.calls.clear()
                self.service.grippers = [None, None]
                inactive = "right" if selected == "left" else "left"
                indices = slice(SIDES.index(inactive) * 7, SIDES.index(inactive) * 7 + 7)
                snapshot = self.q.copy()
                snapshot[indices] += 1e-6  # Fresh FK differs from the last HOLD pose.
                observed = self.observation(snapshot)
                action = self.service._action(self.pose_step(observed, [selected], {inactive: 1}), observed)
                self.assertIsInstance(action, MixedAction)
                self.assertEqual(action.cartesian_arms, (selected,))
                self.assertNotEqual(getattr(action, inactive).pose, self.controller.hold_poses[inactive])
                for tick in range(100):
                    measured = snapshot.copy()
                    measured[indices] += (-1 if tick % 2 else 1) * 1e-4
                    sample = SimpleNamespace(positions=tuple(measured), grippers=(.36, .36))
                    reference, grippers, _ = self.controller.step(sample, action)
                    if tick >= 20:
                        np.testing.assert_allclose(reference[indices], snapshot[indices], atol=1e-12, rtol=0)
                self.assertEqual({side for side, _ in self.ik.calls}, {selected})
                self.assertNotIn(inactive, self.controller.planner.paths)
                self.assertGreater(grippers[SIDES.index(inactive)], .36)
                self.assertEqual(grippers[SIDES.index(selected)], .36)

    def test_both_selected_arms_still_receive_cartesian_ik(self):
        observed = self.observation(self.q)
        action = self.service._action(self.pose_step(observed, SIDES), observed)
        self.controller.step(SimpleNamespace(positions=tuple(self.q), grippers=(.36, .36)), action)
        self.assertEqual(action.cartesian_arms, SIDES)
        self.assertEqual([side for side, _ in self.ik.calls], list(SIDES))
        self.assertEqual(set(self.controller.planner.paths), set(SIDES))

    def test_joint_hold_discards_only_the_inactive_cartesian_path(self):
        observed = self.observation(self.q)
        both = self.service._action(self.pose_step(observed, SIDES), observed)
        sample = SimpleNamespace(positions=tuple(self.q), grippers=(.36, .36))
        self.controller.step(sample, both)
        left_path, right_path = (self.controller.planner.paths[side] for side in SIDES)
        left_only = self.service._action(self.pose_step(observed, ["left"]), observed)
        self.controller.step(sample, left_only)
        self.assertIs(self.controller.planner.paths["left"], left_path)
        self.assertNotIn("right", self.controller.planner.paths)
        self.controller.step(sample, both)
        self.assertIsNot(self.controller.planner.paths["right"], right_path)

    def test_jaw_only_step_remains_joint_control_for_both_arms(self):
        observed = self.observation(self.q + 1e-6)
        action = self.service._action({"grippers": {"left": 1}}, observed)
        self.assertIsInstance(action, JointAction)
        self.controller.step(SimpleNamespace(positions=observed.joint_positions_rad,
                                             grippers=(.36, .36)), action)
        self.assertEqual(self.ik.calls, [])

    def test_mixed_action_validates_arm_selection_and_preserves_it_in_pickle(self):
        observed = self.observation(self.q)
        action = self.service._action(self.pose_step(observed, ["left"]), observed)
        self.assertEqual(pickle.loads(pickle.dumps(action)), action)
        for arms in ((), ("unknown",), ("left", "left")):
            with self.subTest(arms=arms), self.assertRaises(ValueError):
                replace(action, cartesian_arms=arms)

    def test_spawned_simulated_hardware_executes_mixed_action(self):
        # This uses the real IPC/control-worker path with a simulated transport;
        # no physical CAN interface is opened and no hardware is enabled.
        with HardwareSession("both", simulation=True) as session:
            session.arm()
            observed = session.observe()
            action = self.service._action(self.pose_step(observed, ["left"]), observed)
            before = np.asarray(observed.joint_positions_rad)
            for _ in range(80):
                observed = session.observe()
                self.assertNotEqual(observed.mode, "fault", observed.reason)
                session.submit_target(action, observed.observation_id)
                time.sleep(.01)
            observed = session.observe()
            after = np.asarray(observed.joint_positions_rad)
            np.testing.assert_allclose(after[7:], before[7:], atol=1e-5, rtol=0)
            self.assertGreater(np.max(abs(after[:7] - before[:7])), 1e-4)
            self.assertLess(np.linalg.norm(pose_error(observed.left_tcp, action.left.pose)[:3]), .02)


if __name__ == "__main__":
    unittest.main()
