"""Path geometry and shared-controller integration; no hardware connections."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy.cartesian_path import CartesianPath, CartesianPlanner, PathResolution
from agilex_deploy.contract import Action, ArmTarget, JointAction, Pose
from agilex_deploy.backends import SimBackend
from agilex_deploy.controller import Controller
from agilex_deploy.runtime import Robot


class PathTests(unittest.TestCase):
    def test_translation_and_shortest_slerp(self):
        start = Pose([.2, 0, .4], [0, 0, 0, 1])
        end = Pose([.4, .1, .3], Rotation.from_rotvec([0, 0, .8]).as_quat())
        path = CartesianPath(start, end)
        for i, pose in enumerate(path.waypoints):
            fraction = i / (len(path.waypoints) - 1)
            np.testing.assert_allclose(pose.position_m,
                                       np.asarray(start.position_m) + fraction * path.translation)
            np.testing.assert_allclose(Rotation.from_quat(pose.quaternion_xyzw).as_rotvec(),
                                       [0, 0, fraction * .8], atol=1e-12)
        self.assertEqual(path.waypoints[-1], end)
        for a, b in zip(path.waypoints, path.waypoints[1:]):
            self.assertLessEqual(np.linalg.norm(np.subtract(a.position_m, b.position_m)), .01 + 1e-12)
            rotation = Rotation.from_quat(a.quaternion_xyzw).inv() * Rotation.from_quat(b.quaternion_xyzw)
            self.assertLessEqual(rotation.magnitude(), .05 + 1e-12)
        antipodal = CartesianPath(start, Pose(start.position_m, [0, 0, 0, -1]))
        self.assertEqual(len(antipodal.waypoints), 2)
        self.assertEqual(antipodal.angle_rad, 0)

    def test_feedback_not_elapsed_time_advances_path(self):
        start = Pose([0, 0, .4], [0, 0, 0, 1])
        goal = Pose([.2, 0, .4], [0, 0, 0, 1])
        path = CartesianPath(start, goal)
        for _ in range(100):
            path.advance(start)
        self.assertEqual(path.index, 1)
        while path.index < len(path.waypoints) - 1:
            previous = path.index
            path.advance(path.waypoints[previous])
            self.assertEqual(path.index, previous + 1)
        self.assertEqual(path.advance(goal), goal)

    def test_retarget_rebases_only_selected_arm_and_reset_discards_paths(self):
        planner = CartesianPlanner()
        start = Pose([0, 0, .4], [0, 0, 0, 1])
        end = Pose([.2, 0, .4], [0, 0, 0, 1])
        for side in ("left", "right"):
            planner.target(side, start, end)
        right = planner.paths["right"]
        measured = Pose([.03, 0, .4], [0, 0, 0, 1])
        new_goal = Pose([.03, .2, .4], [0, 0, 0, 1])
        planner.target("left", measured, new_goal)
        self.assertEqual(planner.paths["left"].start, measured)
        self.assertIs(planner.paths["right"], right)
        self.assertEqual(planner.revisions, dict(left=2, right=1))
        planner.reset()
        self.assertEqual(planner.status(), {})

    def test_bad_resolution_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PathResolution(translation_m=value)

    def test_fixed_orientation_does_not_require_settling_at_every_translation_sample(self):
        start = Pose([0, 0, .4], [0, 0, 0, 1])
        goal = Pose([.2, 0, .4], [0, 0, 0, 1])
        path = CartesianPath(start, goal)
        measured = Pose(path.waypoints[1].position_m, Rotation.from_rotvec([0, .02, 0]).as_quat())
        waypoint = path.advance(measured)
        self.assertEqual(path.index, 2)
        # Orientation is still commanded; only intermediate progression changes.
        np.testing.assert_allclose(waypoint.quaternion_xyzw, start.quaternion_xyzw)

    def test_stopping_lookahead_can_pass_multiple_samples_without_moving_endpoint(self):
        start = Pose([0, 0, .4], [0, 0, 0, 1])
        goal = Pose([.2, 0, .4], [0, 0, 0, 1])
        path = CartesianPath(start, goal)
        waypoint = path.advance(start, linear_lookahead_m=.05)
        self.assertEqual(path.index, 6)
        self.assertAlmostEqual(waypoint.position_m[0], .06)
        # Removing lookahead does not reverse the selected target.
        self.assertEqual(path.advance(start), waypoint)
        self.assertEqual(path.advance(start, linear_lookahead_m=1), goal)
        self.assertEqual(path.waypoints[-1], goal)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                path.advance(start, linear_lookahead_m=value)

    def test_controller_uses_waypoint_and_keeps_grippers_independent(self):
        class RecordingIK:
            calls = []

            def solve(self, side, pose, seed):
                self.calls.append((side, pose, seed.copy()))
                return seed

        backend = SimBackend()
        try:
            sample = backend.read()
            ik = RecordingIK()
            controller = Controller(backend.kinematics, sample.positions, sample.grippers, ik=ik)
            left, right = (controller.hold_poses[s] for s in ("left", "right"))
            goal = Pose(np.asarray(left.position_m) + [.2, 0, 0], left.quaternion_xyzw)
            action = Action(ArmTarget(goal, 1), ArmTarget(right, None))
            controller.step(sample, action)
            side, waypoint, seed = ik.calls[-1]
            self.assertEqual(side, "left")
            self.assertLess(np.linalg.norm(np.subtract(waypoint.position_m, left.position_m)), .011)
            self.assertNotEqual(waypoint, goal)
            np.testing.assert_array_equal(seed, sample.positions[:7])
            path = controller.planner.paths["left"]
            right_grip = controller.grippers[1]
            controller.step(sample, replace(action, left=ArmTarget(goal, 0)))
            self.assertIs(controller.planner.paths["left"], path)
            self.assertEqual(controller.grippers[1], right_grip)
            self.assertNotIn("right", controller.planner.paths)
            calls = len(ik.calls)
            controller.step(sample, JointAction(action.left, action.right, sample.positions))
            self.assertEqual(len(ik.calls), calls)
            self.assertEqual(controller.planner.status(), {})
            controller.step(sample, action)
            controller.hold(sample.positions, sample.grippers)
            self.assertEqual(controller.planner.status(), {})
        finally:
            backend.close()

    def test_waypoint_solver_failure_holds_and_discards_path(self):
        class FailedIK:
            def solve(self, *_):
                raise RuntimeError("waypoint IK failed")

        with Robot(SimBackend(), ik=FailedIK()) as robot:
            observation = robot.observe()
            left = observation.left_tcp
            action = Action(
                ArmTarget(Pose(np.asarray(left.position_m) + [.1, 0, 0],
                               left.quaternion_xyzw)),
                ArmTarget(observation.right_tcp))
            robot.submit_target(action, observation.observation_id)
            with self.assertRaisesRegex(RuntimeError, "waypoint IK failed"):
                robot.tick()
            self.assertEqual(robot.observe().mode, "fault")
            self.assertIsNone(robot.target)
            self.assertEqual(robot.controller.planner.status(), {})
            np.testing.assert_array_equal(robot.controller.reference,
                                          observation.joint_positions_rad)

    def test_return_to_initial_hold_pose_is_an_active_path(self):
        class RecordingIK:
            calls = []

            def solve(self, side, pose, seed):
                self.calls.append((side, pose))
                return seed

        backend = SimBackend()
        try:
            sample = backend.read()
            ik = RecordingIK()
            controller = Controller(backend.kinematics, sample.positions, sample.grippers, ik=ik)
            controller.reference[0] += .1
            moved = replace(sample, positions=tuple(controller.reference))
            action = Action(*(ArmTarget(controller.hold_poses[s]) for s in ("left", "right")))
            controller.step(moved, action)
            self.assertEqual([s for s, _ in ik.calls], ["left"])
            self.assertEqual(controller.planner.paths["left"].target, action.left.pose)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
