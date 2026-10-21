"""Alternate rotation geometry and recorded-target IK validation, offline only."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy.agent_policy import tcp_pose
from agilex_deploy.backends import SimBackend
from agilex_deploy.cartesian_path import CartesianPath, CartesianPlanner, PathResolution
from agilex_deploy.cartesian_validation import (
    CartesianPathUnavailable, CartesianPlanningCancelled, plan_cartesian_path,
)
from agilex_deploy.contract import ArmTarget, MixedAction, Pose
from agilex_deploy.controller import Controller, pose_error
from agilex_deploy.model import CONTROL_DT, Kinematics
from agilex_deploy.official_ik import OfficialDualIK
from agilex_deploy.runtime import Robot


RECORDED_START = np.array([
    -.04974188368183839, 1.4463892577127409, -1.1795807749603677,
    .5243318138841365, -.4192455396215579, -.0328121899374934, .967104391822578,
    -.029426251188624394, 1.5726987356795705, 1.498592055639891,
    .37573448136933923, -.10314895879286488, -.10805333399096893, 1.4307860141999116,
])
RECORDED_TARGET = tcp_pose(Pose(
    [.32608520132298074, .1330746025756452, .15244532313340578], [.5, -.5, -.5, -.5]))


class RotationGeometryTests(unittest.TestCase):
    def test_alternate_arc_reverses_direction_and_preserves_endpoint(self):
        start = Pose([.2, 0, .4], [0, 0, 0, 1])
        end = Pose([.4, .1, .3], Rotation.from_rotvec([0, 0, 3.1]).as_quat())
        short = CartesianPath(start, end)
        alternate = CartesianPath(start, end, rotation_arc="alternate")
        self.assertAlmostEqual(short.angle_rad + alternate.angle_rad, 2 * np.pi)
        self.assertLess(alternate.rotation[2], 0)
        self.assertEqual(alternate.waypoints[-1], end)
        self.assertEqual(alternate.waypoints[0], start)
        for i, waypoint in enumerate(alternate.waypoints):
            fraction = i / (len(alternate.waypoints) - 1)
            np.testing.assert_allclose(waypoint.position_m,
                                       np.asarray(start.position_m) + fraction * alternate.translation)
        for a, b in zip(alternate.waypoints, alternate.waypoints[1:]):
            delta = Rotation.from_quat(b.quaternion_xyzw) * Rotation.from_quat(a.quaternion_xyzw).inv()
            self.assertLessEqual(delta.magnitude(), .05 + 1e-12)
            self.assertLess(delta.as_rotvec()[2], 0)

    def test_antipodal_equal_orientation_has_no_invented_full_turn(self):
        start = Pose([.2, 0, .4], [0, 0, 0, 1])
        end = Pose([.2, 0, .4], [0, 0, 0, -1])
        alternate = CartesianPath(start, end, rotation_arc="alternate")
        self.assertEqual(alternate.angle_rad, 0)
        self.assertEqual(len(alternate.waypoints), 2)
        self.assertEqual(alternate.waypoints[-1], end)

    def test_arc_changes_rebase_the_planner_without_changing_goal(self):
        planner = CartesianPlanner()
        start = Pose([0, 0, .3], [0, 0, 0, 1])
        end = Pose([.1, 0, .3], Rotation.from_rotvec([0, 0, 2]).as_quat())
        planner.target("left", start, end)
        old = planner.paths["left"]
        planner.target("left", start, end, rotation_arc="alternate")
        self.assertIsNot(old, planner.paths["left"])
        self.assertEqual(planner.status()["left"]["rotation_arc"], "alternate")
        self.assertEqual(planner.status()["left"]["revision"], 2)
        with self.assertRaises(ValueError):
            CartesianPath(start, end, rotation_arc="unknown")

    def test_large_lookahead_cannot_skip_the_alternate_winding(self):
        start = Pose([0, 0, .3], [0, 0, 0, 1])
        end = Pose([0, 0, .3], Rotation.from_euler("z", 30, degrees=True).as_quat())
        path = CartesianPath(start, end, rotation_arc="alternate")
        waypoint = path.advance(start, angular_lookahead_rad=100)
        self.assertLess(path.index, len(path.waypoints) - 1)
        self.assertGreater(path.index, 1)
        self.assertNotEqual(waypoint, end)
        quarter_turn = Rotation.from_euler("z", -90, degrees=True)
        measured = Pose(start.position_m, quarter_turn.as_quat())
        previous = path.index
        path.advance(measured, angular_lookahead_rad=100)
        self.assertGreater(path.index, previous)
        self.assertLess(path.index, len(path.waypoints) - 1)

    def test_coarse_resolution_cannot_remove_the_alternate_winding(self):
        start = Pose([0, 0, .3], [0, 0, 0, 1])
        end = Pose([0, 0, .3], Rotation.from_euler("z", 30, degrees=True).as_quat())
        path = CartesianPath(start, end, PathResolution(rotation_rad=10), rotation_arc="alternate")
        self.assertGreater(len(path.waypoints), 3)
        self.assertNotEqual(path.advance(start), end)

    def test_frozen_start_keeps_validated_direction_across_half_turn_noise(self):
        start = Pose([0, 0, .3], [0, 0, 0, 1])
        end = Pose([0, 0, .3], Rotation.from_euler("z", 179.9, degrees=True).as_quat())
        measured = Pose(start.position_m, Rotation.from_euler("z", -.2, degrees=True).as_quat())
        planner = CartesianPlanner()
        planner.target("left", measured, end, rotation_arc="alternate", validated_start=start)
        frozen = planner.paths["left"]
        self.assertEqual(frozen.start, start)
        self.assertLess(frozen.rotation[2], 0)
        planner.target("left", start, end, rotation_arc="alternate", validated_start=start)
        self.assertIs(planner.paths["left"], frozen)
        planner.target("left", measured, end, rotation_arc="alternate")
        self.assertIsNot(planner.paths["left"], frozen)
        self.assertGreater(planner.paths["left"].rotation[2], 0)

    def test_mixed_action_validates_arc_names_and_keeps_defaults(self):
        arm = ArmTarget(Pose([0, 0, .3], [0, 0, 0, 1]))
        action = MixedAction(arm, arm, ("left",), (0.,) * 14)
        self.assertEqual(action.rotation_arcs, ("shortest", "shortest"))
        self.assertEqual(replace(action, rotation_arcs=["alternate", "shortest"]).rotation_arcs,
                         ("alternate", "shortest"))
        for arcs in [("alternate",), ("alternate", "bad"), ("shortest",) * 3]:
            with self.assertRaises(ValueError):
                replace(action, rotation_arcs=arcs)
        for starts in [(None,), ("bad", None), (None,) * 3]:
            with self.assertRaises(ValueError):
                replace(action, cartesian_starts=starts)

    def test_observation_publishes_current_target_and_path_progress(self):
        class StationaryIK:
            def solve(self, side, target, seed):
                return seed

        with Robot(SimBackend(), ik=StationaryIK()) as robot:
            observed = robot.observe()
            self.assertEqual(observed.motion_paths, {})
            start = observed.left_tcp
            target = Pose(np.asarray(start.position_m) + [.1, 0, 0], start.quaternion_xyzw)
            action = MixedAction(ArmTarget(target), ArmTarget(observed.right_tcp), ("left",),
                                 observed.joint_positions_rad, cartesian_starts=(start, None))
            robot.submit_target(action, observed.observation_id)
            robot.tick()
            row = robot.observe().motion_paths["left"]
            self.assertLess(row["waypoint"], row["segments"])
            self.assertEqual(row["target"], dict(position_m=list(target.position_m),
                                                quaternion_xyzw=list(target.quaternion_xyzw)))
            self.assertEqual(robot.controller.planner.paths["left"].start, start)
            robot.hold()
            self.assertEqual(robot.observe().motion_paths, {})

    def test_runtime_accepts_slow_alternate_progress_but_still_holds_a_stall(self):
        class StationaryIK:
            def solve(self, side, target, seed):
                return seed

        # Isolate the runtime watchdog from dynamics: controller feedback
        # advances one genuine geometric waypoint every half-second while its
        # endpoint error increases. Then freeze progress and require HOLD.
        with Robot(SimBackend(), ik=StationaryIK()) as robot:
            observed = robot.observe()
            start = observed.left_tcp
            target = Pose(start.position_m,
                          (Rotation.from_rotvec([0, 0, .3])
                           * Rotation.from_quat(start.quaternion_xyzw)).as_quat())
            action = MixedAction(ArmTarget(target), ArmTarget(observed.right_tcp), ("left",),
                                 observed.joint_positions_rad, ("alternate", "shortest"))
            path = CartesianPath(start, target, rotation_arc="alternate")
            robot.controller.planner.paths["left"] = path
            calls = [0]
            clock = [time.monotonic()]

            def progressing_step(sample, requested):
                calls[0] += 1
                if calls[0] <= 50 and calls[0] % 5 == 0:
                    path.advance(path.waypoints[path.index])
                return (np.asarray(sample.positions), list(sample.grippers),
                        {"left": (.1 + .001 * calls[0], .1), "right": (0., 0.)})

            with patch("agilex_deploy.runtime.time.monotonic", side_effect=lambda: clock[0]), \
                    patch.object(robot.controller, "step", side_effect=progressing_step):
                for _ in range(50):
                    clock[0] += .1
                    robot.submit_target(action, robot.observe().observation_id)
                    robot.tick()
                    self.assertEqual(robot.mode, "tracking")
                self.assertGreater(path.index, 1)
                for _ in range(32):
                    clock[0] += .1
                    robot.submit_target(action, robot.observe().observation_id)
                    robot.tick()
                    if robot.mode == "hold":
                        break
                self.assertEqual(robot.mode, "hold")
                self.assertEqual(robot.reason, "unreachable or stalled target")
                with self.assertRaisesRegex(RuntimeError, "PATH_STALLED"):
                    robot.submit_target(action, robot.observe().observation_id)
                self.assertIsNone(robot.target)
                self.assertEqual(robot.mode, "hold")
                robot.hold()
                robot.submit_target(action, robot.observe().observation_id)
                self.assertEqual(robot.mode, "tracking")


class SimpleKinematics:
    limits = np.tile([-10., 10.], (14, 1))

    def update(self, q):
        self.q = np.array(q)

    def pose(self, side):
        i = 0 if side == "left" else 7
        return Pose([self.q[i], 0, 0], Rotation.from_rotvec([0, 0, self.q[i + 1]]).as_quat())


class SimpleIK:
    def __init__(self):
        self.calls = []

    def solve(self, side, target, seed):
        self.calls.append((side, target, seed.copy()))
        q = seed.copy()
        q[0] = target.position_m[0]
        q[1] = Rotation.from_quat(target.quaternion_xyzw).as_rotvec()[2]
        return q


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.kin = SimpleKinematics()
        self.ik = SimpleIK()
        self.target = Pose([.03, 0, 0], [0, 0, 0, 1])

    def test_prefers_shortest_and_seeds_each_solve_from_previous_solution(self):
        arc, report = plan_cartesian_path(self.kin, self.ik, "left", self.target, np.zeros(14))
        self.assertEqual(arc, "shortest")
        self.assertEqual(len(report["candidates"]), 1)
        self.assertTrue(report["candidates"][0]["valid"])
        self.assertEqual(len(self.ik.calls), 4)
        self.assertAlmostEqual(self.ik.calls[3][2][0], .02)
        self.assertFalse(report["collision_checked"])

    def test_nominal_solver_success_does_not_hide_pose_error(self):
        class StalledIK:
            def solve(self, side, target, seed):
                return seed

        with self.assertRaises(CartesianPathUnavailable) as raised:
            plan_cartesian_path(self.kin, StalledIK(), "left", self.target, np.zeros(14))
        candidates = raised.exception.report["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all("pose residual" in row["failure"] for row in candidates))
        self.assertTrue(all(not row["valid"] for row in candidates))

    def test_supplied_position_limits_are_checked_without_clipping(self):
        limits = np.tile([-.005, .005], (7, 1))
        with self.assertRaises(CartesianPathUnavailable) as raised:
            plan_cartesian_path(self.kin, self.ik, "left", self.target, np.zeros(14),
                                position_limits=limits)
        self.assertTrue(all("position limits" in row["failure"]
                            for row in raised.exception.report["candidates"]))
        self.assertEqual(raised.exception.report["position_limits_rad"], limits.tolist())

    def test_cancellation_after_solver_stops_before_alternate_candidate(self):
        with self.assertRaises(CartesianPlanningCancelled):
            plan_cartesian_path(self.kin, self.ik, "left", self.target, np.zeros(14),
                                cancelled=lambda: len(self.ik.calls) >= 1)
        self.assertEqual(len(self.ik.calls), 1)

    def test_out_of_bounds_measured_start_rejects_before_solving(self):
        q = np.zeros(14)
        q[0] = .02
        with self.assertRaisesRegex(ValueError, "Planning start"):
            plan_cartesian_path(self.kin, self.ik, "left", self.target, q,
                                position_limits=np.tile([-.01, .01], (14, 1)))
        self.assertEqual(self.ik.calls, [])

    def test_solver_failures_reject_both_arcs_with_diagnostics(self):
        class FailedIK:
            def solve(self, *_):
                raise RuntimeError("solver failed")

        with self.assertRaisesRegex(CartesianPathUnavailable, "solver failed") as raised:
            plan_cartesian_path(self.kin, FailedIK(), "left", self.target, np.zeros(14))
        self.assertEqual(len(raised.exception.report["candidates"]), 2)


class RecordedTargetTests(unittest.TestCase):
    def test_preflight_selects_arc_that_shared_controller_can_complete(self):
        kin = Kinematics()
        ik = OfficialDualIK(kin, RECORDED_START)
        arc, report = plan_cartesian_path(kin, ik, "left", RECORDED_TARGET, RECORDED_START)
        self.assertEqual(arc, "alternate")
        shortest, alternate = report["candidates"]
        self.assertFalse(shortest["valid"])
        self.assertGreater(shortest["orientation_error_rad"], .05)
        self.assertTrue(alternate["valid"])
        self.assertEqual(alternate["samples_checked"], 65)
        self.assertLess(alternate["max_position_error_m"], .0002)
        self.assertLess(alternate["max_orientation_error_rad"], .004)
        with_backend = SimBackend(RECORDED_START)
        try:
            sample = with_backend.read()
            controller = Controller(with_backend.kinematics, sample.positions, sample.grippers)
            action = MixedAction(ArmTarget(RECORDED_TARGET), ArmTarget(controller.hold_poses["right"]),
                                 ("left",), tuple(RECORDED_START), (arc, "shortest"),
                                 (Pose(**report["start_pose"]), None))
            previous_velocity = controller.velocity.copy()
            good = []
            for _ in range(400):
                reference, grippers, _ = controller.step(with_backend.read(), action)
                self.assertTrue(np.all(np.abs(controller.velocity) <= controller.max_velocity + 1e-10))
                self.assertTrue(np.all(np.abs(controller.velocity - previous_velocity)
                                       <= controller.max_acceleration * CONTROL_DT + 1e-10))
                previous_velocity = controller.velocity.copy()
                with_backend.write(reference, grippers)
                with_backend.advance()
                measured = with_backend.read()
                kin.update(measured.positions)
                residual = pose_error(kin.pose("left"), RECORDED_TARGET)
                good.append(np.linalg.norm(residual[:3]) < .005 and np.linalg.norm(residual[3:]) < .05)
            self.assertTrue(all(good[-100:]))
            self.assertEqual(controller.planner.status()["left"]["rotation_arc"], "alternate")
            np.testing.assert_allclose(measured.positions[7:], RECORDED_START[7:], atol=1e-12)
        finally:
            with_backend.close()


if __name__ == "__main__":
    unittest.main()
