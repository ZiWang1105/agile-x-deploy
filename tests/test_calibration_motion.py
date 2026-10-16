"""Offline regression for calibration planning near the physical workspace edge."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from agilex_deploy.calibration_motion import plan_route, run_route, check_route_start
from agilex_deploy.contract import Action, JOINT_NAMES
from agilex_deploy.controller import pose_error
from agilex_deploy.model import Kinematics


class CalibrationMotionTests(unittest.TestCase):
    def test_changed_disabled_posture_reports_joint_without_relaxing_gate(self):
        planned = np.zeros(14)
        changed = planned.copy()
        changed[3] = np.radians(65.05)
        with self.assertRaisesRegex(ValueError, "left_joint4 differs by 65.05 degrees"):
            check_route_start(changed, planned)
        changed[3] = 0.051
        with self.assertRaisesRegex(ValueError, "current disabled-arm feedback"):
            check_route_start(changed, planned)
        changed[3] = 0.049
        check_route_start(changed, planned)

    def test_physical_seed_with_failed_nominal_offset_has_valid_alternatives(self):
        # Recorded disabled-arm encoders that failed the original fixed route.
        q = np.radians([
            -2.115, 90.691, -88.806, 17.212, -9.912, -2.953, 82.14,
            0.552, 84.544, 97.387, 21.964, 6.96, 19.944, 72.632,
        ])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            seed = directory / "seed.json"
            seed.write_text(json.dumps({
                "positions": q.tolist(),
                "joint_names": list(JOINT_NAMES),
                "source": "physical_CAN_receive_only",
            }))
            plan = plan_route(directory / "plan.json", seed=seed)
            self.assertEqual(len(plan["waypoints"]), 9)
            attempts = plan["waypoints"][1]["sampling_attempts"]
            self.assertFalse(attempts[0]["accepted"])
            self.assertTrue(attempts[-1]["accepted"])
            self.assertLess(attempts[-1]["translation_offset_m"][0], 0)
            kin = Kinematics()
            seen = []
            for row in plan["waypoints"]:
                actual = np.asarray(row["q_preview_rad"])
                np.testing.assert_array_equal(actual[7:], q[7:])
                self.assertTrue(np.all(actual >= kin.limits[:, 0]))
                self.assertTrue(np.all(actual <= kin.limits[:, 1]))
                kin.update(actual, (None, None))
                action = Action.from_dict(row["action"])
                error = pose_error(kin.pose("left"), action.left.pose)
                self.assertLess(np.linalg.norm(error[:3]), 0.002)
                self.assertLess(np.linalg.norm(error[3:]), np.radians(1))
                self.assertTrue(all(np.max(np.abs(actual - old)) >= 0.01 for old in seen))
                seen.append(actual)

    def test_missing_plan_reports_the_failed_prerequisite(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Run 'plan' successfully"):
                run_route(Path(directory) / "missing.json")

    def test_raised_route_keeps_approach_out_of_capture_set(self):
        q = np.radians([
            -3.072, 85.579, -44.993, 29.571, -38.865, 7.261, 63.03,
            0.554, 84.544, 97.387, 21.967, 6.96, 19.948, 72.632,
        ])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            seed = directory / "seed.json"
            seed.write_text(json.dumps({
                "positions": q.tolist(), "joint_names": list(JOINT_NAMES),
                "source": "physical_CAN_receive_only",
            }))
            plan = plan_route(directory / "raised.json", seed=seed,
                              lift_m=0.20, pose_count=15, rotation_deg=20)
            approach = [r for r in plan["waypoints"] if not r["capture"]]
            captures = [r for r in plan["waypoints"] if r["capture"]]
            self.assertEqual(len(approach), 4)
            self.assertEqual(len(captures), 15)
            kin = Kinematics()
            kin.update(q, (None, None))
            start = np.asarray(kin.pose("left").position_m)
            np.testing.assert_allclose(plan["capture_center_position_m"],
                                       start + [0, 0, 0.20])
            for index, row in enumerate(approach):
                target = Action.from_dict(row["action"]).left.pose
                np.testing.assert_allclose(target.position_m,
                                           start + [0, 0, index * 0.05])
            seen = []
            for row in plan["waypoints"]:
                actual = np.asarray(row["q_preview_rad"])
                np.testing.assert_array_equal(actual[7:], q[7:])
                kin.update(actual, (None, None))
                error = pose_error(kin.pose("left"),
                                   Action.from_dict(row["action"]).left.pose)
                self.assertLess(np.linalg.norm(error[:3]), 0.002)
                self.assertLess(np.linalg.norm(error[3:]), np.radians(1))
                if row["capture"]:
                    self.assertTrue(all(np.max(np.abs(actual - old)) >= 0.01
                                        for old in seen))
                    seen.append(actual)
