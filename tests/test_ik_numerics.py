"""Numerical-fault injection and incident-pose simulation; no physical CAN I/O."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from agilex_deploy.agent_policy import tcp_pose
from agilex_deploy.backends import SimBackend
from agilex_deploy.contract import Action, ArmTarget, Pose
from agilex_deploy.controller import pose_error
from agilex_deploy.model import Kinematics, initial_positions
from agilex_deploy.official_ik import OfficialDualIK
from agilex_deploy.runtime import Robot


class NumericalIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = initial_positions()
        cls.kin = Kinematics()
        cls.ik = OfficialDualIK(cls.kin, cls.q)

    def test_persistent_nan_holds_without_emitting_invalid_joint_commands(self):
        with Robot(SimBackend()) as robot, TemporaryDirectory() as directory:
            state = robot.observe()
            target = Pose(np.array(state.left_tcp.position_m)+[0, 0, .02], state.left_tcp.quaternion_xyzw)
            robot.submit_target(Action(ArmTarget(target), ArmTarget(state.right_tcp)), state.observation_id)
            solver = robot.controller.ik.solvers["left"]
            with patch("agilex_deploy.official_ik.ROOT", Path(directory)), \
                 patch.object(solver, "solve", side_effect=RuntimeError("Injected Invalid_Number_Detected")) as solve, \
                 patch.object(solver.opti, "stats", return_value=dict(success=False, return_status="Invalid_Number_Detected")), \
                 patch.object(robot.backend, "write", wraps=robot.backend.write) as write:
                with self.assertRaisesRegex(RuntimeError, "Official left IK failed.*diagnostic="):
                    robot.tick()
                self.assertEqual(solve.call_count, 1)
                self.assertEqual(robot.observe().mode, "fault")
                for call in write.call_args_list:
                    np.testing.assert_allclose(call.args[0], state.joint_positions_rad)
                    self.assertTrue(np.isfinite(call.args[0]).all())
            report = json.loads(next((Path(directory)/"logs/ik_failures").glob("*.json")).read_text())
            self.assertEqual(report["failure"]["status"], "Invalid_Number_Detected")

    def test_non_numerical_failure_is_not_retried(self):
        self.kin.update(self.q)
        solver = self.ik.solvers["left"]
        with TemporaryDirectory() as directory, patch("agilex_deploy.official_ik.ROOT", Path(directory)), \
             patch.object(solver, "solve", return_value=self.q[:7]) as solve, \
             patch.object(solver.opti, "stats", return_value=dict(success=False, return_status="Maximum_Iterations_Exceeded")):
            with self.assertRaisesRegex(RuntimeError, "Maximum_Iterations_Exceeded"):
                self.ik.solve("left", self.kin.pose("left"), self.q[:7])
            self.assertEqual(solve.call_count, 1)

    def test_recorded_start_and_target_in_mujoco(self):
        record = json.loads((Path(__file__).parent/"fixtures/ik_start_20261020.json").read_text())
        target = tcp_pose(Pose(**record["pinch_target"]))
        with Robot(SimBackend(joint_positions=record["joint_positions_rad"])) as robot:
            before = robot.observe()
            action = Action(ArmTarget(target), ArmTarget(before.right_tcp))
            for _ in range(700):
                robot.submit_target(action, robot.observe().observation_id)
                robot.tick()
            after = robot.observe()
            error = pose_error(after.left_tcp, target)
            self.assertLess(np.linalg.norm(error[:3]), .005)
            self.assertLess(np.linalg.norm(error[3:]), .05)
            self.assertLess(np.max(abs(np.asarray(after.joint_positions_rad)[7:]-np.asarray(before.joint_positions_rad)[7:])), .001)
            self.assertNotEqual(after.mode, "fault")
