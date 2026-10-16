"""Official solver provenance, explicit local tuning and mounting-frame parity."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from agilex_deploy.contract import Pose, SIDES
from agilex_deploy.model import Kinematics, initial_positions
from agilex_deploy.official_config import SOURCE, TUNING_PATH, official_parameters, solver_parameters
from agilex_deploy.official_ik import OfficialDualIK, make_solver, official_class, pose_matrix


class OfficialIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = Kinematics()
        cls.seed = initial_positions()
        cls.ik = OfficialDualIK(cls.kin, cls.seed)

    def test_pinned_upstream_class_and_nero_parameters_are_used(self):
        parameters = official_parameters()
        expected = dict(w_pos=2., w_ori=2., w_reg=.01, w_smooth=.1,
                        ipopt_max_iter=50, ipopt_tol=1e-4,
                        tool_pre_rot_rpy=[0., 0., 0.],
                        tool_translation_xyz=[.1755, 0., -.0235])
        for key, value in expected.items():
            self.assertEqual(parameters[key], value)
        manifest = json.loads((SOURCE / 'manifest.json').read_text())
        self.assertEqual(manifest['questarmteleop_commit'], '145b80360cf2c5f2e63179086d0aab2cb563826c')
        self.assertEqual(official_class().solve.__code__.co_filename,
                         str(SOURCE / 'arm_ik_pose_node.py'))
        for solver in self.ik.solvers.values():
            self.assertEqual(solver.nq, 7)
            self.assertEqual(solver.active_joint_names(), parameters['output_joint_names'])

    def test_both_stand_transforms_and_official_tcp_match_mujoco_fk(self):
        rng = np.random.default_rng(24)
        limits = self.kin.limits
        for q in [self.seed, *rng.uniform(limits[:, 0], limits[:, 1], (12, 14))]:
            self.kin.update(q, (None, None))
            for index, side in enumerate(SIDES):
                solver = self.ik.solvers[side]
                model = solver.reduced_robot.model
                data = model.createData()
                pin.framesForwardKinematics(model, data, q[index*7:(index+1)*7])
                predicted = self.ik.mounts[side] @ data.oMf[solver.ee_id].homogeneous
                np.testing.assert_allclose(predicted, pose_matrix(self.kin.pose(side)), atol=1e-10)

    def test_local_tuning_changes_only_pose_weights(self):
        upstream, active = official_parameters(), solver_parameters()
        self.assertEqual({k for k in active if active[k] != upstream[k]}, {"w_pos", "w_ori"})
        self.assertEqual(active["w_pos"], 40.)
        self.assertEqual(active["w_ori"], 4.)
        self.assertEqual(active["w_reg"], .01)
        self.assertEqual(upstream["w_reg"], .01)
        active["locked_joints"].append("joint1")
        self.assertNotIn("joint1", official_parameters()["locked_joints"])

    def test_tuning_rejects_invalid_values_and_geometry_overrides(self):
        invalid = ["w_reg: -1", "w_reg: .nan", "w_pos: 0", "w_smooth: true",
                   "ipopt_max_iter: 1.5", "tool_translation_xyz: [0, 0, 0]"]
        for entry in invalid:
            with self.subTest(entry=entry), patch.object(type(TUNING_PATH), "read_text", return_value="overrides:\n  " + entry):
                with self.assertRaises(ValueError):
                    solver_parameters()

    def test_small_reachable_targets_converge_with_tuned_pose_weights(self):
        self.kin.update(self.seed, (None, None))
        for index, side in enumerate(SIDES):
            pose = self.kin.pose(side)
            target = Pose(np.asarray(pose.position_m) + [0, 0, .005], pose.quaternion_xyzw)
            q = self.seed[index*7:(index+1)*7].copy()
            for _ in range(100):
                q = self.ik.solve(side, target, q)
            solver = self.ik.solvers[side]
            model = solver.reduced_robot.model
            data = model.createData()
            pin.framesForwardKinematics(model, data, q)
            actual = self.ik.mounts[side] @ data.oMf[solver.ee_id].homogeneous
            self.assertLess(np.linalg.norm(actual[:3, 3] - target.position_m), .0001)
            self.assertEqual(self.ik.last_status[side], "Solve_Succeeded")

    def test_adapter_matches_direct_upstream_solve_with_same_feedback(self):
        reference = make_solver()
        self.kin.update(self.seed, (None, None))
        for index, side in enumerate(SIDES):
            pose = self.kin.pose(side)
            target = Pose(np.asarray(pose.position_m) + [.05, .02, .04],
                          (Rotation.from_rotvec([.04, -.05, .06]) *
                           Rotation.from_quat(pose.quaternion_xyzw)).as_quat())
            q = self.seed[index*7:(index+1)*7]
            reference.sync_state(q.tolist())
            expected = reference.solve(np.linalg.solve(self.ik.mounts[side], pose_matrix(target)))
            actual = self.ik.solve(side, target, q)
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            # The official objective allows residual pose error. Check that
            # it improves that same objective, not a made-up exact-pose bound.
            opti = reference.opti
            target_objective = float(opti.value(opti.f))
            seed_objective = float(opti.debug.value(opti.f, [reference.var_q == q]))
            self.assertLess(target_objective, seed_objective)

    def test_unsuccessful_iterate_is_not_dispatched(self):
        self.kin.update(self.seed, (None, None))
        solver = self.ik.solvers['left']
        with patch.object(solver, 'solve', return_value=self.seed[:7]), \
             patch.object(solver.opti, 'stats', return_value={'success': False, 'return_status': 'Maximum_Iterations_Exceeded'}):
            with self.assertRaisesRegex(RuntimeError, 'Maximum_Iterations_Exceeded'):
                self.ik.solve('left', self.kin.pose('left'), self.seed[:7])


if __name__ == '__main__':
    unittest.main()
