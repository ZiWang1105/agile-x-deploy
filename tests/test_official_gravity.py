"""Numerical torque adapter checks. No physical I/O is permitted."""
import socket
import unittest
from unittest.mock import patch

import numpy as np

from agilex_deploy.official_gravity import OfficialGravityPreview
from agilex_deploy.model import initial_positions


class OfficialGravityTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket, "socket", side_effect=AssertionError("No physical I/O"))
        guard.start()
        self.addCleanup(guard.stop)
        self.preview = OfficialGravityPreview()
        self.q = initial_positions()

    def test_mounts_rotate_gravity_and_exclude_absent_cameras(self):
        for side in ("left", "right"):
            gravity = self.preview.rotations[side].T @ np.array([0, 0, -9.81])
            self.assertLess(gravity[0], -9.8)
            self.assertLess(abs(gravity[2]), .001)
            self.assertNotIn("camera", self.preview.descriptions[side])
            self.assertIn("gripper_base", self.preview.descriptions[side])

    def test_sides_do_not_share_mutable_dynamics_state(self):
        p = self.preview
        before = p.torques("right", self.q[7:], np.zeros(7), gripper_opening_m=.04)
        p.torques("left", self.q[:7], np.ones(7)*.1, gripper_opening_m=.08)
        after = p.torques("right", self.q[7:], np.zeros(7), gripper_opening_m=.04)
        np.testing.assert_array_equal(before, after)
        for side in ("left", "right"):
            np.testing.assert_allclose(p.arms[side].robot.model.gravity.linear, [0, 0, -9.81])

    def test_rejects_invalid_positions_and_gripper_width(self):
        p = self.preview
        for q, width in (([float("nan")]*7, .04), (self.q[:6], .04),
                         ([10]*7, .04), (self.q[:7], -.1), (self.q[:7], float("nan"))):
            with self.assertRaises(ValueError):
                p.torques("left", q, np.zeros(7), gripper_opening_m=width)

    def test_gripper_positions_use_named_coordinates(self):
        p = self.preview
        q, v = p.model_state("left", self.q[:7], np.zeros(7), .08)
        model = p.arms["left"].robot.model
        for finger, expected in ((1, .04), (2, -.04)):
            joint = model.joints[model.getJointId(f"left_gripper_joint{finger}")]
            self.assertEqual(q[joint.idx_q], expected)
            self.assertEqual(v[joint.idx_v], 0)


if __name__ == "__main__":
    unittest.main()
