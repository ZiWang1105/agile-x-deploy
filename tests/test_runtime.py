"""Contract and closed-loop tests; no CAN device or robot SDK is imported."""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy import Action, ArmTarget, Pose, Robot
from agilex_deploy.backends import RealBackend, SimBackend
from agilex_deploy.controller import MAX_ACCELERATION, MAX_VELOCITY, pose_error
from agilex_deploy.interactive import make_display, read_drag, write_handles
from agilex_deploy.model import CONTROL_DT, Kinematics, build_sim_model


def target(observed):
    arms = []
    for side, y in (("left", -.02), ("right", .02)):
        pose = getattr(observed, f"{side}_tcp")
        q = (Rotation.from_rotvec([0, .18, .1]) * Rotation.from_quat(pose.quaternion_xyzw)).as_quat()
        arms.append(ArmTarget(Pose(np.asarray(pose.position_m) + [.06, y, .07], q), 1))
    return Action(*arms)


class LoopbackTransport:
    """Contract fixture only: hardware-style I/O over a separate simulated plant."""
    def __init__(self):
        self.plant = SimBackend()
        self.holds = 0

    def read_encoders(self):
        return self.plant.read()

    def send_positions(self, positions, grippers):
        self.plant.write(positions, grippers)
        self.plant.advance()

    def hold(self):
        self.holds += 1
        self.plant.hold()

    def close(self):
        self.plant.close()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_sim_model()

    def test_action_round_trip_and_validation(self):
        with Robot(SimBackend()) as r:
            a = target(r.observe())
        self.assertEqual(Action.from_dict(a.to_dict()), a)
        for bad in ([np.nan, 0, 0], [0, 0], [3, 0, 0]):
            with self.assertRaises(ValueError):
                Pose(bad, [0, 0, 0, 1])
        with self.assertRaises(ValueError):
            Pose([0, 0, 0], [0, 0, 0, 0])
        for invalid in (True, .5, 2, "1"):
            with self.assertRaises(ValueError):
                ArmTarget(a.left.pose, invalid)
        with self.assertRaises(ValueError):
            Action.from_dict({"left": a.to_dict()["left"]})

    def test_both_arms_track_translation_rotation_and_limits(self):
        with Robot(SimBackend()) as r:
            a = target(r.observe())
            previous_velocity = np.zeros(14)
            for i in range(700):
                if i % 10 == 0:
                    r.submit_target(a, r.observe().observation_id)
                r.tick()
                v = r.controller.velocity.copy()
                self.assertLessEqual(np.max(abs(v)), MAX_VELOCITY + 1e-9)
                self.assertLessEqual(np.max(abs(v - previous_velocity)) / CONTROL_DT, MAX_ACCELERATION + 1e-8)
                previous_velocity = v
                self.assertTrue(np.all(r.controller.reference >= r.kinematics.limits[:, 0]))
                self.assertTrue(np.all(r.controller.reference <= r.kinematics.limits[:, 1]))
            o = r.observe()
            for side in ("left", "right"):
                e = pose_error(getattr(o, f"{side}_tcp"), getattr(a, side).pose)
                self.assertLess(np.linalg.norm(e[:3]), .001)
                self.assertLess(np.linalg.norm(e[3:]), .01)
            np.testing.assert_allclose(o.gripper_position, [1, 1], atol=.001)

    def test_motion_uses_actuators_not_qpos_assignment(self):
        with Robot(SimBackend()) as r:
            before = np.array(r.observe().joint_positions_rad)
            r.submit_target(target(r.observe()), r.observe().observation_id)
            with patch.object(r.backend, "advance"):
                r.tick()
            np.testing.assert_array_equal(r.observe().joint_positions_rad, before)
            self.assertGreater(np.max(abs(r.controller.reference - before)), 0)

    def test_real_adapter_and_sim_share_observation_and_controller(self):
        transport = LoopbackTransport()
        with Robot(SimBackend()) as sim, Robot(RealBackend(transport)) as real:
            a = target(sim.observe())
            for i in range(300):
                for r in (sim, real):
                    if i % 10 == 0:
                        r.submit_target(a, r.observe().observation_id)
                    r.tick()
                np.testing.assert_allclose(sim.observe().joint_positions_rad,
                                           real.observe().joint_positions_rad, atol=1e-12)
                self.assertEqual(set(sim.observe().to_dict()), set(real.observe().to_dict()))
                self.assertEqual(sim.observe().cameras, ())
            real.hold()
            self.assertGreater(transport.holds, 0)

    def test_hold_stops_goal_and_resumes_from_measurement(self):
        with Robot(SimBackend()) as r:
            a = target(r.observe())
            r.submit_target(a, r.observe().observation_id)
            for _ in range(40):
                r.tick()
            r.hold()
            stop = np.array(r.observe().joint_positions_rad)
            for _ in range(100):
                r.tick()
            self.assertEqual(r.observe().mode, "hold")
            self.assertLess(max(abs(np.array(r.observe().joint_positions_rad) - stop)), .003)
            self.assertLess(max(abs(np.array(r.observe().joint_velocities_rad_s))), .001)
            r.submit_target(a, r.observe().observation_id)
            r.tick()
            self.assertEqual(r.observe().mode, "tracking")

    def test_target_timeout_holds(self):
        with Robot(SimBackend()) as r:
            r.submit_target(target(r.observe()), r.observe().observation_id)
            r.target_deadline = time.monotonic() - 1
            r.tick()
            self.assertEqual(r.observe().mode, "hold")
            self.assertEqual(r.observe().reason, "target lease expired")

    def test_old_observation_rejected(self):
        with Robot(SimBackend()) as r:
            o = r.observe()
            r.observations[o.observation_id] = replace(o, timestamp_s=time.monotonic() - 1)
            with self.assertRaisesRegex(ValueError, "stale"):
                r.submit_target(target(o), o.observation_id)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                r.submit_target(target(o), -1)

    def test_stale_hardware_feedback_latches_fault_and_calls_hold(self):
        transport = LoopbackTransport()
        with Robot(RealBackend(transport)) as r:
            frozen = replace(transport.read_encoders(), timestamp_s=time.monotonic() - 1)
            with patch.object(transport, "read_encoders", return_value=frozen):
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    r.tick()
            self.assertEqual(r.observe().mode, "fault")
            self.assertGreater(transport.holds, 0)
            r.hold()
            self.assertEqual(r.observe().mode, "fault")
            with self.assertRaisesRegex(RuntimeError, "Latched fault"):
                r.submit_target(target(r.observe()), r.observe().observation_id)

    def test_invalid_joint_order_is_not_silently_remapped(self):
        transport = LoopbackTransport()
        with Robot(RealBackend(transport)) as r:
            wrong = replace(transport.read_encoders(), joint_names=tuple(reversed(r.observe().joint_names)))
            with patch.object(transport, "read_encoders", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "names/order"):
                    r.tick()
            self.assertEqual(r.observe().mode, "fault")

    def test_unreachable_goal_stays_finite_and_stall_holds(self):
        with Robot(SimBackend()) as r:
            o = r.observe()
            a = Action(*(ArmTarget(Pose([1.4, y, .6], getattr(o, s + "_tcp").quaternion_xyzw), 1)
                         for s, y in (("left", .4), ("right", -.4))))
            for i in range(700):
                if i % 10 == 0:
                    r.submit_target(a, r.observe().observation_id)
                r.tick()
            self.assertTrue(np.all(np.isfinite(r.observe().joint_positions_rad)))
            r.best_error = 0.
            r.last_progress = time.monotonic() - 4
            r.tick()
            self.assertEqual(r.observe().reason, "unreachable or stalled target")
            self.assertEqual(r.observe().mode, "hold")

    def test_gripper_coupling_and_camera_lock(self):
        with Robot(SimBackend()) as r:
            with self.assertRaises(KeyError):
                r.backend.model.joint("top_camera_stand_joint")
            o = r.observe()
            a = Action(ArmTarget(o.left_tcp, 1), ArmTarget(o.right_tcp, 0))
            for i in range(300):
                if i % 10 == 0:
                    r.submit_target(a, r.observe().observation_id)
                r.tick()
            np.testing.assert_allclose(r.observe().gripper_position, [1, 0], atol=.001)
            for side in ("left", "right"):
                data = r.backend.data
                q1 = data.joint(side + "_gripper_joint1").qpos[0]
                q2 = data.joint(side + "_gripper_joint2").qpos[0]
                self.assertLess(abs(q1 + q2), 1e-5)

    def test_drag_only_accepts_selected_handle_and_reset_discards_input(self):
        with Robot(SimBackend()) as r:
            a = r.observe().as_action()
        kin = make_display()
        write_handles(kin, a)
        left = kin.model.body("left_target")
        kin.data.mocap_pos[left.mocapid[0], 0] += .03
        result = read_drag(kin, a, left.id, True)
        self.assertAlmostEqual(result.left.pose.position_m[0], a.left.pose.position_m[0] + .03)
        self.assertEqual(result.right, a.right)
        self.assertEqual(read_drag(kin, a, left.id, True, reset=True), a)
        self.assertEqual(read_drag(kin, a, kin.model.body("left_gripper_base").id, True), a)
        self.assertEqual(read_drag(kin, a, left.id, False), a)

    def test_jacobian_matches_finite_difference(self):
        with Robot(SimBackend()) as r:
            q = np.asarray(r.observe().joint_positions_rad)
            kin = Kinematics()
            kin.update(q)
            for side in ("left", "right"):
                initial = kin.pose(side)
                analytic = kin.jacobian(side)
                numerical = np.zeros((6, 14))
                for j in range(14):
                    delta = q.copy()
                    delta[j] += 1e-6
                    kin.update(delta)
                    numerical[:, j] = pose_error(initial, kin.pose(side)) / 1e-6
                kin.update(q)
                np.testing.assert_allclose(analytic, numerical, atol=5e-7)

    def test_owned_control_thread_closes(self):
        with Robot(SimBackend()).start() as r:
            time.sleep(.05)
            self.assertTrue(r.thread.is_alive())
        self.assertFalse(r.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            r.observe()


if __name__ == "__main__":
    unittest.main()
