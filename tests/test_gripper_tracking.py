"""Feedback outcomes use measured samples, never command echoes or contact truth."""

import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from agilex_deploy.backends import RealBackend, SimBackend
from agilex_deploy.contract import Action, ArmTarget
from agilex_deploy.gripper import GripperFeedback
from agilex_deploy.gripper_tracking import GripperTracker, TrackingSettings
from agilex_deploy.hardware_session import SimulatedCommissioningTransport
from agilex_deploy.runtime import Robot


class TrackingTests(unittest.TestCase):
    def test_reached_requires_new_measured_endpoint_samples_and_dwell(self):
        tracker = GripperTracker()
        tracker.command(1, 10.0)
        self.assertEqual(tracker.observe(0.4, 10.0, 10.0).phase, "opening")
        self.assertEqual(tracker.observe(1.0, 10.01, 10.01).phase, "opening")
        # Polling the same sample cannot satisfy dwell.
        self.assertEqual(tracker.observe(1.0, 10.01, 10.15).phase, "opening")
        self.assertEqual(tracker.observe(1.0, 10.12, 10.12).phase, "reached")
        # An external displacement invalidates the reached state.
        self.assertEqual(tracker.observe(0.8, 10.13, 10.13).phase, "opening")

    def test_stall_is_not_grasp_success_and_repeated_targets_preserve_window(self):
        tracker = GripperTracker()
        tracker.command(0, 10.0)
        for i in range(1, 13):
            stamp = 10 + i * 0.05
            tracker.command(0, stamp)
            feedback = GripperFeedback(0.04, -0.8, 0x40, stamp)
            state = tracker.observe(0.4, stamp, stamp, feedback)
        self.assertEqual(state.phase, "stalled")
        self.assertEqual(state.force_n, -0.8)
        self.assertEqual(state.width_m, 0.04)
        self.assertTrue(state.enabled)
        self.assertAlmostEqual(state.position_error, -0.4)
        self.assertEqual(tracker.observe(0.3, 10.65, 10.65).phase, "closing")
        tracker.command(1, 10.66)
        self.assertEqual(tracker.observe(0.3, 10.7, 10.7).phase, "opening")

    def test_none_preserves_request_hold_cancels_it(self):
        tracker = GripperTracker()
        tracker.command(0, 10.0)
        tracker.command(None, 10.1)
        self.assertEqual(tracker.observe(0.4, 10.2, 10.2).requested_open, 0)
        tracker.hold()
        state = tracker.observe(0.4, 10.3, 10.3)
        self.assertEqual(state.phase, "hold")
        self.assertIsNone(state.requested_open)
        self.assertIsNone(state.position_error)

    def test_disabled_missing_stale_fault_and_acquisition_gaps(self):
        tracker = GripperTracker()
        tracker.command(1, 10.0)
        self.assertEqual(tracker.observe(None, 10.0, 10.0).phase, "unavailable")
        disabled = GripperFeedback(0.1, 0, 0, 10.01)
        self.assertEqual(tracker.observe(1.0, 10.01, 10.01, disabled).phase, "disabled")
        self.assertEqual(tracker.observe(1.0, 10.02, 10.02).phase, "opening")
        self.assertEqual(tracker.observe(1.0, 10.02, 10.3).phase, "stale")
        self.assertEqual(tracker.observe(1.0, 10.4, 10.4).phase, "opening")
        self.assertEqual(tracker.observe(1.0, 10.51, 10.51).phase, "reached")
        self.assertEqual(tracker.observe(1.0, 10.5, 10.52).phase, "fault")
        self.assertEqual(tracker.observe(1.0, 10.6, 10.6, fault=True).phase, "fault")

    def test_oscillation_is_not_stationary_and_thresholds_are_configurable(self):
        tracker = GripperTracker(TrackingSettings(stationary_window_s=0.2))
        tracker.command(0, 10.0)
        for i in range(1, 20):
            stamp = 10 + i * 0.02
            state = tracker.observe(0.4 + (i % 2) * 0.01, stamp, stamp)
        self.assertEqual(state.phase, "closing")
        self.assertIsNone(
            state.force_n
        )  # No synthesized force for a width-only sensor.
        for bad in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                TrackingSettings(endpoint_dwell_s=bad)

    def test_raw_feedback_validation_and_simulation_unavailable_force(self):
        plant = SimBackend()
        try:
            sample = plant.read()
            self.assertEqual(sample.gripper_feedback, (None, None))
            for field in ("force_n", "width_m"):
                bad = replace(
                    GripperFeedback(0.04, 0.5, 0x40, time.monotonic()),
                    **{field: float("nan")},
                )
                with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                    replace(sample, gripper_feedback=(bad, None)).validate()
        finally:
            plant.close()

    def test_sim_and_transport_share_outcomes_and_hold_without_new_limits(self):
        # Deterministic host acquisition clock, no robot/device connection.
        clock = [time.monotonic()]
        transport = SimulatedCommissioningTransport("left", gripper_control=True)
        with (
            patch("time.monotonic", side_effect=lambda: clock[0]),
            Robot(RealBackend(transport)) as robot,
        ):
            transport.arm()
            robot.hold()
            robot.controller.gripper_rate = None
            first = robot.observe()
            target = Action(
                ArmTarget(first.left_tcp, 1), ArmTarget(first.right_tcp, None)
            )
            robot.submit_target(target, first.observation_id)
            transport.activate_gripper(target)
            for i in range(160):
                clock[0] += 0.01
                if i % 10 == 0:
                    robot.submit_target(target, robot.observe().observation_id)
                robot.tick()
            observation = robot.observe()
            self.assertEqual(observation.gripper_states[0].phase, "reached")
            self.assertEqual(observation.gripper_states[1].phase, "unavailable")
            self.assertIsNone(observation.gripper_states[0].force_n)
            self.assertEqual(
                observation.to_dict()["gripper_states"][0]["requested_open"], 1
            )
            robot.hold()
            self.assertEqual(robot.observe().gripper_states[0].phase, "hold")
            self.assertIsNone(robot.observe().gripper_states[0].requested_open)


if __name__ == "__main__":
    unittest.main()
