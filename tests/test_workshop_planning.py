"""Public planner admission and cancellation regressions; simulation only."""

import asyncio
from contextlib import contextmanager, nullcontext
from concurrent.futures import Future
from dataclasses import asdict
import json
import pickle
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from agilex_deploy.contract import MixedAction
from agilex_deploy.cartesian_validation import CartesianPathUnavailable
from agilex_deploy.hardware_session import HardwareSession, control_worker
from agilex_deploy.model import Kinematics
from agilex_deploy.motion_limits import SIM_ACCELERATION, SIM_VELOCITY
from agilex_deploy.workshop_client import Operation, RobotClient
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_mcp import build_server
from agilex_deploy.workshop_service import WorkshopService
from agilex_deploy.workshop_tasks import WorkshopSim
from test_rotation_arc import RECORDED_START, RECORDED_TARGET


class RecordedWorkshopSim(WorkshopSim):
    """Recorded encoders with the real camera scene and no simulated contacts."""

    def __init__(self, task="place", seed=0):
        super().__init__(task, seed)
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.data.qpos[self.qids] = RECORDED_START
        self.data.qvel[self.dids] = 0
        self.write(RECORDED_START, [.36, .36])
        mujoco.mj_forward(self.model, self.data)


@contextmanager
def public_session(recorded=False):
    factory = (patch("agilex_deploy.workshop_tasks.WorkshopSim", RecordedWorkshopSim)
               if recorded else nullcontext())
    with tempfile.TemporaryDirectory() as directory, factory:
        service = WorkshopService(output=directory)
        server = WorkshopHTTP(service, 0)
        client = RobotClient(server.endpoint, server.token, "planning-test")
        try:
            client.claim()
            # Production MCP runs in its own interpreter. Build its schemas
            # before starting this fixture's simulated control thread.
            client._test_mcp = build_server(client)
            service.enable()
            yield service, client
        finally:
            try:
                client.close()
            finally:
                server.close()
                service.close()


@contextmanager
def paused_planning(service):
    """Delay an already-running future without changing the owned process pool."""
    entered = threading.Event()
    pending = []

    def submit(function, generation, step, joints, limits, pt, rt):
        future = Future()
        future.set_running_or_notify_cancel()
        kin = Kinematics()
        kin.update(joints)
        reports = {side: dict(selected_arc="shortest", start_pose=asdict(kin.pose(side)),
                              candidates=[dict(rotation_arc="shortest", valid=True)],
                              collision_checked=False) for side in step["poses"]}
        pending.append((future, reports))
        entered.set()
        return future

    def release():
        for future, reports in pending:
            if not future.done():
                future.set_result(reports)

    with patch.object(service.planner_pool, "submit", side_effect=submit):
        try:
            yield entered, SimpleNamespace(set=release)
        finally:
            release()


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError("Condition did not become true before the test deadline")


def structured_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result[1]
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    contents = getattr(result, "content", result)
    return json.loads(next(block.text for block in contents if hasattr(block, "text")))


class WorkshopPlanningTests(unittest.TestCase):
    def test_http_observe_camera_and_mcp_stay_live_before_validation_dispatches(self):
        with public_session() as (service, client), paused_planning(service) as (entered, release):
            target = client.observe()["left_tcp"]
            with patch.object(service.robot, "submit_target", wraps=service.robot.submit_target) as dispatch:
                operation = client.move_tcp("left", target)
                self.assertTrue(entered.wait(3))
                first = client.observe()
                first_frame = client.capture()
                time.sleep(.25)
                started = time.monotonic()
                second = client.observe()
                second_frame = client.capture()
                self.assertLess(time.monotonic() - started, 1.)
                self.assertGreater(second["observation_id"], first["observation_id"])
                self.assertGreater(second_frame["timestamp_s"], first_frame["timestamp_s"])
                self.assertLess(time.monotonic() - second["timestamp_s"], .2)
                self.assertLess(time.monotonic() - second_frame["timestamp_s"], .5)
                self.assertEqual(operation.status()["planning"]["state"], "running")
                self.assertEqual(dispatch.call_count, 0)
                np.testing.assert_allclose(second["joint_positions_rad"], first["joint_positions_rad"], atol=.001)

                async def read_mcp():
                    result = await client._test_mcp.call_tool("robot_observe", {})
                    return structured_result(result)

                self.assertIn("motion_paths", asyncio.run(read_mcp()))
                release.set()
                result = operation.wait(timeout_s=10)
                self.assertEqual(result["state"], "succeeded", result)
                self.assertEqual(result["planning"]["state"], "validated")
                self.assertTrue(any(isinstance(call.args[0], MixedAction)
                                    for call in dispatch.call_args_list))

    def test_cancel_and_stop_discard_late_successful_plans(self):
        for stop in (False, True):
            with self.subTest(stop=stop), public_session() as (service, client):
                with paused_planning(service) as (entered, release):
                    target = client.observe()["left_tcp"]
                    with patch.object(service.robot, "submit_target", wraps=service.robot.submit_target) as dispatch:
                        operation = client.move_tcp("left", target)
                        self.assertTrue(entered.wait(3))
                        with service.lock:
                            future = service.operations[operation.id]["_plan_future"]
                        if stop:
                            service.operator_stop()
                        else:
                            operation.cancel()
                        self.assertEqual(operation.status()["state"], "cancelled")
                        release.set()
                        future.result(timeout=5)
                        time.sleep(.15)
                        self.assertEqual(dispatch.call_count, 0)
                        self.assertEqual(operation.status()["state"], "cancelled")
                        self.assertIsNone(client.observe()["session_fault"])
                        self.assertEqual(service.stopped, stop)
                        self.assertIsNone(service.active)

    def test_cancelled_planning_does_not_block_a_new_joint_operation(self):
        with public_session() as (service, client), paused_planning(service) as (entered, release):
            before = client.observe()
            operation = client.move_tcp("left", before["left_tcp"])
            self.assertTrue(entered.wait(3))
            operation.cancel()
            joint_operation = client.move_joints("left", before["joint_positions_rad"][:7])
            # A blocked old solver owns only planning state. The new joint
            # action and fresh observation must not wait for that solver.
            done = threading.Event()
            result = []

            def observe():
                result.append(client.observe())
                done.set()

            reader = threading.Thread(target=observe)
            reader.start()
            try:
                self.assertTrue(done.wait(1), "Cancelled planner blocked fresh observation")
                self.assertEqual(joint_operation.wait(timeout_s=3)["state"], "succeeded")
                self.assertIsNone(result[0]["session_fault"])
            finally:
                release.set()
                reader.join(timeout=5)
                self.assertFalse(reader.is_alive())

    def test_invalid_path_is_an_ordinary_failed_mcp_operation_and_can_recover(self):
        with public_session() as (service, client):
            before = client.observe()
            impossible = dict(position_m=[1.7, 1.7, 1.7],
                              quaternion_xyzw=before["left_tcp"]["quaternion_xyzw"])

            async def submit_mcp():
                result = await client._test_mcp.call_tool(
                    "motion_move_tcp", dict(arm="left", target=impossible,
                                             request_id="unreachable-path", timeout_s=10))
                return structured_result(result)

            with patch.object(service.robot, "submit_target", wraps=service.robot.submit_target) as dispatch:
                accepted = asyncio.run(submit_mcp())
                result = wait_until(lambda: (row if (row := client.call(
                    "operation_get", operation_id=accepted["operation_id"]))["state"] != "running" else None), 10)
                self.assertEqual(result["state"], "failed", result)
                self.assertEqual(result["reason"], "PLANNING_FAILED")
                self.assertEqual(result["planning"]["state"], "failed")
                self.assertEqual(dispatch.call_count, 0)
            after = client.observe()
            self.assertIsNone(after["session_fault"])
            self.assertTrue(after["session_enabled"])
            self.assertFalse(service.stopped)
            np.testing.assert_allclose(after["joint_positions_rad"], before["joint_positions_rad"], atol=.001)
            self.assertEqual(client.gripper_open("left").wait(timeout_s=5)["state"], "succeeded")

    def test_recorded_endpoint_ik_reports_pose_failure_without_moving(self):
        with public_session(recorded=True) as (service, client):
            before = client.observe()
            with patch.object(service.robot, "submit_target", wraps=service.robot.submit_target) as dispatch:
                result = client.ik("left", asdict(RECORDED_TARGET), seed_rad=RECORDED_START[:7].tolist())
                self.assertTrue(result["solver_success"])
                self.assertFalse(result["success"])
                self.assertGreater(result["orientation_error_rad"], .25)
                self.assertLess(result["orientation_error_rad"], .35)
                self.assertTrue(result["within_model_limits"])
                dispatch.assert_not_called()
            np.testing.assert_allclose(client.observe()["joint_positions_rad"], before["joint_positions_rad"], atol=.001)

    def test_recorded_target_alternate_path_completes_through_http(self):
        with public_session(recorded=True) as (service, client):
            before = client.observe()
            operation = client.move_tcp("left", asdict(RECORDED_TARGET), timeout_s=20)
            saw_alternate = False
            deadline = time.monotonic() + 23
            while time.monotonic() < deadline:
                result = operation.status()
                observed = client.observe()
                path = observed.get("motion_paths", {}).get("left", {})
                saw_alternate |= path.get("rotation_arc") == "alternate"
                self.assertIsNone(observed["session_fault"])
                if result["state"] != "running":
                    break
                time.sleep(.05)
            self.assertEqual(result["state"], "succeeded", result)
            self.assertEqual(result["planning"]["arms"]["left"]["selected_arc"], "alternate")
            self.assertTrue(saw_alternate)
            self.assertLess(result["residuals"]["left"]["position_m"], .005)
            self.assertLess(result["residuals"]["left"]["orientation_rad"], .05)
            np.testing.assert_allclose(client.observe()["joint_positions_rad"][7:],
                                       before["joint_positions_rad"][7:], atol=.001)

    def test_alternate_pose_then_gripper_step_does_not_reuse_rotation_completion_gate(self):
        with public_session(recorded=True) as (service, client):
            before = client.observe()
            accepted = service.submit(client.client_id, "alternate-then-jaw", [
                dict(poses={"left": asdict(RECORDED_TARGET)}),
                dict(grippers={"left": 1}),
            ], timeout_s=20)
            result = Operation(client, accepted).wait(timeout_s=23)
            self.assertEqual(result["state"], "succeeded", result)
            self.assertEqual(result["step_index"], 2)
            self.assertEqual(result["progress"], 1.)
            plans = [row for row in service.events if row["event"] == "motion_plan"]
            self.assertEqual(plans[0]["arms"]["left"]["selected_arc"], "alternate")
            after = client.observe()
            self.assertGreater(after["gripper_position"][0], .99)
            self.assertIsNone(after["session_fault"])
            np.testing.assert_allclose(after["joint_positions_rad"][7:],
                                       before["joint_positions_rad"][7:], atol=.001)


class MotionLimitsRPCTests(unittest.TestCase):
    def test_planning_failure_survives_process_result_serialization(self):
        report = dict(side="left", selected_arc=None, candidates=[
            dict(rotation_arc="shortest", failure="pose residual exceeds tolerance", valid=False),
            dict(rotation_arc="alternate", failure="joint position limit", valid=False),
        ])
        error = CartesianPathUnavailable(report)
        restored = pickle.loads(pickle.dumps(error))
        self.assertIsInstance(restored, CartesianPathUnavailable)
        self.assertEqual(restored.report, report)
        self.assertEqual(str(restored), str(error))

    def test_simulated_hardware_session_returns_independent_readback_copies(self):
        session = HardwareSession("both", simulation=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "after arming readback"):
                session.motion_limits()
            session.arm()
            before = session.observe()
            first = session.motion_limits()
            expected = np.asarray(first["position"]).copy()
            first["position"][0][0] = 999.
            second = session.motion_limits()
            np.testing.assert_array_equal(second["position"], expected)
            np.testing.assert_allclose(session.observe().joint_positions_rad,
                                       before.joint_positions_rad, atol=.001)
        finally:
            session.close()

    def test_worker_exports_only_cached_armed_limits_for_single_and_paired_sessions(self):
        for side, count in (("left", 7), ("right", 7), ("both", 14)):
            with self.subTest(side=side):
                connection = Mock()
                connection.poll.return_value = True
                connection.recv.side_effect = [
                    (1, "motion_limits", None), (2, "arm", None),
                    (3, "motion_limits", None), (4, "motion_limits", None),
                    (5, "close", False),
                ]
                with patch("agilex_deploy.hardware_session.signal.signal"), \
                     patch("agilex_deploy.hardware_session.NeroTransport", side_effect=AssertionError("Physical transport forbidden")), \
                     patch("agilex_deploy.hardware_session.DualNeroTransport", side_effect=AssertionError("Physical transport forbidden")):
                    control_worker(connection, side, "unused", "unused", True, None)
                replies = {call.args[0][0]: call.args[0] for call in connection.send.call_args_list}
                self.assertFalse(replies[1][1])
                self.assertIn("after arming readback", replies[1][2])
                self.assertTrue(replies[2][1], replies[2])
                self.assertTrue(replies[3][1], replies[3])
                self.assertEqual(replies[3][2], replies[4][2])
                limits = replies[3][2]
                self.assertEqual(set(limits), {"position", "velocity", "acceleration"})
                np.testing.assert_array_equal(limits["position"], Kinematics().limits[:count])
                np.testing.assert_array_equal(limits["velocity"], np.tile(SIM_VELOCITY, count // 7))
                np.testing.assert_array_equal(limits["acceleration"], np.tile(SIM_ACCELERATION, count // 7))
                self.assertTrue(replies[5][1], replies[5])


class WorkshopCleanupTests(unittest.TestCase):
    @staticmethod
    def fixture():
        service = WorkshopService.__new__(WorkshopService)
        service.closed = False
        service.done = threading.Event()
        service.planning_generation = SimpleNamespace(value=4)
        service.worker = Mock()
        service.worker.name = "offline-test-worker"
        service.worker.is_alive.return_value = True
        service.camera_thread = None
        service.gravity_comp = False
        service.enabled = True
        service.fault = None
        service.backend = "real"
        service.plant = None
        order = []
        service.robot = Mock()
        service.robot.close.side_effect = lambda **kwargs: order.append("robot")
        service.planner_pool = object()
        return service, order

    def test_worker_cleanup_failure_still_closes_robot_before_planner_and_can_retry(self):
        service, order = self.fixture()
        with patch("agilex_deploy.workshop_service.close_planner",
                   side_effect=lambda pool: order.append("planner")) as close_planner:
            with self.assertRaisesRegex(RuntimeError, "offline-test-worker did not exit"):
                service.close()
            self.assertEqual(order, ["robot", "planner"])
            service.robot.close.assert_called_once_with(hand_guiding=False)
            close_planner.assert_called_once()
            self.assertFalse(service.closed)
            self.assertTrue(service.done.is_set())
            self.assertEqual(service.planning_generation.value, 5)
            self.assertIsNone(service.planner_pool)
            service.worker.is_alive.return_value = False
            service.close()
            self.assertTrue(service.closed)
            self.assertEqual(order, ["robot", "planner", "robot"])

    def test_raised_thread_join_still_closes_robot_and_planner(self):
        service, order = self.fixture()
        service.worker.join.side_effect = RuntimeError("unstarted worker")
        with patch("agilex_deploy.workshop_service.close_planner",
                   side_effect=lambda pool: order.append("planner")):
            with self.assertRaisesRegex(RuntimeError, "join failed: unstarted worker"):
                service.close()
        self.assertEqual(order, ["robot", "planner"])
        self.assertFalse(service.closed)
        self.assertIsNone(service.planner_pool)


if __name__ == "__main__":
    unittest.main()
