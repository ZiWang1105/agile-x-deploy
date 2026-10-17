"""Public API, MCP transport, joint execution, and evaluator regressions; no CAN."""

import asyncio
import base64
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import cv2
import numpy as np

from agilex_deploy.workshop_client import RobotClient, APIError
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService
from agilex_deploy.workshop_tasks import WorkshopSim, scene_for
from agilex_deploy.workshop_mcp import build_server


class WorkshopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.service = WorkshopService(output=cls.temp.name)
        cls.server = WorkshopHTTP(cls.service, 0)
        cls.client = RobotClient(cls.server.endpoint, cls.server.token, "test-owner")
        cls.client.claim()
        cls.service.enable()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.close()
        finally:
            cls.server.close()
            cls.service.close()
            cls.temp.cleanup()

    def test_camera_has_registered_geometry_and_no_object_truth(self):
        frame = self.client.capture()
        rgb = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame["rgb_png_base64"]), np.uint8), 1
        )
        depth = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame["depth_png_base64"]), np.uint8), -1
        )
        self.assertEqual(rgb.shape, (*depth.shape, 3))
        self.assertEqual(frame["depth_png_scale_m"], 0.001)
        self.assertNotIn("objects", frame)
        self.assertNotIn("cube_position_m", json.dumps(frame))
        self.assertNotIn("seed", self.client.capabilities())
        with self.assertRaises(APIError):
            self.client.deproject("unknown", [50, 50])
        with self.assertRaises(APIError):
            self.client.deproject(frame["frame_id"], [-1, 0])

    def test_exclusive_owner_and_no_participant_enable_reset_score(self):
        impersonator = RobotClient(
            self.server.endpoint, self.server.token, self.client.client_id
        )
        with self.assertRaises(APIError):
            impersonator.observe()
        with self.assertRaises(APIError):
            impersonator.claim()
        other = RobotClient(self.server.endpoint, self.server.token, "other")
        with self.assertRaises(APIError) as caught:
            other.claim()
        self.assertEqual(caught.exception.code, "OWNER_BUSY")
        for method in ("enable", "reset", "evaluate", "__getattribute__"):
            with self.assertRaises(APIError):
                self.client.call(method)
        with urlopen(self.server.endpoint) as response:
            html = response.read().decode()
        self.assertNotIn(self.server.operator_token, html)
        request = Request(
            self.server.endpoint + "/operator/enable",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.server.token,
            },
        )
        with self.assertRaises(HTTPError):
            urlopen(request)

    def test_duplicate_request_returns_same_operation_and_conflict_rejected(self):
        target = self.client.observe()["left_tcp"]
        op = self.client.move_tcp("left", target, request_id="idempotent-test")
        duplicate = self.client.move_tcp("left", target, request_id="idempotent-test")
        self.assertEqual(op.id, duplicate.id)
        moved = dict(
            target,
            position_m=[target["position_m"][0] + 0.01, *target["position_m"][1:]],
        )
        with self.assertRaises(APIError) as caught:
            self.client.move_tcp("left", moved, request_id="idempotent-test")
        self.assertEqual(caught.exception.code, "REQUEST_CONFLICT")
        self.assertEqual(op.wait()["state"], "succeeded")

    def test_explicit_joint_target_is_not_an_ik_round_trip(self):
        observed = self.client.observe()
        before = np.array(observed["joint_positions_rad"])
        desired = before[:7].copy()
        desired[0] += 0.05
        with patch.object(
            self.service.robot.controller.ik,
            "solve",
            side_effect=AssertionError("Joint command invoked IK"),
        ):
            result = self.client.move_joints("left", desired.tolist()).wait()
            self.assertEqual(result["state"], "succeeded")
        after = np.array(self.client.observe()["joint_positions_rad"])
        np.testing.assert_allclose(after[:7], desired, atol=0.01)
        np.testing.assert_allclose(after[7:], before[7:], atol=0.0001)
        self.assertEqual(
            self.client.move_joints("left", before[:7].tolist()).wait()["state"],
            "succeeded",
        )

    def test_gripper_independence_through_sdk(self):
        before = self.client.observe()
        result = self.client.gripper_open("left").wait()
        self.assertEqual(result["state"], "succeeded")
        after = self.client.observe()
        self.assertGreater(after["gripper_position"][0], 0.99)
        self.assertAlmostEqual(
            after["gripper_position"][1], before["gripper_position"][1], delta=0.001
        )
        self.assertEqual(self.client.gripper_close("left").wait()["state"], "succeeded")

    def test_ik_fk_are_read_only_and_return_residuals(self):
        before = self.client.observe()
        fk = self.client.fk("left", before["joint_positions_rad"][:7])
        np.testing.assert_allclose(
            fk["pose"]["position_m"], before["left_tcp"]["position_m"], atol=1e-8
        )
        ik = self.client.ik("left", fk["pose"])
        self.assertTrue(ik["success"])
        self.assertIn("position_error_m", ik)
        after = self.client.observe()
        np.testing.assert_allclose(
            after["joint_positions_rad"], before["joint_positions_rad"], atol=0.001
        )
        with self.assertRaises(APIError):
            self.client.fk("left", [99] * 7)

    def test_cancel_prevents_further_motion_and_stale_owner_cannot_dispatch(self):
        target = self.client.observe()["left_tcp"]
        target["position_m"][2] += 0.03
        op = self.client.move_tcp("left", target)
        self.assertEqual(op.cancel()["state"], "cancelled")
        before = self.client.observe()["joint_positions_rad"]
        time.sleep(0.15)
        after = self.client.observe()["joint_positions_rad"]
        np.testing.assert_allclose(after, before, atol=0.002)
        other = RobotClient(self.server.endpoint, self.server.token, "not-owner")
        with self.assertRaises(APIError):
            other.gripper_open("right")

    def test_mcp_schemas_and_structured_results_match_http(self):
        async def check():
            mcp = build_server(self.client)
            names = {tool.name for tool in await mcp.list_tools()}
            self.assertIn("motion_move_both", names)
            self.assertIn("gripper_open", names)
            self.assertNotIn("enable", names)
            # Call actual official SDK tool dispatch, not handwritten JSON-RPC.
            result = await mcp.call_tool("robot_capabilities", {})
            self.assertIn("api_version", str(result))

        asyncio.run(check())


class RealAdapterContractTests(unittest.TestCase):
    def test_real_configuration_uses_same_public_api_and_explicit_operator_enable(self):
        from dataclasses import replace
        from agilex_deploy.runtime import Robot
        from agilex_deploy.backends import SimBackend
        from agilex_deploy.agent_sensors import RGBDFrame

        observed_calls = []

        class Hardware:
            def __init__(self, side, **kwargs):
                observed_calls.append((side, kwargs))
                self.robot = Robot(SimBackend())
                self.armed = False

            def observe(self):
                with self.robot.lock:
                    if self.robot.thread is None:
                        self.robot._observe(self.robot._read())
                    value = self.robot.observe()
                    return value if self.armed else replace(value, mode="unarmed")

            def arm(self):
                self.armed = True

            def start(self):
                self.robot.start()

            def submit_target(self, *args):
                return self.robot.submit_target(*args)

            def hold(self):
                self.robot.hold()

            def close(self):
                self.robot.close()

        class Camera:
            def __init__(self, robot, *args, **kwargs):
                self.robot = robot

            def capture(self):
                return self.robot.observe(), RGBDFrame(
                    np.zeros((100, 200, 3), np.uint8),
                    np.ones((100, 200)),
                    np.array([[100.0, 0, 100], [0, 100, 50], [0, 0, 1]]),
                    np.eye(4),
                    time.monotonic(),
                    {"status": "mock"},
                )

            def close(self):
                pass

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agilex_deploy.hardware_session.HardwareSession", Hardware),
            patch("agilex_deploy.agent_zed.ZedStereoCamera", Camera),
        ):
            service = WorkshopService("real", output=directory)
            server = WorkshopHTTP(service, 0)
            client = RobotClient(server.endpoint, server.token)
            try:
                client.claim()
                self.assertEqual(observed_calls, [("both", {"gripper_control": True})])
                self.assertFalse(service.robot.armed)
                self.assertEqual(client.capabilities()["arms"], ["left", "right"])
                self.assertEqual(client.observe()["mode"], "unarmed")
                self.assertIn("T_base_from_optical", client.capture())
                with self.assertRaises(APIError) as caught:
                    client.gripper_open("left")
                self.assertEqual(caught.exception.code, "NOT_ENABLED")
                service.enable()
                self.assertTrue(service.robot.armed)
                self.assertEqual(
                    client.gripper_open("left").wait()["state"], "succeeded"
                )
                self.assertAlmostEqual(
                    client.observe()["gripper_position"][1], 0.36, delta=0.001
                )
            finally:
                client.close()
                server.close()
                service.close()


class TaskTests(unittest.TestCase):
    def test_seeded_tasks_have_distinct_layouts_and_no_initial_success(self):
        self.assertEqual(scene_for("sort", 3), scene_for("sort", 3))
        self.assertNotEqual(
            scene_for("sort", 3)["objects"], scene_for("sort", 4)["objects"]
        )
        for task in ("place", "select", "sort"):
            plant = WorkshopSim(task)
            try:
                for _ in range(20):
                    plant.advance()
                self.assertFalse(plant.evaluate()["success"])
                self.assertEqual(
                    plant.evaluate()["robot_environment_collision_frames"], 0
                )
            finally:
                plant.close()


if __name__ == "__main__":
    unittest.main()
