"""Sensor boundary, grounding, real camera geometry, and offline scoring tests."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agilex_deploy.agent_policy import (
    OpenRouterPolicy,
    packet,
    parse_decision,
    pinch_pose,
    tcp_pose,
)
from agilex_deploy.agent_sensors import (
    RGBDFrame,
    ground_pixel,
    load_calibration,
    quantize_depth,
)
from agilex_deploy.agent_sim import ManipulationSim, SimCamera, evaluate
from agilex_deploy.agent_zed import stereo_geometry
from agilex_deploy.contract import Pose
from agilex_deploy.model import ROOT
from agilex_deploy.runtime import Robot


def frame():
    return RGBDFrame(
        np.zeros((100, 200, 3), np.uint8),
        np.ones((100, 200)),
        np.array([[100.0, 0, 100], [0, 100, 50], [0, 0, 1]]),
        np.eye(4),
        time.monotonic(),
        {"status": "test"},
    )


class AgentTests(unittest.TestCase):
    def test_depth_units_invalid_values_and_point_projection(self):
        np.testing.assert_array_equal(
            quantize_depth(np.array([np.nan, np.inf, -1, 0, 0.5436, 70])),
            [0, 0, 0, 0, 544, 0],
        )
        point, surface = ground_pixel(frame(), [110, 60], [0, 0, 0.2])
        np.testing.assert_allclose(surface, [0.1, 0.1, 1])
        np.testing.assert_allclose(point, [0.1, 0.1, 1.2])
        for pixel in ([0, 0], [110.5, 60]):
            with self.assertRaises(ValueError):
                ground_pixel(frame(), pixel, [0, 0, 0])
        f = frame()
        f.depth_m[60, 110] = 0
        with self.assertRaises(ValueError):
            ground_pixel(f, [110, 60], [0, 0, 0])

    def test_tool_geometry_round_trip(self):
        p = Pose([0.3, 0.2, 0.1], [0.5, -0.5, 0.5, -0.5])
        np.testing.assert_allclose(
            tcp_pose(pinch_pose(p)).position_m, p.position_m, atol=1e-12
        )

    def test_normalized_pixel_coordinates_and_closed_schema(self):
        arm = {
            "position_m": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
            "gripper_open": None,
        }
        output = {
            "left": arm,
            "right": arm,
            "duration_s": 2,
            "done": False,
            "decision_summary": "Inspect",
            "grounding": {
                "arm": "left",
                "pixel_uv_1000": [500, 500],
                "offset_base_m": [0, 0, 0.1],
                "phase": "target",
            },
        }
        decision = parse_decision(output, frame())
        self.assertEqual(decision.grounding["pixel_uv"], [100, 50])
        np.testing.assert_allclose(
            pinch_pose(decision.action.left.pose).position_m, [0, 0, 1.1]
        )
        output["grounding"]["phase"] = "inspect"
        self.assertIsNone(parse_decision(output, frame()).action)
        for extra in ("object_position", "python", "navigation"):
            with self.assertRaises(ValueError):
                parse_decision({**output, extra: None}, frame())
        output["left"] = {**arm, "gripper_open": 0.5}
        with self.assertRaises(ValueError):
            parse_decision(output, frame())

    def test_factory_rectification_transforms_rays_not_only_intrinsics(self):
        base = ROOT / "calibration/2026-10-15_zed35330610"
        t, _ = load_calibration(base / "raised_probe/results_agent_v2/candidate.json")
        maps, k, rectified_t, baseline = stereo_geometry(
            base / "physical/SN35330610.conf", t
        )
        self.assertEqual(maps[0][0].shape, (360, 640))
        self.assertTrue(0.119 < baseline < 0.121)
        self.assertGreater(k[0, 0], 200)
        self.assertFalse(np.allclose(t[:3, :3], rectified_t[:3, :3], atol=1e-6))
        np.testing.assert_allclose(t[:3, 3], rectified_t[:3, 3])

    def test_success_requires_lift_contact_retention_and_clean_final_state(self):
        rows = [
            {
                "time_s": i * 0.01,
                "cube_position_m": [0, 0, 0.25],
                "cube_contacts": ["left_pad1", "left_pad2"],
                "robot_environment_contacts": [],
            }
            for i in range(1100)
        ]
        self.assertTrue(evaluate(rows, 0.1)["success"])
        rows[-1]["cube_contacts"] = ["table"]
        self.assertFalse(evaluate(rows, 0.1)["success"])
        rows[-1]["cube_contacts"] = ["left_pad1", "left_pad2"]
        rows[0]["robot_environment_contacts"] = [["table", "left_link7"]]
        self.assertFalse(evaluate(rows, 0.1)["success"])

    def test_rendered_rgbd_grounding_and_model_boundary(self):
        plant = ManipulationSim(ROOT / "configs/agent/cube_scene.json")
        self.assertEqual(plant.model.ncam, 1)
        self.assertEqual(plant.model.camera(0).name, "head_rgbd")
        body_names = {plant.model.body(i).name for i in range(plant.model.nbody)}
        for side in ("left", "right"):
            self.assertNotIn(f"{side}_camera_stand_link", body_names)
            self.assertNotIn(f"{side}_d435_camera_link", body_names)
        with Robot(plant) as robot:
            for _ in range(10):
                robot.tick()
            camera = SimCamera(robot, plant)
            try:
                observed, f = camera.capture()
                f.validate()
                # Privileged projection belongs only in this offline geometry test.
                xyz = plant.data.body("task_cube").xpos + [0, 0, 0.025]
                optical = np.linalg.inv(f.T_base_from_optical) @ np.r_[xyz, 1]
                uv = f.K @ optical[:3]
                pixel = np.rint(uv[:2] / uv[2]).astype(int).tolist()
                _, recovered = ground_pixel(f, pixel, [0, 0, 0])
                np.testing.assert_allclose(recovered, xyz, atol=0.003)
                metadata = packet(
                    observed,
                    f,
                    "Pick up the cube",
                    {"state": "idle"},
                    ("left", "right"),
                )
                self.assertEqual(
                    set(metadata),
                    {
                        "instruction",
                        "robot",
                        "pinch_poses",
                        "camera",
                        "depth_samples_uv_m",
                        "execution",
                        "available_arms",
                        "tool_geometry",
                    },
                )
                serialized = json.dumps(metadata)
                for forbidden in (
                    "cube_position_m",
                    "cube_contacts",
                    "scene_config",
                    "task_cube",
                    "initial_joint_positions_rad",
                ):
                    self.assertNotIn(forbidden, serialized)
                self.assertEqual(observed.gripper_states[0].force_n, None)
                # Credential handling is tested with a fake key and mock HTTPS only.
                with tempfile.TemporaryDirectory() as directory:
                    key = Path(directory) / "key"
                    key.write_text("test-secret-do-not-log")
                    policy = OpenRouterPolicy(Path(directory) / "policy", key)
                    response = {
                        "choices": [{"message": {"content": "{}"}}],
                        "usage": {"cost": 0.01},
                    }

                    class HTTPResponse:
                        def __enter__(self):
                            return self

                        def __exit__(self, *_):
                            pass

                        def read(self):
                            return json.dumps(response).encode()

                    with (
                        patch(
                            "urllib.request.urlopen", return_value=HTTPResponse()
                        ) as request,
                        self.assertRaises(ValueError),
                    ):
                        policy.act(observed, f, "Pick", {}, ("left", "right"))
                    self.assertEqual(
                        request.call_args.args[0].get_header("Authorization"),
                        "Bearer test-secret-do-not-log",
                    )
                    for file in (Path(directory) / "policy").glob("*.json"):
                        self.assertNotIn("test-secret-do-not-log", file.read_text())
            finally:
                camera.close()


if __name__ == "__main__":
    unittest.main()
