"""Calibration geometry and rejection tests; no devices or motion transports."""

import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy.camera_calibration import (
    fit_extrinsics,
    project,
    validate_camera,
    import_factory_intrinsics,
    new_dataset,
    save_json,
    solve_dataset,
    sha256,
)
from agilex_deploy.calibration_capture import stationary_evidence
from agilex_deploy.model import initial_positions


def camera():
    return dict(
        view="left",
        image_space="raw",
        distortion_model="opencv_brown_conrady",
        size_wh=[1280, 720],
        serial="TEST",
        source="synthetic test fixture",
        K=[[720, 0, 631], [0, 715, 358], [0, 0, 1]],
        distortion=[-0.15, 0.025, 0.001, -0.002, 0.003],
    )


class CameraCalibrationTests(unittest.TestCase):
    def test_distorted_pnp_with_outliers(self):
        rng = np.random.default_rng(84)
        points = rng.uniform([-0.4, -0.3, -0.15], [0.4, 0.3, 0.2], (120, 3))
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [0.1, -0.2, 0.08]).as_matrix()
        transform[:3, 3] = [0.05, -0.07, 1.2]
        uv, _ = project(points, transform, camera())
        uv += rng.normal(0, 0.25, uv.shape)
        uv[:18] += rng.uniform(80, 150, (18, 2))
        result = fit_extrinsics(points, uv, camera())
        estimated = np.asarray(result["T_camera_from_base"])
        self.assertLess(np.linalg.norm(estimated[:3, 3] - transform[:3, 3]), 0.002)
        self.assertLess(
            Rotation.from_matrix(estimated[:3, :3].T @ transform[:3, :3]).magnitude(),
            0.003,
        )
        np.testing.assert_allclose(
            estimated @ result["T_base_from_camera"], np.eye(4), atol=1e-10
        )
        self.assertGreater(result["all_rms_px"], 30)
        self.assertLess(result["inlier_rms_px"], 1)
        self.assertLessEqual(sum(result["inliers"]), 105)

    def test_degenerate_geometry_rejected(self):
        points = np.column_stack((np.linspace(0, 1, 10), np.zeros((10, 2))))
        with self.assertRaisesRegex(ValueError, "spread"):
            fit_extrinsics(points, np.ones((10, 2)), camera())

    def test_rectified_raw_mismatch_rejected(self):
        value = camera()
        value["image_space"] = "rectified"
        with self.assertRaisesRegex(ValueError, "zero distortion"):
            validate_camera(value)

    def test_fisheye_is_not_silently_treated_as_pinhole(self):
        value = camera()
        value["distortion_model"] = "fisheye"
        with self.assertRaisesRegex(ValueError, "Brown"):
            validate_camera(value)

    def test_factory_coefficient_order_and_higher_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "SN123.conf"
            file.write_text(
                "[LEFT_CAM_HD]\nfx=700\nfy=710\ncx=640\ncy=360\nk1=.1\nk2=.2\np1=.3\np2=.4\nk3=.5\nk4=.6\nk5=.7\nk6=.8\n"
            )
            value = import_factory_intrinsics(file, "123")
            self.assertEqual(
                value["distortion"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            )
            self.assertEqual(value["size_wh"], [1280, 720])
            self.assertEqual(value["source"]["sha256"], sha256(file))

    def test_factory_separate_distortion_section_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "SN123.conf"
            file.write_text(
                "[LEFT_CAM_HD]\nfx=700\nfy=710\ncx=640\ncy=360\n"
                "k1=9\nk2=9\np1=9\np2=9\nk3=9\n"
                "[LEFT_DISTO]\nk1=.1\nk2=.2\np1=.3\np2=.4\n"
                "k3=.5\nk4=.6\nk5=.7\nk6=.8\n"
            )
            value = import_factory_intrinsics(file, "123")
            self.assertEqual(
                value["distortion"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            )
            self.assertEqual(value["K"][0][0], 700)
            self.assertEqual(value["source"]["distortion_section"], "LEFT_DISTO")

    def test_stationary_capture_preserves_timing_limit(self):
        samples = [
            dict(timestamp_s=float(t), q=[0.0] * 14) for t in np.linspace(10, 12, 101)
        ]
        evidence = stationary_evidence(samples, 11)
        self.assertTrue(evidence["accepted"])
        self.assertFalse(evidence["timing_qualified"])
        self.assertEqual(evidence["selected_index"], 50)

    def test_nonfinite_capture_gate_rejected(self):
        samples = [
            dict(timestamp_s=float(t), q=[0.0] * 14) for t in np.linspace(10, 12, 101)
        ]
        with self.assertRaisesRegex(ValueError, "Invalid image timestamp"):
            stationary_evidence(samples, float("nan"))
        with self.assertRaisesRegex(ValueError, "stationarity threshold"):
            stationary_evidence(samples, 11, float("nan"))

    def test_repeated_posture_cannot_leak_into_held_out_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "frame.jpg"
            cv2.imwrite(str(image), np.zeros((720, 1280, 3), np.uint8))
            document = new_dataset(camera(), "synthetic")
            document["anchors"]["anchor"] = dict(
                body="left_link3", local_xyz_m=[0, 0, 0]
            )
            for index in range(4):
                pose_id = f"pose_{index}"
                document["poses"].append(
                    dict(
                        id=pose_id,
                        q_measured_rad=initial_positions().tolist(),
                        image="frame.jpg",
                        image_sha256=sha256(image),
                    )
                )
                document["observations"].append(
                    dict(pose_id=pose_id, anchor_id="anchor", uv=[640, 360])
                )
            save_json(directory / "dataset.json", document)
            with self.assertRaisesRegex(ValueError, "Near-duplicate"):
                solve_dataset(
                    directory / "dataset.json",
                    directory / "result.json",
                    allow_synthetic=True,
                )

    def test_moving_capture_rejected(self):
        samples = [
            dict(timestamp_s=float(t), q=[float(t) * 0.01] * 14)
            for t in np.linspace(10, 12, 101)
        ]
        with self.assertRaisesRegex(ValueError, "moved"):
            stationary_evidence(samples, 11)

    def test_missing_bracketing_or_feedback_rejected(self):
        samples = [
            dict(timestamp_s=float(t), q=[0.0] * 14) for t in np.linspace(10, 12, 101)
        ]
        with self.assertRaisesRegex(ValueError, "0.5 s"):
            stationary_evidence(samples, 10.1)
        with self.assertRaisesRegex(ValueError, "gap"):
            stationary_evidence(samples[:20] + samples[60:], 11)

    def test_synthetic_fit_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "dataset.json"
            save_json(p, new_dataset(camera(), "synthetic"))
            with self.assertRaisesRegex(ValueError, "allow-synthetic"):
                solve_dataset(p, Path(directory) / "result.json")

    def test_physical_capture_needs_stationarity_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "frame.jpg"
            cv2.imwrite(str(image), np.zeros((720, 1280, 3), np.uint8))
            d = new_dataset(camera())
            d["poses"] = [
                dict(
                    id="pose",
                    q_measured_rad=initial_positions().tolist(),
                    image="frame.jpg",
                    image_sha256=sha256(image),
                )
            ]
            save_json(directory / "dataset.json", d)
            with self.assertRaisesRegex(ValueError, "stationary"):
                solve_dataset(directory / "dataset.json", directory / "result.json")


if __name__ == "__main__":
    unittest.main()
