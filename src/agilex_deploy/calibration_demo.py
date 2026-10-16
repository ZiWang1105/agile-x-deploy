"""Synthetic end-to-end fixture with known extrinsics; never physical evidence."""

from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_calibration import (
    new_dataset,
    save_json,
    sha256,
    RIGID_ARM,
    project,
    solve_dataset,
)
from .model import Kinematics, initial_positions, MODEL_PATH


def create_demo(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new demo directory")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir()
    camera = dict(
        view="left",
        image_space="raw",
        distortion_model="opencv_brown_conrady",
        size_wh=[1280, 720],
        K=[[700, 0, 640], [0, 700, 360], [0, 0, 1]],
        distortion=[0, 0, 0, 0, 0],
        serial="SYNTHETIC",
        source="synthetic test intrinsics, not ZED factory data",
    )
    doc = new_dataset(camera, "synthetic")
    position = np.array([0.85, 0, 1.1])
    forward = np.array([0.15, 0, 0.4]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = np.column_stack((right, down, forward))
    base_from_camera[:3, 3] = position
    camera_from_base = np.linalg.inv(base_from_camera)
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    quat = Rotation.from_matrix(
        base_from_camera[:3, :3] @ np.diag([1, -1, -1])
    ).as_quat()[[3, 0, 1, 2]]
    spec.worldbody.add_camera(
        name="calibration_truth",
        pos=position,
        quat=quat,
        ipd=0,
        fovy=np.degrees(2 * np.arctan(720 / 1400)),
    )
    kin = Kinematics(spec.compile())
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    rng = np.random.default_rng(731)
    with mujoco.Renderer(kin.model, height=720, width=1280) as renderer:

        def scene(q):
            kin.update(q, (None, None))
            renderer.update_scene(
                kin.data, camera="calibration_truth", scene_option=option
            )

        scene(initial_positions())
        # Seed test anchors by real ray intersections, not by assuming joint
        # centres are visible physical features.
        counts = {}
        for v in range(120, 650, 18):
            for u in range(120, 1160, 18):
                p = np.zeros(3)
                g, f, s = [np.full(1, -1, np.int32) for _ in range(3)]
                body_id = mujoco.mjv_select(
                    kin.model,
                    kin.data,
                    option,
                    1280 / 720,
                    u / 1280,
                    1 - v / 720,
                    renderer.scene,
                    p,
                    g,
                    f,
                    s,
                )
                name = kin.model.body(body_id).name if body_id >= 0 else ""
                if body_id >= 0:
                    roundtrip, _ = project([p], camera_from_base, camera)
                    if np.linalg.norm(roundtrip[0] - [u, v]) > 1:
                        raise AssertionError(
                            ("Camera projection mismatch", roundtrip[0], [u, v])
                        )
                if RIGID_ARM.fullmatch(name) and counts.get(name, 0) < 2:
                    body = kin.data.body(body_id)
                    anchor_id = f"{name}_{counts.get(name, 0)}"
                    doc["anchors"][anchor_id] = dict(
                        body=name,
                        local_xyz_m=(
                            body.xmat.reshape(3, 3).T @ (p - body.xpos)
                        ).tolist(),
                    )
                    counts[name] = counts.get(name, 0) + 1
        if len(doc["anchors"]) < 8:
            raise RuntimeError("Insufficient visible synthetic arm anchors")
        for number in range(6):
            q = initial_positions()
            q[:7] += np.array(
                [
                    0.09 * np.sin(number),
                    0.10 * np.cos(number),
                    0.08 * np.sin(number * 0.7),
                    0.12 * np.sin(number * 0.9),
                    0.07 * number / 6,
                    0,
                    0,
                ]
            )
            q[7:] += np.array(
                [
                    -0.08 * np.sin(number),
                    0.10 * np.cos(number + 0.4),
                    -0.09 * np.sin(number * 0.8),
                    0.1 * np.cos(number),
                    0,
                    0,
                    0,
                ]
            )
            scene(q)
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            pose_id = f"demo_{number:02}"
            for anchor_id, anchor in doc["anchors"].items():
                body = kin.data.body(anchor["body"])
                point = body.xpos + body.xmat.reshape(3, 3) @ anchor["local_xyz_m"]
                uv, depth = project([point], camera_from_base, camera)
                uv = uv[0]
                if (
                    depth[0] <= 0
                    or np.any(uv < 10)
                    or np.any(uv > np.array([1270, 710]))
                ):
                    continue
                direction = point - position
                distance = np.linalg.norm(direction)
                direction /= distance
                geom = np.full(1, -1, np.int32)
                hit = mujoco.mj_ray(
                    kin.model, kin.data, position, direction, None, 1, -1, geom
                )
                if hit < 0 or abs(hit - distance) > 0.004:
                    continue
                observed = uv + rng.normal(0, 0.35, 2)
                doc["observations"].append(
                    dict(pose_id=pose_id, anchor_id=anchor_id, uv=observed.tolist())
                )
                cv2.circle(image, tuple(np.rint(uv).astype(int)), 4, (0, 210, 255), 1)
            cv2.putText(
                image,
                "SYNTHETIC VALIDATION - NOT A REAL CAPTURE",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            image_path = directory / "images" / f"{pose_id}.jpg"
            cv2.imwrite(str(image_path), image)
            doc["poses"].append(
                dict(
                    id=pose_id,
                    q_measured_rad=q.tolist(),
                    image=f"images/{pose_id}.jpg",
                    image_sha256=sha256(image_path),
                    source="synthetic FK state",
                )
            )
    dataset = directory / "dataset.json"
    save_json(dataset, doc)
    result = solve_dataset(
        dataset, directory / "results" / "candidate.json", allow_synthetic=True
    )
    estimated = np.asarray(result["T_base_from_camera"])
    errors = dict(
        translation_error_mm=float(np.linalg.norm(estimated[:3, 3] - position) * 1000),
        rotation_error_deg=float(
            np.degrees(
                Rotation.from_matrix(
                    estimated[:3, :3].T @ base_from_camera[:3, :3]
                ).magnitude()
            )
        ),
        all_rms_px=result["all_rms_px"],
        correspondences=len(doc["observations"]),
    )
    save_json(
        directory / "ground_truth_check.json",
        dict(
            **errors, T_base_from_camera_truth=base_from_camera.tolist(), synthetic=True
        ),
    )
    if errors["translation_error_mm"] > 5 or errors["rotation_error_deg"] > 0.5:
        raise AssertionError(errors)
    return errors
