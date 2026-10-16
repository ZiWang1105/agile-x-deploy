"""Fixed-head camera extrinsics from rigid arm landmarks and measured FK.

Workflow informed by Hexfellow's landmark calibration, reimplemented for NERO
and distorted raw ZED images. No robot transport is opened by this module.
"""

from __future__ import annotations
import configparser
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .contract import JOINT_NAMES
from .model import Kinematics, MODEL_PATH

RIGID_ARM = re.compile(
    r"^(left|right)_(link[1-7]|arm_base|gripper_base|gripper_flange)$"
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_camera(camera):
    if camera.get("view") != "left" or camera.get("image_space") not in (
        "raw",
        "rectified",
    ):
        raise ValueError("Require declared left raw or rectified image space")
    if camera.get("distortion_model") != "opencv_brown_conrady":
        raise ValueError("Only the declared OpenCV Brown-Conrady model is supported")
    k, distortion = (
        np.asarray(camera["K"], float),
        np.asarray(camera["distortion"], float),
    )
    if (
        k.shape != (3, 3)
        or distortion.ndim != 1
        or len(distortion) not in (4, 5, 8, 12, 14)
    ):
        raise ValueError(
            "Expected 3x3 K and 4/5/8/12/14 OpenCV distortion coefficients"
        )
    if not np.isfinite(k).all() or not np.isfinite(distortion).all():
        raise ValueError("Non-finite intrinsics")
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0:
        raise ValueError("Invalid pinhole K")
    if camera["image_space"] == "rectified" and np.any(distortion != 0):
        raise ValueError("Rectified images require zero distortion")
    size = camera.get("size_wh")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(x) is not int or x <= 0 for x in size)
    ):
        raise ValueError("size_wh must contain two positive integers")
    if not camera.get("serial") or not camera.get("source"):
        raise ValueError("Intrinsics require camera serial and source provenance")
    return k, distortion


def import_factory_intrinsics(conf, serial):
    """Import the raw LEFT HD (1280x720) factory section, without rectification."""
    parser = configparser.ConfigParser()
    parser.read(conf)
    row = parser["LEFT_CAM_HD"]
    distortion_section = "LEFT_DISTO" if parser.has_section("LEFT_DISTO") else "LEFT_CAM_HD"
    distortion_row = parser[distortion_section]
    # Do not silently truncate a rational/thin-prism model to five coefficients.
    order = ["k1", "k2", "p1", "p2", "k3"]
    if any(key in distortion_row for key in ("k4", "k5", "k6")):
        order += ["k4", "k5", "k6"]
    if any(key in distortion_row for key in ("s1", "s2", "s3", "s4")):
        order = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]
    # The basic five coefficients must be present; optional model terms default to zero.
    for key in ("fx", "fy", "cx", "cy"):
        float(row[key])
    for key in ("k1", "k2", "p1", "p2", "k3"):
        float(distortion_row[key])
    camera = dict(
        view="left",
        image_space="raw",
        distortion_model="opencv_brown_conrady",
        size_wh=[1280, 720],
        serial=str(serial),
        K=[
            [float(row["fx"]), 0, float(row["cx"])],
            [0, float(row["fy"]), float(row["cy"])],
            [0, 0, 1],
        ],
        distortion=[float(distortion_row.get(key, "0")) for key in order],
        source={
            "file": str(Path(conf).resolve()),
            "sha256": sha256(conf),
            "section": "LEFT_CAM_HD",
            "distortion_section": distortion_section,
        },
        serial_verified_against_capture_device=False,
    )
    validate_camera(camera)
    return camera


def project(points, camera_from_base, camera):
    k, distortion = validate_camera(camera)
    transform = np.asarray(camera_from_base, float)
    points = np.asarray(points, float).reshape(-1, 3)
    rvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    uv = cv2.projectPoints(points, rvec, transform[:3, 3], k, distortion)[0].reshape(
        -1, 2
    )
    depth = (points @ transform[:3, :3].T + transform[:3, 3])[:, 2]
    return uv, depth


def fit_extrinsics(points, pixels, camera, threshold_px=6.0):
    k, distortion = validate_camera(camera)
    points, pixels = np.asarray(points, float), np.asarray(pixels, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or pixels.shape != (len(points), 2)
        or len(points) < 8
    ):
        raise ValueError("Need at least eight 3D-to-2D correspondences")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(pixels).all()
        or not 0 < threshold_px < 100
    ):
        raise ValueError("Invalid correspondences or RANSAC threshold")
    spread = np.linalg.svd(points - points.mean(0), compute_uv=False)
    if spread[-1] < 1e-4 or spread[-1] / spread[0] < 1e-3:
        raise ValueError("Insufficient 3D spread: collect non-coplanar poses/landmarks")
    cv2.setRNGSeed(730)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(points),
        np.ascontiguousarray(pixels),
        k,
        distortion,
        iterationsCount=3000,
        reprojectionError=threshold_px,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 8:
        raise ValueError("Fewer than eight RANSAC inliers; review the feature matches")
    selected = inliers.ravel()

    def residual(x):
        uv = cv2.projectPoints(points[selected], x[:3], x[3:], k, distortion)[
            0
        ].reshape(-1, 2)
        return (uv - pixels[selected]).ravel()

    fit = least_squares(
        residual,
        np.r_[rvec.ravel(), tvec.ravel()],
        loss="soft_l1",
        f_scale=3,
        max_nfev=1000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    if not fit.success:
        raise ValueError("Camera pose refinement did not converge")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    transform[:3, 3] = fit.x[3:]
    uv, depths = project(points, transform, camera)
    if np.any(depths <= 0):
        raise ValueError("Some selected physical features project behind the camera")
    errors = np.linalg.norm(uv - pixels, axis=1)
    mask = errors <= threshold_px
    if mask.sum() < 8:
        raise ValueError("Refined solution has fewer than eight inliers")
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    return dict(
        T_camera_from_base=transform.tolist(),
        T_base_from_camera=np.linalg.inv(transform).tolist(),
        errors_px=errors.tolist(),
        inliers=mask.tolist(),
        all_rms_px=float(np.sqrt(np.mean(errors**2))),
        inlier_rms_px=float(np.sqrt(np.mean(errors[mask] ** 2))),
        max_px=float(errors.max()),
        point_spread_singular_values_m=spread.tolist(),
        jacobian_singular_values=singular.tolist(),
        refinement="RANSAC inliers; soft_l1 loss, 3 px scale",
        threshold_px=threshold_px,
    )


def new_dataset(camera, kind="physical"):
    validate_camera(camera)
    Kinematics()  # Ensure the model exists before fingerprinting it.
    if kind not in ("physical", "synthetic"):
        raise ValueError("Unknown dataset kind")
    return dict(
        schema_version=1,
        kind=kind,
        base_frame="base_link",
        camera_frame="zed_left_optical",
        optical_axes="x right, y down, z forward",
        units="meters_radians_pixels",
        camera=camera,
        joint_names=list(JOINT_NAMES),
        model_sha256=sha256(MODEL_PATH),
        poses=[],
        anchors={},
        observations=[],
    )


def load_dataset(path):
    document = json.loads(Path(path).read_text())
    validate_camera(document["camera"])
    if document.get("schema_version") != 1 or document.get("base_frame") != "base_link":
        raise ValueError("Unsupported calibration dataset")
    if document.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("Joint ordering mismatch")
    if document["model_sha256"] != sha256(MODEL_PATH):
        raise ValueError(
            "Robot model changed since capture; do not reinterpret saved joints"
        )
    if document.get("kind") not in ("physical", "synthetic"):
        raise ValueError("Unknown dataset provenance")
    ids = [p["id"] for p in document["poses"]]
    if any(
        not isinstance(p, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", p)
        for p in ids
    ):
        raise ValueError("Invalid pose ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate pose IDs")
    return document


def anchor_points(document):
    """Calculate each saved rigid anchor at that image's measured joint state."""
    kin = Kinematics()
    poses = {p["id"]: p for p in document["poses"]}
    points, pixels, pose_ids = [], [], []
    seen = set()
    for row in document["observations"]:
        key = (row["pose_id"], row["anchor_id"])
        if key in seen:
            raise ValueError("Duplicate pose/landmark correspondence")
        seen.add(key)
        pose = poses[row["pose_id"]]
        anchor = document["anchors"][row["anchor_id"]]
        if not RIGID_ARM.fullmatch(anchor["body"]):
            raise ValueError(
                "Landmarks must be on rigid arm housings, not fingers or cables"
            )
        q = np.asarray(pose["q_measured_rad"], float)
        if (
            q.shape != (14,)
            or not np.isfinite(q).all()
            or np.any(q < kin.limits[:, 0])
            or np.any(q > kin.limits[:, 1])
        ):
            raise ValueError("Invalid measured joint configuration")
        xyz, uv = np.asarray(anchor["local_xyz_m"], float), np.asarray(row["uv"], float)
        if (
            xyz.shape != (3,)
            or not np.isfinite(xyz).all()
            or uv.shape != (2,)
            or not np.isfinite(uv).all()
        ):
            raise ValueError("Invalid landmark coordinates")
        if np.any(uv < 0) or np.any(uv >= document["camera"]["size_wh"]):
            raise ValueError("Landmark pixel outside original image")
        kin.update(q, (None, None))
        body = kin.data.body(anchor["body"])
        points.append(body.xpos + body.xmat.reshape(3, 3) @ xyz)
        pixels.append(uv)
        pose_ids.append(pose["id"])
    return np.asarray(points), np.asarray(pixels), np.asarray(pose_ids)


def solve_dataset(path, output, allow_synthetic=False):
    path = Path(path)
    doc = load_dataset(path)
    if doc["kind"] == "synthetic" and not allow_synthetic:
        raise ValueError(
            "Synthetic data requires --allow-synthetic and is never a physical calibration"
        )
    for pose in doc["poses"]:
        image = (path.parent / pose["image"]).resolve()
        if not image.is_relative_to(path.parent.resolve()):
            raise ValueError("Image path outside dataset")
        if sha256(image) != pose["image_sha256"]:
            raise ValueError(f"Image changed since capture: {image}")
        if doc["kind"] == "physical" and not pose.get("stationary_capture", {}).get(
            "accepted"
        ):
            raise ValueError(
                "Physical pose missing accepted stationary-capture evidence"
            )
        if doc["kind"] == "physical":
            from .calibration_capture import stationary_evidence

            recorded = pose["stationary_capture"]
            checked = stationary_evidence(
                recorded["samples"],
                float(pose["camera_headers"]["X-Frame-Monotonic"]),
                recorded["max_allowed_span_rad"],
            )
            measured = recorded["samples"][checked["selected_index"]]["q"]
            if not np.allclose(measured, pose["q_measured_rad"], rtol=0, atol=1e-10):
                raise ValueError("Saved FK joints no longer match captured feedback")
    points, pixels, ids = anchor_points(doc)
    unique = np.unique(ids)
    if len(unique) < 4:
        raise ValueError("Collect at least four distinct annotated poses")
    configurations = [
        np.asarray(
            next(p for p in doc["poses"] if p["id"] == pose_id)["q_measured_rad"]
        )
        for pose_id in unique
    ]
    for i, q in enumerate(configurations):
        if any(np.max(abs(q - other)) < 0.01 for other in configurations[:i]):
            raise ValueError(
                "Near-duplicate annotated poses would leak into held-out validation; retain one capture per posture"
            )
    result = fit_extrinsics(points, pixels, doc["camera"])
    errors = np.asarray(result["errors_px"])
    result["per_pose"] = []
    result["leave_one_pose_out"] = []
    for pose_id in unique:
        mask = ids == pose_id
        result["per_pose"].append(
            dict(
                pose_id=str(pose_id),
                count=int(mask.sum()),
                rms_px=float(np.sqrt(np.mean(errors[mask] ** 2))),
            )
        )
        try:
            trained = fit_extrinsics(points[~mask], pixels[~mask], doc["camera"])
            predicted, depth = project(
                points[mask], trained["T_camera_from_base"], doc["camera"]
            )
            held = np.linalg.norm(predicted - pixels[mask], axis=1)
            result["leave_one_pose_out"].append(
                dict(
                    pose_id=str(pose_id),
                    rms_px=float(np.sqrt(np.mean(held**2))),
                    max_px=float(held.max()),
                    positive_depth=bool(np.all(depth > 0)),
                )
            )
        except ValueError as error:
            result["leave_one_pose_out"].append(
                dict(pose_id=str(pose_id), error=str(error))
            )
    result.update(
        status="candidate_not_applied",
        kind=doc["kind"],
        base_frame=doc["base_frame"],
        camera_frame=doc["camera_frame"],
        camera=doc["camera"],
        units="meters",
        dataset_sha256=sha256(path),
        model_sha256=doc["model_sha256"],
        timing_qualified=False,
        active_configuration_changed=False,
        notes="Manual feature identity and kinematic accuracy require review. Reprojection error is not independent metric accuracy.",
    )
    save_json(output, result)
    for pose in doc["poses"]:
        image = cv2.imread(str(path.parent / pose["image"]))
        mask = ids == pose["id"]
        if not mask.any():
            continue
        predicted, _ = project(
            points[mask], result["T_camera_from_base"], doc["camera"]
        )
        for observed, fitted in zip(pixels[mask], predicted):
            a, b = (
                tuple(np.rint(observed).astype(int)),
                tuple(np.rint(fitted).astype(int)),
            )
            cv2.circle(image, a, 5, (0, 220, 0), 2)
            cv2.drawMarker(image, b, (0, 80, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.line(image, a, b, (0, 180, 255), 1)
        cv2.imwrite(str(Path(output).parent / f"residual_{pose['id']}.jpg"), image)
    return result
