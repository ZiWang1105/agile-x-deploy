"""Single-arm calibration routes through the existing robot/official IK interface.

Planning is offline. Physical execution requires a separately reviewed route and
an explicit terminal arming action. No extra speed/acceleration limits are added.
"""

from collections import deque
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_calibration import save_json, sha256, load_dataset
from .contract import Action, ArmTarget, Pose, SIDES, JOINT_NAMES
from .controller import pose_error
from .model import Kinematics, initial_positions, MODEL_PATH


def plan_route(
    output, side="left", seed=None, translation_m=0.06, rotation_deg=15,
    lift_m=0.0, pose_count=9,
):
    from .official_ik import OfficialDualIK

    if not 0 < translation_m < 0.5 or not 0 < rotation_deg < 90:
        raise ValueError("Invalid requested route extent")
    if not 0 <= lift_m < 0.5 or pose_count not in (9, 15):
        raise ValueError("Use a finite lift in [0, 0.5) m and 9 or 15 capture poses")
    kin = Kinematics()
    q = initial_positions()
    source = "nominal_simulation_only"
    if seed:
        saved = json.loads(Path(seed).read_text())
        if saved.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("Seed joint ordering mismatch")
        q = np.asarray(saved["positions"], float)
        source = saved.get("source", "unverified")
    if (
        q.shape != (14,)
        or not np.isfinite(q).all()
        or np.any(q < kin.limits[:, 0])
        or np.any(q > kin.limits[:, 1])
    ):
        raise ValueError("Seed must contain 14 valid model joint positions")
    start = q.copy()
    kin.update(q, (None, None))
    origin = kin.pose(side)
    base = Action(*(ArmTarget(kin.pose(s)) for s in SIDES))
    ik = OfficialDualIK(kin, q)
    rows = []
    idx = slice(0, 7) if side == "left" else slice(7, 14)
    # Translate in three dimensions and change wrist orientation to expose rigid
    # landmarks at different depths. This is an adjustable route, not a speed cap.
    offsets = [
        ([0, 0, 0], [0, 0, 0]),
        ([1, -1, 0], [0, 0, 0]),
        ([1, 1, 0], [0, 0, 0]),
        ([0, 0, 1], [0, 0, 0]),
        ([1, 0, 1], [0, 0, 0]),
        ([0.5, -0.5, 0.5], [0, 1, 0]),
        ([0.5, 0.5, 0.5], [0, -1, 0]),
        ([1, 0, 0.5], [0, 0, 1]),
        ([0.5, 0, 1], [0, 0, -1]),
    ]
    if pose_count == 15:
        offsets.extend([
            ([0, -1, 0], [0, 0, 0]),
            ([0, 1, 0], [0, 0, 0]),
            ([0, 0, -1], [0, 0, 0]),
            ([0, 0, 0], [1, 0, 0]),
            ([0, 0, 0], [-1, 0, 0]),
            ([0, 0, 0.5], [0, 0, 1]),
        ])
    entries = []
    if lift_m > 0:
        # Explicit approach targets, excluded from calibration captures. Their
        # spacing is a path design choice, not a velocity/acceleration limit.
        for height in np.linspace(0, lift_m, int(np.ceil(lift_m / 0.05)) + 1)[:-1]:
            entries.append((np.asarray(origin.position_m) + [0, 0, height],
                            [0, 0, 0], [0, 0, 0], False))
    center = np.asarray(origin.position_m) + [0, 0, lift_m]
    entries.extend((center, delta, rot, True) for delta, rot in offsets)
    for number, (center_position, delta, rot, capture) in enumerate(entries, 1):
        # The recorded posture can be near a workspace boundary. Try alternate
        # sampling poses rather than weakening the official IK or its limits.
        # Each attempt starts from the last accepted state, never a failed solve.
        attempts = []
        seen = set()
        accepted = False
        for scale, x_sign in (
            (1.0, 1), (1.0, -1), (0.5, 1), (0.5, -1), (0.25, 1), (0.25, -1)
        ):
            translation = translation_m * scale * np.asarray(delta, dtype=float)
            translation[0] *= x_sign
            rotation = np.radians(rotation_deg) * scale * np.asarray(rot)
            key = tuple(np.r_[translation, rotation])
            if key in seen:
                continue
            seen.add(key)
            target = Pose(
                center_position + translation,
                (Rotation.from_rotvec(rotation)
                 * Rotation.from_quat(origin.quaternion_xyzw)).as_quat(),
            )
            trial = q.copy()
            attempt = dict(
                scale=scale,
                x_sign=x_sign,
                translation_offset_m=translation.tolist(),
                rotation_offset_deg=np.degrees(rotation).tolist(),
                accepted=False,
            )
            attempts.append(attempt)
            try:
                for iteration in range(120):
                    trial[idx] = ik.solve(side, target, trial[idx])
                    kin.update(trial, (None, None))
                    error = pose_error(kin.pose(side), target)
                    if np.linalg.norm(error[:3]) < 0.002 and np.linalg.norm(
                        error[3:]
                    ) < np.radians(1):
                        break
                attempt.update(
                    iterations=iteration + 1,
                    position_error_m=float(np.linalg.norm(error[:3])),
                    orientation_error_deg=float(np.degrees(np.linalg.norm(error[3:]))),
                )
                if attempt["position_error_m"] >= 0.002 or attempt["orientation_error_deg"] >= 1:
                    attempt["reason"] = "IK residual exceeds unchanged 2 mm / 1 degree gate"
                    continue
                if capture and any(
                    np.max(np.abs(trial - row["q_preview_rad"])) < 0.01
                    for row in rows if row.get("capture", True)
                ):
                    attempt["reason"] = "Near-duplicate calibration posture"
                    continue
            except RuntimeError as exc:
                attempt["reason"] = str(exc)
                continue
            q = trial
            attempt["accepted"] = accepted = True
            if len(attempts) > 1:
                print(f"Waypoint {number}: selected offset {translation.tolist()} m, "
                      f"rotation {np.degrees(rotation).tolist()} degrees "
                      f"after {len(attempts)} attempts", flush=True)
            break
        if not accepted:
            raise ValueError(
                f"Waypoint {number} has no converged, distinct candidate; "
                f"no plan written. Attempts: {json.dumps(attempts)}"
            )
        targets = {s: getattr(base, s) for s in SIDES}
        targets[side] = ArmTarget(target)
        rows.append(
            dict(
                id=f"waypoint_{number:02}",
                phase="capture" if capture else "approach",
                capture=capture,
                action=Action(**targets).to_dict(),
                q_preview_rad=q.tolist(),
                position_error_m=float(np.linalg.norm(error[:3])),
                orientation_error_deg=float(np.degrees(np.linalg.norm(error[3:]))),
                sampling_attempts=attempts,
            )
        )
    result = dict(
        schema_version=1,
        side=side,
        joint_names=list(JOINT_NAMES),
        start_q_rad=start.tolist(),
        seed_source=source,
        seed_sha256=sha256(seed) if seed else None,
        model_sha256=sha256(MODEL_PATH),
        waypoints=rows,
        solver="official AgileX ArmIK",
        requested_translation_m=translation_m,
        requested_rotation_deg=rotation_deg,
        requested_lift_m=lift_m,
        capture_pose_count=pose_count,
        capture_center_position_m=center.tolist(),
        sampling_policy="try requested offset, then mirror X, then half/quarter extents; retain original IK tolerances",
        review="kinematic preview only; contacts, room obstacles, field of view and occlusion are not qualified",
        hardware_commands_sent=0,
    )
    save_json(output, result)
    from .calibration_view import ModelView
    import cv2

    renderer = ModelView()
    try:
        panels = []
        for row in rows:
            frame = renderer.image(row["q_preview_rad"])
            frame = cv2.resize(frame, (384, 288))
            cv2.putText(
                frame, f'{row["id"]} {row["phase"]}', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            panels.append(frame)
        while len(panels) % 3:
            panels.append(np.full_like(panels[0], 255))
        sheet = np.vstack(
            [np.hstack(panels[i : i + 3]) for i in range(0, len(panels), 3)]
        )
        cv2.imwrite(str(Path(output).with_suffix(".jpg")), sheet)
    finally:
        renderer.close()
    return result


def check_route_start(measured, planned):
    """Reject a changed starting posture before arming, with an actionable error."""
    difference = np.abs(np.asarray(measured) - np.asarray(planned))
    index = int(np.argmax(difference))
    if difference[index] > 0.05:
        raise ValueError(
            f"Start posture changed: {JOINT_NAMES[index]} differs by "
            f"{np.degrees(difference[index]):.2f} degrees "
            "(allowed 0.05 rad / 2.86 degrees). No arming or route execution "
            "was requested. Regenerate the approach from the current disabled-arm "
            "feedback; disabling motors can change the posture."
        )


def run_route(
    plan_path,
    mode="sim",
    dataset=None,
    stream_url="http://127.0.0.1:8766",
    reviewed=False,
):
    if not Path(plan_path).is_file():
        raise ValueError(
            f"Plan file does not exist: {plan_path}. Run 'plan' successfully before 'run'; "
            "a failed plan does not produce an executable route."
        )
    plan = json.loads(Path(plan_path).read_text())
    if plan["joint_names"] != list(JOINT_NAMES) or plan["model_sha256"] != sha256(
        MODEL_PATH
    ):
        raise ValueError("Route model/joint ordering mismatch")
    side = plan["side"]
    if mode == "real":
        if (
            not reviewed
            or plan["seed_source"] != "physical_CAN_receive_only"
            or dataset is None
        ):
            raise ValueError(
                "Real execution needs --reviewed-plan, a physical feedback seed, and a physical dataset"
            )
        document = load_dataset(dataset)
        if document["kind"] != "physical":
            raise ValueError("Real execution cannot use a synthetic dataset")
        if document["camera"]["image_space"] != "raw" or document["camera"][
            "size_wh"
        ] != [1280, 720]:
            raise ValueError(
                "Physical UVC route capture requires raw LEFT 1280x720 intrinsics"
            )
        # Fail before arming if the capture server is absent or still running an
        # old build without atomic frame metadata.
        from urllib.parse import urlparse
        from urllib.request import urlopen

        parsed = urlparse(stream_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("Physical capture requires a local camera clock")
        with urlopen(
            stream_url.rstrip("/") + "/snapshot.jpg?view=left", timeout=2
        ) as response:
            if (
                response.headers.get("X-Frame-Timestamp-Source")
                != "host_read_completion_not_exposure"
            ):
                raise ValueError("Restart the updated ZED stream before arming")
            if (
                response.headers.get("X-Camera-View") != "left"
                or response.headers.get("X-Image-Space") != "raw"
            ):
                raise ValueError("Camera source does not match the dataset")
        if (
            input(
                f"{side} arm route: verify the entire path/workspace. Type ARM to enable and execute: "
            ).strip()
            != "ARM"
        ):
            raise ValueError("Arming cancelled")
        from .hardware_session import HardwareSession

        robot = HardwareSession(side)
    else:
        from .runtime import Robot
        from .backends import SimBackend

        robot = Robot(SimBackend(joint_positions=plan["start_q_rad"]))
    rows = []
    try:
        observed = robot.observe()
        check_route_start(observed.joint_positions_rad, plan["start_q_rad"])
        if mode == "real":
            robot.arm()
        for target in plan["waypoints"]:
            action = Action.from_dict(target["action"])
            began = time.monotonic()
            history = deque()
            for tick in range(6000):
                observed = robot.observe()
                if observed.mode == "fault":
                    raise RuntimeError(observed.reason)
                robot.submit_target(action, observed.observation_id)
                if mode == "sim":
                    robot.tick()
                else:
                    time.sleep(0.01)
                observed = robot.observe()
                error = pose_error(
                    getattr(observed, f"{side}_tcp"), getattr(action, side).pose
                )
                # Measured stability is a capture-quality gate, not a motion limit.
                history.append(np.asarray(observed.joint_positions_rad))
                if len(history) > 50:
                    history.popleft()
                stable = (
                    len(history) == 50
                    and np.max(np.ptp(np.asarray(history), axis=0)) < 0.003
                )
                if (
                    stable
                    and np.linalg.norm(error[:3]) < 0.003
                    and np.linalg.norm(error[3:]) < np.radians(1.5)
                ):
                    break
                if time.monotonic() - began > 90:
                    raise RuntimeError("Waypoint timeout; HOLD requested")
            else:
                raise RuntimeError("Waypoint failed to settle; HOLD requested")
            robot.hold()
            row = dict(
                waypoint=target["id"],
                q_measured_rad=list(observed.joint_positions_rad),
                position_error_m=float(np.linalg.norm(error[:3])),
                mode=mode,
            )
            row["capture_requested"] = target.get("capture", True)
            if mode == "real" and row["capture_requested"]:
                from .calibration_capture import capture_pose

                def sample():
                    value = robot.observe()
                    if value.mode == "fault":
                        raise RuntimeError(value.reason)
                    return dict(
                        timestamp_s=value.timestamp_s,
                        q=list(value.joint_positions_rad),
                        joint_names=list(value.joint_names),
                        source="physical_control_owner_feedback",
                    )

                row["capture_id"] = capture_pose(dataset, stream_url, sample)["id"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        try:
            robot.hold()
        finally:
            robot.close()
    report = dict(
        mode=mode,
        results=rows,
        plan_sha256=sha256(plan_path),
        notes="No automatic return motion. Hardware close requests HOLD and leaves motors enabled.",
    )
    save_json(
        Path(plan_path).with_name(f"rehearsal_{mode}_{time.time_ns()}.json"), report
    )
    return report
