"""Backend-independent workshop operations; one owner controls both arms.

Participant code reaches this object only through the transport API. Simulator
truth, reset, enable, and scoring stay in the organizer process.
"""

from collections import OrderedDict
from dataclasses import asdict
import base64
import copy
import json
import secrets
from pathlib import Path
import threading
import time
import uuid
from types import SimpleNamespace

import cv2
import numpy as np

from .agent_policy import pinch_pose, tcp_pose
from .agent_sensors import ground_pixel, quantize_depth
from .contract import Action, ArmTarget, JointAction, Pose, SIDES, finite_tuple
from .controller import pose_error
from .model import Kinematics, ROOT
from .official_ik import OfficialDualIK

API_VERSION = "1.0"
TERMINAL = {"succeeded", "failed", "cancelled"}


class WorkshopError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def pose(value):
    if set(value) != {"position_m", "quaternion_xyzw"}:
        raise ValueError("Pose requires position_m and quaternion_xyzw in base_link")
    return Pose(**value)


def arm_name(arm):
    if arm not in SIDES:
        raise ValueError("arm must be left or right")
    return arm


def number(value, name, lower, upper):
    if isinstance(value, bool) or not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
    return float(value)


class WorkshopService:
    def __init__(
        self,
        backend="sim",
        task="place",
        seed=0,
        output=None,
        calibration=None,
        camera_serial=None,
        lease_s=120,
    ):
        if backend not in ("sim", "real"):
            raise ValueError("backend must be sim or real")
        self.backend, self.task, self.seed = backend, task, seed
        self.lock = threading.RLock()
        self.kin_lock = threading.RLock()
        self.done = threading.Event()
        self.camera_ready = threading.Event()
        self.owner = None
        self.lease_token = None
        self.lease_s = number(lease_s, "lease_s", 1, 3600)
        self.lease_deadline = 0.0
        self.enabled = False
        self.last_enable_s = 0.0
        self.stopped = False
        self.fault = None
        self.active = None
        self.operations = OrderedDict()
        self.requests = {}
        self.frames = OrderedDict()
        self.frame_counter = 0
        self.grippers = [None, None]
        self.target = None
        self.events = []
        self.output = Path(output or ROOT / "logs/workshop" / str(time.time_ns()))
        self.output.mkdir(parents=True, exist_ok=True)
        self.robot = self.plant = self.camera = None
        self.worker = self.camera_thread = None
        self.closed = False
        from .workshop_tasks import TASKS

        self.instruction = TASKS[task]
        try:
            if backend == "sim":
                from .runtime import Robot
                from .workshop_tasks import WorkshopSim

                self.plant = WorkshopSim(task, seed)
                self.robot = Robot(self.plant)
                for _ in range(30):
                    self.robot.tick()
            else:
                from .hardware_session import HardwareSession

                self.robot = HardwareSession("both", gripper_control=True)
            observed = self.robot.observe()
            self.kin = Kinematics()
            self.ik = OfficialDualIK(self.kin, observed.joint_positions_rad)
            self.camera_thread = threading.Thread(
                target=self._camera_loop,
                args=(calibration, camera_serial),
                name="workshop-camera",
                daemon=False,
            )
            self.camera_thread.start()
            if not self.camera_ready.wait(15):
                raise RuntimeError("Camera startup timed out")
            if self.fault:
                raise RuntimeError(self.fault)
            self.worker = threading.Thread(
                target=self._run, name="workshop-execution", daemon=False
            )
            self.worker.start()
            self.event("ready", backend=backend, task=task, seed=seed)
        except BaseException:
            self.close()
            raise

    def event(self, name, **fields):
        row = dict(time_s=time.monotonic(), event=name, **fields)
        self.events.append(row)
        del self.events[:-100]
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def capabilities(self):
        return dict(
            api_version=API_VERSION,
            arms=list(SIDES),
            cameras=["head"],
            pose_frame="base_link",
            position_units="meters",
            joint_units="radians",
            quaternion_order="xyzw",
            tool_frames=["tcp", "pinch"],
            instruction=self.instruction,
            gripper_commands=["open", "close"],
            grippers_independent=True,
            ik="AgileX QuestArmTeleop ArmIK",
            joint_control=True,
            motion_is_collision_planned=False,
        )

    def claim(self, client_id):
        if not isinstance(client_id, str) or not 1 <= len(client_id) <= 128:
            raise ValueError("client_id must contain 1..128 characters")
        with self.lock:
            if self.owner is not None:
                raise WorkshopError(
                    "OWNER_BUSY", "Another participant owns this session"
                )
            self.owner = client_id
            self.lease_token = secrets.token_urlsafe(32)
            self.lease_deadline = time.monotonic() + self.lease_s
            return dict(
                client_id=client_id, lease_s=self.lease_s, lease_token=self.lease_token
            )

    def _authorize(self, client_id, motion=False):
        if self.owner != client_id or client_id is None:
            raise WorkshopError("NOT_OWNER", "Claim the session first")
        if time.monotonic() > self.lease_deadline:
            self._hold_locked("participant lease expired", latch=True)
            self.owner = None
            raise WorkshopError(
                "LEASE_EXPIRED", "Participant lease expired; operator restart required"
            )
        self.lease_deadline = time.monotonic() + self.lease_s
        if motion and (not self.enabled or self.stopped or self.fault):
            raise WorkshopError(
                "NOT_ENABLED", self.fault or "Operator must enable a fresh session"
            )

    def enable(self):
        """Organizer-only action. Never exposed as a participant/MCP tool."""
        with self.lock:
            if self.stopped or self.fault:
                raise WorkshopError("STOPPED", self.fault or "Restart after stop")
            if (
                not self.frames
                or time.monotonic()
                - next(reversed(self.frames.values()))[1].timestamp_s
                > 0.5
            ):
                raise WorkshopError("STALE_CAMERA", "Fresh head-camera frame required")
            if self.enabled:
                return
            if self.backend == "real":
                self.robot.arm()
            self.robot.start()
            self.last_enable_s = time.monotonic()
            self.enabled = True
            self.event("enabled", backend=self.backend)

    def _raw_observe(self):
        with self.lock:
            return self.robot.observe()

    def observe(self):
        with self.lock:
            observed = self.robot.observe()
            return dict(
                observed.to_dict(),
                api_version=API_VERSION,
                pinch_poses={
                    s: asdict(pinch_pose(getattr(observed, s + "_tcp"))) for s in SIDES
                },
                session_enabled=self.enabled and not self.stopped,
                session_fault=self.fault,
            )

    def _camera_loop(self, calibration, serial):
        try:
            if self.backend == "sim":
                from .agent_sim import SimCamera

                self.camera = SimCamera(self.robot, self.plant)
            else:
                from .agent_zed import ZedStereoCamera

                self.camera = ZedStereoCamera(
                    SimpleNamespace(observe=self._raw_observe),
                    calibration
                    or ROOT
                    / "calibration/2026-10-15_zed35330610/raised_probe/results_agent_v2/candidate.json",
                    expected_serial=serial,
                )
            while not self.done.is_set():
                # ZED expects an observation object, not the public JSON envelope.
                observed, frame = self.camera.capture()
                if (
                    frame.timestamp_s < self.last_enable_s
                    and time.monotonic() - frame.timestamp_s > 0.5
                ):
                    continue  # Discard a frame captured before a slow physical arming call.
                frame.validate()
                if self.backend == "sim":
                    import mujoco

                    data = mujoco.MjData(self.plant.model)
                    with self.robot.lock:
                        data.qpos[:] = self.plant.data.qpos
                    mujoco.mj_forward(self.plant.model, data)
                    view = mujoco.MjvCamera()
                    view.lookat[:] = [0.32, 0, 0.23]
                    view.distance, view.azimuth, view.elevation = 1.3, 130, -30
                    self.camera.renderer.update_scene(data, camera=view)
                    rgb = self.camera.renderer.render()
                    _, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    )
                    self.scene_jpeg = encoded.tobytes()
                with self.lock:
                    self.frame_counter += 1
                    self.frames[str(self.frame_counter)] = (observed, frame)
                    while len(self.frames) > 64:
                        self.frames.popitem(last=False)
                self.camera_ready.set()
                self.done.wait(0.2)
        except Exception as error:
            with self.lock:
                self.fault = f"Camera failure: {error}"
                if self.enabled:
                    self._hold_locked(self.fault, latch=True)
            self.camera_ready.set()
        finally:
            if self.camera is not None:
                self.camera.close()

    def capture(self):
        with self.lock:
            if not self.frames:
                raise WorkshopError("NO_FRAME", "Camera has no frame")
            frame_id = next(reversed(self.frames))
            observed, frame = self.frames[frame_id]
            if time.monotonic() - frame.timestamp_s > 0.5:
                raise WorkshopError("STALE_CAMERA", "Camera acquisition is stale")
        rgb_ok, rgb = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        depth_ok, depth = cv2.imencode(".png", quantize_depth(frame.depth_m))
        if not rgb_ok or not depth_ok:
            raise RuntimeError("Frame encoding failed")
        frame_dir = self.output / "policy_frames"
        frame_dir.mkdir(exist_ok=True)
        if not (frame_dir / f"{frame_id}.json").exists():
            (frame_dir / f"{frame_id}_rgb.png").write_bytes(rgb.tobytes())
            (frame_dir / f"{frame_id}_depth.png").write_bytes(depth.tobytes())
            (frame_dir / f"{frame_id}.json").write_text(
                json.dumps(dict(frame.metadata(), robot=observed.to_dict())) + "\n"
            )
        return dict(
            frame_id=frame_id,
            camera="head",
            **frame.metadata(),
            rgb_png_base64=base64.b64encode(rgb).decode(),
            depth_png_base64=base64.b64encode(depth).decode(),
            depth_png_scale_m=0.001,
            observation_id=observed.observation_id,
            robot_timestamp_s=observed.timestamp_s,
            robot_camera_skew_s=abs(frame.timestamp_s - observed.timestamp_s),
        )

    def deproject(self, frame_id, pixel_uv):
        with self.lock:
            if str(frame_id) not in self.frames:
                raise WorkshopError(
                    "UNKNOWN_FRAME", "Frame expired from camera cache; capture again"
                )
            _, frame = self.frames[str(frame_id)]
        point, _ = ground_pixel(frame, pixel_uv, [0, 0, 0])
        return dict(
            position_m=point.tolist(),
            frame="base_link",
            frame_id=str(frame_id),
            acquisition_timestamp_s=frame.timestamp_s,
            age_s=time.monotonic() - frame.timestamp_s,
        )

    def fk(self, arm, joints_rad, tool_frame="tcp"):
        index = SIDES.index(arm_name(arm))
        if tool_frame not in ("tcp", "pinch"):
            raise ValueError("tool_frame must be tcp or pinch")
        q = np.asarray(self.observe()["joint_positions_rad"])
        q[index * 7 : index * 7 + 7] = finite_tuple(joints_rad, 7, "joints_rad")
        with self.kin_lock:
            self._joint_bounds(q)
            self.kin.update(q)
            value = self.kin.pose(arm)
        return dict(
            pose=asdict(pinch_pose(value) if tool_frame == "pinch" else value),
            tool_frame=tool_frame,
            frame="base_link",
        )

    def _joint_bounds(self, q):
        if np.any(q < self.kin.limits[:, 0]) or np.any(q > self.kin.limits[:, 1]):
            raise ValueError("Joint target outside model bounds")

    def ik_solve(self, arm, target, seed_rad=None, tool_frame="tcp"):
        index = SIDES.index(arm_name(arm))
        target = pose(target)
        if tool_frame not in ("tcp", "pinch"):
            raise ValueError("tool_frame must be tcp or pinch")
        tcp = tcp_pose(target) if tool_frame == "pinch" else target
        observed = self.observe()
        q = np.asarray(observed["joint_positions_rad"])
        seed = (
            q[index * 7 : index * 7 + 7]
            if seed_rad is None
            else finite_tuple(seed_rad, 7, "seed_rad")
        )
        with self.kin_lock:
            try:
                result = self.ik.solve(arm, tcp, seed)
            except RuntimeError as error:
                return dict(success=False, joints_rad=None, solver_status=str(error))
            q[index * 7 : index * 7 + 7] = result
            self._joint_bounds(q)
            self.kin.update(q)
            residual = pose_error(self.kin.pose(arm), tcp)
        return dict(
            success=True,
            joints_rad=result.tolist(),
            position_error_m=float(np.linalg.norm(residual[:3])),
            orientation_error_rad=float(np.linalg.norm(residual[3:])),
            solver_status=self.ik.last_status[arm],
            within_model_limits=True,
            collision_checked=False,
            physical_limits_checked=False,
        )

    def submit(
        self,
        client_id,
        request_id,
        steps,
        timeout_s=10,
        position_tolerance_m=0.005,
        orientation_tolerance_rad=0.05,
    ):
        timeout_s = number(timeout_s, "timeout_s", 0.1, 300)
        pt = number(position_tolerance_m, "position_tolerance_m", 0.0001, 0.1)
        rt = number(orientation_tolerance_rad, "orientation_tolerance_rad", 0.001, 1)
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("request_id is required (1..128 characters)")
        if not steps or len(steps) > 100:
            raise ValueError("Expected 1..100 waypoints")
        fingerprint = json.dumps(
            [steps, timeout_s, pt, rt], sort_keys=True, allow_nan=False
        )
        with self.lock:
            self._authorize(client_id, motion=True)
            key = (client_id, request_id)
            if key in self.requests:
                old_fingerprint, op_id = self.requests[key]
                if old_fingerprint != fingerprint:
                    raise WorkshopError(
                        "REQUEST_CONFLICT",
                        "request_id was used with different arguments",
                    )
                return self._public_operation(self.operations[op_id])
            if self.active is not None:
                raise WorkshopError(
                    "BUSY",
                    "Wait or cancel the active operation before submitting another",
                )
            for step in steps:
                self._validate_step(step)
            op_id = uuid.uuid4().hex
            op = dict(
                operation_id=op_id,
                request_id=request_id,
                state="running",
                progress=0,
                steps=copy.deepcopy(steps),
                step_index=0,
                timeout_s=timeout_s,
                position_tolerance_m=pt,
                orientation_tolerance_rad=rt,
                started_s=time.monotonic(),
                settled_since=None,
                action=None,
                residuals={},
            )
            self.operations[op_id] = op
            self.requests[key] = (fingerprint, op_id)
            self.active = op_id
            self.event(
                "operation_start",
                operation_id=op_id,
                request_id=request_id,
                steps=steps,
            )
            return self._public_operation(op)

    def _validate_step(self, step):
        if set(step) - {"poses", "joints", "grippers", "tool_frame"}:
            raise ValueError("Unknown command fields")
        if not any(step.get(k) for k in ("poses", "joints", "grippers")):
            raise ValueError("Empty command")
        if step.get("poses") and step.get("joints"):
            raise ValueError("Choose pose or joint targets, not both")
        if step.get("tool_frame", "tcp") not in ("tcp", "pinch"):
            raise ValueError("tool_frame must be tcp or pinch")
        for side, value in step.get("poses", {}).items():
            arm_name(side)
            pose(value)
        for side, value in step.get("joints", {}).items():
            arm_name(side)
            q = np.asarray(self.robot.observe().joint_positions_rad)
            i = SIDES.index(side) * 7
            q[i : i + 7] = finite_tuple(value, 7, "joints_rad")
            self._joint_bounds(q)
        for side, value in step.get("grippers", {}).items():
            arm_name(side)
            if type(value) is not int or value not in (0, 1):
                raise ValueError(
                    "gripper command must be integer 0 (close) or 1 (open)"
                )

    def _action(self, step, observed):
        arms = {s: ArmTarget(getattr(observed, s + "_tcp"), None) for s in SIDES}
        for s, value in step.get("poses", {}).items():
            target = pose(value)
            arms[s] = ArmTarget(
                tcp_pose(target) if step.get("tool_frame") == "pinch" else target, None
            )
        q = np.asarray(observed.joint_positions_rad).copy()
        if step.get("joints"):
            with self.kin_lock:
                for s, value in step["joints"].items():
                    i = SIDES.index(s) * 7
                    q[i : i + 7] = value
                self.kin.update(q)
                arms = {s: ArmTarget(self.kin.pose(s), None) for s in SIDES}
        for s, value in step.get("grippers", {}).items():
            self.grippers[SIDES.index(s)] = value
        arms = {
            s: ArmTarget(arms[s].pose, self.grippers[i]) for i, s in enumerate(SIDES)
        }
        return (
            JointAction(**arms, joint_positions_rad=tuple(q))
            if step.get("joints")
            else Action(**arms)
        )

    @staticmethod
    def _public_operation(op):
        return copy.deepcopy(
            {
                k: v
                for k, v in op.items()
                if k not in {"action", "steps", "settled_since"}
            }
        )

    def operation(self, operation_id):
        with self.lock:
            if operation_id not in self.operations:
                raise WorkshopError("UNKNOWN_OPERATION", "Unknown operation ID")
            return self._public_operation(self.operations[operation_id])

    def _hold_locked(self, reason, latch=False):
        self.target = None
        self.grippers = [None, None]
        if self.active:
            op = self.operations[self.active]
            op.update(state="cancelled", reason=reason)
            self.active = None
        if latch:
            self.stopped = True
        if self.robot is not None:
            self.robot.hold()
        self.event("hold", reason=reason, latched=latch)

    def hold(self, client_id):
        with self.lock:
            self._authorize(client_id)
            self._hold_locked("participant HOLD")
            return dict(state="holding", motors_disabled=False)

    def cancel(self, client_id, operation_id):
        with self.lock:
            self._authorize(client_id)
            op = self.operation(operation_id)
            if op["state"] not in TERMINAL:
                self._hold_locked("operation cancelled")
            return self.operation(operation_id)

    def release(self, client_id):
        with self.lock:
            self._authorize(client_id)
            self._hold_locked("participant released session")
            self.owner = None
            return dict(released=True)

    def operator_stop(self):
        with self.lock:
            self._hold_locked("operator STOP", latch=True)

    def _run(self):
        while not self.done.is_set():
            if self.fault:
                self.done.wait(0.03)
                continue
            try:
                with self.lock:
                    observed = self.robot.observe()
                    if observed.mode == "fault":
                        raise RuntimeError(observed.reason)
                    if self.owner and time.monotonic() > self.lease_deadline:
                        self._hold_locked("participant lease expired", latch=True)
                        self.owner = None
                    if self.enabled and not self.stopped:
                        if self.active:
                            op = self.operations[self.active]
                            step = op["steps"][op["step_index"]]
                            if op["action"] is None:
                                op["action"] = self._action(step, observed)
                            self.target = op["action"]
                            residuals = {
                                s: pose_error(
                                    getattr(observed, s + "_tcp"),
                                    getattr(self.target, s).pose,
                                )
                                for s in step.get("poses", {})
                            }
                            reached = all(
                                np.linalg.norm(e[:3]) <= op["position_tolerance_m"]
                                and np.linalg.norm(e[3:])
                                <= op["orientation_tolerance_rad"]
                                for e in residuals.values()
                            )
                            if step.get("joints"):
                                reached &= (
                                    max(
                                        abs(
                                            np.asarray(observed.joint_positions_rad)
                                            - self.target.joint_positions_rad
                                        )
                                    )
                                    <= 0.01
                                )
                            grip_reports = {}
                            for s in step.get("grippers", {}):
                                state = observed.gripper_states[SIDES.index(s)]
                                reached &= state.phase == "reached" or (
                                    step["grippers"][s] == 0
                                    and state.phase == "stalled"
                                )
                                grip_reports[s] = asdict(state)
                            op["residuals"] = {
                                s: dict(
                                    position_m=float(np.linalg.norm(e[:3])),
                                    orientation_rad=float(np.linalg.norm(e[3:])),
                                )
                                for s, e in residuals.items()
                            }
                            op["gripper_feedback"] = grip_reports
                            now = time.monotonic()
                            op["settled_since"] = (
                                (op["settled_since"] or now) if reached else None
                            )
                            if op["settled_since"] and now - op["settled_since"] >= 0.1:
                                op["step_index"] += 1
                                op["progress"] = op["step_index"] / len(op["steps"])
                                op["settled_since"] = op["action"] = None
                                if op["step_index"] == len(op["steps"]):
                                    op.update(
                                        state="succeeded",
                                        completion="measured endpoint/settled gripper; not a grasp-success claim",
                                    )
                                    self.active = None
                                    self.robot.hold()
                                    fresh = self.robot.observe()
                                    self.target = Action(
                                        *(
                                            ArmTarget(
                                                getattr(fresh, s + "_tcp"),
                                                self.grippers[i],
                                            )
                                            for i, s in enumerate(SIDES)
                                        )
                                    )
                                    self.event(
                                        "operation_end", **self._public_operation(op)
                                    )
                            elif now - op["started_s"] > op["timeout_s"]:
                                self._hold_locked("operation timed out")
                                op.update(state="failed", reason="TIMEOUT")
                        if self.target is not None:
                            fresh = self.robot.observe()
                            self.robot.submit_target(self.target, fresh.observation_id)
            except Exception as error:
                with self.lock:
                    self.fault = str(error)
                    op = self.operations.get(self.active)
                    try:
                        self._hold_locked(self.fault, latch=True)
                    except Exception as hold_error:
                        self.fault += f"; HOLD unconfirmed: {hold_error}"
                    if op is not None:
                        op.update(state="failed", reason=self.fault)
            self.done.wait(0.03)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.done.set()
        for thread in (self.worker, self.camera_thread):
            if thread is not None:
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError(f"{thread.name} did not exit")
        if self.robot is not None:
            self.robot.close()
        if self.plant is not None:
            (self.output / "score.json").write_text(
                json.dumps(self.plant.evaluate(), indent=2) + "\n"
            )
            (self.output / "truth.json").write_text(json.dumps(self.plant.truth) + "\n")
