"""Backend-independent workshop operations; one owner controls both arms.

Participant code reaches this object only through the transport API. Simulator
truth, reset, enable, and scoring stay in the organizer process.
"""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import base64
import copy
import json
import multiprocessing as mp
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
from .contract import Action, ArmTarget, JointAction, MixedAction, Pose, SIDES, finite_tuple
from .controller import pose_error
from .model import Kinematics, ROOT
from .official_ik import OfficialDualIK
from .motion_limits import bounded_hold_reference
from .workshop_planning import close_planner, initialize_planner, planner_ready, plan_step

API_VERSION = "1.0"
TERMINAL = {"succeeded", "failed", "cancelled"}
CAPTURED_FRAME_LIMIT = 32
POSITION_TOLERANCE_M = .005
ORIENTATION_TOLERANCE_RAD = .05


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
        gravity_comp=False,
    ):
        if backend not in ("sim", "real"):
            raise ValueError("backend must be sim or real")
        self.backend, self.task, self.seed = backend, task, seed
        if type(gravity_comp) is not bool:
            raise ValueError("gravity_comp must be a boolean")
        self.gravity_comp = gravity_comp
        self.lock = threading.RLock()
        # Camera publication and preview must not wait behind arm/mode RPCs.
        self.frames_lock = threading.RLock()
        self.frame_available = threading.Condition(self.frames_lock)
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
        # Model inference can outlast the live ring buffer. Retain frames handed
        # to policies independently; dashboard polling must not evict them.
        self.captured_frames = OrderedDict()
        self.frame_counter = 0
        self.grippers = [None, None]
        self.target = None
        self.events = []
        self.policy_events = []
        self.policy_event_sequence = 0
        self.output = Path(output or ROOT / "logs/workshop" / str(time.time_ns()))
        self.output.mkdir(parents=True, exist_ok=True)
        self.robot = self.plant = self.camera = None
        self.worker = self.camera_thread = None
        self.planner_pool = None
        self.planning_generation = mp.get_context("spawn").Value("Q", 0, lock=False)
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

                self.robot = HardwareSession("both", gripper_control=True, gravity_comp=gravity_comp)
            observed = self.robot.observe()
            self.kin = Kinematics()
            self.action_kin = Kinematics()
            self.ik = OfficialDualIK(self.kin, observed.joint_positions_rad)
            self.planner_pool = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"),
                initializer=initialize_planner,
                initargs=(observed.joint_positions_rad, self.planning_generation))
            if not self.planner_pool.submit(planner_ready).result(timeout=30):
                raise RuntimeError("Cartesian planner initialization failed")
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
            workshop=dict(name="Workshop on Agentic Robotics",
                          url="https://agentic-robotics-workshop.github.io/"),
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
            motion_planner="online_cartesian_waypoints",
            motion_path=dict(position="linear", orientation="validated_shortest_or_complementary_arc",
                             preflight="sampled_official_ik_residuals_and_joint_bounds",
                             feedback="measured", ik="official_agilex",
                             lookahead="joint_stopping_horizon",
                             translation_resolution_m=.01, rotation_resolution_rad=.05),
            differential_ik=False,
            motion_is_collision_planned=False,
            gravity_comp=self.gravity_comp,
            idle_mode="hand_guiding" if self.gravity_comp else "hold",
            normal_policy_exit="hand_guiding" if self.gravity_comp else "hold",
            fault_behavior="latched; no automatic disable or mode recovery",
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
            with self.frames_lock:
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
                with self.kin_lock:
                    self.kin.limits[:] = self.robot.motion_limits()["position"]
            elif self.gravity_comp:
                self.robot.enter_hand_guiding()
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
                if self.backend == "real":
                    # This worker never observes the robot: enable and other
                    # slow RPCs cannot block RGB-D acquisition or publication.
                    observed, frame = None, self.camera.capture_frame()
                else:
                    observed, frame = self.camera.capture()
                if (
                    frame.timestamp_s < self.last_enable_s
                    and time.monotonic() - frame.timestamp_s > 0.5
                ):
                    continue  # Discard a frame captured before a slow arm/mode transition.
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
                with self.frame_available:
                    self.frame_counter += 1
                    self.frames[str(self.frame_counter)] = (observed, frame)
                    while len(self.frames) > 64:
                        self.frames.popitem(last=False)
                    self.frame_available.notify_all()
                self.camera_ready.set()
                self.done.wait(0.03 if self.backend == "real" else 0.2)
        except Exception as error:
            with self.lock:
                self.fault = f"Camera failure: {error}"
                if self.enabled:
                    self._hold_locked(self.fault, latch=True)
            self.camera_ready.set()
        finally:
            if self.camera is not None:
                self.camera.close()

    def capture(self, *, retain=True):
        # Policy requests match fresh robot feedback to the latest camera frame.
        # Dashboard preview does not require robot feedback and stays live during
        # enable. Neither path asks the camera worker to acquire a frame on demand.
        match_feedback = self.backend == "real" and retain
        deadline = time.monotonic() + 1.
        while True:
            observed = self._raw_observe() if match_feedback else None
            with self.frame_available:
                if not self.frames:
                    raise WorkshopError("NO_FRAME", "Camera has no frame")
                frame_id = next(reversed(self.frames))
                stored_observed, frame = self.frames[frame_id]
                if not match_feedback:
                    observed = stored_observed
                if time.monotonic() - frame.timestamp_s > 0.5:
                    raise WorkshopError("STALE_CAMERA", "Camera acquisition is stale")
                if match_feedback and abs(observed.timestamp_s - frame.timestamp_s) > .1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WorkshopError("CAMERA_SYNC", "No robot feedback matched to latest camera frame within 100 ms")
                    self.frame_available.wait(min(.05, remaining))
                    continue
                if retain:
                    self.captured_frames[frame_id] = (observed, frame)
                    self.captured_frames.move_to_end(frame_id)
                    while len(self.captured_frames) > CAPTURED_FRAME_LIMIT:
                        self.captured_frames.popitem(last=False)
                break
        rgb_ok, rgb = cv2.imencode(".png", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        depth_ok, depth = cv2.imencode(".png", quantize_depth(frame.depth_m))
        if not rgb_ok or not depth_ok:
            raise RuntimeError("Frame encoding failed")
        frame_dir = self.output / "policy_frames"
        frame_dir.mkdir(exist_ok=True)
        if retain and not (frame_dir / f"{frame_id}.json").exists():
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
            observation_id=observed.observation_id if observed is not None else None,
            robot_timestamp_s=observed.timestamp_s if observed is not None else None,
            robot_camera_skew_s=abs(frame.timestamp_s - observed.timestamp_s) if observed is not None else None,
        )

    def deproject(self, frame_id, pixel_uv):
        with self.frames_lock:
            stored = self.captured_frames.get(str(frame_id), self.frames.get(str(frame_id)))
            if stored is None:
                raise WorkshopError(
                    "UNKNOWN_FRAME", "Frame is unknown or older than the last 32 policy captures; capture again"
                )
            _, frame = stored
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
        q = bounded_hold_reference(observed["joint_positions_rad"], self.kin.limits)
        seed = (
            q[index * 7 : index * 7 + 7]
            if seed_rad is None
            else finite_tuple(seed_rad, 7, "seed_rad")
        )
        with self.kin_lock:
            try:
                result = self.ik.solve(arm, tcp, seed)
            except RuntimeError as error:
                return dict(success=False, solver_success=False, joints_rad=None, solver_status=str(error))
            q[index * 7 : index * 7 + 7] = result
            self._joint_bounds(q)
            self.kin.update(q)
            residual = pose_error(self.kin.pose(arm), tcp)
        return dict(
            success=bool(np.linalg.norm(residual[:3]) <= POSITION_TOLERANCE_M
                         and np.linalg.norm(residual[3:]) <= ORIENTATION_TOLERANCE_RAD),
            solver_success=True,
            joints_rad=result.tolist(),
            position_error_m=float(np.linalg.norm(residual[:3])),
            orientation_error_rad=float(np.linalg.norm(residual[3:])),
            position_tolerance_m=POSITION_TOLERANCE_M,
            orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
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
        position_tolerance_m=POSITION_TOLERANCE_M,
        orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
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

    def _prepare_action_locked(self, op, step, observed):
        if not step.get("poses"):
            op["action"] = self._action(step, observed)
            return
        future = op.get("_plan_future")
        if future is None:
            op["_plan_start"] = observed
            op["planning"] = dict(state="running")
            op["_plan_future"] = self.planner_pool.submit(
                plan_step, self.planning_generation.value, copy.deepcopy(step),
                observed.joint_positions_rad, self.kin.limits.copy(),
                op["position_tolerance_m"], op["orientation_tolerance_rad"])
            return
        if not future.done():
            return
        try:
            reports = future.result()
        except Exception as error:
            self._hold_locked("Cartesian planning failed", preserve_grippers=True)
            op.update(state="failed", reason="PLANNING_FAILED",
                      planning=dict(state="failed", error=str(error), report=getattr(error, "report", None)))
            self.event("operation_end", **self._public_operation(op))
            return
        # A validated path belongs to its measured start. Replan if the arm
        # changed while validation ran, using the existing feedback envelope
        # and this operation's pose criteria, not a new motion speed limit.
        start = op["_plan_start"]
        for side in step["poses"]:
            i = SIDES.index(side) * 7
            joint_drift = np.max(np.abs(np.asarray(observed.joint_positions_rad)[i:i+7]
                                        - np.asarray(start.joint_positions_rad)[i:i+7]))
            drift = pose_error(getattr(observed, side + "_tcp"), getattr(start, side + "_tcp"))
            if (joint_drift > .02 or np.linalg.norm(drift[:3]) > op["position_tolerance_m"]
                    or np.linalg.norm(drift[3:]) > op["orientation_tolerance_rad"]):
                op.pop("_plan_future")
                op["planning"] = dict(state="replanning", reason="measured start changed")
                return
        op["planning"] = dict(state="validated", arms=reports)
        op["action"] = self._action(step, observed, plans=reports)
        self.event("motion_plan", operation_id=op["operation_id"], step_index=op["step_index"], arms=reports)

    def _action(self, step, observed, plans=None):
        arms = {s: ArmTarget(getattr(observed, s + "_tcp"), None) for s in SIDES}
        for s, value in step.get("poses", {}).items():
            target = pose(value)
            arms[s] = ArmTarget(
                tcp_pose(target) if step.get("tool_frame") == "pinch" else target, None
            )
        q = bounded_hold_reference(observed.joint_positions_rad, self.kin.limits)
        if step.get("joints"):
            # This FK instance belongs to the execution worker. Read-only IK
            # queries and the planning process never acquire its state or lock.
            for s, value in step["joints"].items():
                i = SIDES.index(s) * 7
                q[i : i + 7] = value
            self.action_kin.update(q)
            arms = {s: ArmTarget(self.action_kin.pose(s), None) for s in SIDES}
        for s, value in step.get("grippers", {}).items():
            self.grippers[SIDES.index(s)] = value
        arms = {
            s: ArmTarget(arms[s].pose, self.grippers[i]) for i, s in enumerate(SIDES)
        }
        return (
            JointAction(**arms, joint_positions_rad=tuple(q))
            # Jaw-only steps hold measured joints; they must not create an
            # implicit arm path merely because fresh FK differs slightly.
            if not step.get("poses")
            else MixedAction(**arms, cartesian_arms=tuple(step["poses"]),
                             joint_positions_rad=tuple(q),
                             rotation_arcs=tuple((plans or {}).get(s, {}).get("selected_arc", "shortest")
                                                 for s in SIDES),
                             cartesian_starts=tuple(pose(plans[s]["start_pose"]) if plans and s in plans else None
                                                    for s in SIDES))
        )

    @staticmethod
    def _public_operation(op):
        return copy.deepcopy(
            {
                k: v
                for k, v in op.items()
                if not k.startswith("_") and k not in {"action", "steps", "settled_since"}
            }
        )

    def operation(self, operation_id):
        with self.lock:
            if operation_id not in self.operations:
                raise WorkshopError("UNKNOWN_OPERATION", "Unknown operation ID")
            return self._public_operation(self.operations[operation_id])

    def _hold_locked(self, reason, latch=False, preserve_grippers=False):
        self.planning_generation.value += 1
        self.target = None
        self.grippers = [None, None]
        if self.active:
            op = self.operations[self.active]
            if op.get("_plan_future") is not None:
                op["_plan_future"].cancel()
            op.update(state="cancelled", reason=reason)
            self.active = None
        if latch:
            self.stopped = True
        if self.robot is not None:
            if preserve_grippers:
                self.robot.hold(preserve_grippers=True)
            else:
                self.robot.hold()
        self.event("hold", reason=reason, latched=latch)

    def hold(self, client_id):
        with self.lock:
            self._authorize(client_id)
            self._hold_locked("participant HOLD", preserve_grippers=True)
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
            self._hold_locked("participant released session", preserve_grippers=True)
            if self.gravity_comp and self.enabled and not self.fault:
                self._enter_hand_guiding_locked("policy session ended")
            self.owner = None
            return dict(released=True)

    def operator_stop(self):
        with self.lock:
            self._hold_locked("operator STOP", latch=True)
            if self.gravity_comp and self.enabled and not self.fault:
                self._enter_hand_guiding_locked("operator STOP")

    def _enter_hand_guiding_locked(self, reason):
        if not self.gravity_comp:
            raise RuntimeError("Hand-guiding is disabled (gravity_comp=false)")
        try:
            self.robot.enter_hand_guiding()
            self.event("hand_guiding", reason=reason, motors_disabled=False)
        except Exception as error:
            self.fault = f"Hand-guiding transition failed: {error}"
            self.stopped = True
            self.event("fault", reason=self.fault)
            raise

    def _advance_operation_locked(self, op, step, observed):
        if op.get("_dispatched") and observed.mode == "hold" and observed.reason == "unreachable or stalled target":
            self._hold_locked(observed.reason, preserve_grippers=True)
            op.update(state="failed", reason="PATH_STALLED")
            self.event("operation_end", **self._public_operation(op))
            return
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
        # Endpoint proximity alone cannot complete a complementary rotation
        # before it has traversed its validated waypoint sequence.
        for side, plan in op.get("planning", {}).get("arms", {}).items():
            if side not in step.get("poses", {}):
                continue
            if plan["selected_arc"] == "alternate":
                path = observed.motion_paths.get(side, {})
                reached &= (path.get("target") is not None
                            and pose(path["target"]) == getattr(self.target, side).pose
                            and path.get("waypoint", -1) == path.get("segments", -2))
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
            op.pop("_plan_future", None)
            op.pop("_dispatched", None)
            op["step_index"] += 1
            op["progress"] = op["step_index"] / len(op["steps"])
            op["settled_since"] = op["action"] = None
            # Keep the completed posture fixed while the next step is planned.
            self.robot.hold(preserve_grippers=True)
            fresh = self.robot.observe()
            self.target = JointAction(
                *(ArmTarget(getattr(fresh, s + "_tcp"), self.grippers[i])
                  for i, s in enumerate(SIDES)),
                joint_positions_rad=tuple(bounded_hold_reference(
                    fresh.joint_positions_rad, self.kin.limits)))
            if op["step_index"] == len(op["steps"]):
                op.update(
                    state="succeeded",
                    completion="measured endpoint/settled gripper; not a grasp-success claim",
                )
                self.active = None
                self.event(
                    "operation_end", **self._public_operation(op)
                )
            else:
                op.pop("planning", None)

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
                                if observed.mode == "hand_guiding":
                                    self.robot.begin_control()
                                    self.last_enable_s = time.monotonic()
                                    observed = self.robot.observe()
                                self._prepare_action_locked(op, step, observed)
                            if op["action"] is not None and self.active == op["operation_id"]:
                                self._advance_operation_locked(op, step, observed)
                            if self.active == op["operation_id"] and time.monotonic() - op["started_s"] > op["timeout_s"]:
                                self._hold_locked("operation timed out", preserve_grippers=True)
                                op.update(state="failed", reason="TIMEOUT")
                                self.event("operation_end", **self._public_operation(op))
                        if self.target is not None and not self.done.is_set():
                            fresh = self.robot.observe()
                            if self.active is not None and fresh.mode == "hold" and fresh.reason == "unreachable or stalled target":
                                raise RuntimeError("PATH_STALLED: unreachable or stalled target")
                            self.robot.submit_target(self.target, fresh.observation_id)
                            if self.active is not None:
                                self.operations[self.active]["_dispatched"] = True
            except Exception as error:
                with self.lock:
                    op = self.operations.get(self.active)
                    if op is not None and str(error).startswith(("PATH_STALLED:", "RuntimeError: PATH_STALLED:")):
                        try:
                            self._hold_locked("unreachable or stalled target", preserve_grippers=True)
                            op.update(state="failed", reason="PATH_STALLED")
                            self.event("operation_end", **self._public_operation(op))
                            continue
                        except Exception as hold_error:
                            error = RuntimeError(f"{error}; HOLD unconfirmed: {hold_error}")
                    self.fault = str(error)
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
        self.done.set()
        self.planning_generation.value += 1
        errors = []
        for thread in (self.worker, self.camera_thread):
            if thread is not None:
                try:
                    thread.join(timeout=5)
                    if thread.is_alive():
                        errors.append(f"{thread.name} did not exit")
                except Exception as error:
                    errors.append(f"{thread.name} join failed: {error}")
        # Always close the robot even if another worker failed to join. The
        # offline planner must not delay the hardware owner's HOLD/close path.
        if self.robot is not None:
            try:
                try:
                    if self.gravity_comp and self.enabled and not self.fault:
                        self.robot.enter_hand_guiding()
                finally:
                    if self.backend == "real":
                        self.robot.close(hand_guiding=self.gravity_comp and not bool(self.fault))
                    else:
                        self.robot.close()
            except Exception as error:
                errors.append(f"Robot close failed: {error}")
        if self.planner_pool is not None:
            try:
                close_planner(self.planner_pool)
                self.planner_pool = None
            except Exception as error:
                errors.append(f"Planner close failed: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))
        self.closed = True
        if self.plant is not None:
            (self.output / "score.json").write_text(
                json.dumps(self.plant.evaluate(), indent=2) + "\n"
            )
            (self.output / "truth.json").write_text(json.dumps(self.plant.truth) + "\n")
