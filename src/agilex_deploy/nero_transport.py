"""Explicitly armed, single-arm NERO 1.21 commissioning transport.

This is a supervised free-space commissioning tool, not collision-safe general
teleoperation. No constructor enables motors. Optional gripper control uses the
official codec and starts only after an explicit selected-gripper target.
Wire encoding was reviewed against official pyAgxArm commit
1ae40ad4755ca25ad32bece3ab42325698185fc7 (Nero v121, joint position mode).
"""
import fcntl
import errno
import json
from pathlib import Path
import socket
import struct
import time

import numpy as np

from .backends import EncoderSample
from .can_feedback import PassiveCanFeedback
from .contract import Action, ArmTarget, Pose, JOINT_NAMES, SIDES, finite_tuple
from .controller import pose_error
from .firmware_probe import query_firmware
from .model import Kinematics, ROOT
from .motion_limits import JointLimits, SIM_VELOCITY, SIM_ACCELERATION, bounded_hold_reference
from .limits_probe import query_motion_limits
from .hardware_interlock import require_qualified_enable, require_qualified_hand_guiding

MAX_SPEED = np.tile(SIM_VELOCITY, 2)  # offline fixture only
MAX_ACCELERATION = np.tile(SIM_ACCELERATION, 2)  # offline fixture only
MAX_EXCURSION = .45  # rad from arming pose; cannot be reset by HOLD
TEST_LIFT_M = .03
MAX_TRANSLATION_M = .03
COMMAND_TIMEOUT = .2


def validate_workspace_radius(radius):
    # None is the explicit full-workspace setting, not an invalid/unbounded
    # numeric radius. NaN and infinity are still rejected.
    if radius is None:
        return None
    radius = float(radius)
    if not np.isfinite(radius) or not .03 <= radius <= .2:
        raise ValueError("Workspace radius must be between 3 and 20 cm")
    return radius


def position_frames(q, limits=None):
    values = np.rint(np.degrees(finite_tuple(q, 7, "joint target")) * 1000).astype(np.int64)
    if limits is not None:
        limits = np.asarray(limits, dtype=float)
        if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
            raise ValueError("Invalid joint position limits for CAN encoding")
        lower = np.ceil(np.degrees(limits[:, 0]) * 1000).astype(np.int64)
        upper = np.floor(np.degrees(limits[:, 1]) * 1000).astype(np.int64)
        # Match the feedback decoder's arithmetic, including floating-point
        # round trips at an exactly representable millidegree boundary.
        lower += np.radians(lower * .001) < limits[:, 0]
        upper -= np.radians(upper * .001) > limits[:, 1]
        if np.any(lower > upper):
            raise ValueError("Joint position interval has no representable CAN angle")
        # Guards assess continuous references for speed and acceleration. The
        # wire uses millidegrees, so round inward at bounds without feeding
        # quantization steps back into those derivative checks.
        values = np.clip(values, lower, upper)
    frames = [(0x155 + i, struct.pack(">ii", int(values[2*i]), int(values[2*i+1]))) for i in range(3)]
    frames.append((0x170, struct.pack(">ii", int(values[6]), 0)))
    return frames


class MotionGuard:
    """Pure admission checks shared by the wire transport and offline tests."""
    def __init__(self, side, seed, limits, kinematics=None, workspace_radius_m=None, motion_limits=None,
                 gripper_control=False):
        if side not in SIDES:
            raise ValueError("Select one arm: left or right")
        self.side = side
        self.gripper_control = gripper_control
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.indices = slice(SIDES.index(side)*7, SIDES.index(side)*7+7)
        self.seed = np.asarray(finite_tuple(seed, 14, "arming pose"))
        self.limits = np.asarray(limits).copy()
        self.motion_limits = motion_limits or JointLimits.simulation(self.limits[self.indices])
        self.limits[self.indices] = self.motion_limits.position
        if np.any(self.seed < self.limits[:, 0]) or np.any(self.seed > self.limits[:, 1]):
            raise ValueError("Arming pose outside joint position limits")
        self.previous = self.seed.copy()
        self.velocity = np.zeros(14)
        self.kin = kinematics if kinematics is not None else Kinematics()
        self.kin.update(self.seed, (None, None))
        self.poses = {s: self.kin.pose(s) for s in SIDES}

    def lift_target(self):
        """One absolute target anchored to the same seed used by admission."""
        arms = {s: ArmTarget(self.poses[s], None) for s in SIDES}
        pose = self.poses[self.side]
        arms[self.side] = ArmTarget(Pose(np.asarray(pose.position_m) + [0, 0, TEST_LIFT_M], pose.quaternion_xyzw), None)
        return Action(**arms)

    def action(self, action):
        for side in SIDES:
            arm = getattr(action, side)
            if arm.gripper_open is not None and (not self.gripper_control or side != self.side):
                raise ValueError("Hardware gripper control is unavailable; use None")
            delta = pose_error(self.poses[side], arm.pose)
            if side != self.side:
                if np.linalg.norm(delta[:3]) > .001 or np.linalg.norm(delta[3:]) > .005:
                    raise ValueError("The unselected arm must stay at its measured starting pose")
            elif self.workspace_radius_m is not None and (np.linalg.norm(delta[:3]) > self.workspace_radius_m + 1e-6 or delta[2] < -.0001 or np.linalg.norm(delta[3:]) > .01746):
                raise ValueError(f"Commissioning envelope: <={self.workspace_radius_m * 1000:g} mm from start, no downward target, <=1 degree rotation")

    def command(self, q, measured, dt=.01):
        q = np.asarray(finite_tuple(q, 14, "joint target"))
        measured = np.asarray(finite_tuple(measured, 14, "measured joints"))
        active = self.indices
        if not 0 < dt <= .05:
            raise RuntimeError("Invalid command period")
        outside = (q < self.limits[:, 0]) | (q > self.limits[:, 1])
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            bound = self.limits[index, 0 if q[index] < self.limits[index, 0] else 1]
            raise RuntimeError(
                f"Command violates model-limit bounds: {JOINT_NAMES[index]} "
                f"value={q[index]:.12g} rad, bound={bound:.12g} rad, "
                f"delta={q[index] - bound:+.12g} rad"
            )
        if self.workspace_radius_m is not None and np.max(abs(q[active] - self.seed[active])) > MAX_EXCURSION:
            raise RuntimeError("Command exceeds commissioning joint excursion")
        velocity = (q - self.previous) / dt
        if np.any(abs(velocity[active]) > self.motion_limits.velocity + 1e-6):
            raise RuntimeError("Hardware command speed bound exceeded")
        if np.any(abs(velocity[active] - self.velocity[active]) / dt > self.motion_limits.acceleration + 1e-5):
            raise RuntimeError("Hardware command acceleration bound exceeded")
        # Admit intermediate references as well as end targets. Small numerical
        # and servo tolerances do not constitute collision clearance.
        self.kin.update(q, (None, None))
        delta = pose_error(self.poses[self.side], self.kin.pose(self.side))
        if self.workspace_radius_m is not None and (np.linalg.norm(delta[:3]) > self.workspace_radius_m + .001 or
                delta[2] < -.001 or np.linalg.norm(delta[3:]) > np.radians(1.5)):
            raise RuntimeError("Joint reference leaves commissioning Cartesian envelope")
        self.previous, self.velocity = q.copy(), velocity
        return q[active]

    def hold(self, measured):
        self.previous = bounded_hold_reference(measured, self.limits)
        self.velocity[:] = 0


class NeroTransport:
    """Owns raw CAN I/O inside a separate foreground-owned control process.

HOLD requests the most recent measured position if feedback is valid; if it is
not, the last bounded command is retained. It never disables motors. Sustained
holding after host/power/CAN failure is not certified by this software.
"""
    def __init__(self, side, mapping=ROOT / "configs/hardware_mapping.json",
                 commissioning=ROOT / "configs/commissioning.json", workspace_radius_m=None,
                 gripper_control=False, gripper_force_n=1.0, *, _shared_reader=None, _feedback_sides=None):
        if side not in SIDES:
            raise ValueError("Select one arm")
        config = json.loads(Path(commissioning).read_text())
        if not (config.get("mounting_and_support_confirmed_by_user") and
                config.get("controller_mounting_configuration_verified") and
                config.get("static_model_posture", {}).get("confirmed_by_user")):
            raise RuntimeError("Mounting, support, controller orientation and model posture must be confirmed")
        self.side = side
        self.gripper_control = gripper_control
        self.gripper_active = False
        self.gripper_started = None
        self.gripper_force_n = gripper_force_n
        self.gripper_range_m = None
        if gripper_control:
            from .gripper import validate_force
            validate_force(gripper_force_n)
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.side_index = SIDES.index(side)
        self.indices = slice(self.side_index*7, self.side_index*7+7)
        self.kin = Kinematics()
        self.reader = None
        self.owns_reader = _shared_reader is None
        self.writer = None
        self.lock_file = None
        self.armed = False
        self.guard = None
        self.last = None
        self.last_command = None
        self.last_send = 0.
        self.fault = None
        self.closed = False
        self.control_mode = "follower"
        self.initial_hand_guiding = False
        try:
            if _shared_reader is None:
                arms = json.loads(Path(mapping).read_text())["arms"]
                serials = sorted(arms[s]["usb_identity"]["serial"] for s in SIDES)
                if not all(serial.isalnum() for serial in serials) or len(set(serials)) != 2:
                    raise ValueError("Invalid adapter serial")
                # Lock the pair, not just the selected arm: two commissioning
                # viewers must not independently arm both sides at the same time.
                self.lock_file = open("/tmp/agilex-deploy-" + "-".join(serials) + ".lock", "a")
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                firmware = query_firmware(mapping)
                if firmware["missing"] or any(v["software_version"] != "1.21" for v in firmware["firmware"].values()):
                    raise RuntimeError("This commissioning transport supports verified firmware 1.21 only")
                options = {"gripper_feedback": _feedback_sides or (side,)} if gripper_control else {}
                self.reader = PassiveCanFeedback(mapping, motor_feedback=True, **options)
            else:
                self.reader = _shared_reader
            self.last = self.reader.read(timeout=1.)
            for i, arm in enumerate(SIDES):
                states = self.last.driver_status[i*7:i*7+7]
                if any(v is None for v in states):
                    raise RuntimeError(f"{arm}: leader stream has no motor-status acknowledgement; cannot adopt an unknown enabled state")
                if any(v & 0x40 for v in states) and not (
                    all(v & 0x40 for v in states) and self.last.angle_sources[i] == "leader"
                ):
                    counts = ", ".join(
                        f"{name} {sum(bool(v & 0x40) for v in self.last.driver_status[j*7:j*7+7] if v is not None)}/7 enabled"
                        for j, name in enumerate(SIDES)
                    )
                    statuses = ", ".join(
                        f"{name}={status.hex() if status is not None else 'unavailable'}"
                        for name, status in zip(SIDES, self.last.arm_status or (None, None))
                    )
                    raise RuntimeError(
                        f"START_REQUIRES_DISABLED_ARMS: {counts}; controller status: {statuses}. "
                        "Enabled feedback does not identify who enabled the motors. "
                        "Close other robot controllers, mechanically support BOTH arms, then run "
                        "`.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported`. "
                        "Disabling removes holding torque. Startup sent no motor enable/disable or mode commands."
                    )
            self.initial_hand_guiding = self.last.angle_sources[self.side_index] == "leader"
            if self.initial_hand_guiding:
                self.control_mode = "hand_guiding"
            if gripper_control:
                from .gripper import query_range
                self.gripper_range_m = query_range(self.reader.channels[side])
                self.last = self.reader.read(timeout=1.)
                self._gripper_feedback()
        except BaseException:
            self.close()
            raise

    def _send(self, frames):
        if self.writer is None:
            raise RuntimeError("Hardware writer is not armed")
        deadline = time.monotonic() + .02
        for cid, payload in frames:
            packet = struct.pack("=IB3x8s", cid, len(payload), payload)
            while True:
                try:
                    if self.writer.send(packet) != 16:
                        raise RuntimeError("Short CAN write")
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(f"CAN send deadline exceeded on frame {cid:#x}: {error}") from error
                    # SocketCAN can report writable while its device queue is
                    # full. Back off briefly instead of spinning or accumulating
                    # old position targets in a larger queue. The whole batch
                    # retains the same bounded deadline.
                    time.sleep(min(.001, remaining))
        self.last_send = time.monotonic()

    def read_encoders(self, snapshot=None):
        self.last = self.reader.read(timeout=.01) if snapshot is None else snapshot
        if any(v is not None and v & ~0x40 for v in self.last.driver_status):
            raise RuntimeError("NERO driver reports a fault")
        guiding = getattr(self, "control_mode", "follower") == "hand_guiding"
        if self.armed:
            states = self.last.driver_status[self.indices]
            status = self.last.arm_status[self.side_index]
            if not guiding and not all(v is not None and v & 0x40 for v in states):
                raise RuntimeError("Selected arm lost motor enable")
            if guiding and any(v is not None and not v & 0x40 for v in states):
                raise RuntimeError("Selected arm lost motor enable")
            healthy = (guiding and status is None) or (status is not None and status[1] == 0 and not any(status[6:8]))
            expected = (self.last.angle_sources[self.side_index] == "leader"
                        if guiding else status is not None and status[0] == 1 and status[2] == 1)
            if not healthy or not expected:
                raise RuntimeError(f"Selected arm is not healthy CAN joint mode: {status.hex() if status else 'unavailable'}")
        grippers = [None, None]
        if self.gripper_control and not guiding:
            feedback = self._gripper_feedback()
            grippers[self.side_index] = feedback.normalized(self.gripper_range_m)
            if self.gripper_started is not None and time.monotonic()-self.gripper_started > .5 and not feedback.enabled:
                raise RuntimeError("Selected gripper did not enable or lost enable")
        feedback_status = tuple(dict(angle_source=self.last.angle_sources[i],
                                    velocity_source=self.last.velocity_sources[i],
                                    motor_enabled=(all(v & 0x40 for v in self.last.driver_status[i*7:i*7+7])
                                                   if all(v is not None for v in self.last.driver_status[i*7:i*7+7]) else None),
                                    gripper_feedback_available=bool(self.last.gripper_feedback and self.last.gripper_feedback[i] is not None))
                                for i in range(2))
        return EncoderSample(self.last.sequence, self.last.timestamp_s, JOINT_NAMES,
                             self.last.positions, self.last.velocities, tuple(grippers),
                             tuple(f if i == self.side_index and not guiding else None for i, f in enumerate(self.last.gripper_feedback or (None,None)))
                             if self.gripper_control else (None, None), feedback_status)

    def _gripper_feedback(self):
        if self.last.gripper_feedback is None or self.last.gripper_feedback[self.side_index] is None:
            raise RuntimeError("Selected gripper feedback unavailable")
        feedback = self.last.gripper_feedback[self.side_index]
        feedback.validate(time.monotonic())
        feedback.normalized(self.gripper_range_m)
        return feedback

    def activate_gripper(self, action):
        """Called only after the complete action has passed runtime admission."""
        if getattr(action, self.side).gripper_open is not None:
            if not self.armed or not self.gripper_control:
                raise RuntimeError("Selected gripper control is not armed")
            self._gripper_feedback()
            self.gripper_active = True

    def prepare_arm(self):
        """Read limits and seed targets while both arms remain disabled."""
        require_qualified_enable(self.reader.read(timeout=.03) if getattr(self, "reader", None) else None,
                                 can_mode_sides=())
        if self.armed or self.writer is not None:
            raise RuntimeError("Arming is one-shot; restart to repeat commissioning")
        sample = self.read_encoders()
        adopting = getattr(self, "initial_hand_guiding", False)
        if (not adopting and any(v & 0x40 for v in self.last.driver_status[self.indices])) or max(abs(v) for v in sample.velocities) > .02:
            raise RuntimeError("Arming requires disabled, stationary arms")
        self.writer = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.writer.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b"")
        self.writer.bind((self.reader.channels[self.side],))
        self.writer.setblocking(False)
        # CAN joint position mode, 100% of configured firmware speed. Installation byte 0
        # leaves the already configured mounting orientation unchanged.
        if not adopting:
            # Request normal CAN joint mode and normal feedback while disabled.
            # Never enable as a fallback if firmware ignores mode selection.
            requested_at = time.monotonic()
            self._send([(0x151, bytes([1, 1, 100, 0, 0, 0, 1, 0]))])
            deadline = requested_at + .3
            while True:
                self.last = self.reader.read(timeout=.03)
                require_qualified_enable(self.last, can_mode_sides=())
                status = self.last.arm_status[self.side_index]
                if (self.last.status_timestamp_s > requested_at
                        and status[0] == 1 and status[2] == 1):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{self.side}: CAN position mode was not acknowledged while disabled; "
                        "no motor enable was sent"
                    )
        # Read after selecting CAN mode, while all motors are still disabled.
        # No fallback to another arm's settings, and no limit-setting writes.
        limits = query_motion_limits(self.reader.channels[self.side], self.kin.limits[self.indices])
        sample = self.read_encoders()
        require_qualified_enable(self.last, can_mode_sides=(self.side,))
        if not adopting and any(v & 0x40 for v in self.last.driver_status[self.indices]):
            raise RuntimeError("Motor enable changed during limit readback")
        self.guard = MotionGuard(self.side, sample.positions, self.kin.limits, self.kin,
                                 workspace_radius_m=self.workspace_radius_m, motion_limits=limits,
                                 gripper_control=self.gripper_control)
        print(f"{self.side} controller limits: speed={limits.velocity.tolist()} rad/s, "
              f"acceleration={limits.acceleration.tolist()} rad/s²; firmware speed=100%", flush=True)
        self.last_command = self.guard.previous[self.indices].copy()
        if not adopting:
            self._send(position_frames(self.last_command, self.guard.motion_limits.position))
        return sample

    def arm(self):
        # Outside the recovery handler: no writer or enable has been attempted.
        require_qualified_enable(self.reader.read(timeout=.03) if getattr(self, "reader", None) else None,
                                 can_mode_sides=())
        try:
            sample = self.prepare_arm()
            require_qualified_enable(self.last)
            if getattr(self, "initial_hand_guiding", False):
                self.armed = True
                self.read_encoders()
                return
            # Explicit indices avoid the upstream ambiguity of index 8 versus
            # all joints. No gripper enable or gripper command is sent.
            for joint in range(1, 8):
                self._send([(0x471, bytes([joint, 2]) + bytes(6))])
                time.sleep(.005)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                self.last = self.reader.read(timeout=.03)
                if np.max(abs(np.array(self.last.positions) - sample.positions)) > .015:
                    raise RuntimeError("Posture shifted during arming")
                if any(v & ~0x40 for v in self.last.driver_status):
                    raise RuntimeError("Driver fault during arming")
                self._send(position_frames(self.last_command, self.guard.motion_limits.position))
                states = self.last.driver_status[self.indices]
                status = self.last.arm_status[self.side_index]
                if all(v & 0x40 for v in states) and status[:3] == bytes([1, 0, 1]):
                    self.armed = True
                    return
                time.sleep(.01)
            raise RuntimeError("Arm enable/mode acknowledgement timed out")
        except BaseException as error:
            self.fault = "Arming failed; some motors may be enabled. Maintain physical support."
            try:
                self.hold()
            except Exception:
                self.fault += " CAN HOLD could not be delivered."
            raise RuntimeError(f"{self.fault} Cause: {error}") from error

    def prepare_positions(self, positions, grippers):
        if not self.armed:
            return  # Observe-only startup; zero hardware motion writes.
        if self.fault:
            raise RuntimeError(self.fault)
        if getattr(self, "control_mode", "follower") != "follower":
            raise RuntimeError("Position writes are blocked outside follower control")
        if any(g is not None and (not self.gripper_control or i != self.side_index)
               for i, g in enumerate(grippers)):
            raise ValueError("Hardware grippers are not commissioned")
        if time.monotonic() - self.last_send > COMMAND_TIMEOUT:
            self.fault = "Hardware command watchdog expired"
            self.hold()
            raise RuntimeError(self.fault)
        selected = self.guard.command(positions, self.last.positions)
        frames = position_frames(selected, self.guard.motion_limits.position)
        if self.gripper_active:
            from .gripper import width_frame
            self._gripper_feedback()
            fraction = grippers[self.side_index]
            if fraction is None:
                raise RuntimeError("Active gripper reference unavailable")
            frames.append(width_frame(fraction, self.gripper_range_m, self.gripper_force_n))
        return frames, selected

    def send_positions(self, positions, grippers):
        prepared = self.prepare_positions(positions, grippers)
        if prepared is None:
            return
        self.send_prepared(*prepared)

    def send_prepared(self, frames, selected):
        self._send(frames)
        if self.gripper_active and self.gripper_started is None:
            self.gripper_started = time.monotonic()
        self.last_command = selected.copy()

    def hold(self, *, preserve_grippers=False):
        if getattr(self, "control_mode", "follower") != "follower":
            return  # Never send stale position targets into leader/unknown mode.
        if self.writer is None or self.last_command is None:
            return
        # Keep the last bounded setpoint if fresh feedback cannot be acquired.
        try:
            # Use the same acquisition the runtime just read before HOLD, so
            # the controller and wire guard reset to one reference. Reading
            # again here could introduce encoder jitter between those owners.
            latest = self.last
            latest.validate(time.monotonic())
            self.last = latest
            self.guard.hold(latest.positions)
            self.last_command = self.guard.previous[self.indices].copy()
        except (RuntimeError, OSError):
            pass
        frames = position_frames(self.last_command, self.guard.motion_limits.position)
        gripper_error = None
        if not preserve_grippers and self.gripper_active and self.last is not None and self.last.gripper_feedback:
            feedback = self.last.gripper_feedback[self.side_index]
            # HOLD never enables a disabled/faulted gripper. If feedback is
            # stale, request its last observed position, not the distant goal;
            # actual stopping is unconfirmed when communication is lost.
            if feedback is not None and feedback.enabled and not feedback.status & 0x3F:
                from .gripper import width_frame
                try:
                    frames.append(width_frame(float(np.clip(feedback.normalized(self.gripper_range_m), 0, 1)),
                                              self.gripper_range_m, self.gripper_force_n))
                except (ValueError, RuntimeError) as error:
                    gripper_error = error
        self._send(frames)
        if gripper_error:
            raise RuntimeError(f"Arm HOLD sent; gripper HOLD unavailable: {gripper_error}")

    def _stationary_sample(self, holding):
        deadline = time.monotonic() + 2.
        settled = None
        while time.monotonic() < deadline:
            sample = self.read_encoders()
            now = time.monotonic()
            if max(abs(v) for v in sample.velocities[self.indices]) <= .02:
                settled = settled or now
                if now - settled >= .1:
                    return sample
            else:
                settled = None
            if holding:
                self.hold(preserve_grippers=True)
            time.sleep(.01)
        raise RuntimeError("Mode change requires stationary joints; transition not completed")

    def enter_hand_guiding(self):
        require_qualified_hand_guiding()
        if not self.armed:
            return
        if self.fault:
            raise RuntimeError(self.fault)
        if getattr(self, "control_mode", "follower") == "hand_guiding":
            self.read_encoders()
            return
        self._stationary_sample(holding=True)
        switched = self.request_hand_guiding()
        self.confirm_hand_guiding(switched)

    def request_hand_guiding(self):
        """Paired owner preflights both arms before issuing either request."""
        require_qualified_hand_guiding()
        from .hand_guiding import official_mode_frames
        frames = official_mode_frames("leader")
        # From here on, even failed/partial mode writes must never cause stale
        # follower setpoints to be replayed by HOLD, fault handling, or close.
        self.control_mode = "transition"
        try:
            self._send(frames)
            switched = time.monotonic()
            self.reader.set_angle_source(self.side, "leader", switched)
            return switched
        except Exception as error:
            self.fault = f"Hand-guiding unconfirmed on {self.side}: {error}; no automatic disable"
            raise RuntimeError(self.fault) from error

    def confirm_hand_guiding(self, switched):
        try:
            self._confirm_mode("leader", switched)
            self.control_mode = "hand_guiding"
            self.gripper_active = False
        except Exception as error:
            self.fault = f"Hand-guiding unconfirmed on {self.side}: {error}; no automatic disable"
            raise RuntimeError(self.fault) from error

    def _confirm_mode(self, source, since):
        deadline = time.monotonic() + 1.5
        last_error = "No new feedback"
        while time.monotonic() < deadline:
            try:
                snapshot = self.reader.read(timeout=.03)
                snapshot.validate(time.monotonic())
                status = snapshot.arm_status[self.side_index]
                if snapshot.timestamp_s <= since:
                    raise RuntimeError("Waiting for post-transition telemetry")
                if snapshot.angle_sources[self.side_index] != source:
                    raise RuntimeError("Waiting for requested encoder stream")
                if source == "leader":
                    # A leader angle stream confirms protocol mode only. It is
                    # not an acknowledgement of motor enable or gravity support.
                    self.last = snapshot
                    return snapshot
                if (snapshot.driver_timestamp_s is None or snapshot.driver_timestamp_s <= since
                        or snapshot.status_timestamp_s is None or snapshot.status_timestamp_s <= since):
                    raise RuntimeError("Waiting for fresh follower motor/status feedback")
                if not all(v == 0x40 for v in snapshot.driver_status[self.indices]):
                    raise RuntimeError("Enable lost or driver fault during mode switch")
                if status[1] or any(status[6:8]):
                    raise RuntimeError(f"Unhealthy arm status {status.hex()}")
                if source == "follower" and status[:3] != bytes([1, 0, 1]):
                    raise RuntimeError("Waiting for CAN joint mode")
                leader_stamp = snapshot.leader_timestamps_s[self.side_index]
                if source == "follower" and time.monotonic() - max(since, leader_stamp or since) < .1:
                    raise RuntimeError("Waiting for leader stream to stop before follower control")
                self.last = snapshot
                return snapshot
            except RuntimeError as error:
                last_error = str(error)
            time.sleep(.01)
        raise RuntimeError(last_error)

    def begin_control(self):
        if getattr(self, "control_mode", "follower") == "follower":
            return
        if self.fault or self.control_mode != "hand_guiding":
            raise RuntimeError(self.fault or "Control mode is unconfirmed")
        sample = self._stationary_sample(holding=False)
        switched = self.request_control(sample)
        self.confirm_control(switched)

    def request_control(self, sample):
        from .hand_guiding import official_mode_frames, follower_feedback_frame
        self.guard.hold(sample.positions)
        self.last_command = self.guard.previous[self.indices].copy()
        self.control_mode = "transition"
        try:
            # Seed before leaving leader mode, then resend immediately after.
            # No old policy setpoint survives a hand-guided posture change.
            self._send(position_frames(self.last_command, self.guard.motion_limits.position))
            self._send(official_mode_frames("follower"))
            self._send([follower_feedback_frame()])
            self._send(position_frames(self.last_command, self.guard.motion_limits.position))
            switched = time.monotonic()
            self.reader.set_angle_source(self.side, "follower", switched)
            return switched
        except Exception as error:
            self.fault = f"Follower mode unconfirmed on {self.side}: {error}; no automatic disable"
            raise RuntimeError(self.fault) from error

    def confirm_control(self, switched):
        try:
            snapshot = self._confirm_mode("follower", switched)
            self.guard.hold(snapshot.positions)
            self.last_command = self.guard.previous[self.indices].copy()
            self._send(position_frames(self.last_command, self.guard.motion_limits.position))
            self.control_mode = "follower"
        except Exception as error:
            self.fault = f"Follower mode unconfirmed on {self.side}: {error}; no automatic disable"
            raise RuntimeError(self.fault) from error

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.hold()
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.reader is not None and self.owns_reader:
                self.reader.close()
            if self.lock_file is not None:
                self.lock_file.close()
