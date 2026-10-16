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
from .motion_limits import JointLimits, SIM_VELOCITY, SIM_ACCELERATION
from .limits_probe import query_motion_limits

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


def position_frames(q):
    values = np.rint(np.degrees(finite_tuple(q, 7, "joint target")) * 1000).astype(np.int64)
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
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise RuntimeError("Command violates model-limit bounds")
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
        self.previous = np.asarray(measured).copy()
        self.velocity[:] = 0


class NeroTransport:
    """Owns raw CAN I/O inside a separate foreground-owned control process.

HOLD requests the most recent measured position if feedback is valid; if it is
not, the last bounded command is retained. It never disables motors. Sustained
holding after host/power/CAN failure is not certified by this software.
"""
    def __init__(self, side, mapping=ROOT / "configs/hardware_mapping.json",
                 commissioning=ROOT / "configs/commissioning.json", workspace_radius_m=None,
                 gripper_control=False, gripper_force_n=1.0):
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
        self.writer = None
        self.lock_file = None
        self.armed = False
        self.guard = None
        self.last = None
        self.last_command = None
        self.last_send = 0.
        self.fault = None
        self.closed = False
        try:
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
            options = {"gripper_feedback": (side,)} if gripper_control else {}
            self.reader = PassiveCanFeedback(mapping, motor_feedback=True, **options)
            self.last = self.reader.read(timeout=1.)
            if any(v & 0x40 for v in self.last.driver_status):
                raise RuntimeError("Start with both arms disabled; another controller may own motion")
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

    def read_encoders(self):
        self.last = self.reader.read(timeout=.01)
        if any(v & ~0x40 for v in self.last.driver_status):
            raise RuntimeError("NERO driver reports a fault")
        if self.armed:
            states = self.last.driver_status[self.indices]
            status = self.last.arm_status[self.side_index]
            if not all(v & 0x40 for v in states):
                raise RuntimeError("Selected arm lost motor enable")
            if status[0] != 1 or status[1] != 0 or status[2] != 1 or any(status[6:8]):
                raise RuntimeError(f"Selected arm is not healthy CAN joint mode: {status.hex()}")
        grippers = [None, None]
        if self.gripper_control:
            feedback = self._gripper_feedback()
            grippers[self.side_index] = feedback.normalized(self.gripper_range_m)
            if self.gripper_started is not None and time.monotonic()-self.gripper_started > .5 and not feedback.enabled:
                raise RuntimeError("Selected gripper did not enable or lost enable")
        return EncoderSample(self.last.sequence, self.last.timestamp_s, JOINT_NAMES,
                             self.last.positions, self.last.velocities, tuple(grippers),
                             self.last.gripper_feedback if self.gripper_control else (None, None))

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

    def arm(self):
        if self.armed or self.writer is not None:
            raise RuntimeError("Arming is one-shot; restart to repeat commissioning")
        sample = self.read_encoders()
        if any(v & 0x40 for v in self.last.driver_status) or max(abs(v) for v in sample.velocities) > .02:
            raise RuntimeError("Arming requires disabled, stationary arms")
        self.writer = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.writer.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b"")
        self.writer.bind((self.reader.channels[self.side],))
        self.writer.setblocking(False)
        try:
            # CAN joint position mode, 100% of configured firmware speed. Installation byte 0
            # leaves the already configured mounting orientation unchanged.
            self._send([(0x151, bytes([1, 1, 100, 0, 0, 0, 0, 0]))])
            # Read after selecting CAN mode, while all motors are still disabled.
            # No fallback to another arm's settings, and no limit-setting writes.
            limits = query_motion_limits(self.reader.channels[self.side], self.kin.limits[self.indices])
            sample = self.read_encoders()
            if any(v & 0x40 for v in self.last.driver_status):
                raise RuntimeError("Motor enable changed during limit readback")
            self.guard = MotionGuard(self.side, sample.positions, self.kin.limits, self.kin,
                                     workspace_radius_m=self.workspace_radius_m, motion_limits=limits,
                                     gripper_control=self.gripper_control)
            print(f"{self.side} controller limits: speed={limits.velocity.tolist()} rad/s, "
                  f"acceleration={limits.acceleration.tolist()} rad/s²; firmware speed=100%", flush=True)
            self.last_command = np.asarray(sample.positions[self.indices])
            self._send(position_frames(self.last_command))
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
                self._send(position_frames(self.last_command))
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

    def send_positions(self, positions, grippers):
        if not self.armed:
            return  # Observe-only startup; zero hardware motion writes.
        if self.fault:
            raise RuntimeError(self.fault)
        if any(g is not None and (not self.gripper_control or i != self.side_index)
               for i, g in enumerate(grippers)):
            raise ValueError("Hardware grippers are not commissioned")
        if time.monotonic() - self.last_send > COMMAND_TIMEOUT:
            self.fault = "Hardware command watchdog expired"
            self.hold()
            raise RuntimeError(self.fault)
        selected = self.guard.command(positions, self.last.positions)
        frames = position_frames(selected)
        if self.gripper_active:
            from .gripper import width_frame
            self._gripper_feedback()
            fraction = grippers[self.side_index]
            if fraction is None:
                raise RuntimeError("Active gripper reference unavailable")
            frames.append(width_frame(fraction, self.gripper_range_m, self.gripper_force_n))
        self._send(frames)
        if self.gripper_active and self.gripper_started is None:
            self.gripper_started = time.monotonic()
        self.last_command = selected.copy()

    def hold(self):
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
            self.last_command = np.asarray(latest.positions[self.indices])
            self.guard.hold(latest.positions)
        except (RuntimeError, OSError):
            pass
        frames = position_frames(self.last_command)
        gripper_error = None
        if self.gripper_active and self.last is not None and self.last.gripper_feedback:
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

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.hold()
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.reader is not None:
                self.reader.close()
            if self.lock_file is not None:
                self.lock_file.close()
