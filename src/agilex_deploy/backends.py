"""Only this layer talks to the plant or hardware transport.

RealBackend is an encoder/setpoint adapter, not a bundled AgileX SDK driver.
A commissioned transport must provide timestamped paired feedback, a command
watchdog, unit/sign conversion, atomic fault handling, and a physical HOLD.
"""
from dataclasses import dataclass
from typing import Protocol
import time

import mujoco
import numpy as np

from .contract import JOINT_NAMES, SIDES, finite_tuple
from .model import CONTROL_DT, Kinematics, initial_positions, load_model


@dataclass(frozen=True)
class EncoderSample:
    sequence: int
    timestamp_s: float
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    grippers: tuple[float | None, ...]

    def validate(self):
        if self.joint_names != JOINT_NAMES:
            raise ValueError("Encoder joint names/order do not match contract")
        finite_tuple(self.positions, 14, "encoder positions")
        finite_tuple(self.velocities, 14, "encoder velocities")
        if len(self.grippers) != 2:
            raise ValueError("Expected two gripper measurements (None if unavailable)")
        g = self.grippers
        if not np.isfinite(self.timestamp_s) or any(x is not None and (not np.isfinite(x) or x < -.01 or x > 1.01) for x in g):
            raise ValueError("Invalid acquisition timestamp or gripper feedback")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Invalid encoder sequence")


class Backend(Protocol):
    def read(self) -> EncoderSample: ...
    def write(self, positions: np.ndarray, grippers: np.ndarray) -> None: ...
    def advance(self) -> None: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...


class SimBackend:
    def __init__(self, joint_positions=None):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.kinematics = Kinematics(self.model)
        self.qids, self.dids = self.kinematics.qids, self.kinematics.dids
        q = initial_positions() if joint_positions is None else np.asarray(
            finite_tuple(joint_positions, 14, "initial joint positions"))
        if np.any(q < self.kinematics.limits[:, 0]) or np.any(q > self.kinematics.limits[:, 1]):
            raise ValueError("Initial joint positions outside model limits; no clipping applied")
        self.data.qpos[self.qids] = q
        for side in SIDES:
            for finger, sign in ((1, 1), (2, -1)):
                self.data.qpos[self.model.joint(f"{side}_gripper_joint{finger}").qposadr[0]] = sign * .018
        self.sequence = 0
        self.closed = False
        self.write(q, np.full(2, .36))
        mujoco.mj_forward(self.model, self.data)
        self.timestamp = time.monotonic()

    def read(self):
        # Acquisition occurs here, as it would in a hardware encoder read.
        # Simulation time is deliberately not exposed as host sensor time.
        self.sequence += 1
        self.timestamp = time.monotonic()
        return EncoderSample(self.sequence, self.timestamp, JOINT_NAMES,
            tuple(self.data.qpos[self.qids]), tuple(self.data.qvel[self.dids]),
            tuple(self.data.qpos[self.model.joint(f"{s}_gripper_joint1").qposadr[0]] / .05 for s in SIDES))

    def write(self, positions, grippers):
        if self.closed:
            raise RuntimeError("Backend closed")
        for i, name in enumerate(JOINT_NAMES):
            self.data.ctrl[self.model.actuator(name).id] = positions[i]
        for i, side in enumerate(SIDES):
            self.data.ctrl[self.model.actuator(f"{side}_gripper").id] = grippers[i] * .05

    def advance(self):
        mujoco.mj_step(self.model, self.data, nstep=round(CONTROL_DT / self.model.opt.timestep))
        mujoco.mj_forward(self.model, self.data)
        self.sequence += 1
        self.timestamp = time.monotonic()

    def hold(self):
        sample = self.read()
        self.write(np.array(sample.positions), np.clip(sample.grippers, 0, 1))

    def close(self):
        if not self.closed:
            self.hold()
            self.closed = True


class HardwareTransport(Protocol):
    """Implement with a qualified pyAgxArm/CAN driver; never synthesize feedback.

    read_encoders must return the actual acquisition timestamp in the host's
    monotonic clock, and a new sequence only when BOTH arms have new feedback.
    send_positions must impose its own <= 0.2 s watchdog. An adapter must hold
    both arms if either command/read fails. Gripper travel requires calibration.
    """
    def read_encoders(self) -> EncoderSample: ...
    def send_positions(self, positions, grippers) -> None: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...


class RealBackend:
    def __init__(self, transport: HardwareTransport):
        self.transport = transport

    def read(self):
        sample = self.transport.read_encoders()
        sample.validate()
        return sample

    def write(self, positions, grippers):
        self.transport.send_positions(positions.copy(), tuple(grippers))

    def advance(self):
        pass  # Hardware owns its clock. The runtime must be ticked at 100 Hz.

    def hold(self):
        self.transport.hold()

    def close(self):
        try:
            self.hold()
        finally:
            self.transport.close()
