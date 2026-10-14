"""Version 1 robot contract. Distances are meters, rotations are XYZW quaternions.

All poses use the fixed stand's base_link frame. There is no simulator state,
ground-truth object state, backend selector, or vendor SDK type in this API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol
import numpy as np

SIDES = ("left", "right")
JOINT_NAMES = tuple(f"{side}_joint{i}" for side in SIDES for i in range(1, 8))


def finite_tuple(value, length, label):
    a = np.asarray(value, dtype=float)
    if a.shape != (length,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{label} must contain {length} finite numbers")
    return tuple(float(x) for x in a)


@dataclass(frozen=True)
class Pose:
    position_m: tuple[float, ...]
    quaternion_xyzw: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "position_m", finite_tuple(self.position_m, 3, "position_m"))
        q = np.asarray(finite_tuple(self.quaternion_xyzw, 4, "quaternion_xyzw"))
        if abs(np.linalg.norm(q) - 1) > 1e-3:
            raise ValueError("quaternion_xyzw must have unit norm")
        object.__setattr__(self, "quaternion_xyzw", tuple(q / np.linalg.norm(q)))
        if np.max(np.abs(self.position_m)) > 2:
            raise ValueError("Target exceeds the 2 m contract envelope")


@dataclass(frozen=True)
class ArmTarget:
    pose: Pose
    gripper_open: int = 1

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError("pose must be a Pose")
        if type(self.gripper_open) is not int or self.gripper_open not in (0, 1):
            raise ValueError("gripper_open must be integer 0 or 1")


@dataclass(frozen=True)
class Action:
    left: ArmTarget
    right: ArmTarget

    def __post_init__(self):
        if not all(isinstance(getattr(self, s), ArmTarget) for s in SIDES):
            raise ValueError("Both arm targets are required")

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(SIDES):
            raise ValueError("Action requires exactly left and right")
        arms = {}
        for side in SIDES:
            row = value[side]
            if set(row) != {"position_m", "quaternion_xyzw", "gripper_open"}:
                raise ValueError(f"Unexpected or missing fields for {side}")
            arms[side] = ArmTarget(Pose(row["position_m"], row["quaternion_xyzw"]), row["gripper_open"])
        return cls(**arms)

    def to_dict(self):
        return {s: {**asdict(getattr(self, s).pose), "gripper_open": getattr(self, s).gripper_open}
                for s in SIDES}


@dataclass(frozen=True)
class Observation:
    observation_id: int
    timestamp_s: float  # Host monotonic acquisition time, not render or delivery time.
    joint_names: tuple[str, ...]
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    gripper_position: tuple[float, float]  # Calibrated travel: 0 closed, 1 open.
    left_tcp: Pose
    right_tcp: Pose
    mode: str
    reason: str
    cameras: tuple = ()  # Empty means unavailable; no synthetic RGB-D is returned.
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)

    def as_action(self):
        return Action(*(ArmTarget(getattr(self, f"{s}_tcp"), int(self.gripper_position[i] >= .5))
                        for i, s in enumerate(SIDES)))


class RobotInterface(Protocol):
    def observe(self) -> Observation: ...
    def submit_target(self, action: Action, observation_id: int) -> int: ...
    def hold(self) -> None: ...
    def close(self) -> None: ...
