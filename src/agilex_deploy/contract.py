"""Version 3 robot contract: official NERO TCP, meters and XYZW quaternions.

All poses use the fixed stand's base_link frame. There is no simulator state,
ground-truth object state, backend selector, or vendor SDK type in this API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    gripper_open: int | None = None  # None: leave the gripper unchanged.

    def __post_init__(self):
        if not isinstance(self.pose, Pose):
            raise ValueError("pose must be a Pose")
        if self.gripper_open is not None and (type(self.gripper_open) is not int or self.gripper_open not in (0, 1)):
            raise ValueError("gripper_open must be integer 0, 1, or None (unchanged)")


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
class JointAction(Action):
    """Explicit joint target with its FK poses; executed without an IK round trip."""
    joint_positions_rad: tuple[float, ...]

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "joint_positions_rad",
                           finite_tuple(self.joint_positions_rad, 14, "joint_positions_rad"))


@dataclass(frozen=True)
class MixedAction(Action):
    """Cartesian goals for selected arms and fixed joint holds for the others.

    The joint vector is captured when the action is created. Feedback updates
    must not turn an unselected arm's hold into a new Cartesian IK objective.
    Gripper commands remain independent of each arm's motion mode.
    Optional per-arm starts freeze validated geometry, including the rotation
    direction near a half-turn. Feedback still controls waypoint progression.
    """

    cartesian_arms: tuple[str, ...]
    joint_positions_rad: tuple[float, ...]
    rotation_arcs: tuple[str, str] = ("shortest", "shortest")
    cartesian_starts: tuple[Pose | None, Pose | None] = (None, None)

    def __post_init__(self):
        super().__post_init__()
        arms = tuple(self.cartesian_arms)
        if not arms or any(side not in SIDES for side in arms) or len(set(arms)) != len(arms):
            raise ValueError("cartesian_arms must contain distinct left/right arm names")
        object.__setattr__(self, "cartesian_arms", tuple(side for side in SIDES if side in arms))
        object.__setattr__(self, "joint_positions_rad",
                           finite_tuple(self.joint_positions_rad, 14, "joint_positions_rad"))
        arcs = tuple(self.rotation_arcs)
        if len(arcs) != 2 or any(arc not in ("shortest", "alternate") for arc in arcs):
            raise ValueError("rotation_arcs must contain left/right shortest or alternate")
        object.__setattr__(self, "rotation_arcs", arcs)
        starts = tuple(self.cartesian_starts)
        if len(starts) != 2 or any(start is not None and not isinstance(start, Pose) for start in starts):
            raise ValueError("cartesian_starts must contain left/right Pose or None")
        object.__setattr__(self, "cartesian_starts", starts)


@dataclass(frozen=True)
class GripperState:
    """Measured tracking outcome, not an object-contact or grasp-success claim.

    Error and velocity use normalized configured travel. Force is the signed
    vendor report in N, not a calibrated fingertip force. Missing sensors are
    None, including force in the simulator's uncalibrated jaw model.
    """
    phase: str = "unavailable"
    requested_open: int | None = None
    position_error: float | None = None
    velocity_per_s: float | None = None
    width_m: float | None = None
    force_n: float | None = None
    enabled: bool | None = None
    timestamp_s: float | None = None


@dataclass(frozen=True)
class Observation:
    observation_id: int
    timestamp_s: float  # Host monotonic acquisition time, not render or delivery time.
    joint_names: tuple[str, ...]
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    gripper_position: tuple[float | None, float | None]  # None: unavailable; otherwise 0 closed, 1 open.
    left_tcp: Pose
    right_tcp: Pose
    mode: str
    reason: str
    cameras: tuple = ()  # Empty means unavailable; no synthetic RGB-D is returned.
    schema_version: int = 3
    gripper_states: tuple[GripperState, GripperState] = (GripperState(), GripperState())
    feedback_status: tuple = ()  # Source/availability metadata; unknown motor enable is None.
    motion_paths: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def as_action(self):
        return Action(*(ArmTarget(getattr(self, f"{s}_tcp"), None if self.gripper_position[i] is None else int(self.gripper_position[i] >= .5))
                        for i, s in enumerate(SIDES)))


class RobotInterface(Protocol):
    def observe(self) -> Observation: ...
    def submit_target(self, action: Action, observation_id: int) -> int: ...
    def hold(self) -> None: ...
    def enter_hand_guiding(self) -> None: ...
    def begin_control(self) -> None: ...
    def close(self) -> None: ...
