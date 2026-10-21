"""NERO limits sourced from the manual and controller parameter readback.

The offline fixture is the left NERO 1.21 readback from 2026-10-14. It is used
only for simulation. Real arming must read the selected controller anew.
No motor torque ceiling is inferred from URDF effort or MIT encoding ranges.
"""
from dataclasses import dataclass
import numpy as np

MANUAL_VELOCITY = np.radians([180., 180., 180., 225., 225., 225., 225.])
SIM_VELOCITY = np.array([3.14, 3.14, 3.14, 3.14, 3.92, 3.92, 3.92])
SIM_ACCELERATION = np.array([2., 2., 2.5, 2.5, 2.5, 2.5, 2.5])


def bounded_hold_reference(measured, position_limits):
    """Keep feedback tolerance separate from strict outgoing reference bounds.

    The runtime admits feedback within 0.02 rad of a position bound. Servo
    error and the CAN encoder's millidegree resolution can put valid feedback
    just outside that bound; HOLD must not promote it to an invalid command.
    """
    limits = np.asarray(position_limits, dtype=float)
    try:
        q = np.asarray(measured, dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid measured joints for HOLD") from error
    if q.shape != limits.shape[:-1] or not np.all(np.isfinite(q)):
        raise RuntimeError("Invalid measured joints for HOLD")
    if np.any(q < limits[:, 0] - .02) or np.any(q > limits[:, 1] + .02):
        raise RuntimeError("Measured joint position outside source limits")
    return np.clip(q, limits[:, 0], limits[:, 1])


@dataclass(frozen=True)
class JointLimits:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        for name, shape in (("position", (7, 2)), ("velocity", (7,)), ("acceleration", (7,))):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {name} limits")
            if name != "position" and np.any(value <= 0):
                raise ValueError(f"Nonpositive {name} limits")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.position[:, 0] >= self.position[:, 1]):
            raise ValueError("Empty joint position interval")

    @classmethod
    def simulation(cls, model_limits):
        return cls(model_limits, SIM_VELOCITY, SIM_ACCELERATION)

    @classmethod
    def from_readback(cls, angles, accelerations, model_limits):
        if set(angles) != set(range(1, 8)) or set(accelerations) != set(range(1, 8)):
            raise RuntimeError("Selected controller joint limit readback incomplete; motors were not enabled")
        position = np.radians([[angles[j]["lower_deg"], angles[j]["upper_deg"]] for j in range(1, 8)])
        position[:, 0] = np.maximum(position[:, 0], model_limits[:, 0])
        position[:, 1] = np.minimum(position[:, 1], model_limits[:, 1])
        velocity = np.minimum(MANUAL_VELOCITY, [angles[j]["max_velocity_rad_s"] for j in range(1, 8)])
        return cls(position, velocity, [accelerations[j] for j in range(1, 8)])
