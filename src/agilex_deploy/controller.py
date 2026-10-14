"""Measured-state, damped differential IK with a single bounded integrator.

This is a new NumPy/MuJoCo Jacobian solver, not AgileX's CasADi/IPOPT solver.
Both backends use it unchanged. Limits below bound requested joint motion;
they do not certify measured dynamics, collision clearance, or hardware safety.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SIDES
from .model import CONTROL_DT

from .motion_limits import SIM_VELOCITY, SIM_ACCELERATION

# Offline defaults only. Hardware replaces these with selected-arm live readback.
MAX_VELOCITY = np.tile(SIM_VELOCITY, 2)
MAX_ACCELERATION = np.tile(SIM_ACCELERATION, 2)


def pose_error(actual, target):
    position = np.asarray(target.position_m) - actual.position_m
    rotation = (Rotation.from_quat(target.quaternion_xyzw) *
                Rotation.from_quat(actual.quaternion_xyzw).inv()).as_rotvec()
    return np.r_[position, rotation]


class Controller:
    def __init__(self, kinematics, q, grippers, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION):
        self.kinematics = kinematics
        self.reference = np.array(q, dtype=float)
        self.velocity = np.zeros(14)
        self.grippers = list(grippers)
        self.set_motion_limits(max_velocity, max_acceleration)
        self.home = self.reference.copy()

    def set_motion_limits(self, velocity, acceleration):
        arrays = [np.broadcast_to(np.asarray(v, dtype=float), (14,)).copy()
                  for v in (velocity, acceleration)]
        if any(not np.all(np.isfinite(v)) or np.any(v <= 0) for v in arrays):
            raise ValueError("Joint speed and acceleration limits must be positive and finite")
        self.max_velocity, self.max_acceleration = arrays

    def hold(self, q, grippers):
        # HOLD intentionally overrides trajectory continuity to stop pursuing
        # the old goal; the backend still decelerates through its servos.
        self.reference[:] = q
        self.velocity[:] = 0
        self.grippers[:] = [None if g is None else float(np.clip(g, 0, 1)) for g in grippers]

    def step(self, sample, action):
        q = np.asarray(sample.positions)
        kin = self.kinematics
        kin.update(q, sample.grippers)
        desired = np.zeros(14)
        errors = {}
        for i, side in enumerate(SIDES):
            error = pose_error(kin.pose(side), getattr(action, side).pose)
            errors[side] = (float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
            indices = slice(i * 7, (i + 1) * 7)
            # Weight angular task rows in meters per radian.
            weights = np.array([1, 1, 1, .4, .4, .4])
            jac = weights[:, None] * kin.jacobian(side)[:, indices]
            inverse = jac.T @ np.linalg.solve(jac @ jac.T + .025**2 * np.eye(6), np.eye(6))
            twist = 4 * weights * error
            desired[indices] = inverse @ twist
            # Soft posture preference only in the local null space.
            desired[indices] += (np.eye(7) - inverse @ jac) @ (.08 * (self.home[indices] - q[indices]))
        # Scale each arm's joint vector together. Independent speed clipping
        # distorts the Cartesian direction, especially at commissioning speeds.
        for i in range(len(SIDES)):
            indices = slice(i * 7, (i + 1) * 7)
            ratio = float(np.max(np.abs(desired[indices]) / self.max_velocity[indices]))
            if ratio > 1:
                desired[indices] /= ratio
        lower, upper = kin.limits.T
        # Braking-distance limits approach position bounds without demanding
        # a velocity discontinuity at the last integration step.
        braking_step = self.max_acceleration * CONTROL_DT
        speed_up = np.sqrt(braking_step**2 + 2 * self.max_acceleration * np.maximum(0, upper - self.reference)) - braking_step
        speed_down = np.sqrt(braking_step**2 + 2 * self.max_acceleration * np.maximum(0, self.reference - lower)) - braking_step
        desired = np.clip(desired, -np.minimum(self.max_velocity, speed_down), np.minimum(self.max_velocity, speed_up))
        velocity = np.clip(desired, self.velocity - self.max_acceleration * CONTROL_DT,
                           self.velocity + self.max_acceleration * CONTROL_DT)
        candidate = np.clip(self.reference + velocity * CONTROL_DT, lower, upper)
        self.velocity = (candidate - self.reference) / CONTROL_DT
        self.reference = candidate
        for i, side in enumerate(SIDES):
            goal = getattr(action, side).gripper_open
            if goal is not None:
                if self.grippers[i] is None:
                    raise ValueError("Gripper command requires calibrated feedback")
                self.grippers[i] += float(np.clip(goal - self.grippers[i], -.6 * CONTROL_DT, .6 * CONTROL_DT))
        return self.reference.copy(), self.grippers.copy(), errors
