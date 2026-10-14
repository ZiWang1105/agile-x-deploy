"""Measured-state, damped differential IK with a single bounded integrator.

This is a new NumPy/MuJoCo Jacobian solver, not AgileX's CasADi/IPOPT solver.
Both backends use it unchanged. Limits below bound requested joint motion;
they do not certify measured dynamics, collision clearance, or hardware safety.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SIDES
from .model import CONTROL_DT

MAX_VELOCITY = .6
MAX_ACCELERATION = 2.


def pose_error(actual, target):
    position = np.asarray(target.position_m) - actual.position_m
    rotation = (Rotation.from_quat(target.quaternion_xyzw) *
                Rotation.from_quat(actual.quaternion_xyzw).inv()).as_rotvec()
    return np.r_[position, rotation]


class Controller:
    def __init__(self, kinematics, q, grippers):
        self.kinematics = kinematics
        self.reference = np.array(q, dtype=float)
        self.velocity = np.zeros(14)
        self.grippers = np.array(grippers, dtype=float)
        self.home = self.reference.copy()

    def hold(self, q, grippers):
        # HOLD intentionally overrides trajectory continuity to stop pursuing
        # the old goal; the backend still decelerates through its servos.
        self.reference[:] = q
        self.velocity[:] = 0
        self.grippers[:] = np.clip(grippers, 0, 1)

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
            twist[:3] = np.clip(twist[:3], -.25, .25)
            twist[3:] = np.clip(twist[3:], -.4, .4)
            desired[indices] = inverse @ twist
            # Soft posture preference only in the local null space.
            desired[indices] += (np.eye(7) - inverse @ jac) @ (.08 * (self.home[indices] - q[indices]))
        lower, upper = kin.limits.T
        # Braking-distance limits approach position bounds without demanding
        # a velocity discontinuity at the last integration step.
        braking_step = MAX_ACCELERATION * CONTROL_DT
        speed_up = np.sqrt(braking_step**2 + 2 * MAX_ACCELERATION * np.maximum(0, upper - self.reference)) - braking_step
        speed_down = np.sqrt(braking_step**2 + 2 * MAX_ACCELERATION * np.maximum(0, self.reference - lower)) - braking_step
        desired = np.clip(desired, -np.minimum(MAX_VELOCITY, speed_down), np.minimum(MAX_VELOCITY, speed_up))
        velocity = np.clip(desired, self.velocity - MAX_ACCELERATION * CONTROL_DT,
                           self.velocity + MAX_ACCELERATION * CONTROL_DT)
        candidate = np.clip(self.reference + velocity * CONTROL_DT, lower, upper)
        if np.max(np.abs(candidate - q)) > .2:
            raise RuntimeError("Joint tracking error exceeds 0.2 rad")
        self.velocity = (candidate - self.reference) / CONTROL_DT
        self.reference = candidate
        grip_goal = np.array([getattr(action, s).gripper_open for s in SIDES])
        self.grippers += np.clip(grip_goal - self.grippers, -.6 * CONTROL_DT, .6 * CONTROL_DT)
        return self.reference.copy(), self.grippers.copy(), errors
