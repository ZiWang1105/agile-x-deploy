"""Cartesian path tracking, official IK, and shared joint-limit execution.

Simulation and hardware use the same upstream solver and NERO parameters.
The execution layer limits joint references; it does not certify physical
stopping, torque, contact dynamics or collision clearance.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SIDES, JointAction, MixedAction
from .cartesian_path import CartesianPlanner
from .model import CONTROL_DT
from .official_ik import OfficialDualIK

from .motion_limits import SIM_VELOCITY, SIM_ACCELERATION, bounded_hold_reference

# Offline defaults only. Hardware replaces these with selected-arm live readback.
MAX_VELOCITY = np.tile(SIM_VELOCITY, 2)
MAX_ACCELERATION = np.tile(SIM_ACCELERATION, 2)


def pose_error(actual, target):
    position = np.asarray(target.position_m) - actual.position_m
    rotation = (Rotation.from_quat(target.quaternion_xyzw) *
                Rotation.from_quat(actual.quaternion_xyzw).inv()).as_rotvec()
    return np.r_[position, rotation]


class Controller:
    def __init__(self, kinematics, q, grippers, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION, *, ik=None):
        self.kinematics = kinematics
        self.reference = np.array(q, dtype=float)
        self.velocity = np.zeros(14)
        self.grippers = list(grippers)
        self.gripper_rate = .6  # Simulated actuator reference rate; hardware overrides to None.
        self.set_motion_limits(max_velocity, max_acceleration)
        self.ik = ik if ik is not None else OfficialDualIK(kinematics, q)
        self.planner = CartesianPlanner()
        self.hold(q, grippers)

    def set_motion_limits(self, velocity, acceleration):
        arrays = [np.broadcast_to(np.asarray(v, dtype=float), (14,)).copy()
                  for v in (velocity, acceleration)]
        if any(not np.all(np.isfinite(v)) or np.any(v <= 0) for v in arrays):
            raise ValueError("Joint speed and acceleration limits must be positive and finite")
        self.max_velocity, self.max_acceleration = arrays

    def hold(self, q, grippers):
        # HOLD intentionally overrides trajectory continuity to stop pursuing
        # the old goal; the backend still decelerates through its servos.
        bounded = bounded_hold_reference(q, self.kinematics.limits)
        self.reference[:] = bounded
        self.hold_reference = bounded.copy()
        self.velocity[:] = 0
        self.planner.reset()
        self.grippers[:] = [None if g is None else float(np.clip(g, 0, 1)) for g in grippers]
        self.kinematics.update(q, grippers)
        self.hold_poses = {side: self.kinematics.pose(side) for side in SIDES}

    def step(self, sample, action):
        q = np.asarray(sample.positions)
        kin = self.kinematics
        kin.update(q, sample.grippers)
        desired = np.zeros(14)
        errors = {}
        if isinstance(action, JointAction):
            self.planner.reset()
        for i, side in enumerate(SIDES):
            error = pose_error(kin.pose(side), getattr(action, side).pose)
            errors[side] = (float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
            indices = slice(i * 7, (i + 1) * 7)
            target = getattr(action, side).pose
            path_start = action.cartesian_starts[i] if isinstance(action, MixedAction) else None
            joint_control = isinstance(action, JointAction) or (
                isinstance(action, MixedAction) and side not in action.cartesian_arms)
            if joint_control:
                # A later Cartesian action must start a new path from feedback,
                # even if its endpoint matches a path used before the joint hold.
                self.planner.paths.pop(side, None)
            # An unchanged HOLD target must not move an inactive arm merely
            # because the official objective includes neutral-pose regularization.
            if (target == self.hold_poses[side] and not joint_control
                    and (path_start is None or path_start == target)
                    and np.array_equal(self.reference[indices], self.hold_reference[indices])
                    and np.all(self.velocity[indices] == 0)):
                continue
            # A replacement drag target starts a new Cartesian path at measured
            # feedback. Keep reference velocity continuous through replacement;
            # the existing joint acceleration limiter owns execution timing.
            if joint_control:
                waypoint = target
            else:
                # A nearby intermediate IK goal makes the braking law below
                # slow down even though the Cartesian path continues. Look
                # ahead over the current joint stopping horizon so intermediate
                # samples are not treated as mandatory stopping points.
                # This changes neither the joint speed nor acceleration bounds.
                horizon = float(np.max(abs(self.velocity[indices]) / self.max_acceleration[indices]))
                twist = kin.jacobian(side)[:, indices] @ self.velocity[indices]
                waypoint = self.planner.target(
                    side, kin.pose(side), target,
                    rotation_arc=(action.rotation_arcs[i]
                                  if isinstance(action, MixedAction) else "shortest"),
                    validated_start=path_start,
                    linear_lookahead_m=float(np.linalg.norm(twist[:3]) * horizon),
                    angular_lookahead_rad=float(np.linalg.norm(twist[3:]) * horizon))
            goal = (np.asarray(action.joint_positions_rad)[indices] if joint_control
                    else self.ik.solve(side, waypoint, q[indices]))
            delta = goal - self.reference[indices]
            # Decelerate toward the official joint solution. This is execution
            # timing, not an extra pose objective or a replacement IK solver.
            acceleration = self.max_acceleration[indices]
            step = acceleration * CONTROL_DT
            desired[indices] = np.sign(delta) * np.minimum(
                abs(delta) / CONTROL_DT,
                np.sqrt(step**2 + 2 * acceleration * abs(delta)) - step)
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
                if self.gripper_rate is None:
                    self.grippers[i] = float(goal)  # Firmware owns physical gripper motion timing.
                else:
                    step = self.gripper_rate * CONTROL_DT
                    self.grippers[i] += float(np.clip(goal - self.grippers[i], -step, step))
        return self.reference.copy(), self.grippers.copy(), errors
