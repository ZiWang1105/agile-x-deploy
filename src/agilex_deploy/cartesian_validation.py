"""Validate sampled Cartesian rotation arcs without changing the IK objective.

Callers supply separate planning kinematics and IK instances, or hold their
exclusive lock for this function's entire duration. Both objects are mutated.
This checks sampled pose residuals and joint positions, not collisions,
continuous-path feasibility, or physical tracking and stopping behavior.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .cartesian_path import CartesianPath, PathResolution
from .contract import SIDES, finite_tuple


class CartesianPlanningCancelled(RuntimeError):
    """The operation stopped while its path was being validated."""


class CartesianPathUnavailable(RuntimeError):
    """Neither sampled rotation arc passed the requested pose and joint limits."""

    def __init__(self, report):
        self.report = report
        failures = "; ".join(
            f"{candidate['rotation_arc']}: {candidate['failure']}"
            for candidate in report["candidates"])
        super().__init__(f"No validated Cartesian path for {report['side']}; {failures}")

    def __reduce__(self):
        # Preserve structured failure details across the planner process boundary.
        return type(self), (self.report,)


def plan_cartesian_path(kinematics, ik, side, target, joint_positions_rad, *,
                        position_tolerance_m=.005, orientation_tolerance_rad=.05,
                        position_limits=None, resolution=PathResolution(), cancelled=None):
    """Return ``(rotation_arc, report)`` for the first feasible sampled arc.

    Try the shortest arc first, then its complement. Each candidate starts at
    measured feedback and seeds each unchanged official IK solve with the last
    solution. IPOPT success alone does not establish pose reachability. Residual
    limits must match the requesting operation's completion criteria.

    ``position_limits`` accepts all fourteen joints or the selected seven. The
    provided bounds intersect model bounds; they never widen model constraints.
    ``cancelled`` is an optional no-argument callback returning a boolean. It is
    checked before and after every solve and raises CartesianPlanningCancelled.
    No joint targets are dispatched by this function.
    """
    if side not in SIDES:
        raise ValueError("side must be left or right")
    if any(not np.isfinite(value) or value <= 0 for value in
           (position_tolerance_m, orientation_tolerance_rad)):
        raise ValueError("Pose tolerances must be positive and finite")
    q_start = np.asarray(finite_tuple(joint_positions_rad, 14, "joint_positions_rad"))
    index = SIDES.index(side)
    indices = slice(index * 7, (index + 1) * 7)
    limits = np.asarray(kinematics.limits, dtype=float)[indices].copy()
    if position_limits is not None:
        provided = np.asarray(position_limits, dtype=float)
        if provided.shape == (14, 2):
            provided = provided[indices]
        if (provided.shape != (7, 2) or not np.all(np.isfinite(provided))
                or np.any(provided[:, 0] >= provided[:, 1])):
            raise ValueError("position_limits must contain seven or fourteen finite joint intervals")
        limits[:, 0] = np.maximum(limits[:, 0], provided[:, 0])
        limits[:, 1] = np.minimum(limits[:, 1], provided[:, 1])
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Planning position limits do not overlap model bounds")
    if np.any(q_start[indices] < limits[:, 0]) or np.any(q_start[indices] > limits[:, 1]):
        raise ValueError("Planning start is outside joint position limits")

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise CartesianPlanningCancelled("Cartesian planning cancelled")

    check_cancelled()
    kinematics.update(q_start)
    start = kinematics.pose(side)
    report = dict(side=side, selected_arc=None, position_tolerance_m=float(position_tolerance_m),
                  orientation_tolerance_rad=float(orientation_tolerance_rad),
                  start_joint_positions_rad=q_start.tolist(),
                  start_pose=dict(position_m=list(start.position_m),
                                  quaternion_xyzw=list(start.quaternion_xyzw)),
                  position_limits_rad=limits.tolist(), candidates=[], collision_checked=False)
    for arc in ("shortest", "alternate"):
        check_cancelled()
        path = CartesianPath(start, target, resolution, rotation_arc=arc)
        candidate = dict(rotation_arc=arc, segments=len(path.waypoints) - 1,
                         angle_rad=path.angle_rad, rotation_vector_rad=path.rotation.tolist(),
                         samples_checked=0, valid=False,
                         max_position_error_m=0., max_orientation_error_rad=0.)
        report["candidates"].append(candidate)
        q = q_start.copy()
        seed = q[indices].copy()
        for waypoint_index, waypoint in enumerate(path.waypoints):
            check_cancelled()
            try:
                solution = np.asarray(ik.solve(side, waypoint, seed), dtype=float)
            except RuntimeError as error:
                check_cancelled()
                candidate.update(failure=f"waypoint {waypoint_index} IK failed: {error}",
                                 failed_waypoint=waypoint_index)
                break
            check_cancelled()
            candidate["samples_checked"] += 1
            if solution.shape != (7,) or not np.all(np.isfinite(solution)):
                candidate.update(failure=f"waypoint {waypoint_index} returned invalid joint positions",
                                 failed_waypoint=waypoint_index)
                break
            if np.any(solution < limits[:, 0]) or np.any(solution > limits[:, 1]):
                candidate.update(failure=f"waypoint {waypoint_index} exceeds joint position limits",
                                 failed_waypoint=waypoint_index)
                break
            q[indices] = solution
            kinematics.update(q)
            actual = kinematics.pose(side)
            position_error = float(np.linalg.norm(np.subtract(waypoint.position_m, actual.position_m)))
            orientation_error = float((Rotation.from_quat(waypoint.quaternion_xyzw)
                                       * Rotation.from_quat(actual.quaternion_xyzw).inv()).magnitude())
            if not np.isfinite(position_error) or not np.isfinite(orientation_error):
                candidate.update(failure=f"waypoint {waypoint_index} has nonfinite pose residual",
                                 failed_waypoint=waypoint_index)
                break
            candidate["max_position_error_m"] = max(candidate["max_position_error_m"], position_error)
            candidate["max_orientation_error_rad"] = max(candidate["max_orientation_error_rad"], orientation_error)
            if position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad:
                candidate.update(
                    failure=(f"waypoint {waypoint_index} pose residual "
                             f"{position_error:.6g} m / {orientation_error:.6g} rad exceeds tolerance"),
                    failed_waypoint=waypoint_index, position_error_m=position_error,
                    orientation_error_rad=orientation_error)
                break
            seed = solution.copy()
        else:
            candidate.update(valid=True, endpoint_joints_rad=seed.tolist())
            report["selected_arc"] = arc
            check_cancelled()
            return arc, report
    raise CartesianPathUnavailable(report)
