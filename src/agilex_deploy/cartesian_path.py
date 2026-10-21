"""Online Cartesian waypoint tracking inspired by CaP-X's PyRoKi planner.

The path is linear in position and follows a selected quaternion rotation arc.
Official AgileX IK still supplies every joint goal. This is a geometric path
tracker, not differential IK, a collision planner, or a reachability certificate.
See docs/MOTION_PLANNING.md for provenance and execution semantics.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contract import Pose


@dataclass(frozen=True)
class PathResolution:
    """Geometric sampling resolution, not Cartesian velocity limits."""

    translation_m: float = .01
    rotation_rad: float = .05

    def __post_init__(self):
        if any(not np.isfinite(x) or x <= 0
               for x in (self.translation_m, self.rotation_rad)):
            raise ValueError("Path resolution must be positive and finite")


class CartesianPath:
    def __init__(self, start: Pose, target: Pose, resolution=PathResolution(), *,
                 rotation_arc="shortest"):
        if rotation_arc not in ("shortest", "alternate"):
            raise ValueError("rotation_arc must be shortest or alternate")
        self.start, self.target = start, target
        self.rotation_arc = rotation_arc
        self.translation = np.asarray(target.position_m) - start.position_m
        rotations = Rotation.from_quat([start.quaternion_xyzw, target.quaternion_xyzw])
        self.rotation = (rotations[1] * rotations[0].inv()).as_rotvec()
        self.distance_m = float(np.linalg.norm(self.translation))
        self.angle_rad = float(np.linalg.norm(self.rotation))
        # The complementary arc reaches exactly the same orientation while
        # traversing the opposite direction. Equal orientations have no unique
        # rotation axis; keep their path stationary instead of inventing a spin.
        if rotation_arc == "alternate" and self.angle_rad > 1e-10:
            self.rotation *= (self.angle_rad - 2 * np.pi) / self.angle_rad
            self.angle_rad = float(np.linalg.norm(self.rotation))
        angular_resolution = (min(resolution.rotation_rad, np.pi / 2)
                              if rotation_arc == "alternate" else resolution.rotation_rad)
        count = max(1, int(np.ceil(max(self.distance_m / resolution.translation_m,
                                       self.angle_rad / angular_resolution))))
        if count > 4096:
            raise ValueError("Path exceeds 4096 segments; increase sampling resolution")
        fractions = np.linspace(0., 1., count + 1)
        orientations = (Slerp([0, 1], rotations)(fractions).as_quat()
                        if rotation_arc == "shortest" else
                        (Rotation.from_rotvec(fractions[:, None] * self.rotation)
                         * rotations[0]).as_quat())
        positions = np.asarray(start.position_m) + fractions[:, None] * self.translation
        self.waypoints = tuple(Pose(p, q) for p, q in zip(positions, orientations))
        # Preserve the caller's exact endpoint, including quaternion sign.
        self.waypoints = (*self.waypoints[:-1], target)
        self.index = 1

    def advance(self, measured: Pose, linear_lookahead_m=0., angular_lookahead_rad=0.):
        """Choose a waypoint ahead of feedback over the supplied stopping horizon.

        The controller grows lookahead from reference velocity and joint braking
        limits. Multiple intermediate samples may be passed in one tick without
        stopping at each. The final waypoint remains active so the official
        solver and servo loop can refine it to the operation's completion limits.
        """
        if any(not np.isfinite(value) or value < 0
               for value in (linear_lookahead_m, angular_lookahead_rad)):
            raise ValueError("Path lookahead must be finite and nonnegative")
        segments = len(self.waypoints) - 1
        # These are waypoint switching tolerances, not goal success criteria.
        # Floors accommodate the regularized official IK's nonzero residual.
        position_tolerance = max(.001, .5 * self.distance_m / segments) + linear_lookahead_m
        rotation_tolerance = max(.005, .5 * self.angle_rad / segments) + angular_lookahead_rad
        if self.rotation_arc == "alternate":
            # SO(3) distance wraps at pi. A larger lookahead could accept every
            # sample of a nearly full turn from the unchanged starting pose and
            # skip its selected direction. Require feedback to traverse each
            # quarter-turn before later samples can become active.
            rotation_tolerance = min(rotation_tolerance, np.pi / 2)
        # An unchanged Cartesian component remains an IK objective, but must
        # not demand full settling at every intermediate sample. In particular,
        # a translation with fixed orientation should not stop at every point
        # while transient servo orientation error decays.
        while self.index < segments:
            waypoint = self.waypoints[self.index]
            position_error = np.linalg.norm(np.asarray(waypoint.position_m) - measured.position_m)
            rotation_error = np.linalg.norm(
                (Rotation.from_quat(waypoint.quaternion_xyzw)
                 * Rotation.from_quat(measured.quaternion_xyzw).inv()).as_rotvec())
            position_reached = self.distance_m < 1e-10 or position_error <= position_tolerance
            rotation_reached = self.angle_rad < 1e-10 or rotation_error <= rotation_tolerance
            if not (position_reached and rotation_reached):
                break
            self.index += 1
        return self.waypoints[self.index]


class CartesianPlanner:
    """Independent per-arm paths; changing a jaw command never restarts a path."""

    def __init__(self, resolution=PathResolution()):
        self.resolution = resolution
        self.paths = {}
        self.revisions = {"left": 0, "right": 0}

    def reset(self):
        self.paths.clear()

    def target(self, side, measured, goal, *, linear_lookahead_m=0., angular_lookahead_rad=0.,
               rotation_arc="shortest", validated_start=None):
        if validated_start is not None and not isinstance(validated_start, Pose):
            raise ValueError("validated_start must be a Pose or None")
        path = self.paths.get(side)
        if (path is None or path.target != goal or path.rotation_arc != rotation_arc
                or getattr(path, "validated_start", None) != validated_start):
            # A validated anchor freezes the selected geometry. Recomputing an
            # arc from slightly changed feedback can reverse its axis near pi.
            # Feedback still owns progress along the frozen path below.
            path = CartesianPath(validated_start if validated_start is not None else measured,
                                 goal, self.resolution, rotation_arc=rotation_arc)
            path.validated_start = validated_start
            self.paths[side] = path
            self.revisions[side] += 1
        return path.advance(measured, linear_lookahead_m, angular_lookahead_rad)

    def status(self):
        return {side: dict(waypoint=path.index, segments=len(path.waypoints) - 1,
                           revision=self.revisions[side],
                           rotation_arc=path.rotation_arc,
                           target=dict(position_m=list(path.target.position_m),
                                       quaternion_xyzw=list(path.target.quaternion_xyzw)),
                           distance_m=path.distance_m, angle_rad=path.angle_rad)
                for side, path in self.paths.items()}
