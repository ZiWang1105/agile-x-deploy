"""Adapter around the unmodified AgileX ArmIK class; no ROS or hardware imports.

Pinocchio/CasADi/IPOPT come from the pinned upstream implementation. The NERO
YAML supplies defaults with explicit local tuning in configs/ik.yaml.
Only asset resolution, stand-to-arm
frames, feedback synchronization and result validation are handled here.
"""
import ast
from functools import lru_cache
import hashlib
import inspect
import json
import os
import time
from typing import List
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SIDES
from .official_config import ROOT, SOURCE, official_parameters, solver_parameters


@lru_cache(maxsize=1)
def official_class():
    official_parameters()  # Verify the complete source snapshot before loading.
    try:
        import casadi
        import pinocchio as pin
        from pinocchio import casadi as cpin
    except ImportError as error:
        raise RuntimeError("Official IK requires Pinocchio with CasADi bindings. Run scripts/setup_official_ik.sh and use .venv-official/bin/agilex-interactive; no fallback IK is used.") from error
    path = SOURCE / "arm_ik_pose_node.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ("xyzrpy_to_mat", "ArmIK")]
    if {node.name for node in selected} != {"xyzrpy_to_mat", "ArmIK"}:
        raise RuntimeError("Official ArmIK definition missing")
    namespace = {"os": os, "List": List, "np": np, "pin": pin,
                 "cpin": cpin, "casadi": casadi, "Rotation": Rotation}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["ArmIK"]


@lru_cache(maxsize=1)
def resolved_urdf():
    """Change mesh paths only; retain the original official robot definition."""
    official_parameters()
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    tree = ET.parse(SOURCE / "nero_with_gripper_description.urdf")
    for mesh in tree.findall(".//mesh"):
        asset = manifest["assets"][mesh.get("filename")]
        path = ROOT / asset["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise RuntimeError(f"Official IK mesh checksum mismatch: {path}")
        mesh.set("filename", str(path))
    # Per-process filename avoids racing constructors in independent viewers.
    path = ROOT / "build" / f"nero.official.{os.getpid()}.urdf"
    path.parent.mkdir(exist_ok=True)
    tree.write(path, encoding="unicode")
    return path


def make_solver(*, upstream_defaults=False):
    cls = official_class()
    parameters = official_parameters() if upstream_defaults else solver_parameters()
    kwargs = {name: parameters[name] for name in inspect.signature(cls).parameters
              if name not in ("urdf_path", "package_dirs")}
    # The pinned flattened URDF added a virtual gripper axis after the YAML's
    # lock list was authored. It is not one of the seven output arm joints.
    # Lock only that named gripper axis; this does not change arm geometry.
    kwargs["locked_joints"] = [*parameters["locked_joints"], "gripper"]
    return cls(urdf_path=str(resolved_urdf()), package_dirs=[], **kwargs)


def pose_matrix(pose):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
    value[:3, 3] = pose.position_m
    return value


class OfficialDualIK:
    def __init__(self, kinematics, q):
        self.solvers = {side: make_solver() for side in SIDES}
        self.mounts = {}
        self.last_solve_ms = {}
        self.last_status = {}
        kinematics.update(q, (None, None))
        names = official_parameters()["output_joint_names"]
        for index, side in enumerate(SIDES):
            solver = self.solvers[side]
            if solver.nq != 7 or solver.active_joint_names() != names:
                raise RuntimeError("Official IK joint order does not match NERO command order")
            body = kinematics.data.body(f"{side}_arm_base")
            mount = np.eye(4)
            mount[:3, :3], mount[:3, 3] = body.xmat.reshape(3, 3), body.xpos
            self.mounts[side] = mount
            # Load IPOPT and compile symbolic functions before arming. Discard
            # this warm-up solution; initialization must never command motion.
            sample = np.asarray(q)[index*7:(index+1)*7]
            self.solve(side, kinematics.pose(side), sample)
            solver.sync_state(sample.tolist())

    def solve(self, side, target, measured):
        solver = self.solvers[side]
        measured = np.asarray(measured, dtype=float)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise ValueError("Official IK requires seven finite measured joint angles")
        # Equivalent to the upstream feedback callback before the target callback.
        solver.sync_state(measured.tolist())
        target_in_arm = np.linalg.solve(self.mounts[side], pose_matrix(target))
        began = time.monotonic()
        result = solver.solve(target_in_arm)
        stats = solver.opti.stats()
        self.last_solve_ms[side] = (time.monotonic() - began) * 1000
        self.last_status[side] = stats.get("return_status", "unknown")
        # Upstream solve_limited may return an iteration-limited iterate. Do
        # not silently dispatch that as a successful solution on hardware.
        if not stats.get("success", False):
            raise RuntimeError(f"Official {side} IK failed: {self.last_status[side]}")
        limits = solver.reduced_robot.model
        if (result.shape != (7,) or not np.all(np.isfinite(result)) or
                np.any(result < limits.lowerPositionLimit - 1e-7) or
                np.any(result > limits.upperPositionLimit + 1e-7)):
            raise RuntimeError("Official IK returned invalid joint positions")
        return result
