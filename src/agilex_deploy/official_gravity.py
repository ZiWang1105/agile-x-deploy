"""Offline NERO torque calculation using AgileX's pinned dynamics helper.

This module has no CAN, enable, mode-switch or command-sending API. Its output
is a model prediction, not a verified physical torque command. Mechanical
geometry agreement does not qualify inertias, motor torque scaling, payload,
startup transitions, or behavior on loss of the host process.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import SIDES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/agilex_gravity"


def official_dynamics_class():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    path = SOURCE / "agx_pinocchio.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("Official gravity helper checksum mismatch")
    spec = importlib.util.spec_from_file_location("_agilex_official_gravity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AgxPinocchio


def arm_dynamics_description(side, assembled_urdf=None):
    """Extract a mounted arm's inertias, including gripper but no wrist camera.

    Coordinates of each arm remain in its own base frame. The stand's fixed
    rotation is returned separately for the official gravity-vector transform.
    Mesh geometry is irrelevant to this calculation and is omitted. No inertial
    parameters or joint axes are tuned to match the incident.
    """
    if side not in SIDES:
        raise ValueError("Expected left or right")
    tree = ET.parse(assembled_urdf or ROOT / "build/dual_nero.urdf")
    mount = tree.find(f"joint[@name='{side}_arm_joint']")
    if mount is None or mount.find("parent").get("link") != "base_link":
        raise ValueError("Arm mount must be fixed directly to base_link")
    if mount.get("type") != "fixed":
        raise ValueError("A moving arm mount requires measured base orientation")
    rpy = np.fromstring(mount.find("origin").get("rpy"), sep=" ")
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    links = {x.get("name"): x for x in tree.findall("link")}
    joints = tree.findall("joint")
    root = ET.Element("robot", name=f"{side}_nero_dynamics")

    def append_branch(name):
        link = deepcopy(links[name])
        for tag in ("visual", "collision"):
            for item in link.findall(tag):
                link.remove(item)
        root.append(link)
        for joint in joints:
            child = joint.find("child").get("link")
            if joint.find("parent").get("link") != name:
                continue
            if child == f"{side}_camera_stand_link":
                continue  # User's physical robot has no wrist cameras/mounts.
            root.append(deepcopy(joint))
            append_branch(child)

    append_branch(f"{side}_arm_base")
    return ET.tostring(root, encoding="unicode"), rotation


class OfficialGravityPreview:
    """Independent left/right inverse dynamics; no physical control path."""

    def __init__(self, assembled_urdf=None):
        cls = official_dynamics_class()
        self.arms, self.rotations, self.descriptions = {}, {}, {}
        for side in SIDES:
            xml, rotation = arm_dynamics_description(side, assembled_urdf)
            # The upstream constructor accepts a URDF filename, not XML text.
            with tempfile.TemporaryDirectory(prefix="agilex-gravity-model-") as d:
                path = Path(d) / "arm.urdf"
                path.write_text(xml)
                arm = cls(str(path))
            names = {f"{side}_joint{i}" for i in range(1, 8)}
            names |= {f"{side}_gripper_joint{i}" for i in (1, 2)}
            actual = set(arm.robot.model.names) - {"universe"}
            if actual != names or arm.nq != 9 or arm.nv != 9:
                raise ValueError("Unexpected dynamics joint names/order")
            self.arms[side], self.rotations[side] = arm, rotation
            self.descriptions[side] = xml

    @staticmethod
    def _vector(values, count, name):
        result = np.asarray(values, dtype=float)
        if result.shape != (count,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} requires {count} finite numbers")
        return result

    def model_state(self, side, positions, velocities, gripper_opening_m):
        """Use joint names, never assume Pinocchio's branch traversal order."""
        if side not in SIDES:
            raise ValueError("Expected left or right")
        q_arm = self._vector(positions, 7, "positions")
        v_arm = self._vector(velocities, 7, "velocities")
        if not np.isfinite(gripper_opening_m) or not 0 <= gripper_opening_m <= .1:
            raise ValueError("Measured gripper opening must be in [0, 0.1] m")
        model = self.arms[side].robot.model
        q, v = np.zeros(model.nq), np.zeros(model.nv)
        for i in range(7):
            joint = model.joints[model.getJointId(f"{side}_joint{i+1}")]
            q[joint.idx_q], v[joint.idx_v] = q_arm[i], v_arm[i]
        for finger, sign in ((1, 1), (2, -1)):
            joint = model.joints[model.getJointId(f"{side}_gripper_joint{finger}")]
            q[joint.idx_q] = sign * gripper_opening_m / 2
        if np.any(q < model.lowerPositionLimit) or np.any(q > model.upperPositionLimit):
            raise ValueError("Measured joints outside model limits")
        # Stationary jaws only. An object payload is not included in this model.
        return q, v

    def torques(self, side, positions, velocities, *, gripper_opening_m):
        q, v = self.model_state(side, positions, velocities, gripper_opening_m)
        arm = self.arms[side]
        # Same inverse-dynamics entry point as upstream nero/main_gc.py:
        # zero desired acceleration, measured velocity, rotated world gravity.
        tau = arm.inverse_dynamics(q, v, np.zeros(arm.nv), self.rotations[side]).copy()
        result = np.array([tau[arm.robot.model.joints[
            arm.robot.model.getJointId(f"{side}_joint{i}")].idx_v] for i in range(1, 8)])
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Nonfinite inverse-dynamics output")
        return result
