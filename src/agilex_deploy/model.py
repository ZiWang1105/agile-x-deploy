"""Model construction and encoder-derived kinematics, shared by both backends."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import JOINT_NAMES, SIDES, Pose

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "build/dual_nero.sim.xml"
CONTROL_DT = .01
TCP_OFFSET_M = (0, 0, .1358)  # Nominal CAD finger-root plane, not calibrated pinch point.


def build_sim_model():
    """Create a servo dynamics model without changing the geometry-only preview."""
    tree = ET.parse(ROOT / "build/dual_nero.preview.xml")
    root = tree.getroot()
    root.set("model", "dual_nero_servo_sim")
    root.find("option").attrib.update(gravity="0 0 -9.81", timestep="0.002", integrator="implicitfast")
    for body in root.findall(".//body"):
        # Simulated ideal gravity compensation; no hardware torque claim.
        body.set("gravcomp", "1")
        for joint in body.findall("joint"):
            if joint.get("name") == "top_camera_stand_joint":
                body.remove(joint)
            else:
                joint.set("damping", "0.3")
                joint.set("armature", "0.01")
    actuator = ET.SubElement(root, "actuator")
    equality = ET.SubElement(root, "equality")
    for name in JOINT_NAMES:
        joint = root.find(f".//joint[@name='{name}']")
        ET.SubElement(actuator, "position", name=name, joint=name, kp="350", kv="35",
                      ctrlrange=joint.get("range"), forcerange="-100 100")
    for side in SIDES:
        body = root.find(f".//body[@name='{side}_gripper_base']")
        ET.SubElement(body, "site", name=f"{side}_tcp", pos=" ".join(map(str, TCP_OFFSET_M)),
                      type="sphere", size="0.008", rgba="0.2 1 0.3 1", group="5")
        ET.SubElement(equality, "joint", joint1=f"{side}_gripper_joint2",
                      joint2=f"{side}_gripper_joint1", polycoef="0 -1 0 0 0",
                      solref="0.004 1")
        ET.SubElement(actuator, "position", name=f"{side}_gripper", joint=f"{side}_gripper_joint1",
                      kp="400", kv="10", ctrlrange="0 0.05", forcerange="-20 20")
    ET.indent(root)
    tree.write(MODEL_PATH, encoding="unicode")
    return MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        build_sim_model()
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def initial_positions():
    values = json.loads((ROOT / "previews/display_pose.json").read_text())["joint_positions"]
    return np.array([values.get(n, 0.) for n in JOINT_NAMES])


class Kinematics:
    def __init__(self, model=None):
        self.model = model if model is not None else load_model()
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.dids = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.limits = np.array([self.model.joint(n).range for n in JOINT_NAMES])

    def update(self, q, grippers=(0, 0)):
        self.data.qpos[self.qids] = q
        for i, side in enumerate(SIDES):
            for finger, sign in ((1, 1), (2, -1)):
                # Unknown finger positions are hidden by hardware viewers. This
                # display placeholder does not enter an observation or command.
                value = 0. if grippers[i] is None else grippers[i]
                self.data.qpos[self.model.joint(f"{side}_gripper_joint{finger}").qposadr[0]] = sign * .05 * value
        mujoco.mj_forward(self.model, self.data)

    def pose(self, side):
        site = self.data.site(f"{side}_tcp")
        return Pose(tuple(site.xpos), tuple(Rotation.from_matrix(site.xmat.reshape(3, 3)).as_quat()))

    def jacobian(self, side):
        p, r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, p, r, self.model.site(f"{side}_tcp").id)
        return np.vstack((p[:, self.dids], r[:, self.dids]))
