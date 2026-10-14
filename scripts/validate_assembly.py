"""Cross-check source URDF forward kinematics against the exported MuJoCo tree."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from build_assembly import transform, vector
from preview_assembly import display_pose

ROOT = Path(__file__).resolve().parents[1]


def main():
    robot=ET.parse(ROOT/"build/dual_nero.urdf").getroot()
    joints=robot.findall("joint")
    for mesh in robot.findall(".//mesh"):
        assert (ROOT/"build"/mesh.get("filename")).is_file()
    model=mujoco.MjModel.from_xml_path(str(ROOT/"build/dual_nero.preview.xml"))
    data=mujoco.MjData(model)
    rng=np.random.default_rng(42)
    worst_position=worst_rotation=0.
    for sample in range(22):
        if sample==0:
            data.qpos[:]=0
        elif sample==1:
            display_pose(model,data)
        else:
            for i in range(model.njnt):
                data.qpos[model.jnt_qposadr[i]]=rng.uniform(*model.jnt_range[i])
        mujoco.mj_forward(model,data)
        poses={"base_link":np.eye(4)}
        pending=list(joints)
        while pending:
            progressed=False
            for joint in list(pending):
                parent=joint.find("parent").get("link")
                if parent not in poses:
                    continue
                motion=np.eye(4)
                if joint.get("type")!="fixed":
                    q=data.qpos[model.joint(joint.get("name")).qposadr[0]]
                    axis=vector(joint.find("axis").get("xyz"))
                    axis=axis/np.linalg.norm(axis)
                    if joint.get("type")=="revolute":
                        motion[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix()
                    else:
                        motion[:3,3]=axis*q
                child=joint.find("child").get("link")
                poses[child]=poses[parent]@transform(joint.find("origin"))@motion
                pending.remove(joint);progressed=True
            assert progressed,"Disconnected or cyclic tree"
        for name,t in poses.items():
            actual=data.body(name)
            worst_position=max(worst_position,float(np.linalg.norm(t[:3,3]-actual.xpos)))
            worst_rotation=max(worst_rotation,float(np.linalg.norm(t[:3,:3]-actual.xmat.reshape(3,3))))
    assert worst_position<1e-8, worst_position
    assert worst_rotation<1e-8, worst_rotation
    arm_joints=[j for j in joints if j.get("type")=="revolute" and j.get("name").startswith(("left_joint","right_joint"))]
    assert len(arm_joints)==14
    display_pose(model,data)
    mujoco.mj_forward(model,data)
    # RealSense camera_link uses +X forward; the robot's front is stand +X.
    # Check orientation semantically, not only consistency between two models.
    top_forward=data.body("camera_link").xmat.reshape(3,3)[:,0].copy()
    assert top_forward[0]>0.5 and top_forward[2]<-0.8, top_forward
    assert abs(top_forward[1])<1e-8, top_forward
    for side in ("left", "right"):
        assert data.body(side+"_gripper_base").xpos[0]>0.2
    result={"passed":True,"configurations":22,"link_frames_per_configuration":len(poses),
            "arm_dof":14,"gripper_prismatic_joints":4,"camera_bracket_revolute_joints":1,
            "mujoco_nq":model.nq,"max_fk_position_error_m":worst_position,
            "max_fk_rotation_matrix_error":worst_rotation,
            "top_camera_forward_base_xyz":top_forward.tolist(),
            "top_camera_faces_robot_front_and_down":True,
            "scope":"Tree connectivity, resource resolution, MuJoCo compilation and FK agreement only. No collision, dynamic stability or hardware validation."}
    (ROOT/"build/validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    main()
