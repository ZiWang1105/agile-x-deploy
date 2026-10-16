"""Export posed robot geometry for the mount illustration. No hardware access."""
from pathlib import Path
import json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[2]
model=mujoco.MjModel.from_xml_path(str(ROOT/'build/dual_nero.preview.xml'))
data=mujoco.MjData(model)
pose=json.loads((ROOT/'previews/display_pose.json').read_text())['joint_positions']
# Mounting surface normal from the supplied bracket STL, pointed horizontally.
n=np.array([-.07023567,0,-.99753044]);n/=np.linalg.norm(n)
angle=np.arctan2(n[2],n[0])
q=-.5445-angle
pose['top_camera_stand_joint']=float(q)
for name,v in pose.items():data.qpos[model.joint(name).qposadr[0]]=v
mujoco.mj_forward(model,data)
skip=model.body('top_d435_camera_link').id
parts={};count=0
for i in range(model.ngeom):
 if model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_MESH:continue
 b=model.geom_bodyid[i];anc=b;hidden=False
 while anc:
  if anc==skip:hidden=True;break
  anc=model.body_parentid[anc]
 if hidden:continue
 mid=model.geom_dataid[i];a=model.mesh_vertadr[mid];b=model.mesh_faceadr[mid]
 v=model.mesh_vert[a:a+model.mesh_vertnum[mid]]
 f=model.mesh_face[b:b+model.mesh_facenum[mid]]
 parts[f'v{count}']=v@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
 parts[f'f{count}']=f
 parts[f'c{count}']=(np.array([.12,.13,.14]) if model.geom_bodyid[i]==model.body('top_camera_stand_Link').id else model.geom_rgba[i,:3])
 count+=1
body=data.body('top_camera_stand_Link');rot=body.xmat.reshape(3,3)
x=np.array([0,-1,0]);z=np.cross(x,n)
parts['mount_rotation']=rot@np.column_stack((x,n,z))
parts['mount_origin']=body.xpos+rot@np.array([.0158929443,0,-.0299259135])
parts['count']=np.array(count)
np.savez_compressed('/tmp/zed2i_robot_geometry.npz',**parts)
print('Exported',count,'meshes; preview tilt joint',np.degrees(q),'degrees')
print('Adapter origin:',parts['mount_origin'],'front:',parts['mount_rotation'][:,1])
