"""Render and inspect assembly poses. Joint values are visualization examples only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_pose(model, data):
    data.qpos[:] = 0
    values = {"top_camera_stand_joint": 0.0}
    for side, sign in [("left", -1), ("right", 1)]:
        values.update({f"{side}_joint2": 1.2, f"{side}_joint3": sign * 1.57,
                       f"{side}_joint4": 1.0, f"{side}_gripper_joint1": .018,
                       f"{side}_gripper_joint2": -.018})
    for name, value in values.items():
        joint = model.joint(name)
        assert joint.range[0] <= value <= joint.range[1], name
        data.qpos[joint.qposadr[0]] = value
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Open the local MuJoCo model inspector")
    args = parser.parse_args()
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    model = mujoco.MjModel.from_xml_path(str(ROOT / "build/dual_nero.preview.xml"))
    data = mujoco.MjData(model)
    values = display_pose(model, data)
    mujoco.mj_forward(model, data)
    if args.viewer:
        import mujoco.viewer
        # The standard viewer exposes joint sliders. No hardware or actuator exists.
        mujoco.viewer.launch(model=model, data=data)
        return
    output = ROOT / "previews"
    output.mkdir(exist_ok=True)
    (output / "display_pose.json").write_text(json.dumps({"scope":"Visualization only; not a hardware startup pose.","joint_positions":values}, indent=2)+"\n")
    renderer = mujoco.Renderer(model, height=1000, width=1200)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.08, 0, .34]
    camera.distance = 1.75
    camera.elevation = -12
    for name, azimuth in [("perspective",145),("front",180),("side",90),("rear",0)]:
        camera.azimuth = azimuth
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / f"{name}.png")
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    camera.lookat[:] = [0,0,.34]
    camera.distance = 2.8
    camera.azimuth = 165
    camera.elevation = -12
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(output / "zero_pose.png")
    renderer.close()
    # The camera shell is untextured. Show its actual lens face and the local
    # +X forward axis so the housing's back/top cannot be mistaken for the lens.
    import numpy as np
    detail_renderer=mujoco.Renderer(model,height=600,width=800)
    detail_camera=mujoco.MjvCamera()
    detail_camera.lookat[:]=[.075,0,.61]
    detail_camera.distance=.3
    details=[]
    for azimuth,elevation,label in [(180,-15,"Top/front"),(180,50,"Lens face"),(90,-15,"Side: forward and down")]:
        detail_camera.azimuth=azimuth
        detail_camera.elevation=elevation
        detail_renderer.update_scene(data,camera=detail_camera)
        position=data.body("camera_link").xpos
        direction=data.body("camera_link").xmat.reshape(3,3)[:,0]
        geom=detail_renderer.scene.geoms[detail_renderer.scene.ngeom]
        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_ARROW,np.ones(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.25,.05,1],dtype=np.float32))
        mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_ARROW,.004,position,position+.12*direction)
        detail_renderer.scene.ngeom+=1
        frame=Image.fromarray(detail_renderer.render())
        ImageDraw.Draw(frame).text((16,16),label,fill="#172231",font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",22))
        details.append(frame)
    detail_sheet=Image.new("RGB",(2400,655),"#f4f6fa")
    for i,frame in enumerate(details):
        detail_sheet.paste(frame,(800*i,55))
    ImageDraw.Draw(detail_sheet).text((18,15),"Original top camera mount | orange arrow: optical viewing direction (+X in camera_link)",fill="#172231",font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",24))
    detail_sheet.save(output/"camera_diagnostic.png")
    detail_renderer.close()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 28)
    small = ImageFont.truetype(font_path, 20)
    sheet = Image.new("RGB", (1800, 1690), "#f4f6fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((36,25),"NERO dual-arm stand | supplied assembly", font=font, fill="#172231")
    draw.text((36,68),"Arms posed toward robot front (+X); original camera mounts. Geometry preview only.", font=small, fill="#465465")
    for (name,label),x,y in zip([("perspective","Perspective"),("front","Front"),("side","Side"),("zero_pose","Source zero pose")],[0,900,0,900],[115,115,900,900]):
        frame=Image.open(output/f"{name}.png").resize((900,750),Image.Resampling.LANCZOS)
        sheet.paste(frame,(x,y));draw.text((x+24,y+15),label,font=font,fill="#26364a")
    sheet.save(output/"assembly_overview.png")
    print(json.dumps({"preview":str(output/"assembly_overview.png"),"pose":values}))


if __name__ == "__main__":
    main()
