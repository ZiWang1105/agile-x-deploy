"""Contact-enabled manipulation plant and mounted RGB-D camera.

Cube state/contact truth is used only by record_truth(), never by camera or
policy observations. No attachment, weld, mocap object motion, or grasp shortcut.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .agent_sensors import RGBDFrame, quantize_depth, transform
from .backends import SimBackend
from .model import MODEL_PATH, load_model


def make_scene(scene_path):
    scene = scene_path if isinstance(scene_path, dict) else json.loads(Path(scene_path).read_text())
    visual_model = load_model()
    planar = set()
    for index in range(visual_model.nmesh):
        start, count = (
            visual_model.mesh_vertadr[index],
            visual_model.mesh_vertnum[index],
        )
        vertices = visual_model.mesh_vert[start : start + count]
        if np.linalg.matrix_rank(vertices - vertices.mean(0), tol=1e-7) < 3:
            planar.add(visual_model.mesh(index).name)
    root = ET.parse(MODEL_PATH).getroot()
    # The installed robot has one stand-mounted ZED and no wrist cameras.
    # Remove inherited wrist camera assemblies, including their mass/collisions.
    # Keep the supplied source model intact for provenance.
    wrist_assemblies = {f"{side}_camera_stand_link" for side in ("left", "right")}
    for parent in root.iter():
        for body in list(parent.findall("body")):
            if body.get("name") in wrist_assemblies:
                parent.remove(body)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(MODEL_PATH.parent / mesh.get("file")))
    root.find("option").set("iterations", "100")
    for geom in root.findall(".//geom"):
        geom.set("contype", "2")
        geom.set("conaffinity", "1")
        geom.set("friction", ".8 .005 .0001")
        if geom.get("mesh") in planar:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    root.find("worldbody/geom[@name='floor']").set("contype", "1")
    root.find("worldbody/geom[@name='floor']").set("conaffinity", "3")
    for side in ("left", "right"):
        for finger in (1, 2):
            body = root.find(f".//body[@name='{side}_gripper_link{finger}']")
            body.find("joint").set("solreflimit", ".002 1")
            body.find("joint").set("solimplimit", ".999 .999 .001")
            # Distal inner-face contact proxy, inside the source finger envelope.
            # Source mesh inner face is z=0; no oversized virtual grasp surfaces.
            # Rear black CAD pieces are not distal finger pads.
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_pad{finger}",
                type="box",
                pos="0 -.0128 .002",
                size=".012 .0075 .002",
                rgba=".08 .08 .08 0",
                contype="2",
                conaffinity="1",
                friction="1 .01 .001",
                condim="4",
                solref=".008 1",
                mass="0",
            )
        # Jaw effort is a simulation actuator parameter, not a hardware force model.
        actuator = root.find(f"actuator/position[@name='{side}_gripper']")
        actuator.set("forcerange", "-10 10")
    world = root.find("worldbody")
    table = scene["table"]
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos=" ".join(map(str, table["center_m"])),
        size=" ".join(map(str, table["half_size_m"])),
        rgba=".55 .42 .28 1",
        contype="1",
        conaffinity="3",
        friction=".8 .005 .0001",
    )
    for cube in scene.get("objects", [scene.get("cube")]):
        body = ET.SubElement(
            world, "body", name=cube.get("body_name", "task_cube"), pos=" ".join(map(str, cube["position_m"]))
        )
        ET.SubElement(body, "freejoint", name=cube.get("name", "cube") + "_free")
        ET.SubElement(
            body,
            "geom",
            name=cube.get("name", "cube"),
            type="box",
            size=" ".join([str(cube["half_size_m"])] * 3),
            mass=str(cube["mass_kg"]),
            rgba=" ".join(map(str, cube.get("rgba", [.9, .12, .04, 1]))),
            friction="1 .01 .001",
            contype="1",
            conaffinity="3",
            condim="4",
            solref=".008 1",
        )
    for tray in scene.get("trays", []):
        x, y, z = tray["center_m"]
        hx, hy = tray["half_size_xy_m"]
        rgba = " ".join(map(str, tray["rgba"]))
        for suffix, pos, size in (
            ("floor", [x, y, z], [hx, hy, .003]),
            ("front", [x+hx, y, z+.008], [.003, hy, .008]),
            ("back", [x-hx, y, z+.008], [.003, hy, .008]),
            ("left", [x, y+hy, z+.008], [hx, .003, .008]),
            ("right", [x, y-hy, z+.008], [hx, .003, .008]),
        ):
            ET.SubElement(world, "geom", name=f"{tray['name']}_{suffix}",
                          type="box", pos=" ".join(map(str, pos)),
                          size=" ".join(map(str, size)), rgba=rgba,
                          contype="1", conaffinity="3", friction=".8 .005 .0001")
    t = transform(scene["camera"]["T_base_from_optical"])
    # MuJoCo camera looks along -Z, image up +Y; RGB-D contract uses optical +Z/+Y down.
    rotation = t[:3, :3] @ np.diag([1, -1, -1])
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    ET.SubElement(
        world,
        "camera",
        name="head_rgbd",
        pos=" ".join(map(str, t[:3, 3])),
        quat=" ".join(map(str, quat)),
        fovy=str(scene["camera"]["fovy_deg"]),
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), scene


class ManipulationSim(SimBackend):
    def __init__(self, scene_path):
        model, self.scene_config = make_scene(scene_path)
        super().__init__(
            self.scene_config.get("initial_joint_positions_rad"), model=model
        )
        self.truth = []
        self.video_frames = []

    def advance(self):
        super().advance()
        self.record_truth()
        if len(self.truth) % 10 == 0:
            self.video_frames.append((float(self.data.time), self.data.qpos.copy()))

    def record_truth(self):
        cube_id = self.model.geom("cube").id
        contacts, robot_environment = [], []
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0.001:
                continue
            names = [
                self.model.geom(g).name
                or self.model.body(self.model.geom_bodyid[g]).name
                for g in (a, b)
            ]
            if cube_id in (a, b):
                contacts.append(names[1] if a == cube_id else names[0])
            elif any(n in ("table", "floor") for n in names):
                other = b if names[0] in ("table", "floor") else a
                # Fixed stand/floor contact is installation geometry, not arm collision.
                name = self.model.body(self.model.geom_bodyid[other]).name
                if name.startswith(("left_", "right_")):
                    robot_environment.append(names)
        self.truth.append(
            {
                "time_s": float(self.data.time),
                "cube_position_m": self.data.body("task_cube").xpos.tolist(),
                "cube_contacts": contacts,
                "robot_environment_contacts": robot_environment,
            }
        )


class SimCamera:
    def __init__(self, robot, plant, width=640, height=360):
        self.robot, self.plant = robot, plant
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(plant.model, height=height, width=width)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True

    def capture(self):
        with self.robot.lock:
            stamp = time.monotonic()
            self.robot._observe(self.robot._read())
            observation = self.robot.observe()
            self.renderer.update_scene(self.plant.data, camera="head_rgbd")
            rgb = self.renderer.render().copy()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            camera = self.plant.scene_config["camera"]
            f = self.height / (2 * np.tan(np.deg2rad(camera["fovy_deg"]) / 2))
            k = np.array(
                [[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]]
            )
            frame = RGBDFrame(
                rgb,
                quantize_depth(depth).astype(float) / 1000,
                k,
                transform(camera["T_base_from_optical"]),
                stamp,
                {"status": "configured_sensor_geometry", "source": "head_rgbd_mount"},
            )
            return observation, frame

    def close(self):
        self.renderer.close()


def evaluate(rows, initial_z, lift_m=0.1, retain_s=10.0):
    """Offline only. Opposing fingers, no support and no arm/environment collision."""
    longest = current = 0.0
    previous = None
    for row in rows:
        contacts = row["cube_contacts"]
        paired = any(
            all(
                any(n in contacts for n in (f"{s}_pad{f}", f"{s}_gripper_link{f}"))
                for f in (1, 2)
            )
            for s in ("left", "right")
        )
        okay = (
            row["cube_position_m"][2] >= initial_z + lift_m
            and paired
            and "table" not in contacts
            and "floor" not in contacts
            and not row["robot_environment_contacts"]
        )
        current = (
            current + (row["time_s"] - previous if previous is not None else 0)
            if okay
            else 0
        )
        longest = max(longest, current)
        previous = row["time_s"]
    return {
        "success": bool(
            current >= retain_s
            and not any(r["robot_environment_contacts"] for r in rows)
        ),
        "final_retention_s": current,
        "longest_retention_s": longest,
        "max_lift_m": max(
            (r["cube_position_m"][2] - initial_z for r in rows), default=0
        ),
        "robot_environment_collision_frames": sum(
            bool(r["robot_environment_contacts"]) for r in rows
        ),
        "criterion": "lift >= 0.1 m, opposing fingers, no support, retain >= 10 s through final verification; no arm/environment collision",
    }


def render_video(plant, path):
    """Render the complete recorded trajectory after policy execution ends."""
    import imageio_ffmpeg

    renderer = mujoco.Renderer(plant.model, height=540, width=960)
    data = mujoco.MjData(plant.model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.28, 0, 0.25]
    camera.distance, camera.azimuth, camera.elevation = 1.4, 135, -25
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (960, 540),
        fps=10,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=2,
    )
    writer.send(None)
    try:
        for stamp, q in plant.video_frames:
            data.qpos[:] = q
            mujoco.mj_forward(plant.model, data)
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            cv2.putText(
                rgb,
                f"Actual simulator trajectory | t={stamp:.1f}s",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                2,
            )
            writer.send(rgb)
    finally:
        writer.close()
        renderer.close()
