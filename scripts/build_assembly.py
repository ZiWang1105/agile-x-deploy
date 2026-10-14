"""Expand the supplied assembly without ROS and export a portable URDF and preview MJCF.

The preview is for geometry inspection, not contact simulation or hardware control.
All source joint origins, limits and inertias are retained in the URDF.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import xacro
import xacro.substitution_args

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build"
PACKAGES = {
    "wbcd_urdf": ROOT / "vendor/wbcd_urdf",
    "agx_arm_description": ROOT / "vendor/agx_arm_sim/agx_arm_description",
    "realsense2_description": ROOT / "vendor/agx_arm_sim/realsense2_description",
}


def vector(text, default="0 0 0"):
    return np.fromstring(text or default, sep=" ")


def numbers(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def transform(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, :3] = Rotation.from_euler("xyz", vector(origin.get("rpy"))).as_matrix()
        t[:3, 3] = vector(origin.get("xyz"))
    return t


def pose(origin):
    t = transform(origin)
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return {"pos": numbers(t[:3, 3]), "quat": numbers(q[[3, 0, 1, 2]])}


def resolve(uri):
    if uri.startswith("package://"):
        package, relative = uri[10:].split("/", 1)
        path = PACKAGES[package] / relative
    elif uri.startswith("file://"):
        path = Path(uri[7:])
    else:
        path = Path(uri)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "meshes").mkdir(exist_ok=True)
    (OUT / "preview_meshes").mkdir(exist_ok=True)
    # Use the real Xacro engine; only replace ROS package discovery with a local map.
    xacro.substitution_args._eval_find = lambda name: str(PACKAGES[name])
    source = PACKAGES["wbcd_urdf"] / "urdf/dual_nero.xacro"
    document = xacro.process_file(str(source))
    robot = ET.fromstring(document.toxml())
    links = {e.get("name"): e for e in robot.findall("link")}
    joints = robot.findall("joint")
    assert len(links) == len(robot.findall("link")), "Duplicate link names"
    assert len({j.get("name") for j in joints}) == len(joints), "Duplicate joints"
    children = {}
    for j in joints:
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        assert parent in links and child in links
        assert child not in children, "Multiple parents"
        children[child] = parent
    roots = set(links) - set(children)
    assert roots == {"base_link"}, roots
    for child in links:
        seen = set()
        while child in children:
            assert child not in seen, "Cycle"
            seen.add(child)
            child = children[child]

    assets = {}
    portable = copy.deepcopy(robot)
    for mesh in portable.findall(".//mesh"):
        path = resolve(mesh.get("filename"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        filename = f"{path.stem}_{digest[:12]}{path.suffix}"
        shutil.copyfile(path, OUT / "meshes" / filename)
        mesh.set("filename", f"meshes/{filename}")
        assets[str(path.relative_to(ROOT))] = {"sha256": digest, "output": f"meshes/{filename}"}
    write_xml(portable, OUT / "dual_nero.urdf")

    mj = ET.Element("mujoco", model="dual_nero_geometry_preview")
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mj, "option", gravity="0 0 0")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1200")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2")
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.80 0.85 0.92", rgb2="0.96 0.97 0.99", width="512", height="3072")
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.3 0.3 -1", directional="true")
    ET.SubElement(world, "light", pos="-2 1 2", dir="1 -0.3 -1", diffuse="0.45 0.45 0.45")
    ET.SubElement(world, "geom", name="floor", type="plane", size="3 3 .1", pos="0 0 -0.001", rgba="0.88 0.90 0.93 1", contype="0", conaffinity="0")
    cache = {}

    def mesh_parts(path):
        if path in cache:
            return cache[path]
        scene = trimesh.load(path, force="scene")
        parts = []
        for node in scene.graph.nodes_geometry:
            matrix, key = scene.graph[node]
            geom = scene.geometry[key].copy()
            geom.apply_transform(matrix)
            index = sum(len(v) for v in cache.values()) + len(parts)
            name = f"mesh_{index:03d}"
            filename = f"preview_meshes/{name}.obj"
            # Geometry is already transformed; colors are preserved per DAE material.
            geom.export(OUT / filename, file_type="obj", include_texture=False)
            ET.SubElement(asset, "mesh", name=name, file=filename, inertia="shell")
            material = getattr(geom.visual, "material", None)
            color = getattr(material, "baseColorFactor", None)
            if color is None:
                color = getattr(material, "diffuse", None)
            parts.append((name, None if color is None else np.asarray(color) / 255))
        cache[path] = parts
        return parts

    joint_by_parent = {}
    for j in joints:
        joint_by_parent.setdefault(j.find("parent").get("link"), []).append(j)

    def body(name, parent, joint=None):
        b = ET.SubElement(parent, "body", name=name, **pose(None if joint is None else joint.find("origin")))
        link = links[name]
        inertia = link.find("inertial")
        if inertia is not None:
            info = inertia.find("inertia")
            matrix = np.array([[float(info.get("ixx")), float(info.get("ixy")), float(info.get("ixz"))],
                               [float(info.get("ixy")), float(info.get("iyy")), float(info.get("iyz"))],
                               [float(info.get("ixz")), float(info.get("iyz")), float(info.get("izz"))]])
            t = transform(inertia.find("origin"))
            matrix = t[:3, :3] @ matrix @ t[:3, :3].T
            ET.SubElement(b, "inertial", pos=numbers(t[:3, 3]), mass=inertia.find("mass").get("value"),
                          fullinertia=numbers([matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2]]))
        if joint is not None and joint.get("type") != "fixed":
            limit = joint.find("limit")
            ET.SubElement(b, "joint", name=joint.get("name"), type="slide" if joint.get("type") == "prismatic" else "hinge",
                          axis=joint.find("axis").get("xyz"), range=f"{limit.get('lower')} {limit.get('upper')}")
        for v in link.findall("visual"):
            mesh = v.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"Unhandled visual geometry on {name}")
            assert mesh.get("scale", "1 1 1") == "1 1 1", "Mesh scale needs explicit handling"
            c = v.find("material/color")
            fallback = vector(c.get("rgba") if c is not None else "0.65 0.67 0.70 1")
            for mesh_name, color in mesh_parts(resolve(mesh.get("filename"))):
                ET.SubElement(b, "geom", type="mesh", mesh=mesh_name, **pose(v.find("origin")),
                              rgba=numbers(fallback if color is None else color), mass="0", contype="0", conaffinity="0")
        for j in joint_by_parent.get(name, []):
            body(j.find("child").get("link"), b, j)
        return b

    body("base_link", world)
    write_xml(mj, OUT / "dual_nero.preview.xml")
    joint_records = [{"name":j.get("name"),"type":j.get("type"),"parent":j.find("parent").get("link"),
                      "child":j.find("child").get("link"), "origin":dict(j.find("origin").attrib) if j.find("origin") is not None else {},
                      "limit":dict(j.find("limit").attrib) if j.find("limit") is not None else {}} for j in joints]
    report = {"source":str(source.relative_to(ROOT)), "links":len(links),"joints":len(joints),
              "root":"base_link", "all_meshes_resolved":True, "assets":assets,"joint_records":joint_records,
              "scope":"Geometry/FK preview only; no actuators or contact simulation. Source inertias/limits are unqualified.",
              "notes":["Source wrist camera-stand visual and collision origins differ; retained unchanged in URDF.",
                       "Gripper joints have no mimic constraint in source; preview poses explicitly set both fingers.",
                       "The top camera bracket is a revolute joint in source, not evidence of a physical motor.",
                       "The stand DAE has no material colors; preview uses supplied URDF color."]}
    (OUT / "assembly_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"links":len(links),"joints":len(joints),"unique_source_meshes":len(assets),"outputs":["build/dual_nero.urdf","build/dual_nero.preview.xml"]}))


if __name__ == "__main__":
    main()
