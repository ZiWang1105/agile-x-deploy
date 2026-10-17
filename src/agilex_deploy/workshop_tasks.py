"""Organizer-owned scenes and scoring. Never exported through participant tools."""

import copy
import json

import mujoco
import numpy as np

from .agent_sim import make_scene
from .backends import SimBackend
from .model import ROOT

TASKS = {
    "place": "Put the red cube in the green tray, release it, and retract your hand.",
    "select": "Put the blue cube in the green tray; leave the red cube where it is.",
    "sort": "Put the red cube in the red tray and the blue cube in the blue tray. Release both and retract.",
}
COLORS = {
    "red": [0.9, 0.08, 0.04, 1],
    "blue": [0.04, 0.12, 0.9, 1],
    "green": [0.08, 0.65, 0.12, 1],
}


def scene_for(task="place", seed=0):
    if task not in TASKS:
        raise ValueError(f"Unknown task; choose {list(TASKS)}")
    scene = json.loads((ROOT / "configs/agent/cube_scene.json").read_text())
    rng = np.random.default_rng(seed)
    scene.pop("cube")
    scene["objects"], scene["trays"] = [], []
    assignments = (
        {"red": "green"}
        if task == "place"
        else ({"blue": "green"} if task == "select" else {"red": "red", "blue": "blue"})
    )
    for index, color in enumerate(["red"] if task == "place" else ["red", "blue"]):
        y = 0.13 if index == 0 else -0.13
        jitter = rng.uniform(-0.008, 0.008, 2) if seed else np.zeros(2)
        scene["objects"].append(
            dict(
                name=f"{color}_cube",
                body_name=f"task_{color}",
                position_m=[0.35 + float(jitter[0]), y + float(jitter[1]), 0.126],
                half_size_m=0.025,
                mass_kg=0.04,
                rgba=COLORS[color],
                color=color,
            )
        )
        if color in assignments:
            tray_color = assignments[color]
            scene["trays"].append(
                dict(
                    name=f"{tray_color}_tray",
                    color=tray_color,
                    center_m=[0.49, y, 0.103],
                    half_size_xy_m=[0.065, 0.065],
                    rgba=COLORS[tray_color],
                )
            )
    scene["task"] = dict(
        id=task, instruction=TASKS[task], assignments=assignments, seed=seed
    )
    return scene


class WorkshopSim(SimBackend):
    def __init__(self, task="place", seed=0):
        self.scene_config = scene_for(task, seed)
        model, _ = make_scene(self.scene_config)
        super().__init__(self.scene_config["initial_joint_positions_rad"], model=model)
        self.truth = []
        self.video_frames = []
        self.initial = {
            o["color"]: np.array(o["position_m"]) for o in self.scene_config["objects"]
        }
        self.max_lift = {c: 0.0 for c in self.initial}
        self.max_displacement = {c: 0.0 for c in self.initial}
        self.dwell = {c: 0.0 for c in self.scene_config["task"]["assignments"]}
        self.collisions = 0
        self.last_time = 0.0

    def advance(self):
        super().advance()
        self.record_truth()

    def record_truth(self):
        dt = float(self.data.time) - self.last_time
        self.last_time = float(self.data.time)
        contacts = {c: [] for c in self.initial}
        environment_collisions = []
        for contact in self.data.contact:
            if contact.dist > 0.001:
                continue
            names = [
                self.model.geom(g).name
                or self.model.body(self.model.geom_bodyid[g]).name
                for g in (contact.geom1, contact.geom2)
            ]
            for color in contacts:
                if f"{color}_cube" in names:
                    contacts[color].append(
                        names[1] if names[0] == f"{color}_cube" else names[0]
                    )
            if contact.dist < -0.001 and any(
                n == "table" or n == "floor" or "_tray_" in n for n in names
            ):
                if any(n.startswith(("left_", "right_")) for n in names):
                    environment_collisions.append(names)
        self.collisions += bool(environment_collisions)
        states = {}
        assignments = self.scene_config["task"]["assignments"]
        for obj in self.scene_config["objects"]:
            color = obj["color"]
            body = self.data.body(obj["body_name"])
            center = body.xpos.copy()
            self.max_displacement[color] = max(
                self.max_displacement[color],
                float(np.linalg.norm(center - self.initial[color])),
            )
            self.max_lift[color] = max(
                self.max_lift[color], float(center[2] - self.initial[color][2])
            )
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body.id, velocity, 0
            )
            inside = released = False
            if color in assignments:
                tray = next(
                    t
                    for t in self.scene_config["trays"]
                    if t["color"] == assignments[color]
                )
                # Project all cube corners; center-only scoring accepts overhanging placements.
                half = obj["half_size_m"]
                extent = np.abs(body.xmat.reshape(3, 3)) @ np.full(3, half)
                inside = bool(
                    np.all(
                        np.abs(center[:2] - np.array(tray["center_m"][:2])) + extent[:2]
                        < np.array(tray["half_size_xy_m"]) - 0.003
                    )
                )
                released = not any(
                    n.startswith(("left_", "right_")) for n in contacts[color]
                )
                stable = (
                    inside
                    and released
                    and f"{tray['name']}_floor" in contacts[color]
                    and np.linalg.norm(velocity[3:]) < 0.02
                    and np.linalg.norm(velocity[:3]) < 0.2
                    and self.max_lift[color] >= 0.025
                )
                self.dwell[color] = self.dwell[color] + dt if stable else 0.0
            states[color] = dict(
                position_m=center.tolist(),
                contacts=contacts[color],
                inside=inside,
                released=released,
            )
        self.truth.append(
            dict(
                time_s=self.last_time,
                objects=states,
                environment_collisions=environment_collisions,
            )
        )
        if len(self.truth) % 10 == 0:
            self.video_frames.append((self.last_time, self.data.qpos.copy()))

    def evaluate(self):
        assignments = self.scene_config["task"]["assignments"]
        distractor_ok = all(
            self.max_displacement[c] < 0.01
            for c in self.initial
            if c not in assignments
        )
        return dict(
            success=all(v >= 1.0 for v in self.dwell.values())
            and distractor_ok
            and self.collisions == 0,
            task=self.scene_config["task"]["id"],
            placement_dwell_s=copy.deepcopy(self.dwell),
            max_lift_m=copy.deepcopy(self.max_lift),
            distractor_unchanged=distractor_ok,
            max_displacement_m=copy.deepcopy(self.max_displacement),
            robot_environment_collision_frames=self.collisions,
            criterion="lift >=25 mm; full footprint in tray; released, tray-supported and stable >=1 s; distractor translation <10 mm throughout; no penetrating arm/environment contacts",
        )
