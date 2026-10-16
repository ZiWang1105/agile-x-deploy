"""Offscreen measured-state rendering and exact MuJoCo visual-mesh picking."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import mujoco
import numpy as np

from .camera_calibration import RIGID_ARM
from .model import Kinematics


class ModelView:
    width, height = 1024, 768

    def __init__(self):
        self.kin = Kinematics()
        self.renderer = mujoco.Renderer(
            self.kin.model, height=self.height, width=self.width
        )
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.14, 0, 0.44]
        self.camera.distance = 1.35
        self.camera.azimuth = 145
        self.camera.elevation = -18

    def scene(self, q, view=None):
        view = view or {}
        yaw, pitch, distance = [
            float(view.get(k, v))
            for k, v in [("yaw", 145), ("pitch", -18), ("distance", 1.35)]
        ]
        if (
            not all(np.isfinite([yaw, pitch, distance]))
            or not -89 <= pitch <= 89
            or not 0.15 <= distance <= 4
        ):
            raise ValueError("Invalid model-view settings")
        self.kin.update(q, (None, None))
        self.camera.azimuth, self.camera.elevation, self.camera.distance = (
            yaw,
            pitch,
            distance,
        )
        self.renderer.update_scene(
            self.kin.data, camera=self.camera, scene_option=self.option
        )

    def image(self, q, view=None, anchors=None):
        self.scene(q, view)
        for anchor in (anchors or {}).values():
            body = self.kin.data.body(anchor["body"])
            p = body.xpos + body.xmat.reshape(3, 3) @ anchor["local_xyz_m"]
            scene = self.renderer.scene
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.004] * 3),
                p,
                np.eye(3).ravel(),
                np.array([1, 0.15, 0.03, 1], np.float32),
            )
            scene.ngeom += 1
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def pick(self, q, u, v, view=None):
        if not 0 <= u < self.width or not 0 <= v < self.height:
            raise ValueError("Click outside model image")
        self.scene(q, view)
        point = np.zeros(3)
        geom, flex, skin = [np.full(1, -1, np.int32) for _ in range(3)]
        body_id = mujoco.mjv_select(
            self.kin.model,
            self.kin.data,
            self.option,
            self.width / self.height,
            u / self.width,
            1 - v / self.height,
            self.renderer.scene,
            point,
            geom,
            flex,
            skin,
        )
        name = self.kin.model.body(body_id).name if body_id >= 0 else ""
        if not RIGID_ARM.fullmatch(name):
            raise ValueError(
                "Pick a visible screw/junction on a rigid arm housing; fingers/stand/background are excluded"
            )
        body = self.kin.data.body(body_id)
        local = body.xmat.reshape(3, 3).T @ (point - body.xpos)
        return dict(
            body=name,
            local_xyz_m=local.tolist(),
            selection="MuJoCo visual-mesh ray intersection",
        )

    def close(self):
        self.renderer.close()
