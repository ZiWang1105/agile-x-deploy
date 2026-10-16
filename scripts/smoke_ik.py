"""Run a repeatable dual-arm target through the public API and render evidence."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from agilex_deploy import Action, ArmTarget, Pose, Robot
from agilex_deploy.backends import SimBackend
from agilex_deploy.controller import pose_error
from agilex_deploy.interactive import make_display, write_handles
from agilex_deploy.model import ROOT
from agilex_deploy.official_config import official_parameters, solver_parameters, SOURCE


def main():
    output = ROOT / "previews"
    output.mkdir(exist_ok=True)
    with Robot(SimBackend()) as robot:
        before = robot.observe()
        arms = {}
        for side, dy in (("left", -.02), ("right", .02)):
            p = getattr(before, f"{side}_tcp")
            q = (Rotation.from_rotvec([0, .18, .1]) * Rotation.from_quat(p.quaternion_xyzw)).as_quat()
            arms[side] = ArmTarget(Pose(np.array(p.position_m) + [.06, dy, .07], q), 1)
        action = Action(**arms)
        velocity, max_velocity, max_acceleration = np.zeros(14), 0., 0.
        tick_times = []
        for i in range(700):
            if i % 10 == 0:
                robot.submit_target(action, robot.observe().observation_id)
            began = time.monotonic()
            robot.tick()
            tick_times.append((time.monotonic() - began) * 1000)
            now = robot.controller.velocity.copy()
            max_velocity = max(max_velocity, float(np.max(abs(now))))
            max_acceleration = max(max_acceleration, float(np.max(abs(now - velocity))) / .01)
            velocity = now
        after = robot.observe()
        errors = {}
        for side in ("left", "right"):
            e = pose_error(getattr(after, f"{side}_tcp"), getattr(action, side).pose)
            errors[side] = {"position_mm": float(np.linalg.norm(e[:3]) * 1000),
                            "orientation_deg": float(np.degrees(np.linalg.norm(e[3:])))}
        report = dict(solver="AgileX QuestArmTeleop ArmIK", parameters=solver_parameters(),
                      upstream_parameters=official_parameters(),
                      source=json.loads((SOURCE / "manifest.json").read_text())["questarmteleop_commit"],
                      simulation_seconds=7, target=action.to_dict(), errors=errors,
                      tick_ms=dict(zip(("p50", "p95", "max"), np.percentile(tick_times, [50, 95, 100]).tolist())),
                      max_requested_velocity_rad_s=max_velocity,
                      max_requested_acceleration_rad_s2=max_acceleration,
                      notes="Unmodified official IK with explicit configs/ik.yaml tuning. Actuated dynamics, ideal gravity compensation, contacts disabled; no hardware test.")
        for side in ("left", "right"):
            initial_error = pose_error(getattr(before, f"{side}_tcp"), getattr(action, side).pose)
            assert errors[side]["position_mm"] < .5 * np.linalg.norm(initial_error[:3]) * 1000, report
            assert errors[side]["position_mm"] < 1.0, report
            assert errors[side]["orientation_deg"] < .1, report
        assert after.mode != "fault", after.reason
        (output / "official_ik_report.json").write_text(json.dumps(report, indent=2) + "\n")
        kin = make_display()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.12, 0, .34]
        camera.distance, camera.azimuth, camera.elevation = 1.85, 145, -15
        panels = []
        options = mujoco.MjvOption()
        options.geomgroup[5] = 1
        with mujoco.Renderer(kin.model, height=850, width=900) as renderer:
            for label, o in (("INITIAL | red targets are requested TCPs", before),
                             ("AFTER 7 s | official solver with local pose weights", after)):
                kin.update(o.joint_positions_rad, o.gripper_position)
                write_handles(kin, action)
                mujoco.mj_forward(kin.model, kin.data)
                renderer.update_scene(kin.data, camera=camera, scene_option=options)
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 900, 40), fill="#172638")
                draw.text((16, 13), label, fill="white")
                panels.append(panel)
        combined = Image.new("RGB", (1800, 900), "#172638")
        for i, panel in enumerate(panels):
            combined.paste(panel, (900 * i, 0))
        draw = ImageDraw.Draw(combined)
        draw.text((20, 867), f"OFFICIAL AGILEX IK | w_pos={report['parameters']['w_pos']:g}, w_ori={report['parameters']['w_ori']:g} | same sim/real contract | contacts disabled", fill="white")
        combined.save(output / "official_ik.png")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
