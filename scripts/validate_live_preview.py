"""Exercise simulated IK from live encoders without sending CAN data frames.

Run with MUJOCO_GL=egl. This validates the preview path, not physical motion,
collision clearance, encoder/model alignment, or real stop behavior.
"""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from agilex_deploy.backends import SimBackend
from agilex_deploy.can_feedback import PassiveCanFeedback
from agilex_deploy.contract import Action, ArmTarget, Pose, SIDES
from agilex_deploy.controller import pose_error
from agilex_deploy.interactive import make_display, update_mirror, write_handles
from agilex_deploy.model import ROOT
from agilex_deploy.runtime import Robot


def tx_counts(channels):
    return {side: int(Path(f"/sys/class/net/{channel}/statistics/tx_packets").read_text())
            for side, channel in channels.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/commissioning")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    display = make_display(mirror=True)
    # OpenGL setup may block Python. Open CAN after preparing the renderer.
    with mujoco.Renderer(display.model, height=900, width=1400) as renderer:
        with PassiveCanFeedback() as feedback:
            before_tx = tx_counts(feedback.channels)
            seed = feedback.read(timeout=1.)
            with Robot(SimBackend(joint_positions=seed.positions)) as robot:
                initial = robot.observe()
                action = Action(*(ArmTarget(Pose(
                    np.asarray(getattr(initial, side + "_tcp").position_m) + [0, 0, .02],
                    getattr(initial, side + "_tcp").quaternion_xyzw), 0) for side in SIDES))
                ages, skews = [], []
                for step in range(500):
                    if step % 10 == 0:
                        live = feedback.read(timeout=.1)
                        ages.append(time.monotonic() - live.timestamp_s)
                        skews.append(live.skew_s)
                        robot.submit_target(action, robot.observe().observation_id)
                    robot.tick()
                final = robot.observe()
                live = feedback.read(timeout=.1)
                display.update(final.joint_positions_rad, final.gripper_position)
                update_mirror(display, live)
                write_handles(display, action)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [.08, 0, .34]
                camera.distance, camera.azimuth, camera.elevation = 1.9, 180, -12
                option = mujoco.MjvOption()
                option.geomgroup[3:6] = 1
                renderer.update_scene(display.data, camera, scene_option=option)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 1400, 76), fill="white")
                draw.text((20, 12), "LIVE NERO JOINT MIRROR + SIMULATED IK PREVIEW", fill="black", font_size=23)
                draw.text((20, 44), "Blue = measured arms | Amber = simulated 20 mm upward target | No hardware commands", fill="black", font_size=18)
                draw.rectangle((0, 852, 1400, 900), fill="white")
                draw.text((20, 862), "Finger travel unknown: hidden. Model alignment, TCP and collision clearance are NOT verified.", fill="black", font_size=18)
                image.save(args.output / "live_preview.png")
                after_tx = tx_counts(feedback.channels)
                report = {
                    "validation_kind": "live_receive_only_plus_simulated_IK",
                    "channels": feedback.channels,
                    "hardware_before": seed.report(display.limits),
                    "hardware_after": live.report(display.limits),
                    "hardware_delta_max_deg": float(np.max(np.abs(np.degrees(np.array(live.positions) - seed.positions)))),
                    "netdev_tx_delta": {s: after_tx[s] - before_tx[s] for s in SIDES},
                    "max_feedback_age_s": max(ages), "max_feedback_skew_s": max(skews),
                    "preview_mode": final.mode,
                    "simulated_tcp_position_error_m": {s: float(np.linalg.norm(pose_error(
                        getattr(final, s + "_tcp"), getattr(action, s).pose)[:3])) for s in SIDES},
                    "physical_alignment_verified": False, "physical_motion_tested": False,
                }
                (args.output / "live_preview_report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in report.items() if not k.startswith("hardware_")}, indent=2))
                if final.mode == "fault" or any(report["netdev_tx_delta"].values()):
                    raise RuntimeError("Preview fault or unexpected CAN transmission during validation")
                if max(report["simulated_tcp_position_error_m"].values()) > .002:
                    raise RuntimeError("Simulated IK did not converge within 2 mm")


if __name__ == "__main__":
    main()
