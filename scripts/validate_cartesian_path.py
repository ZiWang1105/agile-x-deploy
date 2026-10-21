"""Compare direct endpoint IK and Cartesian waypoint tracking in servo simulation.

Uses no CAN devices or model API. Video is 1x simulation-time playback, rendered
after execution. It is not a wall-clock recording or collision qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agilex_deploy import Action, ArmTarget, Pose, Robot
from agilex_deploy.backends import SimBackend
from agilex_deploy.controller import pose_error
from agilex_deploy.interactive import make_display, write_handles


def execute(planned, *, adaptive_lookahead=True):
    frames, traces = [], {s: [] for s in ("left", "right")}
    errors = {s: [] for s in traces}
    with Robot(SimBackend()) as robot:
        start = robot.observe().as_action()
        goals = {}
        for side, sign in (("left", 1), ("right", -1)):
            pose = getattr(start, side).pose
            goals[side] = ArmTarget(Pose(np.asarray(pose.position_m) + [.10, -.04 * sign, .14],
                                        pose.quaternion_xyzw), 1 if side == "left" else 0)
        action = Action(**goals)
        if not planned:
            # Reproduce the previous endpoint-to-IK controller for comparison.
            robot.controller.planner.target = lambda side, measured, goal, **_: goal
        elif not adaptive_lookahead:
            # Reproduce the original short-lookahead waypoint implementation.
            original = robot.controller.planner.target
            robot.controller.planner.target = lambda side, measured, goal, **_: original(side, measured, goal)
        speed_ratio = acceleration_ratio = 0.
        previous_velocity = np.zeros(14)
        for tick in range(600):
            if tick % 5 == 0:
                robot.submit_target(action, robot.observe().observation_id)
            robot.tick()
            observed = robot.observe()
            assert observed.mode == "tracking", observed.reason
            velocity = robot.controller.velocity.copy()
            speed_ratio = max(speed_ratio, float(np.max(abs(velocity) / robot.controller.max_velocity)))
            acceleration_ratio = max(acceleration_ratio, float(np.max(
                abs(velocity - previous_velocity) / .01 / robot.controller.max_acceleration)))
            previous_velocity = velocity
            for side in traces:
                traces[side].append(getattr(observed, side + "_tcp").position_m)
                error = pose_error(getattr(observed, side + "_tcp"), getattr(action, side).pose)
                errors[side].append([np.linalg.norm(error[:3]), np.linalg.norm(error[3:])])
            if tick % 4 == 0:
                frames.append(observed)
        metrics = {}
        for side, trace in traces.items():
            a = np.array(getattr(start, side).pose.position_m)
            b = np.array(getattr(action, side).pose.position_m)
            displacement = b - a
            actual = np.array(trace)
            fraction = np.clip((actual - a) @ displacement / (displacement @ displacement), 0, 1)
            deviation = np.linalg.norm(actual - (a + fraction[:, None] * displacement), axis=1)
            residual = pose_error(getattr(observed, side + "_tcp"), getattr(action, side).pose)
            metrics[side] = dict(max_cross_track_mm=float(deviation.max() * 1000),
                                 final_position_mm=float(np.linalg.norm(residual[:3]) * 1000),
                                 final_rotation_rad=float(np.linalg.norm(residual[3:])))
            measured_errors = np.array(errors[side])
            for label, tolerance in (("settled_5mm_005rad_s", .005), ("settled_1mm_005rad_s", .001)):
                inside = (measured_errors[:, 0] <= tolerance) & (measured_errors[:, 1] <= .05)
                stable_suffix = np.logical_and.accumulate(inside[::-1])[::-1]
                candidates = np.flatnonzero(stable_suffix)
                metrics[side][label] = None if not len(candidates) else float((candidates[0] + 1) * .01)
            assert metrics[side]["final_position_mm"] < 1, metrics
            assert metrics[side]["final_rotation_rad"] < .01, metrics
        assert speed_ratio <= 1 + 1e-8
        assert acceleration_ratio <= 1 + 1e-7
        np.testing.assert_allclose(observed.gripper_position, [1, 0], atol=.001)
        metrics.update(max_velocity_limit_ratio=speed_ratio,
                       max_acceleration_limit_ratio=acceleration_ratio,
                       grippers=list(observed.gripper_position))
        return frames, start, action, traces, metrics


def render(path, frames, start, action, traces, label="Braking-distance lookahead"):
    width, height, fps = 960, 720, 25
    kin = make_display()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.13, 0, .34]
    camera.distance, camera.azimuth, camera.elevation = 1.75, 150, -16
    options = mujoco.MjvOption()
    options.geomgroup[5] = 1
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        macro_block_size=2, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        with mujoco.Renderer(kin.model, height=height, width=width) as renderer:
            for index, observed in enumerate(frames):
                kin.update(observed.joint_positions_rad, observed.gripper_position)
                write_handles(kin, action)
                renderer.update_scene(kin.data, camera=camera, scene_option=options)
                for side in traces:
                    positions = np.linspace(getattr(start, side).pose.position_m,
                                            getattr(action, side).pose.position_m, 35)
                    for points, color, size in (
                        (positions, [1, .7, .1, 1], .003),
                        (traces[side][:index * 4 + 1:4], [.1, .9, .4, 1], .002),
                    ):
                        for point in points:
                            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                                raise RuntimeError("Path visualization exceeded scene geometry capacity")
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                               [size] * 3, point, np.eye(3).flatten(), color)
                            renderer.scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle([0, 0, width, 88], fill="#132334")
                draw.text((18, 10), "Dual NERO | " + label, font=font, fill="white")
                draw.text((18, 38), "Yellow: planned line   Green: measured TCP trail   Red: final target", font=font, fill="white")
                draw.text((18, 63), f"{index / fps:.2f} s | Left jaws open, right jaws closed", font=font, fill="white")
                draw.rectangle([0, height - 35, width, height], fill="#132334")
                draw.text((18, height - 28), "SIM ONLY | 1x simulation time | No obstacle avoidance or hardware commands",
                          font=font, fill="white")
                writer.send(np.asarray(image))
                if index == len(frames) - 1:
                    image.save(path.with_suffix(".png"))
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/validation/cartesian_path"))
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    *_, baseline = execute(False)
    old_frames, old_start, old_action, old_traces, old_planned = execute(True, adaptive_lookahead=False)
    frames, start, action, traces, planned = execute(True)
    for side in ("left", "right"):
        assert planned[side]["max_cross_track_mm"] < baseline[side]["max_cross_track_mm"]
        assert planned[side]["settled_5mm_005rad_s"] < old_planned[side]["settled_5mm_005rad_s"]
    report = dict(simulation_only=True, duration_sim_s=6, direct_endpoint=baseline,
                  previous_waypoint_pacing=old_planned,
                  cartesian_waypoints=planned, collision_checked=False,
                  description="Same official IK, joint limits and actuator dynamics; only waypoint layer differs.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.video:
        render(args.output / "cartesian_path.mp4", frames, start, action, traces,
               label=f"After: {planned['left']['settled_5mm_005rad_s']:.2f} s to 5 mm / 0.05 rad")
        render(args.output / "previous_pacing.mp4", old_frames, old_start, old_action, old_traces,
               label=f"Before: {old_planned['left']['settled_5mm_005rad_s']:.2f} s to 5 mm / 0.05 rad")
        comparison = args.output / "pacing_comparison.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-i", str(args.output / "previous_pacing.mp4"),
            "-i", str(args.output / "cartesian_path.mp4"),
            "-filter_complex", "hstack=inputs=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ], check=True)
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
            "-i", str(comparison), "-filter_complex",
            "fps=15,scale=1280:-1:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
            "-loop", "0", str(comparison.with_suffix(".gif")),
        ], check=True)


if __name__ == "__main__":
    main()
