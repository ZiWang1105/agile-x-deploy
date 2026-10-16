"""Record real servo-simulation results, with scripted targets through RobotInterface.

The recording is a 1x simulation-time playback, not a mouse/screen capture.
Rendering happens after control execution so encoding cannot stall the runtime.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from agilex_deploy import Action, ArmTarget, Pose, Robot
from agilex_deploy.backends import SimBackend
from agilex_deploy.controller import pose_error, MAX_VELOCITY, MAX_ACCELERATION
from agilex_deploy.interactive import make_display, write_handles
from agilex_deploy.model import ROOT
from agilex_deploy.official_config import solver_parameters

FPS, WIDTH, HEIGHT = 25, 1440, 900
OUTPUT = ROOT / "previews/official_nero_ik_validation.mp4"
PHASES = [
    (0, "READY / HOLD", "Measured startup pose"),
    (160, "01 / DUAL-ARM TRANSLATION", "Move forward and upward; open both grippers"),
    (720, "02 / ORIENTATION IK", "Rotate the two TCP targets in opposite directions"),
    (1280, "03 / LEFT GRIPPER", "Close left gripper; keep right gripper open"),
    (1640, "04 / RIGHT GRIPPER", "Open left gripper; close right gripper"),
    (2000, "05 / MOVE THEN HOLD", "Submit a new target before interrupting motion"),
    (2080, "06 / HOLD INTERRUPT", "Cancel the target and hold measured joint positions"),
    (2320, "07 / RESUME + RETURN", "Return to the initial TCP poses; open both grippers"),
    (2960, "08 / FINAL HOLD", "Target cancelled; servos settle at the measured pose"),
]


def with_grippers(action, left, right):
    return Action(ArmTarget(action.left.pose, left), ArmTarget(action.right.pose, right))


def execute():
    frames, checks = [], []
    peak_speed = peak_acceleration = 0.
    phase_index = 0
    previous_velocity = np.zeros(14)
    with Robot(SimBackend()) as robot:
        initial = robot.observe().as_action()
        translated, rotated, interrupt = {}, {}, {}
        for side, sign in (("left", 1), ("right", -1)):
            initial_pose = getattr(initial, side).pose
            position = np.asarray(initial_pose.position_m) + [.10, -.04 * sign, .14]
            translated[side] = ArmTarget(Pose(position, initial_pose.quaternion_xyzw), 1)
            orientation = (Rotation.from_rotvec([0, .12, .35 * sign]) *
                           Rotation.from_quat(initial_pose.quaternion_xyzw)).as_quat()
            rotated[side] = ArmTarget(Pose(position, orientation), 1)
            interrupt[side] = ArmTarget(Pose(position + [0, .06 * sign, .06], orientation), 1)
        goals = {1: Action(**translated), 2: Action(**rotated),
                 3: with_grippers(Action(**rotated), 0, 1),
                 4: with_grippers(Action(**rotated), 1, 0),
                 5: Action(**interrupt), 7: with_grippers(initial, 1, 1)}
        active = None
        display_target = initial
        hold_reference = None
        for tick in range(3200):
            if phase_index + 1 < len(PHASES) and tick == PHASES[phase_index + 1][0]:
                if active is not None and phase_index != 5:
                    observed = robot.observe()
                    residuals = [pose_error(getattr(observed, f"{s}_tcp"), getattr(active, s).pose)
                                 for s in ("left", "right")]
                    positions = [float(np.linalg.norm(e[:3]) * 1000) for e in residuals]
                    rotations = [float(np.degrees(np.linalg.norm(e[3:]))) for e in residuals]
                    checks.append(dict(phase=PHASES[phase_index][1], position_mm=positions,
                                       orientation_deg=rotations, grippers=list(observed.gripper_position)))
                    # Official NERO weights use soft pose costs. Record the
                    # residual instead of asserting the old solver's accuracy.
                    assert np.all(np.isfinite(positions + rotations)), checks[-1]
                    assert max(positions) < 1.0 and max(rotations) < .1, checks[-1]
                    expected_grip = [getattr(active, s).gripper_open for s in ("left", "right")]
                    np.testing.assert_allclose(observed.gripper_position, expected_grip, atol=.01)
                phase_index += 1
                active = goals.get(phase_index)
                if active is None:
                    robot.hold()
                    display_target = robot.observe().as_action()
                    hold_reference = np.array(robot.observe().joint_positions_rad)
                else:
                    display_target = active
            if active is not None and tick % 10 == 0:
                robot.submit_target(active, robot.observe().observation_id)
            robot.tick()
            observation = robot.observe()
            assert observation.mode != "fault", observation.reason
            velocity = robot.controller.velocity.copy()
            peak_speed = max(peak_speed, float(np.max(abs(velocity))))
            assert np.all(abs(velocity) <= MAX_VELOCITY + 1e-8)
            # HOLD explicitly cancels the motion reference; continuity limits
            # apply to tracking steps, not the emergency trajectory override.
            if active is not None and tick != PHASES[phase_index][0]:
                assert np.all(abs(velocity - previous_velocity) / .01 <= MAX_ACCELERATION + 1e-7)
                peak_acceleration = max(peak_acceleration, float(np.max(abs(velocity - previous_velocity))) / .01)
            previous_velocity = velocity
            if tick in (2319, 3199):
                drift = float(np.max(abs(np.array(observation.joint_positions_rad) - hold_reference)))
                speed = float(np.max(abs(np.array(observation.joint_velocities_rad_s))))
                checks.append(dict(phase=PHASES[phase_index][1], hold_drift_rad=drift,
                                   settled_speed_rad_s=speed))
                assert drift < .01 and speed < .001, checks[-1]
            if tick % 4 == 0:
                frames.append((tick / 100, phase_index, observation, display_target))
    report = dict(duration_s=32, fps=FPS, frames=len(frames),
                  source="Official AgileX ArmIK with configs/ik.yaml tuning; scripted RobotInterface targets; actuated MuJoCo simulation; no mouse capture",
                  parameters=solver_parameters(),
                  max_requested_speed_rad_s=peak_speed,
                  max_requested_acceleration_rad_s2=peak_acceleration, checks=checks,
                  limitations="Contacts disabled; official nominal TCP; soft objective retains pose residuals; no hardware validation")
    return frames, report


def render(frames):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    small = ImageFont.truetype(font_path, 17)
    regular = ImageFont.truetype(font_path, 21)
    title = ImageFont.truetype(bold_path, 28)
    subheading = ImageFont.truetype(bold_path, 21)
    kin = make_display()
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.13, 0, .34]
    camera.distance, camera.azimuth, camera.elevation = 1.75, 150, -16
    options = mujoco.MjvOption()
    options.geomgroup[5] = 1
    OUTPUT.parent.mkdir(exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(OUTPUT), (WIDTH, HEIGHT), fps=FPS,
        codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=8,
        macro_block_size=2, ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        with mujoco.Renderer(kin.model, height=HEIGHT, width=1100) as renderer:
            for frame_index, (elapsed, phase, observation, action) in enumerate(frames):
                kin.update(observation.joint_positions_rad, observation.gripper_position)
                write_handles(kin, action)
                mujoco.mj_forward(kin.model, kin.data)
                renderer.update_scene(kin.data, camera=camera, scene_option=options)
                canvas = Image.new("RGB", (WIDTH, HEIGHT), "#132334")
                canvas.paste(Image.fromarray(renderer.render()), (0, 0))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, WIDTH, 111), fill="#132334")
                draw.text((28, 17), "DUAL NERO  /  IK VALIDATION", font=title, fill="white")
                draw.text((28, 62), PHASES[phase][1], font=subheading, fill="#6ae0c1")
                draw.text((28, 846), PHASES[phase][2], font=regular, fill="#193345")
                x = 1122
                draw.text((x, 135), f"{elapsed:05.2f} / 32.00 s", font=title, fill="white")
                draw.text((x, 176), "1x simulation-time playback", font=small, fill="#b7c6d6")
                color = "#6ae0c1" if observation.mode == "tracking" else "#ffd58b"
                draw.text((x, 223), observation.mode.upper(), font=title, fill=color)
                draw.text((x, 267), "Red: requested TCP", font=regular, fill="#ff9390")
                draw.text((x, 296), "Green: measured TCP", font=regular, fill="#81e998")
                for i, side in enumerate(("left", "right")):
                    y = 354 + i * 155
                    error = pose_error(getattr(observation, f"{side}_tcp"), getattr(action, side).pose)
                    draw.text((x, y), side.upper() + " ARM", font=subheading, fill="white")
                    draw.text((x, y + 35), f"Position  {np.linalg.norm(error[:3])*1000:7.2f} mm", font=regular, fill="#cfdae7")
                    draw.text((x, y + 66), f"Rotation  {np.degrees(np.linalg.norm(error[3:])):6.2f} deg", font=regular, fill="#cfdae7")
                    draw.text((x, y + 97), f"Gripper travel  {np.clip(observation.gripper_position[i],0,1)*100:5.1f}%", font=regular, fill="#cfdae7")
                for y, line in enumerate(("Scripted targets through", "the interactive viewer's", "shared robot API.", "", "100 Hz control", "500 Hz actuator simulation")):
                    draw.text((x, 685 + y * 25), line, font=small, fill="#b7c6d6")
                draw.rectangle((0, 881, WIDTH, 899), fill="#132334")
                draw.text((20, 882), "SIMULATION ONLY  |  Contacts disabled  |  Nominal CAD TCP  |  No hardware commands", font=ImageFont.truetype(font_path, 13), fill="#bdcddd")
                draw.rectangle((0, 109, int(WIDTH * elapsed / 32), 112), fill="#6ae0c1")
                writer.send(np.asarray(canvas))
                if frame_index in (0, 240, 540, 799):
                    canvas.save(OUTPUT.with_name(f"video_check_{frame_index:03d}.png"))
                if frame_index % 125 == 0:
                    print(f"Rendered {frame_index}/{len(frames)} frames", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    frames, report = execute()
    print(json.dumps(report, indent=2), flush=True)
    render(frames)
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(OUTPUT))
    assert count == len(frames) and abs(duration - 32) < .1, (count, duration)
    report.update(encoded_frames=count, encoded_duration_s=duration, bytes=OUTPUT.stat().st_size)
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {OUTPUT}: {count} frames, {duration:.2f} seconds", flush=True)
