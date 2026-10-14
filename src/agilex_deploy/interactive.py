"""Native MuJoCo drag targets over the shared RobotInterface.

The viewer owns a separate display model. It reconstructs that display from
observations, and cannot step, reset, perturb, or command the backend plant.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contract import Action, ArmTarget, Pose, SIDES, RobotInterface
from .controller import pose_error
from .model import Kinematics, MODEL_PATH


@contextmanager
def managed_viewer(model, data, **kwargs):
    """Wait for native teardown before Python's GLFW atexit handler runs.

    MuJoCo 3.10 launch_passive creates a daemon _launch_internal thread and
    Handle.close only requests exit. Join that newly created renderer thread
    to avoid racing glfw.terminate at interpreter shutdown on this platform.
    """
    import mujoco.viewer
    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, **kwargs)
    render_threads = [t for t in set(threading.enumerate()) - before
                      if "_launch_internal" in t.name]
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in render_threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError("Native viewer did not finish shutdown")


def make_display():
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    world = root.find("worldbody")
    for side in SIDES:
        body = ET.SubElement(world, "body", name=f"{side}_target", mocap="true")
        ET.SubElement(body, "geom", type="sphere", size=".023", rgba="1 .15 .12 .85",
                      contype="0", conaffinity="0", group="5")
        for axis in range(3):
            end, color = np.zeros(3), np.full(4, .15)
            end[axis], color[axis], color[3] = .08, 1, 1
            ET.SubElement(body, "geom", type="capsule", size=".003",
                          fromto="0 0 0 " + " ".join(map(str, end)), rgba=" ".join(map(str, color)),
                          contype="0", conaffinity="0", group="5")
    # Absolute asset paths permit in-memory loading without rewriting plant files.
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str(MODEL_PATH.parent / mesh.get("file")))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return Kinematics(model)


def write_handles(kin, action):
    for side in SIDES:
        pose = getattr(action, side).pose
        index = kin.model.body(f"{side}_target").mocapid[0]
        kin.data.mocap_pos[index] = pose.position_m
        kin.data.mocap_quat[index] = np.asarray(pose.quaternion_xyzw)[[3, 0, 1, 2]]


def read_drag(kin, action, selected, active, reset=False):
    """Only the selected mocap handle can become a robot target."""
    if reset or not active:
        return action
    arms = {s: getattr(action, s) for s in SIDES}
    for side in SIDES:
        body = kin.model.body(f"{side}_target")
        if selected == body.id:
            index = body.mocapid[0]
            candidate = Pose(tuple(kin.data.mocap_pos[index]),
                             tuple(kin.data.mocap_quat[index][[1, 2, 3, 0]]))
            delta = pose_error(arms[side].pose, candidate)
            if np.linalg.norm(delta) > 1e-7:
                arms[side] = ArmTarget(candidate, arms[side].gripper_open)
    return Action(**arms)


def run_viewer(robot: RobotInterface, duration=None, log_path=None, on_ready=None):
    import mujoco.viewer

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Desktop DISPLAY required. Run agilex-interactive in a desktop terminal.")
    kin = make_display()
    events = queue.SimpleQueue()
    observed = robot.observe()
    targets = observed.as_action()
    kin.update(observed.joint_positions_rad, observed.gripper_position)
    write_handles(kin, targets)
    selected_ids = [kin.model.body(f"{s}_target").id for s in SIDES]
    tracking = False
    message = "Ready. Drag a red sphere to begin."
    last_submit = 0.
    start = time.monotonic()
    record = open(log_path, "a") if log_path else None

    def log(kind, **fields):
        if record:
            record.write(json.dumps(dict(timestamp_s=time.monotonic(), kind=kind, **fields)) + "\n")
            record.flush()

    try:
        with managed_viewer(kin.model, kin.data, key_callback=events.put,
                show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [.12, 0, .34]
                viewer.cam.distance = 1.9
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -15
                viewer.opt.geomgroup[5] = 1
            if on_ready is not None:
                on_ready()
            start = time.monotonic()
            while viewer.is_running() and (duration is None or time.monotonic() - start < duration):
                began = time.monotonic()
                observed = robot.observe()
                fresh = began - observed.timestamp_s < .2
                usable = fresh and observed.mode != "fault"
                reset = False
                changed = False
                if tracking and (not usable or observed.mode == "hold"):
                    tracking, reset = False, True
                    message = observed.reason
                while not events.empty():
                    key = events.get()
                    if key in (ord("Q"), 256):
                        return
                    if key in (32, ord("R")):
                        robot.hold()
                        observed = robot.observe()
                        reset, tracking = True, False
                        message = "HOLD. Drag a target to resume."
                        log("hold")
                    if key in (ord("1"), ord("2")) and usable:
                        side = SIDES[key - ord("1")]
                        arms = {s: getattr(targets, s) for s in SIDES}
                        arms[side] = ArmTarget(arms[side].pose, 1 - arms[side].gripper_open)
                        targets, changed = Action(**arms), True
                with viewer.lock():
                    kin.update(observed.joint_positions_rad, observed.gripper_position)
                    kin.data.time = 1.
                    if reset:
                        targets = observed.as_action()
                        viewer.perturb.active = viewer.perturb.select = 0
                    write_handles(kin, targets)
                viewer.sync(state_only=True)
                with viewer.lock():
                    gui_reset = kin.data.time != 1.
                    if gui_reset:
                        robot.hold()
                        observed = robot.observe()
                        targets, tracking = observed.as_action(), False
                        message = "HOLD. Display reset; plant preserved."
                        log("display_reset_hold")
                    try:
                        updated = read_drag(kin, targets, viewer.perturb.select,
                                            viewer.perturb.active, reset=reset or gui_reset or not usable)
                        changed = changed or updated != targets
                        targets = updated
                    except ValueError as error:
                        message = str(error)
                    kin.update(observed.joint_positions_rad, observed.gripper_position)
                    kin.data.xfrc_applied[:] = 0
                    kin.data.qfrc_applied[:] = 0
                    write_handles(kin, targets)
                    if viewer.perturb.select not in selected_ids:
                        viewer.perturb.active = 0
                if usable and (changed or (tracking and began - last_submit >= .1)):
                    try:
                        latest = robot.observe()
                        request = robot.submit_target(targets, latest.observation_id)
                        tracking, last_submit = True, began
                        if changed:
                            message = f"Tracking target {request}"
                            log("target", request_id=request, observation_id=latest.observation_id,
                                action=targets.to_dict())
                    except (ValueError, RuntimeError) as error:
                        message, tracking = str(error), False
                error = [pose_error(getattr(observed, f"{s}_tcp"), getattr(targets, s).pose) for s in SIDES]
                text = f"{message}\nMode: {observed.mode} | {observed.reason}\n"
                text += "TCP error L/R: " + " / ".join(f"{np.linalg.norm(e[:3])*1000:.1f} mm, {np.degrees(np.linalg.norm(e[3:])):.1f} deg" for e in error)
                text += f"\nGripper travel L/R: {observed.gripper_position[0]:.2f} / {observed.gripper_position[1]:.2f}"
                if not usable:
                    text += "\nFeedback stale or fault latched. Restart after inspection."
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "DUAL NERO | Interactive IK", text),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "Double-click red sphere | Ctrl+right drag: move | Ctrl+left drag: rotate",
                     "Space / R: HOLD | 1 / 2: grippers | Q: quit\nSimulation prototype: contacts disabled; TCP is nominal CAD frame")])
                time.sleep(max(0., began + 1 / 30 - time.monotonic()))
    finally:
        try:
            robot.hold()
        finally:
            if record:
                record.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim",), default="sim",
                        help="Only sim is commissioned in this release")
    parser.add_argument("--duration", type=float, help="Close after this many seconds (smoke testing)")
    parser.add_argument("--log", type=Path, help="Append operator actions as JSONL")
    args = parser.parse_args()
    from .backends import SimBackend
    from .runtime import Robot
    with Robot(SimBackend()) as robot:
        # Native window/GL initialization can hold the GIL. Start the control
        # clock after the window is ready, not while loading the renderer.
        run_viewer(robot, duration=args.duration, log_path=args.log, on_ready=robot.start)
        if robot.observe().mode == "fault":
            raise RuntimeError(robot.observe().reason)


if __name__ == "__main__":
    main()
