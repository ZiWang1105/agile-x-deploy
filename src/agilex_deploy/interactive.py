"""Native MuJoCo drag targets over the shared RobotInterface.

The viewer owns a separate display model. It reconstructs that display from
observations, and cannot step, reset, perturb, or command the backend plant.
"""
from __future__ import annotations

import argparse
import copy
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

from .contract import Action, ArmTarget, Pose, SIDES, JOINT_NAMES, RobotInterface
from .controller import pose_error
from .model import Kinematics, MODEL_PATH, ROOT
from .nero_transport import MAX_TRANSLATION_M, TEST_LIFT_M, validate_workspace_radius


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


def make_display(mirror=False):
    tree = ET.parse(MODEL_PATH)
    root = tree.getroot()
    world = root.find("worldbody")
    if mirror:
        base = world.find("body[@name='base_link']")
        for side in SIDES:
            arm = base.find(f"body[@name='{side}_arm_base']")
            measured = copy.deepcopy(arm)
            for element in measured.iter():
                if "name" in element.attrib:
                    element.set("name", "measured_" + element.get("name"))
                if element.tag == "geom":
                    element.set("rgba", ".15 .55 .9 1")
                    element.set("group", "4")
                if element.tag == "site":
                    element.set("rgba", "0 0 0 0")
            for geom in arm.iter("geom"):
                geom.set("rgba", "1 .65 .12 .28")
                geom.set("group", "3")
            # Finger travel is uncalibrated. Do not display an invented opening
            # as measured geometry, or expose gripper controls in this mode.
            for tree, prefix in ((arm, ""), (measured, "measured_")):
                for finger in (1, 2):
                    body = tree.find(f".//body[@name='{prefix}{side}_gripper_link{finger}']")
                    for geom in body.iter("geom"):
                        geom.set("rgba", "0 0 0 0")
            base.append(measured)
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


def update_mirror(kin, snapshot):
    """Update only display qpos; hardware feedback never passes through a plant."""
    for name, value in zip(JOINT_NAMES, snapshot.positions):
        kin.data.qpos[kin.model.joint("measured_" + name).qposadr[0]] = value
    mujoco.mj_forward(kin.model, kin.data)


def write_handles(kin, action):
    for side in SIDES:
        pose = getattr(action, side).pose
        index = kin.model.body(f"{side}_target").mocapid[0]
        kin.data.mocap_pos[index] = pose.position_m
        kin.data.mocap_quat[index] = np.asarray(pose.quaternion_xyzw)[[3, 0, 1, 2]]


def read_drag(kin, action, selected, active, reset=False, sides=SIDES):
    """Only the selected mocap handle can become a robot target."""
    if reset or not active:
        return action
    arms = {s: getattr(action, s) for s in SIDES}
    for side in sides:
        body = kin.model.body(f"{side}_target")
        if selected == body.id:
            index = body.mocapid[0]
            candidate = Pose(tuple(kin.data.mocap_pos[index]),
                             tuple(kin.data.mocap_quat[index][[1, 2, 3, 0]]))
            delta = pose_error(arms[side].pose, candidate)
            if np.linalg.norm(delta) > 1e-7:
                arms[side] = ArmTarget(candidate, arms[side].gripper_open)
    return Action(**arms)


def constrain_commissioning_drag(action, lift_target, side, workspace_radius_m=MAX_TRANSLATION_M):
    """Keep the displayed handle inside the admitted translation workspace.

    The arming anchor comes from the control owner, not drifting observations.
    Bounded commissioning input changes position only. Full-workspace input
    permits translation and rotation without an arming-relative envelope.
    The transport still independently validates every submitted target.
    """
    if workspace_radius_m is None:
        arms = {s: getattr(lift_target, s) for s in SIDES}
        arms[side] = getattr(action, side)
        result = Action(**arms)
        return result, result != action
    anchor = np.asarray(getattr(lift_target, side).pose.position_m) - [0, 0, TEST_LIFT_M]
    delta = np.asarray(getattr(action, side).pose.position_m) - anchor
    delta[2] = max(0., delta[2])
    distance = float(np.linalg.norm(delta))
    if distance > workspace_radius_m:
        delta *= workspace_radius_m / distance
    arms = {s: getattr(lift_target, s) for s in SIDES}
    arms[side] = ArmTarget(Pose(anchor + delta, arms[side].pose.quaternion_xyzw), None)
    result = Action(**arms)
    limited = any(np.linalg.norm(pose_error(getattr(action, s).pose, getattr(result, s).pose)) > 1e-7
                  for s in SIDES)
    return result, limited


def configure_handles(kin, sides):
    """Gray inactive handles and return only the IDs permitted to command."""
    allowed = []
    for side in SIDES:
        body = kin.model.body(f"{side}_target")
        if side in sides:
            allowed.append(body.id)
        else:
            kin.model.geom_rgba[kin.model.geom_bodyid == body.id] = [.4, .4, .4, .4]
    return allowed


def run_viewer(robot: RobotInterface, duration=None, log_path=None, on_ready=None, feedback=None,
               hardware_control=False):
    import mujoco.viewer

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("Desktop DISPLAY required. Run agilex-interactive in a desktop terminal.")
    kin = make_display(mirror=feedback is not None)
    hardware_snapshot = None
    hardware_fault = None
    events = queue.SimpleQueue()
    observed = robot.observe()
    for i, side in enumerate(SIDES):
        if observed.gripper_position[i] is None:
            for geom_id in range(kin.model.ngeom):
                body = kin.model.body(kin.model.geom_bodyid[geom_id]).name
                if body.startswith(f"{side}_gripper_link"):
                    kin.model.geom_rgba[geom_id, 3] = 0
    targets = observed.as_action()
    kin.update(observed.joint_positions_rad, observed.gripper_position)
    write_handles(kin, targets)
    if feedback is not None:
        hardware_snapshot = feedback.read(timeout=1.)
        update_mirror(kin, hardware_snapshot)
    draggable_sides = (robot.side,) if hardware_control else SIDES
    workspace_radius_m = robot.workspace_radius_m if hardware_control else MAX_TRANSLATION_M
    workspace_label = "FULL WORKSPACE" if workspace_radius_m is None else f"{workspace_radius_m * 100:g} cm radius"
    selected_ids = configure_handles(kin, draggable_sides)
    tracking = False
    lift_target = None
    message = "Ready. Drag a red sphere to begin."
    last_submit = 0.
    last_feedback_log = 0.
    start = time.monotonic()
    record = open(log_path, "a") if log_path else None
    last_state = None

    def log(kind, **fields):
        if record:
            record.write(json.dumps(dict(timestamp_s=time.monotonic(), kind=kind, **fields)) + "\n")
            record.flush()
        if kind in ("target_rejected", "controller_state", "operator_arm", "hold"):
            print(f"{kind}: {json.dumps(fields)}", flush=True)

    if hardware_control:
        log("session_config", side=robot.side, simulation=robot.simulation,
            workspace_radius_m=workspace_radius_m)

    try:
        with managed_viewer(kin.model, kin.data, key_callback=events.put,
                show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [.12, 0, .34]
                viewer.cam.distance = 1.9
                viewer.cam.azimuth = 145
                viewer.cam.elevation = -15
                viewer.opt.geomgroup[5] = 1
                viewer.opt.geomgroup[3] = 1
                viewer.opt.geomgroup[4] = 1
            if on_ready is not None:
                on_ready()
            start = time.monotonic()
            while viewer.is_running() and (duration is None or time.monotonic() - start < duration):
                began = time.monotonic()
                if feedback is not None and hardware_fault is None:
                    try:
                        hardware_snapshot = feedback.read()
                        q = np.asarray(hardware_snapshot.positions)
                        if np.any(q < kin.limits[:, 0]) or np.any(q > kin.limits[:, 1]):
                            raise RuntimeError("Measured joints outside model limits")
                    except (RuntimeError, OSError) as error:
                        hardware_fault = str(error)
                        robot.hold()  # The robot is the preview simulator in this mode.
                        log("hardware_feedback_fault", reason=hardware_fault)
                observed = robot.observe()
                state = (observed.mode, observed.reason)
                if hardware_control and state != last_state:
                    log("controller_state", mode=observed.mode, reason=observed.reason,
                        observation_id=observed.observation_id,
                        joint_positions_rad=observed.joint_positions_rad)
                    last_state = state
                fresh = began - observed.timestamp_s < .2
                usable = fresh and observed.mode not in ("fault", "unarmed") and hardware_fault is None
                reset = False
                changed = False
                constrained = False
                if tracking and (not usable or observed.mode == "hold"):
                    tracking, reset = False, True
                    message = observed.reason
                while not events.empty():
                    key = events.get()
                    if key in (ord("Q"), 256):
                        return
                    if key == ord("E") and hardware_control and observed.mode == "unarmed":
                        try:
                            lift_target = robot.arm()
                            observed = robot.observe()
                            reset, tracking = True, False
                            message = "ARMED. Holding current posture. G: request 30 mm upward test."
                            log("operator_arm", side=robot.side)
                        except RuntimeError as error:
                            message = str(error)
                    if key == ord("G") and hardware_control and usable and lift_target is not None:
                        targets, changed = lift_target, True
                    if key in (32, ord("R")):
                        robot.hold()
                        observed = robot.observe()
                        reset, tracking = True, False
                        message = "HOLD. Drag a target to resume."
                        log("hold")
                    if key == ord("M") and feedback is not None:
                        with viewer.lock():
                            viewer.opt.geomgroup[3] = 1 - viewer.opt.geomgroup[3]
                    if key in (ord("1"), ord("2")) and usable and feedback is None and not hardware_control:
                        side = SIDES[key - ord("1")]
                        arms = {s: getattr(targets, s) for s in SIDES}
                        arms[side] = ArmTarget(arms[side].pose, 1 - arms[side].gripper_open)
                        targets, changed = Action(**arms), True
                with viewer.lock():
                    kin.update(observed.joint_positions_rad, observed.gripper_position)
                    if hardware_snapshot is not None:
                        update_mirror(kin, hardware_snapshot)
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
                                            viewer.perturb.active, reset=reset or gui_reset or not usable,
                                            sides=draggable_sides)
                        if hardware_control and lift_target is not None and updated != targets:
                            raw_drag = updated
                            updated, constrained = constrain_commissioning_drag(updated, lift_target, robot.side,
                                                                               workspace_radius_m)
                            log("drag_input", selected_side=robot.side, constrained=constrained,
                                requested=raw_drag.to_dict(), admitted=updated.to_dict())
                        changed = changed or updated != targets
                        targets = updated
                    except ValueError as error:
                        message = str(error)
                    kin.update(observed.joint_positions_rad, observed.gripper_position)
                    kin.data.xfrc_applied[:] = 0
                    kin.data.qfrc_applied[:] = 0
                    if hardware_snapshot is not None:
                        update_mirror(kin, hardware_snapshot)
                    write_handles(kin, targets)
                    if viewer.perturb.select not in selected_ids:
                        viewer.perturb.active = 0
                if usable and (changed or (tracking and began - last_submit >= .1)):
                    try:
                        latest = robot.observe()
                        request = robot.submit_target(targets, latest.observation_id)
                        tracking, last_submit = True, began
                        if changed:
                            message = (f"Target constrained to {workspace_label}. " if constrained else "") + f"Tracking target {request}"
                            log("target", request_id=request, observation_id=latest.observation_id,
                                action=targets.to_dict())
                    except (ValueError, RuntimeError) as error:
                        rejected = targets
                        robot.hold()
                        observed = robot.observe()
                        targets, tracking = observed.as_action(), False
                        message = f"TARGET REJECTED; HOLD requested: {error}"
                        with viewer.lock():
                            viewer.perturb.active = viewer.perturb.select = 0
                            write_handles(kin, targets)
                        log("target_rejected", reason=str(error), action=rejected.to_dict(),
                            mode=observed.mode, controller_reason=observed.reason)
                error = [pose_error(getattr(observed, f"{s}_tcp"), getattr(targets, s).pose) for s in SIDES]
                if hardware_control and began - last_feedback_log >= .2:
                    # Mode changes alone cannot establish whether the physical
                    # arm followed a target. Retain measured feedback as well.
                    log("feedback", observation=observed.to_dict(), target=targets.to_dict(),
                        tracking=tracking)
                    last_feedback_log = began
                text = f"{message}\nMode: {observed.mode} | {observed.reason}\n"
                text += "TCP error L/R: " + " / ".join(f"{np.linalg.norm(e[:3])*1000:.1f} mm, {np.degrees(np.linalg.norm(e[3:])):.1f} deg" for e in error)
                text += "\nGripper travel L/R: " + " / ".join("unavailable" if g is None else f"{g:.2f}" for g in observed.gripper_position)
                if not fresh or observed.mode == "fault" or hardware_fault:
                    text += "\nFeedback stale or fault latched. Restart after inspection."
                title = "DUAL NERO | Interactive IK"
                footer = "Space / R: HOLD | 1 / 2: grippers | Q: quit\nSimulation prototype: contacts disabled; TCP is nominal CAD frame"
                if feedback is not None:
                    title = "DUAL NERO | REAL FEEDBACK + SIM IK | NO HARDWARE COMMANDS"
                    enabled = sum(bool(v & 0x40) for v in hardware_snapshot.driver_status)
                    age = max(0., time.monotonic() - hardware_snapshot.timestamp_s)
                    text += f"\nCAN age: {age*1000:.0f} ms | skew: {hardware_snapshot.skew_s*1000:.1f} ms | enabled: {enabled}/14"
                    text += "\nBLUE: measured joints | AMBER: simulated IK preview"
                    text += "\nHardware velocity and gripper travel: unavailable; fingers hidden"
                    if hardware_fault:
                        text += "\nFROZEN hardware display: " + hardware_fault
                    footer = "Space / R: stop SIM preview | M: show/hide preview | Q: quit\nNo collision admission; mount/signs/TCP not yet physically verified"
                if hardware_control:
                    title = f"NERO {'TEST SIMULATION' if robot.simulation else 'PHYSICAL ROBOT'} | {robot.side.upper()} ARM | {workspace_label}"
                    text += "\nE: ENABLE selected arm at current pose (only while UNARMED)"
                    text += "\nG: target 30 mm above the original arming pose (does not stack)"
                    text += ("\nFull workspace: model joint limits apply; NO collision avoidance" if workspace_radius_m is None
                             else f"\nWorkspace: {workspace_label}, at/above start height")
                    if lift_target is not None:
                        anchor = np.asarray(getattr(lift_target, robot.side).pose.position_m) - [0, 0, TEST_LIFT_M]
                        target_delta = (np.asarray(getattr(targets, robot.side).pose.position_m) - anchor) * 1000
                        measured_delta = (np.asarray(getattr(observed, robot.side + "_tcp").position_m) - anchor) * 1000
                        text += "\nTarget offset XYZ (mm): " + ", ".join(f"{v:+.1f}" for v in target_delta)
                        text += "\nMeasured offset XYZ (mm): " + ", ".join(f"{v:+.1f}" for v in measured_delta)
                        if workspace_radius_m is not None:
                            remaining = max(0., workspace_radius_m * 1000 - np.linalg.norm(target_delta))
                            text += f"\nWorkspace radius remaining: {remaining:.1f} mm"
                            if remaining < .1:
                                text += " | AT LIMIT: drag back toward the starting position"
                    footer = "Space / R: servo HOLD | Q: HOLD and close (motors stay enabled)\nSupervised free-space test only. No collision protection or physical E-stop guarantee."
                drag_help = (f"Only {robot.side.upper()} red target is active; gray target is locked | Double-click red sphere, Ctrl+right drag: move"
                             if hardware_control else "Double-click red sphere | Ctrl+right drag: move | Ctrl+left drag: rotate")
                if hardware_control and workspace_radius_m is None:
                    drag_help += " | Ctrl+left drag: rotate"
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     title, text),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     drag_help,
                     footer)])
                time.sleep(max(0., began + 1 / 30 - time.monotonic()))
    finally:
        try:
            try:
                robot.hold()
            except Exception as error:
                # The enclosing session still closes its transport and joins
                # the owner. A failed HOLD must remain visible, but must not
                # mask Ctrl+C or prevent that cleanup from running.
                print(f"HOLD unconfirmed during viewer exit: {error}", flush=True)
        finally:
            if record:
                record.close()


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "real-preview", "real-commissioning", "commissioning-sim"), default="sim",
                        help="sim/real-preview never move hardware; real-commissioning starts unarmed and E enables one arm")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--side", choices=SIDES, help="Single arm selected for commissioning")
    parser.add_argument("--workspace", choices=("bounded", "full"), default="full",
                        help="full removes arming-relative motion caps; joint position/speed/acceleration limits remain, without collision avoidance")
    parser.add_argument("--workspace-radius-cm", type=float,
                        help="Commissioning target radius from arming pose, 3–20 cm (default: 3). Joint guards still apply.")
    parser.add_argument("--duration", type=float, help="Close after this many seconds (smoke testing)")
    parser.add_argument("--log", type=Path, help="Append operator actions as JSONL")
    args = parser.parse_args()
    from .backends import SimBackend
    from .runtime import Robot
    if args.backend in ("real-commissioning", "commissioning-sim"):
        if args.workspace != "bounded" and args.workspace_radius_cm is not None:
            parser.error("Use --workspace bounded to opt into --workspace-radius-cm")
        try:
            workspace_radius_m = validate_workspace_radius(None if args.workspace == "full" else
                (3. if args.workspace_radius_cm is None else args.workspace_radius_cm) / 100)
        except ValueError as error:
            parser.error(str(error))
        if args.side is None:
            parser.error("Commissioning requires --side left or --side right")
        from .hardware_session import HardwareSession
        if args.log is None:
            args.log = ROOT / "logs/commissioning" / f"{args.backend}-{time.time_ns()}.jsonl"
        args.log.parent.mkdir(parents=True, exist_ok=True)
        print(f"Session log: {args.log}", flush=True)
        with HardwareSession(args.side, mapping=args.mapping,
                             simulation=args.backend == "commissioning-sim",
                             workspace_radius_m=workspace_radius_m) as robot:
            if workspace_radius_m is None:
                print("FULL WORKSPACE: no starting-position radius, height floor, rotation cap or joint-excursion cap. Joint position limits and controller-reported speed/acceleration limits apply. NO collision avoidance.", flush=True)
            else:
                print(f"Workspace radius: {workspace_radius_m * 100:g} cm. Joint excursion remains limited to 0.45 rad; not every target is reachable.", flush=True)
            print("Starts UNARMED. E enables the selected arm at its current posture.", flush=True)
            print("G requests 30 mm above the arming pose; repeated presses do not stack. Space HOLDs; closing does NOT disable motors.", flush=True)
            run_viewer(robot, duration=args.duration, log_path=args.log, hardware_control=True)
        return
    if args.backend == "real-preview":
        from .can_feedback import PassiveCanFeedback
        with PassiveCanFeedback(args.mapping) as feedback:
            sample = feedback.read(timeout=1.)
            print("Receive-only hardware mapping:", feedback.channels, flush=True)
            print("Drag targets control SIMULATION ONLY. No hardware command path is connected.", flush=True)
            with Robot(SimBackend(joint_positions=sample.positions)) as robot:
                run_viewer(robot, duration=args.duration, log_path=args.log,
                           on_ready=robot.start, feedback=feedback)
                if robot.observe().mode == "fault":
                    raise RuntimeError(robot.observe().reason)
        return
    with Robot(SimBackend()) as robot:
        # Native window/GL initialization can hold the GIL. Start the control
        # clock after the window is ready, not while loading the renderer.
        run_viewer(robot, duration=args.duration, log_path=args.log, on_ready=robot.start)
        if robot.observe().mode == "fault":
            raise RuntimeError(robot.observe().reason)


def main():
    try:
        _main()
    except KeyboardInterrupt:
        # Context managers have already requested HOLD and joined the owner.
        print("Controller closed after Ctrl+C. Motors are NOT disabled.", flush=True)


if __name__ == "__main__":
    main()
