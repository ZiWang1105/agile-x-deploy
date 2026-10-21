"""Run one RGB-D policy through the shared robot interface; default is simulation."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from agilex_deploy.agent_execution import AgentExecutor, AgentStopped
from agilex_deploy.agent_policy import OpenRouterPolicy
from agilex_deploy.agent_sensors import save_frame
from agilex_deploy.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument("--gravity-comp", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to physical hand-guiding on enable/exit (default: false)")
    parser.add_argument(
        "--scene", type=Path, default=ROOT / "configs/agent/cube_scene.json"
    )
    parser.add_argument(
        "--key-file", type=Path, default=ROOT / "openrouter_key.txt",
        help="OpenRouter key file (default: openrouter_key.txt in the repository root)",
    )
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--max-calls", type=int, default=16)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument(
        "--instruction",
        "--prompt",
        dest="instruction",
        help="Natural-language task sent to the policy on every observation",
        default="Pick up the red cube with either hand, lift it at least 10 cm from its starting height, and hold it securely in the air for at least 10 seconds. Avoid touching the table with the robot.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Run immediately without the browser dashboard",
    )
    parser.add_argument(
        "--port", type=int, default=8770, help="Local dashboard port (default: 8770)"
    )
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--camera-serial",
        help="Require this ZED video serial; defaults to the calibration file's serial",
    )
    parser.add_argument(
        "--camera-device",
        type=Path,
        help="Optional V4L2 path, still verified against the calibration serial",
    )
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help="Allow physical execution (browser Start required unless --no-web); otherwise monitor only",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT
        / "calibration/2026-10-15_zed35330610/raised_probe/results_agent_v2/candidate.json",
    )
    args = parser.parse_args()
    if args.max_calls < 1 or not 0 < args.max_cost < float("inf"):
        parser.error("Require a positive call count and finite positive cost budget")
    if args.execute_real and args.backend != "real":
        parser.error("--execute-real requires --backend real")
    directory = args.output or ROOT / "logs/agent" / time.strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "source"
    source.mkdir()
    files = (
        list((ROOT / "src/agilex_deploy").glob("*.py"))
        + list((ROOT / "src/agilex_deploy").glob("agent_*.html"))
        + [Path(__file__).resolve()]
    )
    hashes = {}
    for file in files:
        content = file.read_bytes()
        (source / file.name).write_bytes(content)
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    (directory / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if args.backend == "sim":
        (directory / "scene.json").write_bytes(args.scene.read_bytes())
    plant = camera = robot = executor = dashboard = None
    report = {
        "backend": args.backend,
        "model": args.model,
        "physical_motion_requested": args.execute_real,
        "status": "starting",
    }
    policy = OpenRouterPolicy(
        directory / "policy", args.key_file, args.model, args.max_calls, args.max_cost
    )
    try:
        if not args.no_web:
            from agilex_deploy.agent_dashboard import AgentDashboard

            dashboard = AgentDashboard(
                args.port, args.backend, args.instruction, args.model
            )
        if args.backend == "sim":
            from agilex_deploy.agent_sim import ManipulationSim, SimCamera
            from agilex_deploy.runtime import Robot

            plant = ManipulationSim(args.scene)
            robot = Robot(plant)
            for _ in range(30):
                robot.tick()
            camera = SimCamera(robot, plant)
            observation, frame = (
                camera.capture()
            )  # Warm rendering before timed control.
            if dashboard:
                dashboard.observation(observation, frame)
                dashboard.watch_sim(robot, plant)
                # No control thread or simulated time advances while awaiting Start.
                args.instruction = dashboard.wait_start()
            available = ("left", "right") if args.side == "both" else (args.side,)
        else:
            from agilex_deploy.agent_zed import ZedStereoCamera
            from agilex_deploy.hardware_session import HardwareSession

            robot = HardwareSession(args.side, gripper_control=True, gravity_comp=args.gravity_comp)
            available = ("left", "right") if args.side == "both" else (args.side,)
            # All hardware IPC, including live monitoring before enable, shares one lock.
            executor = AgentExecutor(robot, available)
            camera = ZedStereoCamera(
                executor,
                args.calibration,
                expected_serial=args.camera_serial,
                device=args.camera_device,
            )
            observation, frame = (
                camera.capture()
            )  # Validate RGB-D before any arm enable.
            if dashboard:
                dashboard.observation(observation, frame)
                dashboard.update(
                    selected_arm=args.side,
                    execute_real=args.execute_real,
                    calibration=frame.calibration,
                )
                with dashboard.lock:
                    dashboard.halt = executor.halt
                dashboard.watch_real(executor, camera)
            if not args.execute_real:
                save_frame(frame, directory, 0)
                (directory / "robot.json").write_text(
                    json.dumps(observation.to_dict(), indent=2) + "\n"
                )
                report["status"] = "read_only_preflight"
                if dashboard:
                    dashboard.update(
                        phase="monitoring",
                        can_start=False,
                        message="Live ZED and robot feedback. Physical execution is disabled for this session.",
                    )
                    while not dashboard.exited.wait(0.1):
                        dashboard.check_stop()
                return
            if dashboard:
                args.instruction = dashboard.wait_start()
                dashboard.check_stop()
            # Reacquire aligned RGB-D/feedback after the user has reviewed the live view.
            observation, frame = camera.capture()
            if observation.mode == "fault":
                raise RuntimeError(observation.reason)
            if dashboard:
                dashboard.check_stop()
                dashboard.update(
                    phase="arming",
                    can_start=False,
                    message=f"Enabling {args.side} arm(s) at measured postures…",
                )
            executor.arm()
        if executor is None:
            executor = AgentExecutor(robot, available)
        executor.start()
        report["instruction"] = args.instruction
        if dashboard:
            with dashboard.lock:
                dashboard.halt = executor.halt
            dashboard.check_stop()
            dashboard.event("Episode started: " + args.instruction)
        if args.backend == "real":
            camera.robot = executor  # Serialize camera and heartbeat IPC reads.
        execution = {"state": "idle"}
        turns = []
        for turn in range(args.max_calls):
            if dashboard:
                dashboard.check_stop()
            # The executor maintains measured arm HOLD and existing jaw commands
            # while this thread waits for camera/network I/O.
            observation, frame = camera.capture()
            if dashboard:
                dashboard.observation(observation, frame)
                dashboard.update(
                    phase="thinking",
                    turn=turn + 1,
                    message="Waiting for the model. Robot holds its current arm targets.",
                    api_calls=policy.calls + 1,
                    api_cost_usd=policy.cost,
                )
            try:
                decision = policy.act(
                    observation, frame, args.instruction, execution, available
                )
            except (ValueError, json.JSONDecodeError) as error:
                execution = {"state": "rejected", "reason": str(error)}
                if dashboard:
                    dashboard.event("Decision rejected: " + str(error))
                    dashboard.update(execution=execution)
                print(json.dumps({"turn": turn, **execution}), flush=True)
                continue
            if dashboard:
                dashboard.check_stop()
                dashboard.event(decision.summary)
                dashboard.update(
                    phase="inspecting" if decision.action is None else "executing",
                    message=decision.summary,
                    action=None
                    if decision.action is None
                    else decision.action.to_dict(),
                    api_calls=policy.calls,
                    api_cost_usd=policy.cost,
                )
            print(
                json.dumps(
                    {
                        "turn": turn,
                        "summary": decision.summary,
                        "action": None
                        if decision.action is None
                        else decision.action.to_dict(),
                        "grounding": decision.grounding,
                    }
                ),
                flush=True,
            )
            try:
                execution = (
                    {"state": "inspected", "grounding": decision.grounding}
                    if decision.action is None
                    else executor.execute(decision)
                )
            except ValueError as error:
                execution = {"state": "rejected", "reason": str(error)}
            if dashboard:
                dashboard.update(execution=execution)
                dashboard.event(
                    "Execution: "
                    + execution["state"]
                    + (
                        f" · arrived={execution['arrived']}"
                        if "arrived" in execution
                        else ""
                    )
                    + (" · " + execution["reason"] if "reason" in execution else "")
                )
            turns.append(
                {"turn": turn, "decision": asdict(decision), "execution": execution}
            )
            (directory / "turns.json").write_text(json.dumps(turns, indent=2) + "\n")
            if decision.done and execution["state"] != "rejected":
                report["policy_declared_done"] = True
                break
        # Fixed final verification interval, independent of privileged success.
        # Do not release a closed gripper or stop physics when the model finishes.
        until = time.monotonic() + 11
        if dashboard:
            dashboard.update(
                phase="verifying", message="Observing the final hold for 11 seconds."
            )
        while time.monotonic() < until:
            if dashboard:
                dashboard.check_stop()
            executor.observe()
            time.sleep(0.1)
        observation, frame = camera.capture()
        save_frame(frame, directory, 999)
        report["final_observation"] = observation.to_dict()
        report["status"] = "completed"
    except AgentStopped as error:
        report["status"], report["error"] = "stopped", str(error)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        if dashboard:
            dashboard.exited.set()
    except Exception as error:  # noqa: BLE001 - preserve failures for browser review after cleanup.
        if dashboard and dashboard.cancelled.is_set():
            report["status"] = "stopped"
            report["error"] = (
                "Operator stopped session; pending request ended: "
                + type(error).__name__
            )
        else:
            report["status"], report["error"] = (
                "failed",
                f"{type(error).__name__}: {error}",
            )
    finally:
        cleanup_errors = []
        if dashboard:
            dashboard.update(
                phase="finalizing", message="Closing control and saving the report…"
            )
            try:
                dashboard.freeze()
            except Exception as error:  # noqa: BLE001 - rendering errors must not skip robot cleanup.
                cleanup_errors.append(f"Dashboard shutdown: {error}")
        for resource in (executor, robot, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:  # noqa: BLE001 - attempt every owner even if HOLD fails.
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            report["status"] = "failed"
        try:
            report.update(api_calls=policy.calls, api_cost_usd=policy.cost)
            # Only after policy/control termination may the offline evaluator read truth.
            if plant is not None:
                from agilex_deploy.agent_sim import evaluate, render_video

                truth = plant.truth
                (directory / "evaluator_truth.json").write_text(
                    json.dumps(truth) + "\n"
                )
                report["evaluation"] = evaluate(
                    truth, plant.scene_config["cube"]["position_m"][2]
                )
                if args.video:
                    render_video(plant, directory / "trajectory.mp4")
            (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if dashboard:
                message = (
                    "Episode ended. Review the result below. Close session to exit."
                )
                if report["status"] == "completed" and "evaluation" in report:
                    result = report["evaluation"]
                    message = (
                        "Pickup passed. "
                        if result["success"]
                        else "Pickup criterion not met. "
                    ) + (
                        f"Max lift {100 * result['max_lift_m']:.1f} cm; "
                        f"final retention {result['final_retention_s']:.1f} s. Close session to exit."
                    )
                if report["status"] == "stopped":
                    message = "Policy stopped; control session closed. No later model decision can execute."
                elif report["status"] == "failed":
                    message = report.get(
                        "error", "Session cleanup failed; inspect the report."
                    )
                dashboard.update(
                    phase=report["status"],
                    result=report,
                    api_calls=policy.calls,
                    api_cost_usd=policy.cost,
                    message=message,
                )
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "final_observation"}
                ),
                flush=True,
            )
        finally:
            if dashboard:
                try:
                    if not cleanup_errors and sys.exc_info()[0] is None:
                        while not dashboard.exited.wait(0.2):
                            pass
                except KeyboardInterrupt:
                    pass
                finally:
                    dashboard.close()
            if cleanup_errors:
                raise RuntimeError("Session cleanup failed; see report.json")
    if report["status"] == "failed":
        raise RuntimeError(report.get("error", "Session failed; see report.json"))


if __name__ == "__main__":
    main()
