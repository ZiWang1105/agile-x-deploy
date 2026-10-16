"""Run one RGB-D policy through the shared robot interface; default is simulation."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from agilex_deploy.agent_execution import AgentExecutor
from agilex_deploy.agent_policy import OpenRouterPolicy
from agilex_deploy.agent_sensors import save_frame
from agilex_deploy.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument(
        "--scene", type=Path, default=ROOT / "configs/agent/cube_scene.json"
    )
    parser.add_argument(
        "--key-file", type=Path, default=Path.home() / "projects/openrouter_key.txt"
    )
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--max-calls", type=int, default=16)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument(
        "--instruction",
        default="Pick up the red cube with either hand, lift it at least 10 cm from its starting height, and hold it securely in the air for at least 10 seconds. Avoid touching the table with the robot.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--execute-real",
        action="store_true",
        help="Explicitly enable physical motion; default real mode only captures observations",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT
        / "calibration/2026-10-15_zed35330610/raised_probe/results_agent_v2/candidate.json",
    )
    args = parser.parse_args()
    if args.max_calls < 1 or not 0 < args.max_cost < float('inf'):
        parser.error('Require a positive call count and finite positive cost budget')
    if args.execute_real and args.backend != 'real':
        parser.error('--execute-real requires --backend real')
    directory = args.output or ROOT / "logs/agent" / time.strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "source"
    source.mkdir()
    files = list((ROOT / "src/agilex_deploy").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {}
    for file in files:
        content = file.read_bytes()
        (source / file.name).write_bytes(content)
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    (directory / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if args.backend == "sim":
        (directory / "scene.json").write_bytes(args.scene.read_bytes())
    plant = camera = robot = executor = None
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
        if args.backend == "sim":
            from agilex_deploy.agent_sim import ManipulationSim, SimCamera
            from agilex_deploy.runtime import Robot

            plant = ManipulationSim(args.scene)
            robot = Robot(plant)
            for _ in range(30):
                robot.tick()
            camera = SimCamera(robot, plant)
            camera.capture()  # Warm the render context before starting timed control.
            available = ("left", "right")
        else:
            from agilex_deploy.agent_zed import ZedStereoCamera
            from agilex_deploy.hardware_session import HardwareSession

            robot = HardwareSession(args.side, gripper_control=True)
            camera = ZedStereoCamera(robot, args.calibration)
            available = (args.side,)
            observation, frame = camera.capture()  # Validate RGB-D before any arm enable.
            if not args.execute_real:
                save_frame(frame, directory, 0)
                (directory / "robot.json").write_text(
                    json.dumps(observation.to_dict(), indent=2) + "\n"
                )
                report["status"] = "read_only_preflight"
                return
            robot.arm()
        executor = AgentExecutor(robot, available).start()
        if args.backend == "real":
            camera.robot = executor  # Serialize camera and heartbeat IPC reads.
        execution = {"state": "idle"}
        turns = []
        for turn in range(args.max_calls):
            # The executor maintains measured arm HOLD and existing jaw commands
            # while this thread waits for camera/network I/O.
            observation, frame = camera.capture()
            try:
                decision = policy.act(
                    observation, frame, args.instruction, execution, available
                )
            except (ValueError, json.JSONDecodeError) as error:
                execution = {"state": "rejected", "reason": str(error)}
                print(json.dumps({"turn": turn, **execution}), flush=True)
                continue
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
        while time.monotonic() < until:
            executor.observe()
            time.sleep(0.1)
        observation, frame = camera.capture()
        save_frame(frame, directory, 999)
        report["final_observation"] = observation.to_dict()
        report["status"] = "completed"
    except BaseException as error:
        report["status"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for resource in (executor, robot, camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:  # Cleanup must attempt every owner, including on HOLD failure.
                    cleanup_errors.append(f'{type(error).__name__}: {error}')
        if cleanup_errors:
            report['cleanup_errors'] = cleanup_errors
            report['status'] = 'failed'
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
            print(
                json.dumps(
                    {k: v for k, v in report.items() if k != "final_observation"}
                ),
                flush=True,
            )
        finally:
            if cleanup_errors:
                raise RuntimeError('Session cleanup failed; see report.json')


if __name__ == "__main__":
    main()
