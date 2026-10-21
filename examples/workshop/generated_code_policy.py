"""OpenRouter writes Python fences; an isolated worker executes the public SDK."""

from agilex_deploy.policy_launcher import add_launch_arguments, launch_policy

import argparse
import json
from pathlib import Path
import re
import time
import urllib.request

from agilex_deploy.code_execution import execute_code
from agilex_deploy.policy_model import (ModelResponseGuard, add_model_arguments,
                                      model_options, request_model, response_metadata,
                                      require_active_session)
from agilex_deploy.policy_trace import PolicyTrace, scrub
from agilex_deploy.workshop_client import RobotClient
from agent_policy import SYSTEM

CODE_SYSTEM = (
    SYSTEM
    + """
This run uses CODE-AS-POLICY, not model tool calling. Reply with exactly one
```python code fence defining run(robot), or a final plain-text conclusion only
after inspecting the result of your code. The host executes the fence and returns
stdout, its JSON-serializable return value, errors, and captured camera images.
The robot argument is already connected and claimed; do not construct a client.
Use the Python SDK:
robot.observe() -> joint_positions_rad (left 7 then right 7), left_tcp/right_tcp,
  pinch_poses, gripper_position, gripper_states and session_enabled.
robot.capture() -> frame_id, rgb_png_base64, depth_png_base64 (uint16 mm), K,
  T_base_from_optical, size_wh; frames are RGB-D, not simulator ground truth.
robot.deproject(frame_id, [u,v]) -> see its returned dictionary; print it if unsure.
robot.ik(arm, target, seed_rad=None, tool_frame="tcp") is READ ONLY.
Its success checks pose residuals; solver_success only describes optimizer convergence.
robot.fk(arm, joints_rad, tool_frame="tcp") is READ ONLY.
robot.move_tcp(arm, target, tool_frame="pinch", timeout_s=12),
robot.move_joints(arm, joints_rad), robot.move_both(targets),
robot.gripper_open(arm), robot.gripper_close(arm) return Operation objects.
Call op.wait() and require result["state"]=="succeeded" before the next motion.
robot.hold() stops arm motion while preserving the last gripper actuator targets.
It does not open the jaws or prove an object is grasped.
Poses: {"position_m":[x,y,z], "quaternion_xyzw":[x,y,z,w]}.
Python numpy (np), OpenCV (cv2), base64, json, math and time are available.
No files outside scratch, network, shell, or subprocess access. Never read keys.
Use camera geometry to locate the cube and tray. You may inspect image pixels
visually and deproject them, or write your own color/depth processing in Python.
When using a visible cube top surface, lower its Z by 0.025 m for pinch center.
The tray floor can be sampled at an interior pixel away from the rim.
Avoid returning raw image base64 or printing entire camera frames. Capture again
after manipulation so the host can show you the result. A fence has 90 seconds.
Each fence ends with arm HOLD and retained gripper targets; do not leave unfinished asynchronous actions.
The host provides a fresh image after cleanup. Verify that image before claiming a retained grasp;
an image captured inside your fence may precede cleanup and cannot prove the final state.
The first response must contain executable Python, not a final success claim.
"""
)


def add_frame(messages, frame, label="Head RGB-D metadata"):
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": label + ": " + json.dumps(scrub(frame)),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + frame["rgb_png_base64"]
                    },
                },
            ],
        }
    )


def ensure_active(robot):
    require_active_session(robot.observe())


def capture_after_cleanup(robot, after_s, timeout_s=3):
    """Do not verify a grasp using a cached image from before worker cleanup."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ensure_active(robot)
        frame = robot.capture()
        if frame["timestamp_s"] >= after_s:
            return frame
        time.sleep(.05)
    raise RuntimeError("No fresh camera frame after code cleanup; final state is unverified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "openrouter_key.txt",
    )
    add_model_arguments(parser)
    parser.add_argument("--prompt")
    parser.add_argument("--max-calls", type=int, default=100,
                        help="Maximum model calls per prompt (default: 100)")
    parser.add_argument(
        "--output", type=Path
    )
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError("Empty API key")
    launch_policy(args, "code", lambda: run_policy(args, key))


def run_policy(args, key):
    report = {
        "mode": "code",
        "model": args.model,
        "status": "call_budget_exhausted",
        "generations": [],
        "task_success": None,
    }
    trace = None
    try:
        with RobotClient.from_config(args.config) as robot:
            instruction = args.prompt or robot.capabilities()["instruction"]
            trace = PolicyTrace(args.config, "code", args.model, instruction)
            messages = [
                {"role": "system", "content": CODE_SYSTEM},
                *getattr(args, "conversation_history", []),
                {
                    "role": "user",
                    "content": instruction
                    + "\nRobot state: "
                    + json.dumps(robot.observe()),
                },
            ]
            add_frame(messages, robot.capture())
            responses = ModelResponseGuard()
            for turn in range(args.max_calls):
                if not robot.observe()["session_enabled"]:
                    raise RuntimeError("Session stopped; policy cancelled")
                trace.emit("model_request", turn=turn + 1)
                started = time.monotonic()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(
                        {"model": args.model, "messages": messages, "max_tokens": responses.max_tokens,
                         **model_options(args)}
                    ).encode(),
                    headers={
                        "Authorization": "Bearer " + key,
                        "Content-Type": "application/json",
                    },
                )
                result = request_model(req, trace, lambda: ensure_active(robot))
                trace.emit(
                    "model_response",
                    turn=turn + 1,
                    duration_s=time.monotonic() - started,
                    **response_metadata(result),
                )
                message = responses.message(result, trace)
                if message is None:
                    continue
                messages.append(message)
                content = message.get("content") or ""
                fences = re.findall(r"```(?:python|py)\s*\n(.*?)```", content, re.S)
                if not fences:
                    if (
                        report["generations"]
                        and report["generations"][-1]["result"]["ok"]
                    ):
                        report.update(status="finished", final=content)
                        trace.emit("run_finished", text=content)
                        break
                    messages.append(
                        {
                            "role": "user",
                            "content": "Provide one executable python fence defining run(robot). No successful execution has completed.",
                        }
                    )
                    continue
                if len(fences) != 1:
                    messages.append(
                        {"role": "user", "content": "Provide exactly one Python fence."}
                    )
                    continue
                code = fences[0]
                trace.emit("code_generated", turn=turn + 1, code=code)
                outcome, frames = execute_code(code, robot, trace)
                cleanup_finished_s = time.monotonic()
                report["generations"].append(
                    {"turn": turn + 1, "code": code, "result": outcome}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Python execution result: " + json.dumps(outcome),
                    }
                )
                for frame in frames:
                    add_frame(messages, frame, "Image captured during code execution")
                final_frame = capture_after_cleanup(robot, cleanup_finished_s)
                add_frame(messages, final_frame, "Fresh post-cleanup verification image; use this to assess the final state")
            if report["status"] != "finished":
                raise RuntimeError("Model call budget exhausted")
    except KeyboardInterrupt:
        report.update(status="cancelled", error="Interrupted by operator")
        if trace:
            trace.emit("run_failed", error=report["error"])
    except Exception as error:
        report.update(status="failed", error=str(error).replace(key, "[REDACTED]"))
        if trace:
            trace.emit("run_failed", error=report["error"])
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}))
    if report["status"] == "cancelled":
        raise KeyboardInterrupt
    if report["status"] != "finished":
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
