"""Model-driven MCP policy starter. Only public robot tools are available."""

from agilex_deploy.policy_launcher import add_launch_arguments, launch_policy

import argparse
import json
from pathlib import Path
import urllib.request
import time

from agilex_deploy.policy_model import (ModelResponseGuard, add_model_arguments,
                                      model_options, request_model, response_metadata,
                                      require_active_session)
from agilex_deploy.policy_trace import PolicyTrace

from mcp_baseline import MCPRobotClient

SYSTEM = """You are a robot policy for the Workshop on Agentic Robotics (https://agentic-robotics-workshop.github.io/). Complete the user's task using only the provided MCP tools.
Both arms and grippers are independent; choose left for positive base Y and right for negative base Y.
One fixed head RGB-D camera, no wrist cameras. Poses are base_link meters and XYZW quaternions.
The simulator cubes are 50 mm wide. The nominal pinch frame is the midpoint between distal fingers.
For top-down pinch poses use quaternion_xyzw [0.5,-0.5,-0.5,-0.5].
Claim the session, get capabilities, and inspect the camera before movement. Camera pixels are actual
[u,v] pixels in the returned size_wh, NOT normalized coordinates. Use camera_deproject on visible
interior cube/tray surfaces to ground targets. A cube pinch center is about 25 mm below its top face.
A tray placement pinch center is about 31 mm above its inner floor. Use about 120 mm vertical clearance
for approach, lift, transfer and retraction, adjusting from measured geometry if needed.
Open ONLY the selected gripper; move above the cube; descend; close; lift; inspect.
For a placement task, then transfer above the tray; lower; open; retract; inspect release.
For a pickup/hold task, keep the grasp after lifting and verify it with the camera.
Other arms/grippers retain their targets automatically.
IK is read-only: success means <=5 mm position and <=0.05 rad orientation residual,
not merely optimizer convergence. A one-seed IK failure does not prove the pose unreachable.
Pose motion validates sampled paths using official IK before dispatch. It tries the shortest
rotation arc, then its complement with the same endpoint. This is not collision checking.
PLANNING_FAILED means no validated path was sent; inspect the reported residuals and adjust the
explicit approach or target. Do not treat an inaccurate one-shot IK solution as a joint command.
Motion calls return operation_id. ALWAYS call operation_wait and inspect completion before proceeding.
Wait expiry may still report running. A failed operation is not arrival. A settled/stalled gripper
is not proof of a grasp: use camera and lift observations. No collision planner exists. Do not sweep
an arm through the table, stand, other arm or camera. Do not command both grippers unless both need it.
Use unique request_id strings per new command; the same ID is only for identical retries.
Do not issue concurrent motion tools. Finish only after verifying the user's requested outcome.
Do not call session_release until done; the host releases automatically when your final response ends.
Previous-round reports provide conversation context, not verified current state. Inspect fresh
robot feedback and camera images before acting on a follow-up instruction.
"""


def ensure_active(robot):
    require_active_session(robot.observe())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser)
    parser.add_argument(
        "--key-file",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "openrouter_key.txt",
        help="OpenRouter key file (default: openrouter_key.txt in the repository root)",
    )
    add_model_arguments(parser)
    parser.add_argument("--prompt")
    parser.add_argument("--max-calls", type=int, default=100,
                        help="Maximum model calls per prompt (default: 100)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError("Empty API key")
    launch_policy(args, "mcp", lambda: run_policy(args, key))


def run_policy(args, key):
    report = {
        "model": args.model,
        "calls": 0,
        "status": "call_budget_exhausted",
        "tools": [],
    }
    with MCPRobotClient(args.config) as robot:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in robot.tools()
            if t.name not in ("session_claim", "session_release")
        ]
        instruction = args.prompt or robot.capabilities()["instruction"]
        trace = PolicyTrace(args.config, "mcp", args.model, instruction)
        messages = [
            {"role": "system", "content": SYSTEM},
            *getattr(args, "conversation_history", []),
            {"role": "user", "content": instruction},
        ]
        responses = ModelResponseGuard()
        for turn in range(args.max_calls):
            payload = dict(
                model=args.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tokens=responses.max_tokens,
                **model_options(args),
            )
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
            )
            trace.emit("model_request", turn=turn + 1)
            request_started = time.monotonic()
            try:
                result = request_model(request, trace, lambda: ensure_active(robot))
            except Exception as error:
                report.update(status="failed", error=str(error).replace(key, "[REDACTED]"))
                break
            trace.emit(
                "model_response",
                turn=turn + 1,
                duration_s=time.monotonic() - request_started,
                **response_metadata(result),
            )
            report["calls"] += 1
            try:
                message = responses.message(result, trace)
            except RuntimeError as error:
                report.update(status="failed", error=str(error))
                break
            if message is None:
                continue
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                trace.emit("run_finished", text=message.get("content"))
                report.update(status="finished", final=message.get("content"))
                break
            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                trace.emit(
                    "tool_call", call_id=call["id"], tool=name, arguments=arguments
                )
                call_started = time.monotonic()
                try:
                    # Recheck the latched operator stop after every model wait.
                    observed = robot.call("robot_observe")
                    require_active_session(observed)
                    result = robot.call(name, **arguments)
                    log_result = {
                        k: v for k, v in result.items() if not k.endswith("_base64")
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(log_result),
                        }
                    )
                    if name == "camera_capture":
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Head RGB for frame {result['frame_id']}; use this frame_id for deprojection.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + result["rgb_png_base64"]
                                        },
                                    },
                                ],
                            }
                        )
                except Exception as error:
                    log_result = {"error": str(error).replace(key, "[REDACTED]")}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(log_result),
                        }
                    )
                trace.emit(
                    "tool_result",
                    call_id=call["id"],
                    tool=name,
                    duration_s=time.monotonic() - call_started,
                    result=log_result,
                )
                if log_result.get("error") and any(
                    word in log_result["error"]
                    for word in ("STOPPED", "LEASE_EXPIRED", "faulted", "NOT_ENABLED")
                ):
                    trace.emit("run_failed", error=log_result["error"])
                    raise RuntimeError(log_result["error"])
                row = dict(tool=name, arguments=arguments, result=log_result)
                report["tools"].append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "tools"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "finished":
        error = report.get("error", report["status"])
        trace.emit("run_failed", error=error)
        raise RuntimeError(error)
    return report


if __name__ == "__main__":
    main()
