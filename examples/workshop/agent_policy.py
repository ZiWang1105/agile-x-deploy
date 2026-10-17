"""Model-driven MCP policy starter. Only public robot tools are available."""

import argparse
import json
from pathlib import Path
import urllib.request

from mcp_baseline import MCPRobotClient

SYSTEM = """You are a workshop robot policy. Complete the user's task using only the provided MCP tools.
Both arms and grippers are independent; choose left for positive base Y and right for negative base Y.
One fixed head RGB-D camera, no wrist cameras. Poses are base_link meters and XYZW quaternions.
The workshop cubes are 50 mm wide. The nominal pinch frame is the midpoint between distal fingers.
For top-down pinch poses use quaternion_xyzw [0.5,-0.5,-0.5,-0.5].
Claim the session, get capabilities, and inspect the camera before movement. Camera pixels are actual
[u,v] pixels in the returned size_wh, NOT normalized coordinates. Use camera_deproject on visible
interior cube/tray surfaces to ground targets. A cube pinch center is about 25 mm below its top face.
A tray placement pinch center is about 31 mm above its inner floor. Use about 120 mm vertical clearance
for approach, lift, transfer and retraction, adjusting from measured geometry if needed.
Open ONLY the selected gripper; move above the cube; descend; close; lift; inspect; transfer above tray;
lower; open; retract; inspect release. Other arms/grippers retain their targets automatically.
Motion calls return operation_id. ALWAYS call operation_wait and inspect completion before proceeding.
Wait expiry may still report running. A failed operation is not arrival. A settled/stalled gripper
is not proof of a grasp: use camera and lift observations. No collision planner exists. Do not sweep
an arm through the table, stand, other arm or camera. Do not command both grippers unless both need it.
Use unique request_id strings per new command; the same ID is only for identical retries.
Do not issue concurrent motion tools. Finish only after visible release in the tray and retraction.
Do not call session_release until done; the host releases automatically when your final response ends.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="logs/workshop/client.json")
    parser.add_argument(
        "--key-file", type=Path, default=Path.home() / "projects/openrouter_key.txt"
    )
    parser.add_argument("--model", default="google/gemini-3.8-flash")
    parser.add_argument("--prompt")
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    if not key:
        raise ValueError("Empty API key")
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
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instruction},
        ]
        for turn in range(args.max_calls):
            payload = dict(
                model=args.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tokens=2048,
            )
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
            report["calls"] += 1
            message = result["choices"][0]["message"]
            messages.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                report.update(status="finished", final=message.get("content"))
                break
            for call in calls:
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                try:
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
                row = dict(tool=name, arguments=arguments, result=log_result)
                report["tools"].append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != "tools"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
