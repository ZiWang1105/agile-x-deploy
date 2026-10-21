"""Organizer validation: launch sim, run a separate SDK policy, then score truth."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from agilex_deploy.agent_sim import render_video
from agilex_deploy.model import ROOT
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["place", "select", "sort"], default="place")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/workshop/validation"
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--transport", choices=["python", "mcp"], default="python")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use the live OpenRouter MCP policy instead of the scripted RGB-D baseline",
    )
    parser.add_argument("--policy-mode", choices=["code", "mcp"], default="mcp")
    args = parser.parse_args()
    service = server = None
    try:
        service = WorkshopService("sim", args.task, args.seed, args.output)
        server = WorkshopHTTP(service, 0, args.output / "client.json")
        service.enable()
        example = (
            (
                "generated_code_policy.py"
                if args.policy_mode == "code"
                else "agent_policy.py"
            )
            if args.agent
            else ("code_policy.py" if args.transport == "python" else "mcp_baseline.py")
        )
        command = [
            sys.executable,
            str(ROOT / "examples/workshop" / example),
            "--config",
            str(args.output / "client.json"),
        ]
        if args.agent:
            command += ["--output", str(args.output / "agent_report.json")]
        result = subprocess.run(command, cwd=ROOT)
        with service.robot.lock:
            score = service.plant.evaluate()
        score["policy_exit_code"] = result.returncode
        score["transport"] = (
            ("python" if args.policy_mode == "code" else "mcp")
            if args.agent
            else args.transport
        )
        score["policy_type"] = "model" if args.agent else "scripted_rgbd_baseline"
        (args.output / "validation.json").write_text(json.dumps(score, indent=2) + "\n")
        print(json.dumps(score, indent=2), flush=True)
        time.sleep(3)  # Allow the dashboard to observe the terminal event before shutdown.
    finally:
        if server:
            server.close()
        if service:
            service.close()
    if args.video:
        render_video(service.plant, args.output / "episode.mp4")
    if not score["success"] or score["policy_exit_code"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
