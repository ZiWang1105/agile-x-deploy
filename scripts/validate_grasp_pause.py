"""Replay recorded generated code in simulation and measure grip through cleanup.

No model requests or physical hardware access. Simulator truth is used only by
this organizer check, never provided to the generated policy.
"""

from agilex_deploy.policy_launcher import launch_policy  # Configure offscreen rendering first.

import argparse
import base64
import json
from pathlib import Path
import time

from agilex_deploy.code_execution import execute_code
from agilex_deploy.workshop_client import RobotClient
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService


class Trace:
    def emit(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recorded = json.loads(args.report.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    service = server = None
    result = {"source_report": str(args.report), "backend": "sim", "model_calls": 0}
    try:
        service = WorkshopService("sim", "place", 0, args.output)
        server = WorkshopHTTP(service, 0)
        service.enable()
        with RobotClient(server.endpoint, server.token) as robot:
            for generation in recorded["generations"]:
                outcome, _ = execute_code(generation["code"], robot, Trace())
                if not outcome["ok"]:
                    raise RuntimeError(outcome["error"])
            with service.robot.lock:
                start = len(service.plant.truth)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                robot.observe()  # Maintain the participant lease while awaiting retention.
                time.sleep(.1)
            frame = robot.capture()
            (args.output / "after_cleanup.png").write_bytes(base64.b64decode(frame["rgb_png_base64"]))
            with service.robot.lock:
                rows = service.plant.truth[start:]
                initial_z = service.plant.initial["red"][2]
                minimum_lift = min(r["objects"]["red"]["position_m"][2] - initial_z for r in rows)
                opposed = all({"left_pad1", "left_pad2"}.issubset(r["objects"]["red"]["contacts"]) for r in rows)
                supported = any("table" in r["objects"]["red"]["contacts"] for r in rows)
            result.update(retention_s=10, minimum_lift_m=minimum_lift,
                          opposing_finger_contacts=opposed, table_support=supported)
        # Normal session release must retain grip while arms enter hand-guiding.
        time.sleep(1)
        with service.robot.lock:
            final = service.plant.truth[-1]["objects"]["red"]
        result.update(final_mode=service.observe()["mode"], after_release=final)
        result["success"] = (minimum_lift >= .1 and opposed and not supported
                             and {"left_pad1", "left_pad2"}.issubset(final["contacts"]))
    finally:
        if server:
            server.close()
        if service:
            service.close()
        (args.output / "retention.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
