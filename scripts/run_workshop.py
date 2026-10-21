"""Run the Workshop on Agentic Robotics robot service and dashboard.

https://agentic-robotics-workshop.github.io/"""

import argparse
from pathlib import Path
import time

from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService
from agilex_deploy.workshop_tasks import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sim", "real"], default="sim")
    parser.add_argument("--task", choices=list(TASKS), default="place")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument(
        "--client-config", type=Path, default=Path("logs/workshop/client.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--camera-serial")
    parser.add_argument("--gravity-comp", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to hand-guiding on enable/idle/exit (default: false)")
    parser.add_argument("--auto-start-sim", action="store_true")
    parser.add_argument(
        "--duration", type=float, help="Optional foreground smoke-test duration"
    )
    args = parser.parse_args()
    if args.auto_start_sim and args.backend != "sim":
        parser.error("--auto-start-sim cannot enable hardware")
    service = server = None
    try:
        service = WorkshopService(
            args.backend,
            args.task,
            args.seed,
            args.output,
            args.calibration,
            args.camera_serial,
            gravity_comp=args.gravity_comp,
        )
        server = WorkshopHTTP(service, args.port, args.client_config)
        if args.auto_start_sim:
            service.enable()
        print(
            f"Workshop on Agentic Robotics: https://agentic-robotics-workshop.github.io/\nRobot dashboard: {server.endpoint}\nOperator link: {args.client_config.with_name('operator_url.txt')}\nParticipant configuration: {args.client_config}\nLogs: {service.output}",
            flush=True,
        )
        deadline = time.monotonic() + args.duration if args.duration else float("inf")
        while time.monotonic() < deadline:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.close()
        if service:
            service.close()


if __name__ == "__main__":
    main()
