"""Foreground owner for a policy, robot service, and shared sim/real dashboard."""

import os
import argparse
import json
from pathlib import Path
import time
import uuid
import webbrowser

# Set before either runner imports its numerical libraries. Explicit user values
# still win; simulation and physical control use the same launch environment.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "egl")


def add_launch_arguments(parser):
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument("--backend", choices=("sim", "real"),
                            help="Launch the robot service and dashboard (default: sim)")
    connection.add_argument("--config", type=Path,
                            help="Attach to an existing workshop service instead of launching one")
    parser.add_argument("--task", choices=("place", "select", "sort"), default="place")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8780, help="Dashboard port; use 0 for an available port")
    parser.add_argument("--camera-serial")
    parser.add_argument("--gravity-comp", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to hand-guiding on enable/idle/exit (default: false; physical startup remains interlocked)")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--no-browser", action="store_true", help="Print the organizer URL without opening a browser")
    parser.add_argument("--exit-after-policy", action="store_true",
                        help="Close the owned service after execution instead of keeping the result page open")


def launch_policy(args, mode, run):
    """Run a callback after organizer enable, or attach with explicit --config.

    The callback owns its participant lease. This owner never enables hardware
    automatically and never detaches a service. STOP before enable cancels the
    queued policy. After execution, the same session accepts another prompt until
    Close or Ctrl-C. Follow-up rounds do not re-arm or reset the robot.
    """
    if args.config is not None:
        if args.output is None:
            args.output = args.config.parent / (mode + "_policy_report.json")
        return run()

    from .model import ROOT
    from .policy_trace import scrub
    from .workshop_http import WorkshopHTTP
    from .workshop_service import WorkshopService

    directory = ROOT / "logs/workshop" / ("policy_" + time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    args.config = directory / "client.json"
    if args.output is None:
        args.output = directory / "policy_report.json"
    first_output = args.output
    history = []
    round_number = 0
    service = server = None
    failure = None
    try:
        service = WorkshopService(backend=args.backend or "sim", task=args.task,
                                  seed=args.seed, output=directory,
                                  calibration=args.calibration, camera_serial=args.camera_serial,
                                  gravity_comp=getattr(args, "gravity_comp", False))
        if args.prompt:
            service.instruction = args.prompt
        args.prompt = service.instruction
        server = WorkshopHTTP(service, args.port, args.config)
        server.launcher = dict(mode=mode, model=args.model, prompt=args.prompt,
                               state="waiting_for_enable", error=None, round=0)
        url = server.endpoint + "/#" + server.operator_token
        print(f"\n{'PHYSICAL ROBOT' if service.backend == 'real' else 'SIMULATION'} · {mode} policy\n"
              "Workshop on Agentic Robotics: https://agentic-robotics-workshop.github.io/\n"
              f"Organizer dashboard: {url}\n"
              "Review the camera and prompt, then click Enable & start policy.\n"
              f"Logs: {directory}\n", flush=True)
        if not args.no_browser:
            try:
                opened = webbrowser.open(url, new=2)
            except Exception:
                opened = False
            if not opened:
                print("Browser could not open automatically; open the organizer URL above.", flush=True)

        while True:
            with service.lock:
                cancelled = server.close_requested.is_set() or service.stopped
                fault = service.fault
                enabled = service.enabled
                state = server.launcher["state"]
            if fault:
                if round_number:
                    failure = failure or RuntimeError(fault)
                    server.launcher.update(state="failed", error=scrub(fault))
                    break  # Keep the fault visible; follow-up prompts stay blocked.
                raise RuntimeError(fault)
            if cancelled:
                server.launcher.update(state="cancelled")
                break
            if enabled and state in ("waiting_for_enable", "queued"):
                with service.lock:
                    # Claim this round before releasing the lock; a second POST
                    # cannot replace its prompt or launch a concurrent policy.
                    if service.stopped or service.fault or server.close_requested.is_set():
                        continue
                    round_number += 1
                    args.prompt = server.launcher["prompt"]
                    args.conversation_history = list(history[-20:])
                    args.output = (first_output if round_number == 1 else
                                   first_output.with_name(f"{first_output.stem}_round_{round_number:03d}{first_output.suffix}"))
                    server.launcher.update(state="running", round=round_number, error=None)
                failure = None
                try:
                    report = run() or {}
                except (Exception, SystemExit) as error:
                    failure = error
                    report = dict(status="failed", error=scrub(str(error)))
                    with service.lock:
                        server.launcher.update(state="cancelled" if service.stopped and not service.fault else "failed",
                                               error=scrub(str(error)))
                else:
                    with service.lock:
                        server.launcher.update(state="cancelled" if service.stopped else "finished")
                # Carry bounded conversational summaries, not old camera images
                # or reusable motion targets. Each runner reads current state.
                summary = scrub({k: report[k] for k in ("status", "final", "error", "task_success") if k in report})
                history.extend([dict(role="user", content=args.prompt),
                                dict(role="assistant", content="Previous round report (not verified current state): " + json.dumps(summary))])
                service.event("policy_round_end", round=round_number,
                              prompt=args.prompt, report=str(args.output), summary=summary)
                next_step = ("Session stopped or faulted; another prompt cannot resume it. Close session / Ctrl-C to exit."
                             if service.stopped or service.fault else
                             "Submit another prompt in the dashboard, or Close session / Ctrl-C to exit.")
                print(f"Policy round {round_number} {server.launcher['state']}. {next_step}", flush=True)
                if args.exit_after_policy:
                    break
            server.close_requested.wait(.1)

        print("Policy " + server.launcher["state"] + ". Review the dashboard; Close session or Ctrl-C exits.", flush=True)
        if not args.exit_after_policy:
            while not server.close_requested.wait(.2):
                pass
    except KeyboardInterrupt:
        if server is not None:
            server.launcher.update(state="cancelled")
        if service is not None:
            service.operator_stop()
    finally:
        # Both owners must close even when one cleanup path fails. Their close
        # methods join the control, camera, and HTTP threads/processes.
        try:
            if server is not None:
                server.close()
        finally:
            if service is not None:
                service.close()
    if failure is not None:
        raise failure
