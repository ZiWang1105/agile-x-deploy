#!/usr/bin/env python3
"""NERO fixed-head ZED calibration: plan, capture, annotate, fit and validate."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("import-intrinsics")
    p.add_argument("--conf", type=Path, required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("init")
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p = commands.add_parser("plan")
    p.add_argument("--side", choices=["left", "right"], default="left")
    p.add_argument("--seed", type=Path)
    p.add_argument("--translation-m", type=float, default=0.06)
    p.add_argument("--rotation-deg", type=float, default=15)
    p.add_argument("--lift-m", type=float, default=0.0,
                   help="Raise the sampling centre along base Z; approach poses are not captured")
    p.add_argument("--poses", type=int, choices=[9, 15], default=9)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--mode", choices=["sim", "real"], default="sim")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--reviewed-plan", action="store_true")
    p.add_argument("--stream-url", default="http://127.0.0.1:8766")
    p = commands.add_parser("capture")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--stream-url", default="http://127.0.0.1:8766")
    p.add_argument(
        "--mapping",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/hardware_mapping.json",
    )
    p.add_argument("--max-span-rad", type=float, default=0.003)
    p = commands.add_parser("annotate")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--port", type=int, default=8772)
    p = commands.add_parser("solve")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-synthetic", action="store_true")
    p = commands.add_parser("demo")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from agilex_deploy.camera_calibration import (
        save_json,
        new_dataset,
        import_factory_intrinsics,
        solve_dataset,
    )

    if args.command == "import-intrinsics":
        if args.output.exists():
            raise ValueError("Output already exists; choose a new filename")
        save_json(args.output, import_factory_intrinsics(args.conf, args.serial))
    elif args.command == "init":
        if args.dataset.exists():
            raise ValueError("Dataset already exists; refusing to overwrite")
        save_json(args.dataset, new_dataset(json.loads(args.intrinsics.read_text())))
    elif args.command == "plan":
        from agilex_deploy.calibration_motion import plan_route

        if args.output.exists():
            raise ValueError("Plan already exists; choose a new filename")
        plan_route(
            args.output, args.side, args.seed, args.translation_m, args.rotation_deg,
            args.lift_m, args.poses,
        )
    elif args.command == "run":
        from agilex_deploy.calibration_motion import run_route

        run_route(
            args.plan, args.mode, args.dataset, args.stream_url, args.reviewed_plan
        )
    elif args.command == "capture":
        from agilex_deploy.calibration_capture import passive_capture

        print(
            json.dumps(
                passive_capture(
                    args.dataset, args.stream_url, args.mapping, args.max_span_rad
                ),
                indent=2,
            )
        )
    elif args.command == "annotate":
        from agilex_deploy.calibration_ui import serve

        serve(args.dataset, args.port)
    elif args.command == "solve":
        if args.output.exists():
            raise ValueError("Result already exists; choose a new filename")
        result = solve_dataset(args.dataset, args.output, args.allow_synthetic)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "status",
                        "kind",
                        "inlier_rms_px",
                        "all_rms_px",
                        "leave_one_pose_out",
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "demo":
        from agilex_deploy.calibration_demo import create_demo

        print(json.dumps(create_demo(args.output), indent=2))


if __name__ == "__main__":
    main()
