"""Open jaws while holding measured arm positions, then disable supported arms."""

import argparse
import json
from pathlib import Path

from agilex_deploy.contract import SIDES
from agilex_deploy.model import ROOT
from agilex_deploy.gripper_open import open_with_arm_hold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument(
        "--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--arms-supported",
        action="store_true",
        help="Temporarily enable each arm at its measured joint positions, then restore disabled arms; BOTH arms must be supported",
    )
    args = parser.parse_args()
    if not args.arms_supported:
        parser.error("--arms-supported is required: support BOTH arms before running")
    sides = SIDES if args.side == "both" else (args.side,)
    result = open_with_arm_hold(args.mapping, sides, arms_supported=True)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)




if __name__ == "__main__":
    main()
