"""Check real RobotInterface feedback and process cleanup without arming."""
import argparse
import json
from pathlib import Path
import time

from agilex_deploy.hardware_session import HardwareSession
from agilex_deploy.can_feedback import PassiveCanFeedback, mapped_channels
from agilex_deploy.model import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    args = parser.parse_args()
    channels = mapped_channels(ROOT / "configs/hardware_mapping.json")
    before = {s: int(Path(f"/sys/class/net/{c}/statistics/tx_packets").read_text()) for s, c in channels.items()}
    with HardwareSession(args.side) as session:
        for _ in range(20):
            observation = session.observe()
            if observation.mode != "unarmed":
                raise RuntimeError("Expected unarmed session")
            time.sleep(.025)
        report = {"observation": observation.to_dict(), "selected_arm": args.side}
    after = {s: int(Path(f"/sys/class/net/{c}/statistics/tx_packets").read_text()) for s, c in channels.items()}
    report["tx_delta"] = {s: after[s] - before[s] for s in channels}
    report["control_process_joined"] = not session.process.is_alive()
    with PassiveCanFeedback() as reader:
        report["driver_status_after"] = reader.read(timeout=1.).driver_status
    dest = ROOT / "logs/commissioning/real_session_unarmed.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    assert report["tx_delta"] == {"left": 1, "right": 1}, "Expected one firmware read per arm, no other frames"
    assert not any(report["driver_status_after"]), "Expected disabled healthy motors"
    assert report["control_process_joined"]
    print("Real RobotInterface passed: unarmed, physical feedback, two firmware reads only, no enabled motors, child joined.")


if __name__ == "__main__":
    main()
