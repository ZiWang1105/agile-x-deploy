"""Offline replay of NERO mode transitions through the real transport and MuJoCo.

Every CAN writer and feedback source is replaced by the firmware simulator.
This checks protocol behavior and reference continuity, not physical gravity
compensation. No robot connection or model API is used.
"""
from agilex_deploy.policy_launcher import launch_policy  # Configure EGL before MuJoCo imports.

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

from agilex_deploy.model import ROOT
from agilex_deploy.hardware_power import _disable_and_verify

sys.path.insert(0, str(ROOT / "tests"))
from nero_firmware_sim import FirmwareSim
from test_dual_control import fake_hardware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/hand_guiding_fix_20261020")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bus = FirmwareSim()
    report = {"backend": "MuJoCo + NERO 1.21 firmware emulator",
              "physical_commands_sent": 0, "physical_gravity_compensation_verified": False,
              "physical_deployment_qualified": False, "cycles": []}
    try:
        with fake_hardware(bus) as (transport, _):
            transport.arm()
            q_start = np.asarray(transport.read_encoders().positions)
            max_uncommanded = 0.
            for cycle in range(3):
                transport.enter_hand_guiding()
                deadline = time.monotonic()+1
                while time.monotonic() < deadline:
                    sample = transport.read_encoders()
                    max_uncommanded = max(max_uncommanded, float(np.max(np.abs(np.asarray(sample.positions)-q_start))))
                    time.sleep(.01)
                if any(r["motor_enabled"] is not None for r in sample.feedback_status):
                    raise AssertionError("Leader status must be explicitly unknown")
                leader_feedback = sample.feedback_status
                transport.begin_control()
                after = transport.read_encoders()
                report["cycles"].append({"index": cycle, "leader_feedback": leader_feedback,
                                         "follower_feedback": after.feedback_status,
                                         "mode_after_return": transport.control_mode})
            report["max_joint_drift_rad_ideal_gravity"] = max_uncommanded
            if max_uncommanded > .001:
                raise AssertionError("Unexpected simulated movement during mode transitions")
            transport.enter_hand_guiding()
            transport.close()
            with patch("agilex_deploy.hardware_power.socket.socket",
                       side_effect=lambda *a: closing(bus.socket(*a))):
                report["supported_shutdown"] = _disable_and_verify(bus)
            report["success"] = True
    finally:
        bus.plant.close()
        (args.output / "simulation_report.json").write_text(json.dumps(report, indent=2)+"\n")
        (args.output / "simulation_trace.json").write_text(json.dumps(bus.timeline, indent=2)+"\n")
        (args.output / "can_commands.json").write_text(json.dumps([
            {"arm": s, "can_id": hex(cid), "payload": data.hex()} for s, cid, data in bus.frames
        ], indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
