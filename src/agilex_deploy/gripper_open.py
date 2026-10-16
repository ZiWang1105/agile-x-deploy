"""Open grippers with measured-joint holding and restore disabled arm motors."""

import time
from dataclasses import asdict

from .contract import SIDES


def open_with_arm_hold(mapping, sides, *, arms_supported=False):
    """Use the existing commissioning transport with constant measured joint targets."""
    if not arms_supported:
        raise ValueError("Mechanically support BOTH arms before opening jaws and disabling")
    import numpy as np

    from agilex_deploy.contract import Action, ArmTarget
    from agilex_deploy.hardware_power import disable_supported_arms
    from agilex_deploy.nero_transport import NeroTransport

    result = {
        "status": "fully_open_confirmed",
        "grippers": {},
        "arm_trajectory": "constant measured joint positions; original disabled state restored",
    }
    for side in sides:
        transport = NeroTransport(side, mapping=mapping, gripper_control=True)
        arm_requested = False
        try:
            arm_requested = True
            transport.arm()
            sample = transport.read_encoders()
            held = np.asarray(sample.positions)
            transport.kin.update(held)
            action = Action(
                *(
                    ArmTarget(transport.kin.pose(s), 1 if s == side else None)
                    for s in SIDES
                )
            )
            transport.activate_gripper(action)
            grips = [None, None]
            grips[SIDES.index(side)] = 1.0
            deadline, settled_since, max_change = time.monotonic() + 10, None, 0.0
            while time.monotonic() < deadline:
                sample = transport.read_encoders()
                max_change = max(
                    max_change,
                    float(np.max(np.abs(np.asarray(sample.positions) - held))),
                )
                transport.send_positions(held, grips)
                feedback = transport.last.gripper_feedback[SIDES.index(side)]
                if (
                    feedback.enabled
                    and abs(feedback.width_m - transport.gripper_range_m) <= 0.001
                ):
                    settled_since = settled_since or time.monotonic()
                    if time.monotonic() - settled_since >= 0.1:
                        result["grippers"][side] = {
                            **asdict(feedback),
                            "target_width_m": transport.gripper_range_m,
                            "max_observed_arm_change_rad": max_change,
                        }
                        print(
                            f"{side}: fully open verified at {feedback.width_m * 1000:.2f} mm",
                            flush=True,
                        )
                        break
                else:
                    settled_since = None
                time.sleep(0.01)
            else:
                raise RuntimeError(
                    f"{side}: gripper did not reach fully open; last width {feedback.width_m * 1000:.2f} mm"
                )
        finally:
            try:
                transport.close()
            finally:
                if arm_requested:
                    disable_supported_arms(mapping, arms_supported=True)
    result["arms_disabled"] = True
    return result
