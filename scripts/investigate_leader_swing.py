"""Offline sensitivity experiment, NOT a replay of the physical incident.

No CAN or robot driver is opened. Each run releases only the left position
servo and varies the left body's ideal gravity compensation. The right servo
stays active. These artificial faults expose what the previous ideal-gravity
test could not detect; they do not identify the real controller's torque law.
"""

import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch

import mujoco
import numpy as np

from agilex_deploy.backends import SimBackend
from agilex_deploy.hand_guiding import official_mode_frames


def experiment(factor, duration_s=1.0, *, retained_position_servo=False):
    plant = SimBackend()
    left = plant.arm_actuators[:7]
    if not retained_position_servo:
        plant.model.actuator_gainprm[left] = 0
        plant.model.actuator_biasprm[left] = 0
    for body in range(plant.model.nbody):
        name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if name.startswith("left_"):
            plant.model.body_gravcomp[body] = factor
    mujoco.mj_forward(plant.model, plant.data)
    start = np.asarray(plant.read().positions)
    start_tcp = plant.data.site("left_tcp").xpos.copy()
    target = start.copy()
    if retained_position_servo:
        # A deliberately injected old target, not an incident measurement.
        target[0] += np.radians(20)
        plant.data.ctrl[left] = target[:7]
    rows = []
    while plant.data.time < duration_s - 1e-9:
        plant.advance()
        rows.append({
            "simulation_time_s": float(plant.data.time),
            "q_rad": np.asarray(plant.read().positions).tolist(),
            "left_tcp_m": plant.data.site("left_tcp").xpos.tolist(),
        })
    delta = np.asarray([row["q_rad"] for row in rows]) - start
    result = {
        "artificial_left_gravity_compensation_factor": factor,
        "duration_s": duration_s,
        "left_max_joint_displacement_deg": float(np.degrees(abs(delta[:, :7]).max())),
        "right_max_joint_displacement_deg": float(np.degrees(abs(delta[:, 7:]).max())),
        "left_final_tcp_displacement_m": (plant.data.site("left_tcp").xpos - start_tcp).tolist(),
        "initial_q_rad": start.tolist(),
        "initial_left_tcp_m": start_tcp.tolist(),
        "retained_position_servo": retained_position_servo,
        "hypothetical_target_q_rad": target.tolist() if retained_position_servo else None,
    }
    if retained_position_servo:
        before_push = np.asarray(plant.read().positions)
        plant.data.qfrc_applied[plant.dids[0]] = 1.0
        for _ in range(20):
            plant.advance()
        pushed = np.asarray(plant.read().positions)
        plant.data.qfrc_applied[:] = 0
        for _ in range(80):
            plant.advance()
        released = np.asarray(plant.read().positions)
        result["synthetic_push"] = {
            "joint": "left_joint1",
            "torque_nm": 1.0,
            "duration_s": .2,
            "displacement_during_push_deg": float(np.degrees(pushed[0]-before_push[0])),
            "error_from_old_target_after_release_deg": float(np.degrees(released[0]-target[0])),
            "release_wait_s": .8,
        }
    plant.close()
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("validation/leader_swing_investigation_20261020"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    traces = {}
    # Prevent accidental hardware/network access even if an import changes.
    with patch.object(socket, "socket", side_effect=RuntimeError("Offline investigation: sockets forbidden")):
        frames = [{"can_id": hex(cid), "payload": payload.hex()}
                  for cid, payload in official_mode_frames("leader")]
        for name, factor in (("ideal", 1.0), ("20_percent_under", .8),
                             ("absent", 0.0), ("opposite_torque", -1.0)):
            result, trace = experiment(factor)
            result["case"] = name
            results.append(result)
            traces[name] = trace
        result, trace = experiment(1.0, retained_position_servo=True)
        result["case"] = "hypothetical_old_target_and_position_servo_retained"
        results.append(result)
        traces[result["case"]] = trace
    assert results[0]["left_max_joint_displacement_deg"] < .01
    assert results[2]["left_max_joint_displacement_deg"] > 1.0
    assert all(row["right_max_joint_displacement_deg"] < .01 for row in results)
    assert results[-1]["left_max_joint_displacement_deg"] > 15
    assert abs(results[-1]["synthetic_push"]["error_from_old_target_after_release_deg"]) < .1
    report = {
        "physical_commands_sent": 0,
        "physical_incident_reproduced": False,
        "physical_root_cause_established": False,
        "real_deployment_qualified": False,
        "purpose": "Demonstrate sensitivity hidden by ideal gravity compensation, not fit or reconstruct the observed swing.",
        "limitations": [
            "Initial posture is the preview posture, not a recorded incident posture.",
            "No incident CAN TX/RX or continuous joint trajectory was saved in the session directory.",
            "Gravity factors are hypothetical faults, not measured firmware settings.",
            "The simulation does not implement the firmware's internal controller, torque limits, friction or mounting logic.",
            "Opposite torque is not a calibrated model of either official side-mount setting.",
            "The retained-servo case injects a 20-degree target error; neither that target nor that error is known to have occurred on the real robot.",
        ],
        "official_leader_frames_encoded_offline": frames,
        "experiments": results,
    }
    (args.output / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "sensitivity_traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
