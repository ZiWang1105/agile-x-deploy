"""Offline cross-engine gravity validation; does not qualify physical control."""

import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from agilex_deploy.official_gravity import OfficialGravityPreview, ROOT
from agilex_deploy.model import initial_positions


def fixed_jaw_model(preview, side, opening):
    root = ET.fromstring(preview.descriptions[side])
    # Preserve fixed-link inertias explicitly. With this model, the installed
    # MuJoCo URDF static fusion changes the resulting gravity torques; retaining
    # the fixed bodies agrees with independent Pinocchio RNEA to roundoff.
    compiler = ET.SubElement(root, "mujoco")
    ET.SubElement(compiler, "compiler", fusestatic="false")
    for finger, sign in ((1, 1), (2, -1)):
        joint = root.find(f"joint[@name='{side}_gripper_joint{finger}']")
        origin = joint.find("origin")
        rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        xyz = np.fromstring(origin.get("xyz"), sep=" ")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        origin.set("xyz", " ".join(map(str, xyz + rotation @ axis * sign * opening / 2)))
        joint.set("type", "fixed")
        joint.remove(joint.find("axis"))
        joint.remove(joint.find("limit"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.gravity[:] = preview.rotations[side].T @ np.array([0., 0., -9.81])
    model.opt.timestep = .001
    model.body_gravcomp[:] = 0
    assert model.nu == 0  # No position servos or motor gains conceal a bad torque.
    return model


def simulate(preview, side, q, case, duration_s=.5):
    model = fixed_jaw_model(preview, side, .04)
    data = mujoco.MjData(model)
    qids = [model.joint(f"{side}_joint{i}").qposadr[0] for i in range(1, 8)]
    dids = [model.joint(f"{side}_joint{i}").dofadr[0] for i in range(1, 8)]
    data.qpos[qids] = q
    mujoco.mj_forward(model, data)
    predicted = preview.torques(side, q, np.zeros(7), gripper_opening_m=.04)
    mismatch = float(np.max(np.abs(predicted - data.qfrc_bias[dids])))
    trace = []
    target_rotation = preview.rotations[side].copy()
    if case == "wrong_upright_assumption":
        preview.rotations[side] = np.eye(3)
    termination = "duration_completed"
    max_drift_rad = 0.
    try:
        for tick in range(round(duration_s / model.opt.timestep)):
            if tick % 5 == 0:  # Official example's nominal 200 Hz control rate.
                data.qfrc_applied[:] = 0
                if case != "no_compensation":
                    try:
                        data.qfrc_applied[dids] = preview.torques(
                            side, data.qpos[qids], data.qvel[dids], gripper_opening_m=.04)
                    except ValueError:
                        if case == "mounted_official_compensation":
                            raise
                        termination = "fault_injection_exceeded_model_limits"
                        break
            mujoco.mj_step(model, data)
            max_drift_rad = max(max_drift_rad, float(np.max(np.abs(data.qpos[qids] - q))))
            if tick % 10 == 0:
                trace.append({"time_s": float(data.time), "q_rad": data.qpos[qids].tolist()})
    finally:
        preview.rotations[side] = target_rotation
    return {
        "side": side, "case": case, "duration_s": duration_s,
        "elapsed_simulation_s": float(data.time), "termination": termination,
        "initial_torque_nm": predicted.tolist(),
        "pinocchio_mujoco_static_torque_max_error_nm": mismatch,
        "max_joint_drift_deg": float(np.rad2deg(max_drift_rad)),
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/official_gravity_20261020")
    args = parser.parse_args()
    q = initial_positions()
    if args.incident:
        rows = [json.loads(line) for line in args.incident.read_text().splitlines()]
        q = np.asarray(next(row["positions"] for row in rows if row.get("event") == "feedback"))
    with patch.object(socket, "socket", side_effect=RuntimeError("Offline validation forbids sockets")):
        preview = OfficialGravityPreview()
        results = [simulate(preview, side, q[i*7:i*7+7], case)
                   for i, side in enumerate(("left", "right"))
                   for case in ("mounted_official_compensation", "no_compensation", "wrong_upright_assumption")]
    for result in results:
        assert result["pinocchio_mujoco_static_torque_max_error_nm"] < 1e-6, result
        if result["case"] == "mounted_official_compensation":
            assert result["max_joint_drift_deg"] < .01, result
        else:
            assert result["max_joint_drift_deg"] > .1, result
    report = {
        "physical_commands_sent": 0,
        "physical_gravity_compensation_verified": False,
        "physical_root_cause_established": False,
        "host_control_rate_hz": 200,
        "source_incident": str(args.incident) if args.incident else None,
        "initial_q_rad": q.tolist(),
        "limitations": [
            "Both engines use the same uncalibrated URDF inertias; numerical agreement is not hardware identification.",
            "No grasped payload; gripper opening fixed at 40 mm for this simulation, not inferred from the incident.",
            "No collisions, hardware torque scaling, friction identification, CAN latency, or firmware mode transitions modeled.",
            "No physical writer, enable path, stop recovery, or deployment qualification is provided by this preview.",
            "This does not establish whether the observed descent preceded or followed emergency stop.",
        ],
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output / "report.json"),
                      "results": [{k: v for k, v in r.items() if k != "trace"} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
