"""Read NERO joint limits with fixed 0x472 queries; never modify settings.

Protocol reviewed against pyAgxArm commit
1ae40ad4755ca25ad32bece3ab42325698185fc7, Nero get_joint_angle_vel_limits
and Piper encode_472/decode_473. Each request uses search_content=1.
"""
import argparse
import json
from pathlib import Path
import select
import socket
import struct
import time

import numpy as np

from .can_feedback import mapped_channels
from .model import Kinematics, ROOT


def decode_limit(data):
    if len(data) != 8 or not 1 <= data[0] <= 7:
        raise ValueError("Invalid joint limit response")
    upper, lower, speed = struct.unpack(">hhH", data[1:7])
    if lower >= upper or speed == 0:
        raise ValueError("Invalid or unavailable firmware joint limits")
    return {"joint": data[0], "lower_deg": lower * .1,
            "upper_deg": upper * .1, "max_velocity_rad_s": speed * .01}


def query_limits(mapping, timeout=.3):
    channels = mapped_channels(mapping)
    limits, sockets = {s: {} for s in channels}, {}
    try:
        for side, channel in channels.items():
            sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sockets[sock] = side
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                            struct.pack("=II", 0x473, 0xC00007FF))
            sock.settimeout(.2)
            sock.bind((channel,))
        for joint in range(1, 8):
            for sock in sockets:
                sock.send(struct.pack("=IB3x8s", 0x472, 8, bytes([joint, 1]) + bytes(6)))
            deadline = time.monotonic() + timeout
            while not all(joint in row for row in limits.values()) and time.monotonic() < deadline:
                ready, _, _ = select.select(list(sockets), [], [], max(0., deadline - time.monotonic()))
                for sock in ready:
                    cid, dlc, data = struct.unpack("=IB3x8s", sock.recv(16))
                    if cid == 0x473 and dlc == 8 and data[0] == joint:
                        limits[sockets[sock]][joint] = decode_limit(data)
        return {"channels": channels, "limits": limits,
                "complete": all(len(row) == 7 for row in limits.values()),
                "read_request_count": 14, "motion_or_configuration_commands_sent": 0}
    finally:
        for sock in sockets:
            sock.close()



def decode_acceleration(data):
    """Official NERO decode_47C_motor_max_acc_limit: uint16 * 1e-3 rad/s²."""
    if len(data) != 8 or not 1 <= data[0] <= 7:
        raise ValueError("Invalid acceleration limit response")
    value = int.from_bytes(data[1:3], "big") * .001
    if value <= 0:
        raise ValueError("Invalid or unavailable acceleration limit")
    return value


def query_motion_limits(channel, model_limits, timeout=.15):
    """Read selected controller settings. Only 0x472 requests are transmitted."""
    from .motion_limits import JointLimits
    angles, accelerations = {}, {}
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
        sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                        b"".join(struct.pack("=II", cid, 0xC00007FF) for cid in (0x473, 0x47C)))
        sock.settimeout(.2)
        sock.bind((channel,))
        for kind, response, rows, decode in ((1, 0x473, angles, decode_limit),
                                            (2, 0x47C, accelerations, decode_acceleration)):
            for joint in range(1, 8):
                sock.send(struct.pack("=IB3x8s", 0x472, 8, bytes([joint, kind]) + bytes(6)))
                deadline = time.monotonic() + timeout
                while joint not in rows and time.monotonic() < deadline:
                    ready, _, _ = select.select([sock], [], [], max(0., deadline - time.monotonic()))
                    if not ready:
                        break
                    cid, dlc, data = struct.unpack("=IB3x8s", sock.recv(16))
                    if cid == response and dlc == 8 and data[0] == joint:
                        rows[joint] = decode(data)
    return JointLimits.from_readback(angles, accelerations, np.asarray(model_limits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = Kinematics()
    report = query_limits(args.mapping)
    for side_index, side in enumerate(("left", "right")):
        for joint, row in report["limits"][side].items():
            row["model_limits_deg"] = np.degrees(model.limits[side_index * 7 + joint - 1]).tolist()
            row["intersection_deg"] = [max(row["lower_deg"], row["model_limits_deg"][0]),
                                       min(row["upper_deg"], row["model_limits_deg"][1])]
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not report["complete"]:
        raise SystemExit("Joint limit query incomplete; no limits were changed")


if __name__ == "__main__":
    main()
