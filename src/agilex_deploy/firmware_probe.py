"""Query NERO firmware with one fixed read request per arm; never enable motors.

This utility is separate from the receive-only viewer. Protocol reference:
pyAgxArm's Piper Codec.encode_4AF_req_firmware returns [0x01] (DLC 1), and
Nero Driver.get_firmware reads major/minor from response bytes 6 and 7.
https://github.com/agilexrobotics/pyAgxArm/tree/master/pyAgxArm/protocols/can_protocol
"""
import argparse
import json
from pathlib import Path
import select
import socket
import struct
import time

from .can_feedback import mapped_channels
from .model import ROOT


def profile_for(major, minor):
    if major != 1:
        return None  # Do not guess compatibility for a new major version.
    if minor >= 21:
        return "v121"
    return {20: "v120", 12: "v112", 11: "v111"}.get(minor, "default" if minor <= 10 else None)


def query_firmware(mapping, timeout=2.):
    channels = mapped_channels(mapping)
    sockets, results = {}, {}
    try:
        for side, channel in channels.items():
            sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            sockets[sock] = side
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", 0x4AF, 0xC00007FF))
            sock.settimeout(.2)
            sock.bind((channel,))
        for sock in sockets:
            # Fixed, reviewed read request. No general-purpose transmit API.
            sock.send(struct.pack("=IB3x8s", 0x4AF, 1, b"\x01"))
        deadline = time.monotonic() + timeout
        while len(results) < 2 and time.monotonic() < deadline:
            ready, _, _ = select.select(list(sockets), [], [], max(0., deadline - time.monotonic()))
            for sock in ready:
                cid, dlc, data = struct.unpack("=IB3x8s", sock.recv(16))
                if cid != 0x4AF or dlc != 8 or data == b"\x01" + bytes(7) or not any(data):
                    continue
                results[sockets[sock]] = {
                    "software_version": f"{data[6]}.{data[7]:02d}",
                    "sdk_profile": profile_for(data[6], data[7]),
                    "raw_response_hex": data.hex(),
                }
        return {"channels": channels, "firmware": results,
                "missing": [s for s in channels if s not in results],
                "requests": {"can_id": "0x4AF", "dlc": 1, "payload_hex": "01", "count": 2},
                "motion_commands_sent": 0}
    finally:
        for sock in sockets:
            sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = query_firmware(args.mapping)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if report["missing"]:
        raise SystemExit("Firmware query incomplete; no profile should be assumed")


if __name__ == "__main__":
    main()
