"""Receive-only raw CAN recording for supported controller diagnosis.

Never enables, disables, changes modes, sends queries, or configures interfaces.
Local command echoes and bus receive frames are recorded separately. A local
echo is not a controller acknowledgement or proof that a target was accepted.
The recorder cannot reconstruct an incident that happened before it started.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import select
import socket
import struct
import time

from .can_feedback import mapped_channels, SO_TIMESTAMPNS, SO_RXQ_OVFL
from .model import ROOT


def decode_record(side, packet, ancillary, flags):
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(packet) != 16:
        raise RuntimeError("Truncated CAN packet or timestamp metadata")
    can_id, length, data = struct.unpack("=IB3x8s", packet)
    if length > 8:
        raise RuntimeError("Invalid classic CAN data length")
    acquired_ns = None
    dropped = 0
    for level, kind, value in ancillary:
        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
            sec, nsec = struct.unpack("@ll", value)
            acquired_ns = sec * 1_000_000_000 + nsec
        elif level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
            dropped = struct.unpack("=I", value)[0]
    if acquired_ns is None:
        raise RuntimeError("Missing kernel CAN acquisition timestamp")
    return dict(event="frame", side=side, kernel_realtime_ns=acquired_ns,
                observed_monotonic_ns=time.monotonic_ns(), can_id=hex(can_id),
                payload=data[:length].hex(), flags=flags, dropped_frames=dropped,
                origin="host_echo" if flags & socket.MSG_DONTROUTE else "bus_rx")


def capture(channels, output, duration_s):
    if not 0 < duration_s <= 3600:
        raise ValueError("Duration must be greater than 0 and at most 3600 seconds")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sockets = {}
    count = 0
    try:
        # Exclusive creation avoids overwriting earlier evidence.
        with output.open("x") as stream:
            def record(value):
                stream.write(json.dumps(value, allow_nan=False) + "\n")

            record(dict(event="start", receive_only=True, channels=channels,
                        monotonic_ns=time.monotonic_ns(), realtime_ns=time.time_ns()))
            stream.flush()
            try:
                for side, channel in channels.items():
                    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                    sockets[sock] = side
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
                    sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
                    sock.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
                    sock.bind((channel,))
                    sock.setblocking(False)
                deadline = time.monotonic() + duration_s
                last_flush = time.monotonic()
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(list(sockets), [], [],
                                                max(0, min(.1, deadline-time.monotonic())))
                    for sock in ready:
                        packet, ancillary, flags, _ = sock.recvmsg(16, 128)
                        row = decode_record(sockets[sock], packet, ancillary, flags)
                        record(row)
                        count += 1
                        if row["dropped_frames"]:
                            raise RuntimeError("CAN capture lost frames; trace is incomplete")
                    if time.monotonic() - last_flush >= .25:
                        stream.flush()
                        last_flush = time.monotonic()
            except BaseException as error:
                record(dict(event="incomplete", error=type(error).__name__ + ": " + str(error)))
                raise
            finally:
                record(dict(event="end", frames=count, monotonic_ns=time.monotonic_ns()))
                stream.flush()
    finally:
        for sock in sockets:
            sock.close()
    return dict(output=str(output), frames=count, receive_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration <= 3600:
        parser.error("--duration must be greater than 0 and at most 3600 seconds")
    output = args.output or ROOT / "logs/diagnostics" / ("can_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
    print("Receive-only capture; no robot commands will be sent. Output: " + str(output), flush=True)
    print(json.dumps(capture(mapped_channels(args.mapping), output, args.duration)))


if __name__ == "__main__":
    main()
