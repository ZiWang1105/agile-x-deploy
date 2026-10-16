"""Disable supported arms, optionally opening both grippers with temporary arm holding."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import socket
import struct
import time

from .can_feedback import PassiveCanFeedback
from .contract import SIDES
from .model import ROOT
from .gripper_open import open_with_arm_hold


def disable_supported_arms(mapping=ROOT / "configs/hardware_mapping.json", *, arms_supported=False):
    if not arms_supported:
        raise ValueError("Mechanically support BOTH arms before disabling; motor release can let them fall")
    arms = json.loads(Path(mapping).read_text())["arms"]
    serials = sorted(arms[s]["usb_identity"]["serial"] for s in SIDES)
    if not all(s.isalnum() for s in serials) or len(set(serials)) != 2:
        raise ValueError("Invalid adapter serials")
    # Same pair-wide ownership lock as NeroTransport. Never disable underneath
    # one of our running controllers. External applications must also be closed.
    with open("/tmp/agilex-deploy-" + "-".join(serials) + ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Close the robot control window before disabling motors") from error
        with PassiveCanFeedback(mapping) as feedback:
            return _disable_and_verify(feedback)


def _disable_and_verify(feedback):
    snapshot = feedback.read(timeout=1.)
    if not any(v & 0x40 for v in snapshot.driver_status):
        return {"left_enabled": 0, "right_enabled": 0, "commands_sent": 0}
    sent = 0
    with ExitStack() as stack:
        writers = {}
        for side in SIDES:
            writer = stack.enter_context(socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW))
            writer.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b"")
            writer.settimeout(.02)
            writer.bind((feedback.channels[side],))
            writers[side] = writer
        deadline = time.monotonic() + 2.
        last_send = 0.
        disabled_since = None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now - last_send >= .1:
                for i, side in enumerate(SIDES):
                    for j, status in enumerate(snapshot.driver_status[i*7:(i+1)*7], 1):
                        if status & 0x40:
                            # Official NERO disable(j): 0x471, j=1..7, flag=1.
                            # No broadcast/gripper index, mode switch or setpoint.
                            payload = bytes([j, 1]) + bytes(6)
                            if writers[side].send(struct.pack("=IB3x8s", 0x471, 8, payload)) != 16:
                                raise RuntimeError("Short disable write; some joints may remain enabled")
                            sent += 1
                last_send = now
            snapshot = feedback.read(timeout=.1)
            if any(v & 0x40 for v in snapshot.driver_status):
                disabled_since = None
            elif disabled_since is None:
                disabled_since = time.monotonic()
            elif time.monotonic() - disabled_since >= .2:
                return {"left_enabled": 0, "right_enabled": 0, "commands_sent": sent}
            time.sleep(.01)
    raise RuntimeError("Disable acknowledgement timed out; keep support in place and inspect the robot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--arms-supported", action="store_true",
                        help="Confirm BOTH arms are mechanically supported against falling or swinging")
    parser.add_argument("--open-grippers", action="store_true",
                        help="Open both jaws with temporary measured-joint holding, then disable both arms")
    args = parser.parse_args()
    if not args.arms_supported:
        parser.error("Support BOTH arms first, then explicitly pass --arms-supported. Disabling releases motor holding torque.")
    try:
        result = disable_supported_arms(args.mapping, arms_supported=True)
        if args.open_grippers:
            print("Opening both jaws with temporary arm holding; keep both arms supported.", flush=True)
            result["grippers"] = open_with_arm_hold(args.mapping, SIDES, arms_supported=True)
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Requested operation not confirmed: {error}\nKeep both arms supported.\n")
    print("DISABLED confirmed: left 0/7 enabled; right 0/7 enabled.")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
