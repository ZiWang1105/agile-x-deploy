"""AgxGripper width control using the pinned official pyAgxArm codec.

No SDK arm driver is created. Importing this module never opens CAN. The status
CLI receives feedback and optionally issues only a parameter READ request.
"""

import argparse
import json
import math
import select
import socket
import struct
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

SDK_COMMIT = "d90e234ba8e408cfa964b38aa9120b3131fc9f85"
DEFAULT_FORCE_N = 1.0  # Official move_gripper_m default, not a measured jaw force.


@lru_cache(maxsize=1)
def official_codec():
    try:
        from pyAgxArm.protocols.can_protocol.drivers.effector.agx_gripper.default.parser import (
            Codec,
        )
        from pyAgxArm.protocols.can_protocol.msgs.effector.agx_gripper.default import (
            ArmMsgFeedbackGripper,
            ArmMsgFeedbackGripperTeachingPendantParam,
            ArmMsgGripperCtrl,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-gripper.txt in .venv-official first"
        ) from error
    return (
        Codec(),
        ArmMsgFeedbackGripper,
        ArmMsgGripperCtrl,
        ArmMsgFeedbackGripperTeachingPendantParam,
    )


@dataclass(frozen=True)
class GripperFeedback:
    width_m: float
    force_n: float
    status: int
    timestamp_s: float

    @property
    def enabled(self):
        return bool(self.status & 0x40)

    @property
    def zeroed(self):
        return bool(self.status & 0x80)

    def validate(self, now):
        if not math.isfinite(self.width_m) or not math.isfinite(self.force_n):
            raise RuntimeError("Nonfinite gripper width or force feedback")
        if (
            not math.isfinite(self.timestamp_s)
            or not -0.01 <= now - self.timestamp_s <= 0.2
        ):
            raise RuntimeError("Gripper feedback stale or clock invalid")
        if self.status & 0x3F:
            raise RuntimeError(f"Gripper driver fault: status=0x{self.status:02x}")

    def normalized(self, max_width_m):
        validate_range(max_width_m)
        fraction = self.width_m / max_width_m
        # Match the shared encoder contract tolerance, without inventing travel.
        if not math.isfinite(fraction) or not -0.01 <= fraction <= 1.01:
            raise RuntimeError(
                "Gripper width outside configured travel; inspect zero/range"
            )
        return fraction


def decode_feedback(data, acquired):
    if len(data) != 8 or data[7] != 0:
        raise ValueError(
            "Require an eight-byte gripper WIDTH feedback frame; angle mode is unsupported"
        )
    codec, feedback_type, _, _ = official_codec()
    message = feedback_type()
    codec.decode_2A8_gripper(message, bytearray(data))
    return GripperFeedback(message.value, message.force, message.status_code, acquired)


def validate_range(width):
    if not math.isfinite(width) or width not in (0.07, 0.1):
        raise ValueError(
            "Require controller-reported gripper travel of 0.07 or 0.1 m; no guessed fallback"
        )


def validate_force(force):
    if not math.isfinite(force) or not 0 <= force <= 3:
        raise ValueError(
            "Gripper force must be within the official documented 0–3 N range"
        )


def width_frame(fraction, max_width_m, force_n=DEFAULT_FORCE_N):
    validate_range(max_width_m)
    validate_force(force_n)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Gripper command must be normalized travel in [0, 1]")
    codec, _, command_type, _ = official_codec()
    # Same scaling/status as official Driver.move_gripper_m. Never clear errors
    # or set_zero=0xAE, and never address arm enable/broadcast motor indices.
    message = command_type(
        value=round(fraction * max_width_m * 1e6),
        force=round(force_n * 1e3),
        status_code=1,
        set_zero=0,
    )
    return 0x159, bytes(codec.encode_159_gripper_ctrl(message))


def query_range(channel, timeout=0.5):
    codec, _, _, feedback_type = official_codec()
    with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as sock:
        sock.setsockopt(
            socket.SOL_CAN_RAW,
            socket.CAN_RAW_FILTER,
            struct.pack("=II", 0x47E, 0xC00007FF),
        )
        sock.bind((channel,))
        sock.settimeout(timeout)
        # Official get_gripper_teaching_pendant_param: param_enquiry=4;
        # all configuration and enable fields remain zero.
        sock.send(struct.pack("=IB3x8s", 0x477, 8, bytes([4]) + bytes(7)))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [sock], [], [], max(0.0, deadline - time.monotonic())
            )
            if not ready:
                break
            cid, dlc, data = struct.unpack("=IB3x8s", sock.recv(16))
            if cid != 0x47E or dlc != 8:
                continue
            message = feedback_type()
            codec.decode_47E_gripper_teaching_pendant_param(message, bytearray(data))
            validate_range(message.max_range_config)
            return message.max_range_config
    raise RuntimeError(
        "No gripper range readback; no configuration or mode changes attempted"
    )


def main():
    from .can_feedback import PassiveCanFeedback
    from .contract import SIDES
    from .model import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json"
    )
    parser.add_argument(
        "--query-parameters",
        action="store_true",
        help="Send only the official range read request",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with PassiveCanFeedback(args.mapping, gripper_feedback=SIDES) as reader:
        sample = reader.read(timeout=1.0)
        result = {
            "channels": reader.channels,
            "motion_commands_sent": 0,
            "grippers": {},
        }
        for side, feedback in zip(SIDES, sample.gripper_feedback):
            row = asdict(feedback)
            row.update(
                enabled=feedback.enabled,
                zeroed=feedback.zeroed,
                driver_fault_bits=feedback.status & 0x3F,
            )
            if args.query_parameters:
                row["configured_max_width_m"] = query_range(reader.channels[side])
                row["normalized_travel"] = feedback.normalized(
                    row["configured_max_width_m"]
                )
            result["grippers"][side] = row
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
