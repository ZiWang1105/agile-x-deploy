"""Receive-only NERO joint feedback for commissioning, not a motion backend.

No SDK instance is created and no CAN frames are transmitted. SocketCAN remains
in its existing bus mode; normal hardware ACKs are not application transmissions.
The partial snapshot intentionally does not implement EncoderSample: velocity
and calibrated gripper travel are unavailable and must not be fabricated.
"""
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import select
import socket
import struct
import time

import numpy as np

from .contract import JOINT_NAMES, SIDES
from .model import ROOT

ANGLE_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A9)
LEADER_ANGLE_IDS = (0x155, 0x156, 0x157, 0x170)  # Official NERO >=1.12.
DRIVER_IDS = tuple(range(0x261, 0x268))
MOTOR_IDS = tuple(range(0x251, 0x258))
MAX_AGE_S = .2
MAX_SKEW_S = .03
SO_TIMESTAMPNS = 35  # Linux SO_TIMESTAMPNS_OLD, native timespec on this host.
SO_RXQ_OVFL = 40


@dataclass(frozen=True)
class JointSnapshot:
    sequence: int
    timestamp_s: float  # Oldest kernel receive time, converted to monotonic time.
    skew_s: float
    joint_names: tuple[str, ...]
    positions: tuple[float, ...]
    driver_status: tuple[int | None, ...]
    driver_timestamp_s: float | None
    velocities: tuple[float, ...] | None = None
    grippers: None = None
    arm_status: tuple[bytes | None, bytes | None] | None = None
    status_timestamp_s: float | None = None
    gripper_feedback: tuple | None = None
    angle_sources: tuple[str, str] = ("follower", "follower")
    leader_timestamps_s: tuple[float | None, float | None] = (None, None)
    velocity_sources: tuple[str, str] = ("motor", "motor")

    def validate(self, now):
        if not -.01 <= now - self.timestamp_s <= MAX_AGE_S:
            raise RuntimeError("Hardware joint feedback stale or clock invalid")
        if self.driver_timestamp_s is not None and not -.01 <= now - self.driver_timestamp_s <= MAX_AGE_S:
            raise RuntimeError("Hardware driver feedback stale or clock invalid")
        if self.skew_s > MAX_SKEW_S:
            raise RuntimeError("Paired hardware joint feedback exceeds 30 ms skew")
        if self.status_timestamp_s is not None and not -.01 <= now - self.status_timestamp_s <= MAX_AGE_S:
            raise RuntimeError("Hardware arm status stale or clock invalid")
        for feedback in self.gripper_feedback or ():
            if feedback is not None and not -.01 <= now - feedback.timestamp_s <= MAX_AGE_S:
                raise RuntimeError("Hardware gripper feedback stale or clock invalid")

    def report(self, limits=None):
        result = asdict(self)
        if self.arm_status is not None:
            result["arm_status"] = [s.hex() if s is not None else None for s in self.arm_status]
        result["positions_deg"] = np.degrees(self.positions).tolist()
        result["motor_enabled"] = [bool(v & 0x40) if v is not None else None for v in self.driver_status]
        result["source"] = "physical_CAN_receive_only"
        if limits is not None:
            q = np.asarray(self.positions)
            result["limit_margin_deg"] = np.degrees(np.minimum(q - limits[:, 0], limits[:, 1] - q)).tolist()
        return result


class FeedbackAssembler:
    """Publish only when all eight angle frames have advanced across both arms."""
    def __init__(self, motor_feedback=False, gripper_feedback=(), discover_leaders=False):
        self.angles = {}
        self.drivers = {}
        self.dirty = set()
        self.sequence = 0
        self.latest = None
        self.motor_feedback = motor_feedback
        self.motors = {}
        self.status = {}
        self.motor_dirty = set()
        self.gripper_sides = tuple(gripper_feedback)
        if any(s not in SIDES for s in self.gripper_sides) or len(set(self.gripper_sides)) != len(self.gripper_sides):
            raise ValueError("Invalid gripper feedback sides")
        self.grippers = {}
        self.gripper_dirty = set()
        self.angle_sources = dict.fromkeys(SIDES, "follower")
        self.discover_leaders = set(SIDES) if discover_leaders else set()
        self.source_since = dict.fromkeys(SIDES, float("-inf"))
        self.leader_timestamps = dict.fromkeys(SIDES, None)
        self.previous_leader = {}

    def set_angle_source(self, side, source, since):
        if side not in SIDES or source not in ("leader", "follower"):
            raise ValueError("Invalid feedback source")
        self.angle_sources[side] = source
        self.source_since[side] = since
        self.discover_leaders.discard(side)
        self.angles = {k: v for k, v in self.angles.items() if k[0] != side}
        self.dirty = {k for k in self.dirty if k[0] != side}
        self.motor_dirty = {k for k in self.motor_dirty if k[0] != side}
        self.gripper_dirty.discard(side)
        self.drivers = {k: v for k, v in self.drivers.items() if k[0] != side}
        self.status.pop(side, None)
        self.grippers.pop(side, None)
        self.previous_leader.pop(side, None)
        self.latest = None  # Never acknowledge a transition with a cached sample.

    def feed(self, side, can_id, data, acquired):
        if side not in SIDES or len(data) != 8:
            return
        if acquired < self.source_since[side]:
            return
        if can_id in LEADER_ANGLE_IDS:
            self.leader_timestamps[side] = acquired
            if side in self.discover_leaders:
                self.set_angle_source(side, "leader", acquired)
            if self.angle_sources[side] != "leader":
                return
            can_id = ANGLE_IDS[LEADER_ANGLE_IDS.index(can_id)]
        elif can_id in ANGLE_IDS and self.angle_sources[side] == "leader":
            return
        if can_id in ANGLE_IDS:
            key = (side, can_id)
            if key in self.angles and acquired < self.angles[key][1]:
                raise RuntimeError("CAN acquisition time moved backwards")
            values = struct.unpack(">ii", data) if can_id != 0x2A9 else (struct.unpack(">i", data[:4])[0],)
            self.angles[key] = (tuple(np.radians(np.array(values) * .001)), acquired)
            self.dirty.add(key)
        elif can_id in DRIVER_IDS:
            self.drivers[side, can_id] = (data[5], acquired)
        elif self.motor_feedback and can_id in MOTOR_IDS:
            key = (side, can_id)
            self.motors[key] = (struct.unpack(">h", data[:2])[0] * .001, acquired)
            self.motor_dirty.add(key)
        elif self.motor_feedback and can_id == 0x2A1:
            self.status[side] = (bytes(data), acquired)
        elif can_id == 0x2A8 and side in self.gripper_sides:
            from .gripper import decode_feedback
            self.grippers[side] = decode_feedback(data, acquired)
            self.gripper_dirty.add(side)
        else:
            return
        followers = [s for s in SIDES if self.angle_sources[s] == "follower"]
        if self.motor_feedback and (any((s, i) not in self.motor_dirty for s in followers for i in MOTOR_IDS)
                                    or any(s not in self.status for s in followers)):
            return
        if any(s not in self.gripper_dirty for s in self.gripper_sides if s in followers):
            return
        if len(self.dirty) == 8 and all((s, i) in self.drivers for s in followers for i in DRIVER_IDS):
            rows = [self.angles[s, i] for s in SIDES for i in ANGLE_IDS]
            states = [self.drivers[s, i] for s in followers for i in DRIVER_IDS]
            stamps = [r[1] for r in rows]
            motors = [self.motors[s, i] for s in followers for i in MOTOR_IDS] if self.motor_feedback else []
            stamps += [t for _, t in motors]
            stamps += [self.grippers[s].timestamp_s for s in self.gripper_sides if s in followers]
            velocities = []
            waiting = False
            if self.motor_feedback:
                for s in SIDES:
                    if s in followers:
                        velocities.extend(self.motors[s, i][0] for i in MOTOR_IDS)
                        continue
                    # Leader packets contain angles, not motor velocity/status.
                    # Differentiate two fresh complete angle sets and label the
                    # estimate; never synthesize enable, fault or jaw feedback.
                    q = np.array([v for cid in ANGLE_IDS for v in self.angles[s, cid][0]])
                    stamp = min(self.angles[s, cid][1] for cid in ANGLE_IDS)
                    previous = self.previous_leader.get(s)
                    if previous is None or stamp <= previous[1]:
                        waiting = True
                    else:
                        velocities.extend((q - previous[0]) / (stamp - previous[1]))
                    self.previous_leader[s] = (q, stamp)
            if waiting:
                self.dirty.clear()
                self.motor_dirty.clear()
                self.gripper_dirty.clear()
                return
            self.sequence += 1
            self.latest = JointSnapshot(self.sequence, min(stamps), max(stamps) - min(stamps),
                JOINT_NAMES, tuple(v for row, _ in rows for v in row),
                tuple(self.drivers[s, i][0] if s in followers else None for s in SIDES for i in DRIVER_IDS),
                min((t for _, t in states), default=None),
                velocities=tuple(velocities) if self.motor_feedback else None,
                arm_status=tuple(self.status[s][0] if s in followers else None for s in SIDES) if self.motor_feedback else None,
                status_timestamp_s=min((self.status[s][1] for s in followers), default=None) if self.motor_feedback else None,
                gripper_feedback=tuple(self.grippers.get(s) if s in followers else None for s in SIDES) if self.gripper_sides else None,
                angle_sources=tuple(self.angle_sources[s] for s in SIDES),
                leader_timestamps_s=tuple(self.leader_timestamps[s] for s in SIDES),
                velocity_sources=tuple(("motor" if s in followers else "angle_difference")
                                       if self.motor_feedback else "unavailable" for s in SIDES))
            self.dirty.clear()
            self.motor_dirty.clear()
            self.gripper_dirty.clear()

    def snapshot(self, now):
        if self.latest is None:
            raise RuntimeError("Waiting for complete paired joint and driver feedback")
        self.latest.validate(now)
        return self.latest


def mapped_channels(path, sys_class_net=Path("/sys/class/net")):
    """Resolve current interface names from the physically confirmed USB serials."""
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("automatic_hardware_enable") is not False:
        raise ValueError("Expected a version 1 mapping with automatic_hardware_enable=false")
    channels = {}
    for side in SIDES:
        identity = config["arms"][side]["usb_identity"]
        matches = []
        for interface in sys_class_net.iterdir():
            if not (interface / "device").exists():
                continue
            device = (interface / "device").resolve()
            for parent in (device, *device.parents):
                if (parent / "serial").exists():
                    if all((parent / key).exists() and (parent / key).read_text().strip() == identity[key]
                           for key in ("serial", "idVendor", "idProduct")):
                        matches.append(interface.name)
                    break
        if len(matches) != 1:
            raise RuntimeError(f"{side}: expected one CAN adapter matching confirmed USB identity, found {matches}")
        channels[side] = matches[0]
    if len(set(channels.values())) != 2:
        raise RuntimeError("Both arms resolve to the same adapter")
    return channels


class PassiveCanFeedback:
    def __init__(self, mapping=ROOT / "configs/hardware_mapping.json", motor_feedback=False, gripper_feedback=()):
        if gripper_feedback:
            from .gripper import official_codec
            official_codec()  # Import/warm up before sockets can accumulate feedback.
        self.channels = mapped_channels(mapping)
        self.assembler = FeedbackAssembler(motor_feedback=motor_feedback, gripper_feedback=gripper_feedback,
                                           discover_leaders=True)
        self.sockets = {}
        self.clock_offset = time.monotonic() - time.time()
        try:
            for side, channel in self.channels.items():
                sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sockets[sock] = side
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
                sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
                sock.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
                # Match standard data frames only; reject EFF/RTR flags.
                # Error reception is disabled by SocketCAN's default ERR_FILTER.
                ids = (*ANGLE_IDS, *LEADER_ANGLE_IDS, *DRIVER_IDS, *MOTOR_IDS, 0x2A1) if motor_feedback else (*ANGLE_IDS, *LEADER_ANGLE_IDS, *DRIVER_IDS)
                if gripper_feedback:
                    ids = (*ids, 0x2A8)
                filters = b"".join(struct.pack("=II", cid, 0xC00007FF) for cid in ids)
                sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
                sock.bind((channel,))
                sock.setblocking(False)
        except BaseException:
            self.close()
            raise

    def read(self, timeout=0.):
        deadline = time.monotonic() + timeout
        if abs(time.monotonic() - time.time() - self.clock_offset) > .01:
            raise RuntimeError("Host wall clock changed; restart CAN timestamp conversion")
        while True:
            # Bound draining so a busy bus cannot monopolize the rendering thread.
            drained = False
            for _ in range(4096):
                ready, _, _ = select.select(list(self.sockets), [], [], 0)
                if not ready:
                    drained = True
                    break
                for sock in ready:
                    frame, ancillary, flags, _ = sock.recvmsg(16, 128)
                    # Leader feedback uses the same IDs as host position commands.
                    # SocketCAN marks locally generated frames with MSG_DONTROUTE;
                    # never mistake our writer's loopback for encoder feedback.
                    if flags & socket.MSG_DONTROUTE:
                        continue
                    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(frame) != 16:
                        raise RuntimeError("Truncated CAN frame or acquisition metadata")
                    acquired = None
                    for level, kind, data in ancillary:
                        if level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS:
                            sec, nsec = struct.unpack("@ll", data)
                            acquired = sec + nsec / 1e9 + self.clock_offset
                        if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL and struct.unpack("=I", data)[0]:
                            raise RuntimeError("CAN receive queue overflow; restart after inspection")
                    if acquired is None:
                        raise RuntimeError("Missing kernel CAN acquisition timestamp")
                    cid, dlc, data = struct.unpack("=IB3x8s", frame)
                    if dlc == 8:
                        self.assembler.feed(self.sockets[sock], cid, data, acquired)
            if not drained:
                raise RuntimeError("CAN receive backlog exceeds bounded drain")
            try:
                return self.assembler.snapshot(time.monotonic())
            except RuntimeError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                select.select(list(self.sockets), [], [], min(.01, remaining))

    def set_angle_source(self, side, source, since):
        self.assembler.set_angle_source(side, source, since)

    def close(self):
        for sock in self.sockets:
            sock.close()
        self.sockets.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from .model import Kinematics
    kin = Kinematics()
    with PassiveCanFeedback(args.mapping) as feedback:
        sample = feedback.read(timeout=1.)
        report = sample.report(kin.limits)
        report["channels"] = feedback.channels
        report["age_s"] = time.monotonic() - sample.timestamp_s
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
