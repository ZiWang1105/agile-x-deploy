"""Supported, single-arm commissioning of the official NERO leader mode.

This is a physical test, not a qualified policy backend. Supports must remain in
place throughout enable, dragging and shutdown. No joint, TCP, MIT, gripper,
mounting, reset or disable commands are sent. Enable can activate an unknown
firmware target before leader entry; a host-side monitor cannot prevent all
motion. A fault, Ctrl-C or the test deadline requests the official damped stop.
That stop can permit descent and is NOT a verified motor disable or power-off.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime
import fcntl
from functools import lru_cache
import json
from pathlib import Path
import signal
import socket
import struct
import time
from types import SimpleNamespace

import numpy as np

from .can_feedback import PassiveCanFeedback
from .contract import SIDES
from .firmware_probe import query_firmware
from .hand_guiding import official_mode_frames
from .model import Kinematics, ROOT


# These are stop thresholds for this supported diagnostic, not robot speed or
# acceleration settings. They do not constrain the production motion planner.
STARTUP_DRIFT_RAD = np.deg2rad(2)
DRAG_EXCURSION_RAD = np.deg2rad(15)
DRAG_SPEED_RAD_S = .35
TRANSITION_TIMEOUT_S = 1.5
SETTLE_S = 2.


@lru_cache(maxsize=1)
def official_test_frames():
    """Run pinned SDK enable/stop methods against an encoder, never live I/O."""
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v121.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v112.parser import Codec
    from pyAgxArm.protocols.can_protocol.drivers.nero.default.driver import ArmMsgMotionCtrl

    class Encoder(Driver):
        def __init__(self):
            self.frames = []
            self.codec = Codec()
            self._ctx = SimpleNamespace(_wait_for_response=lambda condition, timeout: condition())

        def get_arm_status(self):
            # The actual test requires fresh CAN-mode feedback before enabling.
            return SimpleNamespace(msg=SimpleNamespace(ctrl_mode=self.ARM_STATUS.CtrlMode.CAN_CTRL))

        def get_joints_enable_status_list(self):
            return [True] * 7  # Codec extraction only; not physical evidence.

        def _send_msg(self, msg):
            if isinstance(msg, self._MSG_MotorEnableDisableConfig):
                frame = (0x471, bytes(self.codec.encode_471_motor_enable_disable_config(msg)))
            elif isinstance(msg, ArmMsgMotionCtrl):
                frame = (0x150, bytes(self.codec.encode_150_motion_ctrl(msg)))
            else:
                raise TypeError("Unexpected official diagnostic message")
            self.frames.append(frame)

    encoder = Encoder()
    encoder.enable(255, timeout=0.)
    enable = tuple(encoder.frames)
    encoder.frames.clear()
    encoder.electronic_emergency_stop()
    stop = tuple(encoder.frames)
    if enable != ((0x471, bytes([8, 2]) + bytes(6)),) or stop != ((0x150, bytes([1]) + bytes(7)),):
        raise RuntimeError("Installed SDK diagnostic wire format differs from reviewed NERO 1.21")
    return enable, stop


class GravityCompTest:
    """Single command owner; injected reader/writer allow the exact path offline."""

    def __init__(self, reader, writer, side, limits, record, announce=print):
        if side not in SIDES:
            raise ValueError("Select left or right")
        self.reader, self.writer, self.side = reader, writer, side
        self.index = SIDES.index(side)
        self.selected = slice(self.index * 7, self.index * 7 + 7)
        self.other = slice((1-self.index) * 7, (1-self.index) * 7 + 7)
        self.limits = np.asarray(limits)
        self.record, self.announce = record, announce
        self.enable_frames, self.stop_frames = official_test_frames()
        self.leader_frames = official_mode_frames("leader")
        self.possibly_enabled = False
        self.last_sequence = None
        self.origin = None
        self.stop_requested = False
        self.stopping = False
        self.max_startup_drift_rad = 0.
        self.leader_confirmed = False
        self.phase = "preflight"

    def send(self, frames):
        for cid, payload in frames:
            # Closed allow-list prevents adding arbitrary targets to this tool.
            if (cid, payload) not in (*self.enable_frames, *self.stop_frames, *self.leader_frames):
                raise RuntimeError("Frame is outside the diagnostic allow-list")
            self.record({"event": "tx_attempt", "side": self.side,
                         "can_id": hex(cid), "payload": payload.hex()})
            if self.writer.send(struct.pack("=IB3x8s", cid, len(payload), payload)) != 16:
                raise RuntimeError("Short CAN write")
            self.record({"event": "tx_sent", "side": self.side, "can_id": hex(cid)})

    def sample(self):
        # PassiveCanFeedback.read returns the latest complete snapshot even if
        # no new CAN cycle has arrived. Immediately after a command it is normal
        # to receive the same still-fresh sample. Wait for acquisition progress,
        # never treat a duplicate as acknowledgement or fabricate its timestamp.
        deadline = time.monotonic() + .05
        while True:
            sample = self.reader.read(timeout=max(0., deadline-time.monotonic()))
            sample.validate(time.monotonic())
            if self.last_sequence is None or sample.sequence > self.last_sequence:
                break
            if sample.sequence < self.last_sequence:
                raise RuntimeError("Joint feedback sequence moved backwards")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Joint feedback sequence did not advance within 50 ms")
            time.sleep(min(.001, remaining))
        self.last_sequence = sample.sequence
        self.record({"event": "feedback", "phase": self.phase, **sample.report()})
        q = np.asarray(sample.positions)
        velocity = np.asarray(sample.velocities, dtype=float)
        if q.shape != (14,) or velocity.shape != (14,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError("Invalid joint position/velocity feedback")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise RuntimeError("Measured joints outside model limits")
        return sample

    @staticmethod
    def normal_status(sample, index, *, disabled=False):
        status = sample.arm_status[index] if sample.arm_status else None
        allowed = (0, 6) if disabled else (0,)
        if status is None or len(status) != 8 or status[0] != 1 or status[2] != 1 or status[1] not in allowed or any(status[6:8]):
            raise RuntimeError("Require healthy CAN joint mode; no automatic mode reset or fault clearing")

    def preflight(self, duration_s=.3):
        """Fresh, stationary, disabled feedback from both arms; zero commands."""
        start = time.monotonic()
        while True:
            sample = self.sample()
            if sample.angle_sources != ("follower", "follower") or any(v != 0 for v in sample.driver_status):
                raise RuntimeError("Start with BOTH arms disabled and normal driver feedback")
            for i in range(2):
                self.normal_status(sample, i, disabled=True)
            q = np.asarray(sample.positions)
            if self.origin is None:
                self.origin = q.copy()
            if np.max(np.abs(sample.velocities)) > .02 or np.max(np.abs(q-self.origin)) > .005:
                raise RuntimeError("Both supported arms must remain stationary during preflight")
            if time.monotonic() - start >= duration_s:
                self.origin = q.copy()
                return sample
            time.sleep(.01)

    def guard(self, sample, *, dragging=False):
        q = np.asarray(sample.positions)
        if any(v != 0 for v in sample.driver_status[self.other]):
            raise RuntimeError("Untested arm enabled or lost healthy driver feedback")
        if sample.angle_sources[1-self.index] != "follower":
            raise RuntimeError("Untested arm changed feedback mode")
        self.normal_status(sample, 1-self.index, disabled=True)
        if np.max(np.abs(q[self.other]-self.origin[self.other])) > STARTUP_DRIFT_RAD:
            raise RuntimeError("Untested arm moved")
        excursion = float(np.max(np.abs(q[self.selected]-self.origin[self.selected])))
        if not dragging:
            self.max_startup_drift_rad = max(self.max_startup_drift_rad, excursion)
        if excursion > (DRAG_EXCURSION_RAD if dragging else STARTUP_DRIFT_RAD):
            raise RuntimeError("Joint excursion exceeded the supported-test stop threshold")
        speeds = np.abs(np.asarray(sample.velocities)[self.selected])
        if np.max(speeds) > DRAG_SPEED_RAD_S:
            joint = int(np.argmax(speeds)) + 1
            raise RuntimeError(
                f"Joint speed exceeded the supported-test stop threshold: "
                f"{self.side} joint {joint}, {speeds[joint-1]:.6f} rad/s > "
                f"{DRAG_SPEED_RAD_S:.6f} rad/s, phase={self.phase}")

    def start(self):
        # The cancel window must drain live traffic. Sleeping with subscribed
        # sockets open overflows their queues before the first feedback read.
        self.preflight(duration_s=3.)
        self.announce(f"ENABLING {self.side.upper()} ONLY. Keep both arms supported and hands clear of pinch points.", flush=True)
        # Set before send: a short write or interrupted syscall is uncertain.
        self.possibly_enabled = True
        self.phase = "enabling"
        self.send(self.enable_frames)
        deadline = time.monotonic() + TRANSITION_TIMEOUT_S
        while True:
            sample = self.sample()
            self.guard(sample)
            states = sample.driver_status[self.selected]
            if any(v not in (0, 0x40) for v in states):
                raise RuntimeError("Selected arm driver fault during enable")
            self.normal_status(sample, self.index, disabled=True)
            if all(v == 0x40 for v in states) and sample.arm_status[self.index][1] == 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Enable acknowledgement timed out")
            time.sleep(.005)
        # No pre-enable position seed and no position servo loop. Do not pretend
        # the firmware's retained target is known or that enable is motion-free.
        self.phase = "entering_leader"
        self.send(self.leader_frames)
        switched = time.monotonic()
        self.reader.set_angle_source(self.side, "leader", switched)
        deadline = switched + TRANSITION_TIMEOUT_S
        while True:
            # The reader deliberately needs time to receive a complete new
            # leader angle set. A stale/missing sample fails after the bounded
            # reader timeout; it is never reused as a valid acknowledgement.
            sample = self.sample()
            self.guard(sample)
            if sample.timestamp_s > switched and sample.angle_sources[self.index] == "leader":
                self.leader_confirmed = True
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Leader feedback acknowledgement timed out")
            time.sleep(.005)
        self.announce("LEADER ENCODERS RECEIVED; motor status is unavailable in this mode. Keep still for 2 seconds.", flush=True)
        self.phase = "leader_settling"
        until = time.monotonic() + SETTLE_S
        while time.monotonic() < until:
            self.guard(self.sample())
            time.sleep(.01)

    def stop(self):
        """Request damping, never disable or replay a joint target on cleanup."""
        if not self.possibly_enabled:
            return
        self.stopping = True
        self.phase = "stopping"
        # Diagnostics (disk full, broken stdout) must not prevent stop delivery.
        try:
            self.announce("STOP: requesting official damped emergency stop. Keep support in place; descent is possible.", flush=True)
        except Exception:
            pass
        failures = []
        first_stop_attempt_s = time.monotonic()
        # Bounded retries also cover a prior partial write. There is no claim
        # that this host process is a certified or independent safety system.
        for _ in range(3):
            try:
                for cid, payload in self.stop_frames:
                    if self.writer.send(struct.pack("=IB3x8s", cid, len(payload), payload)) != 16:
                        raise RuntimeError("Short emergency-stop write")
                self.stop_requested = True
            except Exception as error:
                failures.append(str(error))
            time.sleep(.02)
        try:
            self.record({"event": "stop_requested", "sent": self.stop_requested,
                         "first_attempt_monotonic_s": first_stop_attempt_s,
                         "can_id": hex(self.stop_frames[0][0]), "payload": self.stop_frames[0][1].hex(),
                         "attempts": 3, "errors": failures})
            self.announce("STOP delivery is not proof of stopping. Motors were NOT disabled. If motion persists, use the physical emergency stop.", flush=True)
        except Exception:
            pass
        if not self.stop_requested:
            raise RuntimeError("EMERGENCY STOP COULD NOT BE SENT. Use the physical emergency stop; keep support in place.")
        # Preserve post-stop evidence if feedback remains available. Stationary
        # encoders alone do not acknowledge emergency state or motor disable.
        deadline = time.monotonic() + .3
        last_post_sequence = self.last_sequence
        while time.monotonic() < deadline:
            try:
                sample = self.reader.read(timeout=.03)
                sample.validate(time.monotonic())
                # A cached snapshot acquired before stop is not post-stop
                # evidence, even if read() returns it after the stop frame.
                if (sample.timestamp_s >= first_stop_attempt_s and
                        (last_post_sequence is None or sample.sequence > last_post_sequence)):
                    self.record({"event": "post_stop_feedback", **sample.report()})
                    last_post_sequence = sample.sequence
            except BaseException:
                break  # Stop frames were already attempted; no mode recovery.
            time.sleep(.01)

    def run(self, seconds):
        if not np.isfinite(seconds) or not 1 <= seconds <= 120:
            raise ValueError("Test duration must be 1..120 seconds")
        try:
            self.start()
            self.phase = "supported_drag"
            self.announce(f"SUPPORTED DRAG TEST ({seconds:g} seconds): gently move ONLY {self.side}; keep supports in place. Ctrl-C requests damped stop.", flush=True)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                sample = self.sample()
                if sample.angle_sources[self.index] != "leader":
                    raise RuntimeError("Selected arm left leader feedback mode")
                self.guard(sample, dragging=True)
                time.sleep(.01)
            return {"result": "protocol_test_completed", "leader_feedback_confirmed": self.leader_confirmed,
                    "max_startup_drift_rad": self.max_startup_drift_rad,
                    "physical_gravity_compensation_verified": False,
                    "note": "Operator must assess force, stiffness and gravity support; encoder feedback cannot verify these."}
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--arms-supported", action="store_true",
                        help="Confirm both arms are mechanically supported against falling/swinging for the ENTIRE test, sweep clear, emergency stop accessible")
    parser.add_argument("--check-only", action="store_true", help="Inspect readiness without enabling or changing modes")
    parser.add_argument("--seconds", type=float, default=30., help="Supported dragging window, 1..120 seconds (default 30)")
    parser.add_argument("--mapping", type=Path, default=ROOT / "configs/hardware_mapping.json")
    parser.add_argument("--commissioning", type=Path, default=ROOT / "configs/commissioning.json")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
        parser.error("--seconds must be 1..120")
    if not args.check_only and not args.arms_supported:
        parser.error("Physical enable requires --arms-supported; use --check-only for readiness inspection")
    official_test_frames()  # Warm up all imports/encoders before hardware access.
    official_mode_frames("leader")
    kin = Kinematics()
    evidence = json.loads(args.commissioning.read_text())
    if not evidence.get("controller_mounting_configuration_verified"):
        parser.error("Controller mounting evidence is missing")
    mapping = json.loads(args.mapping.read_text())
    serials = sorted(mapping["arms"][s]["usb_identity"]["serial"] for s in SIDES)
    if len(set(serials)) != 2 or not all(s.isalnum() for s in serials):
        parser.error("Invalid CAN identities")
    output = ROOT / "logs/gravity_comp" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {output}", flush=True)
    report = {"side": args.side, "check_only": args.check_only,
              "physical_gravity_compensation_verified": False}
    test = None
    code = 0
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(*_):
        if not (test and test.stopping):
            raise KeyboardInterrupt
    for signum in previous_signals:
        signal.signal(signum, interrupted)
    try:
        with ExitStack() as stack:
            trace = stack.enter_context((output / "events.jsonl").open("x"))
            def record(event):
                trace.write(json.dumps({"monotonic_s": time.monotonic(), **event}, allow_nan=False) + "\n")
                trace.flush()
            lock = stack.enter_context(open("/tmp/agilex-deploy-" + "-".join(serials) + ".lock", "a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Close the policy/interactive controller first; robot ownership is already held") from error
            firmware = query_firmware(args.mapping)
            record({"event": "firmware_readback", **firmware})
            if firmware["missing"] or set(firmware["firmware"]) != set(SIDES) or any(v["software_version"] != "1.21" for v in firmware["firmware"].values()):
                raise RuntimeError("This diagnostic requires fresh NERO 1.21 firmware readback from both arms")
            reader = stack.enter_context(PassiveCanFeedback(args.mapping, motor_feedback=True))
            writer = None
            if not args.check_only:
                writer = stack.enter_context(socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW))
                writer.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, b"")
                writer.settimeout(.02)
                writer.bind((reader.channels[args.side],))
            test = GravityCompTest(reader, writer, args.side, kin.limits, record)
            if args.check_only:
                # Exercise the same receive window used before physical enable.
                test.preflight(duration_s=3.)
                report["result"] = "preflight_passed_no_enable"
            else:
                print("PHYSICAL COMMISSIONING: supports must remain throughout. Firmware enable may move before leader mode. No policy or gripper commands. Checking disabled-arm feedback for 3 seconds; Ctrl-C cancels before enable.", flush=True)
                report.update(test.run(args.seconds))
    except KeyboardInterrupt:
        report["result"] = "interrupted"
        code = 130
    except Exception as error:
        report.update(result="failed", error=f"{type(error).__name__}: {error}")
        code = 1
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
        report.update(stop_requested=bool(test and test.stop_requested),
                      stop_confirmed=False, motor_disable_command_sent=False,
                      motor_state_after_test=("disabled_at_preflight" if report.get("result") == "preflight_passed_no_enable" else "unknown"),
                      enable_attempted=bool(test and test.possibly_enabled))
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
