"""Coordinated dual NERO ownership using the existing official CAN codecs.

Both controllers are prepared before either is enabled. A paired target is
validated completely before either bus is written. CAN writes on independent
adapters are sequential within one control tick, not hardware-synchronized.
"""

from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np

from .contract import Action, ArmTarget, SIDES
from .nero_transport import MotionGuard, NeroTransport, position_frames
from .hardware_interlock import require_qualified_enable, require_qualified_hand_guiding


def selected_sides(side):
    if side == "both":
        return SIDES
    if side in SIDES:
        return (side,)
    raise ValueError("Select left, right, or both")


class DualMotionGuard:
    def __init__(self, guards):
        self.guards = guards
        self.indices = slice(0, 14)
        self.motion_limits = SimpleNamespace(
            **{
                name: np.concatenate(
                    [getattr(guards[s].motion_limits, name) for s in SIDES]
                )
                for name in ("position", "velocity", "acceleration")
            }
        )
        # Each guard checks the full vector against the same controller limits.
        for guard in guards.values():
            guard.limits = self.motion_limits.position.copy()

    @classmethod
    def simulation(cls, seed, limits, **kwargs):
        return cls({s: MotionGuard(s, seed, limits, **kwargs) for s in SIDES})

    def lift_target(self):
        return Action(*(getattr(self.guards[s].lift_target(), s) for s in SIDES))

    def action(self, action):
        for side, guard in self.guards.items():
            # Single-arm admission remains intact; the paired owner admits the
            # other arm separately rather than pretending it is stationary.
            local = Action(
                *(
                    getattr(action, s) if s == side else ArmTarget(guard.poses[s], None)
                    for s in SIDES
                )
            )
            guard.action(local)

    def command(self, q, measured, dt=0.01):
        for guard in self.guards.values():
            guard.command(q, measured, dt)
        return np.asarray(q).copy()

    def hold(self, measured):
        for guard in self.guards.values():
            guard.hold(measured)


class DualNeroTransport:
    def __init__(self, side="both", **kwargs):
        if side != "both":
            raise ValueError("Dual transport requires side=both")
        self.side = side
        self.endpoints = {}
        self.reader = None
        self.guard = None
        self.armed = False
        self.closed = False
        self.fault = None
        try:
            left = NeroTransport("left", _feedback_sides=SIDES, **kwargs)
            self.endpoints["left"] = left
            self.reader = left.reader
            self.endpoints["right"] = NeroTransport(
                "right", _shared_reader=self.reader, **kwargs
            )
        except BaseException:
            self.close()
            raise

    def _snapshot(self, timeout=0.01):
        snapshot = self.reader.read(timeout=timeout)
        for endpoint in self.endpoints.values():
            endpoint.last = snapshot
        return snapshot

    def read_encoders(self):
        try:
            snapshot = self._snapshot()
            samples = [self.endpoints[s].read_encoders(snapshot) for s in SIDES]
            return replace(
                samples[0], grippers=tuple(samples[i].grippers[i] for i in range(2)),
                gripper_feedback=tuple(samples[i].gripper_feedback[i] for i in range(2))
            )
        except Exception as error:
            if self.armed:
                raise self._fail(error) from error
            raise

    def _fail(self, error):
        self.fault = str(error)
        try:
            self.hold()
        except Exception as hold_error:
            self.fault += f"; paired HOLD unconfirmed: {hold_error}"
        return RuntimeError(self.fault)

    def arm(self):
        # Check both disabled controllers before preparing either CAN writer.
        require_qualified_enable(self._snapshot(.03), can_mode_sides=())
        if self.armed or self.fault:
            raise RuntimeError(self.fault or "Arming is one-shot")
        try:
            for endpoint in self.endpoints.values():
                endpoint.prepare_arm()
            self.guard = DualMotionGuard(
                {s: e.guard for s, e in self.endpoints.items()}
            )
            adopting = [e.initial_hand_guiding for e in self.endpoints.values()]
            if any(adopting):
                if not all(adopting):
                    raise RuntimeError("Both arms must be disabled or both in hand-guiding before paired enable")
                for endpoint in self.endpoints.values():
                    endpoint.armed = True
                self.armed = True
                self.read_encoders()
                return
            # Refresh and seed BOTH controllers after potentially slow readbacks.
            sample = self.read_encoders()
            require_qualified_enable(self.endpoints["left"].last)
            if any(v & 0x40 for v in self.endpoints["left"].last.driver_status):
                raise RuntimeError("Motor enable changed during paired preparation")
            if max(abs(v) for v in sample.velocities) > 0.02:
                raise RuntimeError("Arming requires stationary arms")
            for endpoint in self.endpoints.values():
                endpoint.guard.hold(sample.positions)
                endpoint.last_command = endpoint.guard.previous[endpoint.indices].copy()
                endpoint._send(position_frames(endpoint.last_command, endpoint.guard.motion_limits.position))
            for joint in range(1, 8):
                for endpoint in self.endpoints.values():
                    endpoint._send([(0x471, bytes([joint, 2]) + bytes(6))])
                time.sleep(0.005)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                snapshot = self._snapshot(0.03)
                if (
                    np.max(abs(np.asarray(snapshot.positions) - sample.positions))
                    > 0.015
                ):
                    raise RuntimeError("Posture shifted during paired arming")
                if any(v & ~0x40 for v in snapshot.driver_status):
                    raise RuntimeError("Driver fault during paired arming")
                for endpoint in self.endpoints.values():
                    endpoint._send(position_frames(endpoint.last_command, endpoint.guard.motion_limits.position))
                if all(v & 0x40 for v in snapshot.driver_status) and all(
                    status[:3] == bytes([1, 0, 1]) and not any(status[6:8])
                    for status in snapshot.arm_status
                ):
                    for endpoint in self.endpoints.values():
                        endpoint.armed = True
                    self.armed = True
                    return
                time.sleep(0.01)
            raise RuntimeError("Both arms did not acknowledge enable/mode")
        except BaseException as error:
            raise self._fail(
                RuntimeError(
                    f"Paired arming failed; some motors may be enabled. Maintain physical support. Cause: {error}"
                )
            ) from error

    def activate_gripper(self, action):
        # Validate all requested grippers before activating either.
        for side, endpoint in self.endpoints.items():
            if getattr(action, side).gripper_open is not None:
                if not self.armed or not endpoint.gripper_control:
                    raise RuntimeError("Gripper control is not armed")
                endpoint._gripper_feedback()
        for endpoint in self.endpoints.values():
            endpoint.activate_gripper(action)

    @property
    def control_mode(self):
        modes = {getattr(e, "control_mode", "follower") for e in self.endpoints.values()}
        return next(iter(modes)) if len(modes) == 1 else "transition"

    def enter_hand_guiding(self):
        require_qualified_hand_guiding()
        if not self.armed:
            return
        if self.fault:
            raise RuntimeError(self.fault)
        if self.control_mode == "hand_guiding":
            self.read_encoders()
            return
        try:
            # Stop both arms before changing either controller's mode.
            self.read_encoders()
            self.hold(preserve_grippers=True)
            for endpoint in self.endpoints.values():
                endpoint._stationary_sample(holding=True)
            # Issue both requests before waiting for either acknowledgement.
            # CAN buses are not atomic: write failures still latch explicitly.
            requests = {s: e.request_hand_guiding() for s, e in self.endpoints.items()}
            for side, endpoint in self.endpoints.items():
                endpoint.confirm_hand_guiding(requests[side])
            self.read_encoders()
        except Exception as error:
            raise self._fail(error) from error

    def begin_control(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.control_mode == "follower":
            return
        if self.control_mode != "hand_guiding":
            raise RuntimeError("Both arm modes must be confirmed before follower transition")
        try:
            for endpoint in self.endpoints.values():
                endpoint._stationary_sample(holding=False)
            sample = self.read_encoders()
            requests = {s: e.request_control(sample) for s, e in self.endpoints.items()}
            for side, endpoint in self.endpoints.items():
                endpoint.confirm_control(requests[side])
            self.read_encoders()
            self.hold(preserve_grippers=True)  # Refresh both guards without changing jaw commands.
        except Exception as error:
            raise self._fail(error) from error

    def send_positions(self, positions, grippers):
        if not self.armed:
            return
        if self.fault:
            raise RuntimeError(self.fault)
        try:
            prepared = []
            for index, side in enumerate(SIDES):
                local = [None, None]
                local[index] = grippers[index]
                prepared.append(
                    self.endpoints[side].prepare_positions(positions, local)
                )
            for side, frames in zip(SIDES, prepared):
                self.endpoints[side].send_prepared(*frames)
        except BaseException as error:
            raise self._fail(error) from error

    def hold(self, *, preserve_grippers=False):
        errors = []
        for side, endpoint in self.endpoints.items():
            try:
                if preserve_grippers:
                    endpoint.hold(preserve_grippers=True)
                else:
                    endpoint.hold()
            except Exception as error:
                errors.append(f"{side}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        # Close the reader/lock owner last so the pair stays exclusively owned
        # until BOTH writers have attempted HOLD and closed.
        for side in reversed(SIDES):
            if side in self.endpoints:
                try:
                    self.endpoints[side].close()
                except Exception as error:
                    errors.append(f"{side}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))
