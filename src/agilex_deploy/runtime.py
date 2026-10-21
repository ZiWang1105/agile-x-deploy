"""Single control owner used by the viewer, scripted policies, sim, and real.

The runtime thread is owned and joined by Robot.close(); no detached process
or timer is used. Rendering cannot change actuator commands or step the plant.
"""
from collections import OrderedDict
from dataclasses import replace
import threading
import time

import numpy as np

from .backends import Backend
from .contract import Action, JOINT_NAMES, Observation
from .controller import Controller
from .gripper_tracking import GripperTracker
from .model import CONTROL_DT, Kinematics

FEEDBACK_MAX_AGE_S = .2
TARGET_LEASE_S = .5


class Robot:
    def __init__(self, backend: Backend, *, ik=None):
        self.backend = backend
        self.kinematics = Kinematics()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.closed = False
        self.mode, self.reason = "hold", "startup"
        self.target = None
        self.target_deadline = 0.
        self.goal_started = 0.
        self.best_error = float("inf")
        self.path_progress = {}
        self.last_progress = 0.
        self.request_id = 0
        self.observation_id = 0
        self.observations = OrderedDict()
        self.last_sample = None
        self.last_sequence = -1
        self.errors = {}
        self.gripper_trackers = (GripperTracker(), GripperTracker())
        sample = self._read()
        self.controller = Controller(self.kinematics, sample.positions, sample.grippers, ik=ik)
        # Official solver construction/warm-up can exceed the feedback lease.
        # Acquire fresh encoders before exposing the initial observation.
        sample = self._read()
        self.controller.hold(sample.positions, sample.grippers)
        self._observe(sample)

    def _read(self):
        sample = self.backend.read()
        sample.validate()
        age = time.monotonic() - sample.timestamp_s
        if not -.01 <= age <= FEEDBACK_MAX_AGE_S:
            raise RuntimeError("Encoder feedback stale or timestamp is in the future")
        if sample.sequence < self.last_sequence:
            raise RuntimeError("Encoder sequence moved backwards")
        if self.last_sample is not None:
            if sample.timestamp_s < self.last_sample.timestamp_s:
                raise RuntimeError("Encoder acquisition time moved backwards")
            if sample.sequence == self.last_sequence and sample != self.last_sample:
                raise RuntimeError("Encoder sample changed without a new sequence")
        limits = self.kinematics.limits
        q = np.asarray(sample.positions)
        if np.any(q < limits[:, 0] - .02) or np.any(q > limits[:, 1] + .02):
            raise RuntimeError("Measured joint position outside source limits")
        self.last_sequence = sample.sequence
        self.last_sample = sample
        return sample

    def _observe(self, sample):
        self.kinematics.update(sample.positions, sample.grippers)
        self.observation_id += 1
        self.observation = Observation(self.observation_id, sample.timestamp_s, JOINT_NAMES,
            tuple(sample.positions), tuple(sample.velocities), tuple(sample.grippers),
            self.kinematics.pose("left"), self.kinematics.pose("right"), self.mode, self.reason)
        self.observation = replace(self.observation, gripper_states=tuple(
            tracker.observe(position, sample.timestamp_s, time.monotonic(), feedback,
                            fault=self.mode == "fault")
            for tracker, position, feedback in zip(self.gripper_trackers, sample.grippers, sample.gripper_feedback)),
            feedback_status=sample.feedback_status,
            motion_paths=self.controller.planner.status())
        self.observations[self.observation_id] = self.observation
        while len(self.observations) > 64:
            self.observations.popitem(last=False)

    def observe(self):
        with self.lock:
            self._ensure_open()
            return self.observation

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("Robot is closed")

    def submit_target(self, action: Action, observation_id: int):
        with self.lock:
            self._ensure_open()
            if self.mode == "fault":
                raise RuntimeError(f"Latched fault: {self.reason}; close and restart after inspection")
            if self.mode == "hold" and self.reason == "unreachable or stalled target":
                # A heartbeat can race the tick that detects a stalled path.
                # Do not resurrect it after HOLD; an explicit hold() clears
                # this reason before a caller starts a new operation.
                raise RuntimeError(f"PATH_STALLED: {self.reason}")
            if not isinstance(action, Action):
                raise ValueError("submit_target requires a validated Action")
            observed = self.observations.get(observation_id)
            if observed is None or time.monotonic() - observed.timestamp_s > FEEDBACK_MAX_AGE_S:
                raise ValueError("Unknown or stale observation; observe again")
            sample = self._read()
            if self.mode == "hand_guiding":
                self.begin_control()
                sample = self._read()
            for i, side in enumerate(("left", "right")):
                if sample.grippers[i] is None and getattr(action, side).gripper_open is not None:
                    raise ValueError(f"{side} gripper feedback unavailable; use gripper_open=None")
            now = time.monotonic()
            if action != self.target:
                self.goal_started = self.last_progress = now
                self.best_error = float("inf")
                self.path_progress.clear()
            self.target = action
            for side, tracker in zip(("left", "right"), self.gripper_trackers):
                tracker.command(getattr(action, side).gripper_open, now)
            self.target_deadline = now + TARGET_LEASE_S
            self.mode, self.reason = "tracking", "target accepted"
            self.request_id += 1
            return self.request_id

    def _hold(self, reason, fault=False, preserve_grippers=False):
        self.target = None
        preserve_grippers = preserve_grippers and not fault
        for tracker in self.gripper_trackers:
            tracker.hold()
        self.mode, self.reason = ("fault" if fault else "hold"), reason
        if self.last_sample is not None:
            grip = tuple(self.controller.grippers) if preserve_grippers else self.last_sample.grippers
            self.controller.hold(self.last_sample.positions, grip)
        # On read failures, delegate to the transport's own hold rather than
        # transmitting an old encoder snapshot as a new hardware position.
        if preserve_grippers:
            self.backend.hold(preserve_grippers=True)
        else:
            self.backend.hold()

    def hold(self, *, preserve_grippers=False):
        with self.lock:
            self._ensure_open()
            was_fault = self.mode == "fault"
            if self.mode == "hand_guiding":
                self._observe(self._read())
                return  # HOLD must not silently recapture a hand-guided arm.
            try:
                self._read()
                self._hold(self.reason if was_fault else "operator hold", fault=was_fault,
                           preserve_grippers=preserve_grippers)
                self._observe(self.last_sample)
            except Exception as error:
                self._hold(str(error), fault=True)
                raise

    def enter_hand_guiding(self):
        with self.lock:
            self._ensure_open()
            if self.mode == "fault":
                raise RuntimeError(f"Hand-guiding refused during fault: {self.reason}")
            if self.mode == "hand_guiding":
                self._observe(self._read())
                return
            try:
                self.hold(preserve_grippers=True)
                self.backend.enter_hand_guiding()
                self.mode, self.reason = "hand_guiding", "hand-guiding with no arm position targets; see feedback_status for motor telemetry availability"
                self._observe(self._read())
            except Exception as error:
                self._hold(str(error), fault=True)
                raise

    def begin_control(self):
        with self.lock:
            self._ensure_open()
            if self.mode == "fault":
                raise RuntimeError(self.reason)
            if self.mode != "hand_guiding":
                return
            try:
                self.backend.begin_control()
                sample = self._read()
                self.controller.hold(sample.positions, tuple(self.controller.grippers))
                self.target = None
                self.mode, self.reason = "hold", "follower control seeded from measured posture"
                self._observe(sample)
            except Exception as error:
                self._hold(str(error), fault=True)
                raise

    def tick(self):
        """Advance one controller period. Manual stepping is for deterministic tests."""
        with self.lock:
            self._ensure_open()
            try:
                sample = self._read()
                now = time.monotonic()
                if self.target is not None and now > self.target_deadline:
                    self._hold("target lease expired")
                if self.target is not None:
                    q, grip, self.errors = self.controller.step(sample, self.target)
                    error = sum(p + .2 * r for p, r in self.errors.values())
                    if error < self.best_error - .001:
                        self.best_error, self.last_progress = error, now
                    # A complementary arc can initially move away from its
                    # endpoint. Advancing measured-feedback waypoints is also
                    # progress, while an unchanged waypoint retains the same
                    # three-second stall protection as before.
                    for side, path in self.controller.planner.status().items():
                        if path["rotation_arc"] == "alternate":
                            progress = (path["revision"], path["waypoint"])
                            previous = self.path_progress.get(side)
                            if (previous is None or progress[0] != previous[0]
                                    or progress[1] > previous[1]):
                                self.last_progress = now
                            self.path_progress[side] = progress
                    if error > .025 and now - self.last_progress > 3.:
                        self._hold("unreachable or stalled target")
                    else:
                        self.backend.write(q, grip)
                elif self.mode not in ("fault", "hand_guiding"):
                    self.backend.write(self.controller.reference, self.controller.grippers)
                self.backend.advance()
                self._observe(self._read())
            except Exception as error:
                try:
                    self._hold(str(error), fault=True)
                except Exception as hold_error:
                    self.reason = f"{error}; HOLD failed: {hold_error}"
                finally:
                    if self.last_sample is not None:
                        self._observe(self.last_sample)
                raise

    def start(self):
        with self.lock:
            self._ensure_open()
            if self.thread is not None:
                raise RuntimeError("Control thread already started")
            self.thread = threading.Thread(target=self._run, name="agilex-control", daemon=False)
            self.thread.start()
        return self

    def _run(self):
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                return  # Fault was latched and backend HOLD attempted in tick().
            deadline += CONTROL_DT
            delay = deadline - time.monotonic()
            if delay < -.1:
                with self.lock:
                    self._hold("control loop overrun", fault=True)
                    self._observe(self.last_sample)
                return
            self.stop_event.wait(max(0., delay))

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
            if self.thread.is_alive():
                raise RuntimeError("Control thread did not stop; hardware transport must bound I/O")
        with self.lock:
            if not self.closed:
                try:
                    self.backend.close()
                finally:
                    self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
