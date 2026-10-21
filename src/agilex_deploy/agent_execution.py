"""Policy-independent execution owner; HTTPS latency never owns robot timing."""

import threading
import time

import numpy as np

from .contract import SIDES, Action, ArmTarget
from .controller import pose_error


class AgentStopped(RuntimeError):
    """Operator ended this policy session; later decisions must not execute."""


class AgentExecutor:
    def __init__(self, robot, available_arms=SIDES):
        self.robot, self.available_arms = robot, tuple(available_arms)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.target = None
        self.deadline = 0.0
        self.idle_deadline = time.monotonic() + 120
        self.error = None
        self.report = {"state": "idle"}
        self.thread = threading.Thread(
            target=self._run, name="agent-execution", daemon=False
        )

    def start(self):
        with self.lock:
            if self.stop_event.is_set():
                raise AgentStopped("Policy stopped by operator")
            self.robot.start()
            self.thread.start()
        return self

    def arm(self):
        """Explicit physical enable, serialized with feedback and cancellation."""
        with self.lock:
            if self.stop_event.is_set():
                raise AgentStopped("Policy stopped by operator")
            return self.robot.arm()

    def observe(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.robot.observe()

    def execute(self, decision):
        with self.lock:
            if self.stop_event.is_set():
                raise AgentStopped("Policy stopped by operator")
            observed = self.observe()
            for side in SIDES:
                if side not in self.available_arms:
                    arm = getattr(decision.action, side)
                    error = pose_error(getattr(observed, f"{side}_tcp"), arm.pose)
                    if arm.gripper_open is not None or np.linalg.norm(error) > 0.001:
                        raise ValueError("Policy commanded an unavailable arm")
            self.target = decision.action
            self.deadline = time.monotonic() + decision.duration_s
            self.idle_deadline = self.deadline + 120
            self.report = {"state": "executing", "grounding": decision.grounding}
        while time.monotonic() < self.deadline:
            if self.stop_event.is_set():
                raise AgentStopped("Policy stopped by operator")
            if self.error:
                raise RuntimeError(self.error)
            time.sleep(0.03)
        # Wait for the owner to capture measured HOLD arm references.
        while self.report["state"] == "executing" and not self.error:
            time.sleep(0.01)
        if self.error:
            raise RuntimeError(self.error)
        return self.report.copy()

    def halt(self):
        """Latch cancellation and HOLD under the same lock as target submission."""
        with self.lock:
            self.stop_event.set()
            self.target = None
            self.robot.hold()
            if not self.error and self.robot.observe().mode not in ("fault", "unarmed"):
                self.robot.enter_hand_guiding()
            self.report = {"state": "stopped", "reason": "Operator stopped policy; hand-guiding requested"}

    def _run(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    observed = self.robot.observe()
                    if observed.mode == "fault":
                        raise RuntimeError(observed.reason)
                    if time.monotonic() > self.idle_deadline:
                        raise RuntimeError("Policy session idle lease expired")
                    if self.target is not None:
                        if (
                            time.monotonic() >= self.deadline
                            and self.report["state"] == "executing"
                        ):
                            errors = {
                                s: pose_error(
                                    getattr(observed, f"{s}_tcp"),
                                    getattr(self.target, s).pose,
                                ).tolist()
                                for s in SIDES
                            }
                            self.target = Action(
                                *(
                                    ArmTarget(
                                        getattr(observed, f"{s}_tcp"),
                                        getattr(self.target, s).gripper_open,
                                    )
                                    for s in SIDES
                                )
                            )
                            arrived = all(
                                np.linalg.norm(e[:3]) <= 0.008
                                and np.linalg.norm(e[3:]) <= 0.1
                                for e in errors.values()
                            )
                            self.report = {
                                **self.report,
                                "state": "holding_arms",
                                "arrived": arrived,
                                "tcp_error_xyz_rotvec": errors,
                            }
                        self.robot.submit_target(self.target, observed.observation_id)
                self.stop_event.wait(0.05)
        except Exception as error:  # noqa: BLE001 - every owner failure must latch and attempt HOLD.
            self.error = f"{type(error).__name__}: {error}"
            try:
                self.robot.hold()
            except Exception:  # noqa: BLE001 - record an unconfirmed HOLD without restarting motion.
                self.error += "; HOLD unconfirmed"

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Execution owner failed to stop")
        self.robot.hold()
        if not self.error and self.robot.observe().mode not in ("fault", "unarmed"):
            self.robot.enter_hand_guiding()
