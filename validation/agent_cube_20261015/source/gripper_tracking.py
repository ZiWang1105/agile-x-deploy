"""Shared feedback supervision around the official width/force command path.

This does not implement another force servo. The official firmware receives
width and force targets unchanged. Diagnostic thresholds never limit motion,
change force, infer contact, or stop a legitimate firmware-controlled grasp.
"""

import math
from collections import deque
from dataclasses import dataclass

from .contract import GripperState


@dataclass(frozen=True)
class TrackingSettings:
    # Local observation criteria, not manufacturer parameters or motion limits.
    endpoint_tolerance: float = 0.01
    endpoint_dwell_s: float = 0.1
    stationary_span: float = 0.002
    stationary_window_s: float = 0.5
    feedback_max_age_s: float = 0.2

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Tracking settings must be positive and finite")


class GripperTracker:
    def __init__(self, settings=None):
        self.settings = settings or TrackingSettings()
        self.request = None
        self.since = None
        self.samples = deque()
        self.at_endpoint_since = None
        self.last_stamp = None
        self.last_position = None
        self.velocity = None
        self.phase = "idle"

    def command(self, request, stamp):
        if request is None:
            return  # The public None request leaves the existing goal unchanged.
        if type(request) is not int or request not in (0, 1):
            raise ValueError("Gripper request must be integer 0 or 1")
        if request != self.request:
            self.request, self.since = request, stamp
            self.samples.clear()
            self.at_endpoint_since = None
            self.phase = "opening" if request else "closing"

    def hold(self):
        self.request = self.since = None
        self.samples.clear()
        self.at_endpoint_since = None
        self.phase = "hold"

    def observe(self, position, stamp, now, feedback=None, fault=False):
        if feedback is not None:
            stamp = feedback.timestamp_s
        error = (
            None
            if position is None or self.request is None
            else self.request - position
        )
        phase = self.phase
        fresh = (
            math.isfinite(stamp)
            and -0.01 <= now - stamp <= self.settings.feedback_max_age_s
        )
        if fault:
            phase = "fault"
        elif position is None:
            phase = "unavailable"
        elif not fresh:
            phase = "stale"
        elif feedback is not None and not feedback.enabled:
            phase = "disabled"
        elif self.last_stamp is not None and stamp < self.last_stamp:
            phase = "fault"
        else:
            if stamp != self.last_stamp:
                dt = None if self.last_stamp is None else stamp - self.last_stamp
                self.velocity = (
                    None
                    if dt is None or dt > self.settings.feedback_max_age_s
                    else (position - self.last_position) / dt
                )
                if dt is not None and dt > self.settings.feedback_max_age_s:
                    self.samples.clear()
                    self.at_endpoint_since = None
                self.last_stamp, self.last_position = stamp, position
                if self.request is not None and stamp > self.since:
                    phase = self._track(position, error, stamp)
                    self.phase = phase
        if phase in ("fault", "stale", "disabled", "unavailable"):
            self.samples.clear()
            self.at_endpoint_since = None
            # A later fresh report must establish progress/settling again.
            self.phase = (
                ("opening" if self.request else "closing")
                if self.request is not None
                else self.phase
            )
        return GripperState(
            phase=phase,
            requested_open=self.request,
            position_error=error,
            velocity_per_s=self.velocity
            if fresh and phase not in ("fault", "disabled", "unavailable")
            else None,
            width_m=None if feedback is None else feedback.width_m,
            force_n=None if feedback is None else feedback.force_n,
            enabled=None if feedback is None else feedback.enabled,
            timestamp_s=stamp,
        )

    def _track(self, position, error, stamp):
        settings = self.settings
        self.samples.append((stamp, position))
        while (
            len(self.samples) > 2
            and self.samples[1][0] <= stamp - settings.stationary_window_s
        ):
            self.samples.popleft()
        if abs(error) <= settings.endpoint_tolerance:
            if self.at_endpoint_since is None:
                self.at_endpoint_since = stamp
            if stamp - self.at_endpoint_since + 1e-9 >= settings.endpoint_dwell_s:
                return "reached"
        else:
            self.at_endpoint_since = None
            if (
                len(self.samples) >= 3
                and stamp - self.samples[0][0] + 1e-9 >= settings.stationary_window_s
            ):
                positions = [row[1] for row in self.samples]
                if max(positions) - min(positions) <= settings.stationary_span:
                    return "stalled"  # Could be an object, hard stop, or failure; not grasp success.
        return "opening" if self.request else "closing"
