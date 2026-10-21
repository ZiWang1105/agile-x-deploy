"""Position-start admission and a separate physical hand-guiding interlock.

Ordinary position control is restored at the operator's explicit request. It
requires fresh disabled follower feedback and retains transport arming guards.
This does not establish the cause of the historical swing or prove firmware
startup is motion-free. The unqualified leader transition remains blocked.
There is no CLI/environment/configuration bypass of the leader interlock.
"""

import time

import numpy as np


def require_qualified_enable(snapshot=None, *, can_mode_sides=("left", "right")):
    """Check disabled startup; require CAN mode on already-prepared sides.

    During mode preparation, a disabled controller may still report teaching
    input mode (6). Both sides must acknowledge CAN mode before any enable.
    This does not admit leader feedback or relax motor/fault/velocity checks.
    """
    if any(side not in ("left", "right") for side in can_mode_sides):
        raise ValueError("Unknown side in CAN mode admission")
    if snapshot is None:
        raise RuntimeError("POSITION_START_REJECTED: fresh disabled-arm feedback is required")
    snapshot.validate(time.monotonic())
    if snapshot.angle_sources != ("follower", "follower"):
        raise RuntimeError("POSITION_START_REJECTED: both arms must report follower feedback; no leader adoption")
    if len(snapshot.driver_status) != 14 or any(v != 0 for v in snapshot.driver_status):
        raise RuntimeError("POSITION_START_REJECTED: both arms must be disabled with healthy motor feedback")
    if snapshot.driver_timestamp_s is None or snapshot.status_timestamp_s is None:
        raise RuntimeError("POSITION_START_REJECTED: motor/status acquisition timestamps required")
    if snapshot.arm_status is None or len(snapshot.arm_status) != 2:
        raise RuntimeError("POSITION_START_REJECTED: both controller status reports required")
    for side, status in zip(("left", "right"), snapshot.arm_status):
        allowed_modes = (1,) if side in can_mode_sides else (1, 6)
        if (status is None or len(status) != 8 or status[0] not in allowed_modes or status[2] != 1
                or status[1] not in (0, 6) or any(status[6:8])):
            raise RuntimeError(f"POSITION_START_REJECTED: {side} requires healthy CAN joint mode; status={status.hex() if status else None}; no automatic fault reset")
    q = np.asarray(snapshot.positions, dtype=float)
    v = np.asarray(snapshot.velocities, dtype=float)
    if (q.shape != (14,) or v.shape != (14,) or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(v)) or np.max(np.abs(v)) > .02):
        raise RuntimeError("POSITION_START_REJECTED: fresh stationary joint feedback required")


def require_qualified_hand_guiding():
    raise RuntimeError(
        "PHYSICAL_HAND_GUIDING_BLOCKED: leader/gravity-compensation behavior is "
        "unqualified. Use the default gravity_comp=False position-control session."
    )
