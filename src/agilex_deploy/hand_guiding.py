"""Official NERO mode methods, encoded without a second CAN owner.

The installed, pinned pyAgxArm driver supplies the leader/follower methods and
codec. This adapter only replaces its I/O: it opens no sockets or SDK threads.
The existing transport remains the sole writer and checks acquired feedback.
"""

from functools import lru_cache


@lru_cache(maxsize=2)
def official_mode_frames(mode):
    if mode not in ("leader", "follower"):
        raise ValueError("Expected leader or follower")
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v121.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v112.parser import Codec
    from pyAgxArm.protocols.can_protocol.msgs.nero.default import ArmMsgLeaderFollowerModeConfig

    class Encoder(Driver):
        def __init__(self):
            # Preserve the commissioned mounting orientation and firmware speed.
            self._msg_mode = self._MSG_ModeCtrl(move_spd_rate_ctrl=100, installation_pos=0)
            self.frames = []
            self.codec = Codec()

        def _send_msg(self, message):
            if isinstance(message, ArmMsgLeaderFollowerModeConfig):
                cid = 0x470
                payload = self.codec.encode_470_leader_follower_mode_config(message)
            elif isinstance(message, self._MSG_ModeCtrl):
                cid = 0x151
                payload = self.codec.encode_151_mode_ctrl(message)
            else:
                raise TypeError("Unexpected SDK mode message")
            self.frames.append((cid, bytes(payload)))

    encoder = Encoder()
    getattr(encoder, "set_" + mode + "_mode")()
    return tuple(encoder.frames)


def follower_feedback_frame():
    """Restore normal telemetry after the official follower method disables it.

    Installation=0 preserves the configured mount. No enable or position target
    is encoded here. Leader mode does not promise normal telemetry on firmware
    1.21, so the reader explicitly exposes that telemetry as unavailable.
    """
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v121.driver import Driver
    from pyAgxArm.protocols.can_protocol.drivers.nero.versions.v112.parser import Codec
    msg = Driver._MSG_ModeCtrl(move_spd_rate_ctrl=100, installation_pos=0, enable_can_push=1)
    return (0x151, bytes(Codec().encode_151_mode_ctrl(msg)))
