"""Verify diagnostic probes never emit enable, motion, or setting-write frames."""
import struct
import unittest
import numpy as np
from agilex_deploy.motion_limits import JointLimits, MANUAL_VELOCITY
from agilex_deploy.model import Kinematics
from unittest.mock import patch

from agilex_deploy import firmware_probe, limits_probe


class ProbeSocket:
    def __init__(self):
        self.sent = []
        self.pending = []
        self.closed = False

    def setsockopt(self, *_):
        pass

    def settimeout(self, *_):
        pass

    def bind(self, address):
        self.channel = address[0]

    def send(self, frame):
        cid, dlc, data = struct.unpack("=IB3x8s", frame)
        self.sent.append((cid, dlc, data[:dlc]))
        if cid == 0x4AF:
            self.pending.append(struct.pack("=IB3x8s", 0x4AF, 8, bytes([2, 3, 4, 5, 6, 7, 1, 21])))
        elif cid == 0x472 and data[1] == 2:
            response = bytes([data[0]]) + (2000 if data[0] < 3 else 2500).to_bytes(2, "big") + bytes(5)
            self.pending.append(struct.pack("=IB3x8s", 0x47C, 8, response))
        elif cid == 0x472:
            response = bytes([data[0]]) + struct.pack(">hhH", 900, -900, 100) + bytes(1)
            self.pending.append(struct.pack("=IB3x8s", 0x473, 8, response))
        else:
            raise AssertionError(f"Unexpected CAN transmit ID {cid:#x}")
        return len(frame)

    def recv(self, _):
        return self.pending.pop(0)

    def __enter__(self): return self

    def __exit__(self, *_): self.close()

    def close(self):
        self.closed = True


def ready(sockets, *_):
    return [s for s in sockets if s.pending], [], []


class ProbeTests(unittest.TestCase):
    def test_firmware_probe_transmits_only_one_fixed_read_per_arm(self):
        sockets = [ProbeSocket(), ProbeSocket()]
        with patch.object(firmware_probe, "mapped_channels", return_value={"left": "can1", "right": "can0"}), \
             patch.object(firmware_probe.socket, "socket", side_effect=sockets), \
             patch.object(firmware_probe.select, "select", side_effect=ready):
            result = firmware_probe.query_firmware("unused")
        self.assertFalse(result["missing"])
        for s in sockets:
            self.assertEqual(s.sent, [(0x4AF, 1, b"\x01")])
            self.assertTrue(s.closed)
        self.assertEqual(result["firmware"]["left"]["sdk_profile"], "v121")

    def test_limit_probe_requests_each_joint_without_modification(self):
        sockets = [ProbeSocket(), ProbeSocket()]
        with patch.object(limits_probe, "mapped_channels", return_value={"left": "can1", "right": "can0"}), \
             patch.object(limits_probe.socket, "socket", side_effect=sockets), \
             patch.object(limits_probe.select, "select", side_effect=ready):
            result = limits_probe.query_limits("unused")
        self.assertTrue(result["complete"])
        for s in sockets:
            self.assertEqual(s.sent, [(0x472, 8, bytes([j, 1]) + bytes(6)) for j in range(1, 8)])
            self.assertTrue(s.closed)
        self.assertEqual(result["limits"]["left"][1]["lower_deg"], -90)

    def test_timeout_does_not_enable_or_switch_mode_to_retry(self):
        sockets = [ProbeSocket(), ProbeSocket()]
        with patch.object(limits_probe, "mapped_channels", return_value={"left": "can1", "right": "can0"}), \
             patch.object(limits_probe.socket, "socket", side_effect=sockets):
            result = limits_probe.query_limits("unused", timeout=0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["limits"], {"left": {}, "right": {}})
        self.assertTrue(all(cid == 0x472 for s in sockets for cid, _, _ in s.sent))

    def test_motion_limits_query_is_selected_channel_read_only(self):
        sock = ProbeSocket()
        with patch.object(limits_probe.socket, "socket", return_value=sock), \
             patch.object(limits_probe.select, "select", side_effect=ready):
            result = limits_probe.query_motion_limits("selected-arm", Kinematics().limits[:7])
        self.assertEqual(sock.channel, "selected-arm")
        self.assertEqual(sock.sent, [(0x472, 8, bytes([j, k]) + bytes(6))
                                    for k in (1, 2) for j in range(1, 8)])
        self.assertTrue(sock.closed)
        np.testing.assert_allclose(result.velocity, np.ones(7))
        np.testing.assert_allclose(result.acceleration, [2, 2, 2.5, 2.5, 2.5, 2.5, 2.5])

    def test_motion_readback_timeout_has_no_fallback(self):
        sock = ProbeSocket()
        with patch.object(limits_probe.socket, "socket", return_value=sock):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                limits_probe.query_motion_limits("selected-arm", Kinematics().limits[:7], timeout=0)
        self.assertTrue(all(cid == 0x472 for cid, _, _ in sock.sent))
        self.assertTrue(sock.closed)

    def test_manual_speed_ceiling_and_invalid_acceleration(self):
        model = Kinematics().limits[:7]
        angles = {j: {"lower_deg": -180, "upper_deg": 180, "max_velocity_rad_s": 10}
                  for j in range(1, 8)}
        limits = JointLimits.from_readback(angles, {j: 2 for j in angles}, model)
        np.testing.assert_allclose(limits.velocity, MANUAL_VELOCITY)
        np.testing.assert_allclose(limits.position, model)
        for data in (bytes(8), bytes([1]) + bytes(7), bytes(7)):
            with self.assertRaises(ValueError):
                limits_probe.decode_acceleration(data)
        with self.assertRaises(ValueError):
            JointLimits(model, np.ones(7), [1, 1, 1, 1, 1, 1, np.nan])

    def test_unknown_firmware_is_not_assumed_compatible(self):
        self.assertEqual(firmware_probe.profile_for(1, 21), "v121")
        self.assertEqual(firmware_probe.profile_for(1, 20), "v120")
        self.assertEqual(firmware_probe.profile_for(1, 12), "v112")
        self.assertEqual(firmware_probe.profile_for(1, 11), "v111")
        self.assertEqual(firmware_probe.profile_for(1, 10), "default")
        self.assertIsNone(firmware_probe.profile_for(1, 15))
        self.assertIsNone(firmware_probe.profile_for(2, 0))

    def test_invalid_limit_response_rejected(self):
        for data in (bytes(7), bytes(8), bytes([1]) + struct.pack(">hhH", -900, 900, 100) + bytes(1)):
            with self.assertRaises(ValueError):
                limits_probe.decode_limit(data)


if __name__ == "__main__":
    unittest.main()
