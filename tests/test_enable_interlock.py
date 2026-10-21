"""Containment tests for the real incident path; all CAN I/O is replaced."""

import struct
import time
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np

from agilex_deploy.hardware_session import control_worker
from agilex_deploy.nero_transport import NeroTransport, position_frames
from agilex_deploy.hardware_interlock import require_qualified_enable
from nero_firmware_sim import FirmwareSim
from test_dual_control import fake_hardware, BusPlant


class EnableInterlockTests(unittest.TestCase):
    def test_disabled_teaching_mode_is_prepared_on_both_sides_before_enable(self):
        bus = BusPlant()
        bus.ctrl_modes = dict(left=6, right=6)
        with fake_hardware(bus, allow_simulated_enable=False) as (transport, _):
            transport.arm()
            first_enable = next(i for i, (_, cid, _) in enumerate(bus.frames) if cid == 0x471)
            for side in ("left", "right"):
                self.assertTrue(any(s == side and cid == 0x151 and payload[0] == 1
                                    for s, cid, payload in bus.frames[:first_enable]))
            self.assertEqual(bus.ctrl_modes, dict(left=1, right=1))
            self.assertEqual(bus.enabled, [0x40]*14)
            self.assertFalse(any(cid == 0x470 for _, cid, _ in bus.frames))

    def test_ignored_mode_request_never_enables_either_arm(self):
        for unresponsive_side in ("left", "right"):
            with self.subTest(side=unresponsive_side):
                bus = BusPlant()
                bus.ctrl_modes[unresponsive_side] = 6
                bus.ignore_mode = True
                with fake_hardware(bus, allow_simulated_enable=False) as (transport, _):
                    with self.assertRaisesRegex(RuntimeError, "not acknowledged while disabled"):
                        transport.arm()
                    self.assertEqual(bus.enabled, [0]*14)
                    self.assertFalse(any(cid == 0x471 for _, cid, _ in bus.frames))

    def test_mode_preparation_does_not_relax_motor_fault_or_leader_checks(self):
        good = BusPlant().snapshot()
        teaching = bytes([6, 6, 1]) + bytes(5)
        sample = replace(good, arm_status=(teaching, teaching))
        require_qualified_enable(sample, can_mode_sides=())
        with self.assertRaises(RuntimeError):
            require_qualified_enable(sample)
        for invalid in (
            replace(sample, driver_status=(0x40,)*14),
            replace(sample, angle_sources=("leader", "follower")),
            replace(sample, arm_status=(bytes([6, 1, 1])+bytes(5), teaching)),
            replace(sample, velocities=(.1,)*14),
            replace(sample, timestamp_s=time.monotonic()-1),
        ):
            with self.assertRaises(RuntimeError):
                require_qualified_enable(invalid, can_mode_sides=())

    def test_enabled_left_start_reports_counts_and_recovery_without_writes(self):
        bus = BusPlant()
        bus.enabled[:7] = [0x40] * 7
        with self.assertRaises(RuntimeError) as raised:
            with fake_hardware(bus, allow_simulated_enable=False):
                self.fail("An already-enabled arm must not be adopted")
        message = str(raised.exception)
        self.assertIn("left 7/7 enabled, right 0/7 enabled", message)
        self.assertIn("hardware_power --arms-supported", message)
        self.assertIn("Disabling removes holding torque", message)
        self.assertEqual(bus.frames, [])
        self.assertEqual(bus.enabled, [0x40]*7 + [0]*7)

    def test_single_arm_is_blocked_before_any_transport_access(self):
        transport = NeroTransport.__new__(NeroTransport)
        with self.assertRaisesRegex(RuntimeError, "POSITION_START_REJECTED"):
            transport.arm()
        with self.assertRaisesRegex(RuntimeError, "POSITION_START_REJECTED"):
            transport.prepare_arm()

    def test_gravity_opt_in_rejects_enable_before_sending_motion(self):
        bus = FirmwareSim()
        bus.targets[0] += .6  # Latent old target must never be activated.
        try:
            with fake_hardware(bus, allow_simulated_enable=False) as (transport, _):
                connection = Mock()
                connection.poll.return_value = True
                connection.recv.side_effect = [(1, "arm", None), (2, "begin_control", None),
                                               (3, "close", None)]
                with patch("agilex_deploy.hardware_session.DualNeroTransport", return_value=transport), \
                     patch("agilex_deploy.hardware_session.signal.signal"):
                    control_worker(connection, "both", "unused", "unused", False, None, True,
                                   gravity_comp=True)
                replies = {c.args[0][0]: c.args[0] for c in connection.send.call_args_list}
                self.assertFalse(replies[1][1])
                self.assertIn("PHYSICAL_HAND_GUIDING_BLOCKED", replies[1][2])
                self.assertFalse(replies[2][1])
                self.assertTrue(replies[3][1])
                self.assertEqual(bus.frames, [])
                self.assertEqual(bus.enabled, [0]*14)
        finally:
            bus.plant.close()

    def test_default_worker_runs_position_mode_with_all_interlocks_active(self):
        with fake_hardware(allow_simulated_enable=False) as (transport, bus):
            connection = Mock()
            connection.poll.return_value = True
            connection.recv.side_effect = [(1, "arm", None), (2, "observe", None), (3, "close", None)]
            with patch("agilex_deploy.hardware_session.DualNeroTransport", return_value=transport), \
                 patch("agilex_deploy.hardware_session.signal.signal"):
                control_worker(connection, "both", "unused", "unused", False, None, True)
            replies = {c.args[0][0]: c.args[0] for c in connection.send.call_args_list}
            self.assertTrue(replies[1][1], replies[1])
            self.assertEqual(replies[2][2].mode, "hold")
            self.assertTrue(replies[3][1], replies[3])
            self.assertEqual(bus.enabled, [0x40]*14)
            self.assertFalse(any(cid == 0x470 for _, cid, _ in bus.frames))
            self.assertFalse(any(cid == 0x150 for _, cid, _ in bus.frames))
            self.assertFalse(any(cid == 0x471 and payload[1] == 1 for _, cid, payload in bus.frames))

    def test_position_preflight_rejects_fault_stale_enabled_and_leader(self):
        good = BusPlant().snapshot()
        require_qualified_enable(good)
        disabled_status = bytes([1, 6, 1]) + bytes(5)
        require_qualified_enable(replace(good, arm_status=(disabled_status, disabled_status)))
        bad = [
            replace(good, driver_status=(0x40,)*14),
            replace(good, driver_status=(None,)*7+(0,)*7),
            replace(good, angle_sources=("leader", "follower")),
            replace(good, timestamp_s=time.monotonic()-1),
            replace(good, arm_status=(bytes([1, 1, 1])+bytes(5), good.arm_status[1])),
            replace(good, velocities=(.1,)+(0.,)*13),
            replace(good, positions=(float("nan"),)*14),
        ]
        for sample in bad:
            with self.subTest(sample=sample), self.assertRaises(RuntimeError):
                require_qualified_enable(sample)

    def test_direct_leader_request_is_blocked_without_mode_frames(self):
        with fake_hardware(allow_simulated_enable=False) as (transport, bus):
            transport.arm()
            bus.frames.clear()
            for target in (transport, transport.endpoints["left"], transport.endpoints["right"]):
                with self.assertRaisesRegex(RuntimeError, "PHYSICAL_HAND_GUIDING_BLOCKED"):
                    target.enter_hand_guiding()
            self.assertEqual(bus.frames, [])

    def test_firmware_does_not_accept_disabled_targets_or_modes(self):
        bus = FirmwareSim()
        try:
            # Unknown old target deliberately differs from the measured pose.
            bus.targets[0] += .6
            retained = bus.targets.copy()
            writer = bus.socket()
            writer.bind(("left",))
            frames = position_frames(bus.q[:7]) + [(0x470, b"\xfa" + bytes(7))]
            for cid, data in frames:
                writer.send(struct.pack("=IB3x8s", cid, 8, data))
            np.testing.assert_array_equal(bus.targets, retained)
            self.assertEqual(bus.device_modes["left"], "follower")
            self.assertEqual(len(bus.ignored_disabled_frames), 5)
        finally:
            bus.plant.close()

    def test_leader_packets_can_exist_with_position_stiffness(self):
        bus = FirmwareSim()
        bus.leader_servo_active.add("left")
        try:
            with fake_hardware(bus) as (transport, _):
                transport.arm()
                transport.enter_hand_guiding()
                sample = transport.read_encoders()
                self.assertEqual(sample.feedback_status[0]["angle_source"], "leader")
                self.assertIsNone(sample.feedback_status[0]["motor_enabled"])
                target = bus.targets[0]
                bus.plant.data.qpos[bus.plant.qids[0]] += .1
                # Protocol-only acknowledgement must not be interpreted as a
                # compliance measurement: this hypothetical firmware still
                # pulls the joint back to its stored target.
                for _ in range(80):
                    bus.last_step = time.monotonic() - .02
                    bus.snapshot()
                self.assertLess(abs(bus.q[0]-target), .01)
        finally:
            bus.plant.close()


if __name__ == "__main__":
    unittest.main()
