"""Verify disable-only protocol and acknowledgement without opening CAN."""
import contextlib
import io
import struct
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agilex_deploy.hardware_power import disable_supported_arms, _disable_and_verify, main
from agilex_deploy.gripper_open import open_with_arm_hold


class DisableTests(unittest.TestCase):
    def test_support_acknowledgement_required_before_any_hardware_access(self):
        with patch("agilex_deploy.hardware_power.PassiveCanFeedback") as reader:
            with self.assertRaisesRegex(ValueError, "support BOTH arms"):
                disable_supported_arms()
            reader.assert_not_called()

    def test_only_enabled_joint_disable_frames_are_sent_and_feedback_is_verified(self):
        states = [0x40] * 7 + [0] * 7
        frames = []

        class Writer:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def setsockopt(self, *_): pass
            def settimeout(self, *_): pass
            def bind(self, address): self.side = address[0]
            def send(self, packet):
                cid, dlc, data = struct.unpack("=IB3x8s", packet)
                frames.append((self.side, cid, dlc, data))
                states[data[0]-1] = 0
                return 16

        class Reader:
            channels = {"left": "left", "right": "right"}
            def read(self, **_):
                return SimpleNamespace(driver_status=tuple(states), driver_timestamp_s=time.monotonic())

        with patch("agilex_deploy.hardware_power.socket.socket", side_effect=lambda *_: Writer()):
            result = _disable_and_verify(Reader())
        self.assertEqual(result, {"left_enabled": 0, "right_enabled": 0, "commands_sent": 7})
        self.assertEqual(frames, [("left", 0x471, 8, bytes([j, 1]) + bytes(6)) for j in range(1, 8)])

    def test_already_disabled_does_not_open_a_writer(self):
        reader = SimpleNamespace(read=lambda **_: SimpleNamespace(driver_status=(0,)*14))
        with patch("agilex_deploy.hardware_power.socket.socket") as writer:
            self.assertEqual(_disable_and_verify(reader)["commands_sent"], 0)
            writer.assert_not_called()

    def test_no_acknowledgement_never_reports_success(self):
        reader = SimpleNamespace(read=lambda **_: SimpleNamespace(driver_status=(0x40,)*14),
                                 channels={"left": "left", "right": "right"})
        with patch("agilex_deploy.hardware_power.socket.socket"), patch(
                "agilex_deploy.hardware_power.time.monotonic", side_effect=[0., 3.]):
            with self.assertRaisesRegex(RuntimeError, "acknowledgement timed out"):
                _disable_and_verify(reader)


class CombinedCommandTests(unittest.TestCase):
    def run_cli(self, flags, disable_error=None, open_error=None):
        calls = []

        def disable(*args, **kwargs):
            calls.append("disable")
            if disable_error:
                raise disable_error
            return {"left_enabled": 0, "right_enabled": 0}

        def open_jaws(mapping, sides, *, arms_supported):
            self.assertEqual(tuple(sides), ("left", "right"))
            self.assertTrue(arms_supported)
            calls.append("open")
            if open_error:
                raise open_error
            return {"status": "fully_open_confirmed", "arms_disabled": True}

        output = io.StringIO()
        error = io.StringIO()
        code = 0
        with patch("sys.argv", ["hardware_power", *flags]), patch(
            "agilex_deploy.hardware_power.disable_supported_arms", side_effect=disable
        ), patch(
            "agilex_deploy.hardware_power.open_with_arm_hold", side_effect=open_jaws
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            try:
                main()
            except SystemExit as exc:
                code = exc.code
        return calls, code, output.getvalue(), error.getvalue()

    def test_combined_command_disables_before_opening_both(self):
        calls, code, output, _ = self.run_cli(["--arms-supported", "--open-grippers"])
        self.assertEqual(calls, ["disable", "open"])
        self.assertEqual(code, 0)
        self.assertIn("fully_open_confirmed", output)
        self.assertIn("DISABLED confirmed", output)

    def test_disable_only_never_opens(self):
        calls, code, _, _ = self.run_cli(["--arms-supported"])
        self.assertEqual((calls, code), (["disable"], 0))

    def test_missing_support_acknowledgement_never_touches_hardware(self):
        calls, code, _, _ = self.run_cli(["--open-grippers"])
        self.assertEqual((calls, code), ([], 2))
        with self.assertRaisesRegex(ValueError, "support BOTH arms"):
            open_with_arm_hold("nonexistent.json", ("left", "right"))

    def test_failed_initial_disable_never_opens(self):
        calls, code, output, _ = self.run_cli(
            ["--arms-supported", "--open-grippers"], disable_error=RuntimeError("busy")
        )
        self.assertEqual((calls, code), (["disable"], 1))
        self.assertNotIn("DISABLED confirmed", output)

    def test_failed_open_never_reports_success(self):
        calls, code, output, error = self.run_cli(
            ["--arms-supported", "--open-grippers"], open_error=RuntimeError("timeout")
        )
        self.assertEqual((calls, code), (["disable", "open"], 1))
        self.assertNotIn("DISABLED confirmed", output)
        self.assertIn("not confirmed: timeout", error)


if __name__ == "__main__":
    unittest.main()
