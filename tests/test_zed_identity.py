"""Prevent a generic ZED USB label from selecting a different physical camera."""

import unittest
from pathlib import Path
from unittest.mock import patch

from agilex_deploy.zed_identity import read_video_serial, select_video_device


class ZedIdentityTests(unittest.TestCase):
    def test_selects_calibrated_camera_among_multiple_generic_devices(self):
        with (
            patch(
                "agilex_deploy.zed_identity.video_candidates",
                return_value=[Path("/dev/video4"), Path("/dev/video8")],
            ),
            patch(
                "agilex_deploy.zed_identity.read_video_serial",
                side_effect=["99999999", "35330610"],
            ),
        ):
            self.assertEqual(
                select_video_device("35330610"), (Path("/dev/video8"), "35330610")
            )

    def test_single_wrong_camera_and_explicit_wrong_path_are_rejected(self):
        with (
            patch(
                "agilex_deploy.zed_identity.video_candidates",
                return_value=[Path("/dev/video4")],
            ),
            patch(
                "agilex_deploy.zed_identity.read_video_serial", return_value="99999999"
            ),
        ):
            for device in (None, "/dev/video4"):
                with self.assertRaisesRegex(RuntimeError, "found 0"):
                    select_video_device("35330610", device)

    def test_unreadable_and_duplicate_identity_have_no_fallback(self):
        with patch(
            "agilex_deploy.zed_identity.video_candidates",
            return_value=[Path("/dev/video4"), Path("/dev/video8")],
        ):
            with (
                patch(
                    "agilex_deploy.zed_identity.read_video_serial",
                    side_effect=OSError("No identity"),
                ),
                self.assertRaisesRegex(RuntimeError, "found 0"),
            ):
                select_video_device("35330610")
            with (
                patch(
                    "agilex_deploy.zed_identity.read_video_serial",
                    return_value="35330610",
                ),
                self.assertRaisesRegex(RuntimeError, "found 2"),
            ):
                select_video_device("35330610")

    def test_serial_comes_from_video_response_and_read_request_is_fixed(self):
        requests = []

        def ioctl(fd, request, pointer):
            q = pointer._obj
            requests.append(q.query)
            self.assertEqual((fd, q.unit, q.selector), (123, 4, 2))
            if q.query == 0x85:
                q.data[0], q.data[1] = 128, 1  # 384-byte USB3 extension control.
            elif q.query == 0x01:
                payload = bytes(q.data[: q.size])
                self.assertEqual(payload[:3], bytes([0x51, 0xA1, 0x03]))
                self.assertEqual(int.from_bytes(payload[5:9], "big"), 0x18000)
                self.assertEqual(int.from_bytes(payload[11:13], "big"), 64)
            elif q.query == 0x81:
                response = b"OV" + bytes.fromhex("35330610")
                for i, byte in enumerate(response, 17):
                    q.data[i] = byte
            return 0

        with (
            patch("agilex_deploy.zed_identity.os.open", return_value=123),
            patch("agilex_deploy.zed_identity.os.close") as close,
            patch("agilex_deploy.zed_identity.ctypes.CDLL") as library,
        ):
            library.return_value.ioctl.side_effect = ioctl
            self.assertEqual(read_video_serial("/dev/video4"), "35330610")
            self.assertEqual(requests, [0x85, 0x01, 0x81])
            close.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
