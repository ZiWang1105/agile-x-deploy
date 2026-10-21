import json
from pathlib import Path
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

from agilex_deploy.can_trace import capture, decode_record
from agilex_deploy.can_feedback import SO_TIMESTAMPNS, SO_RXQ_OVFL


def metadata():
    return [(socket.SOL_SOCKET, SO_TIMESTAMPNS, struct.pack("@ll", 123, 456))]


class CanTraceTests(unittest.TestCase):
    def test_command_echo_cannot_masquerade_as_bus_feedback(self):
        packet = struct.pack("=IB3x8s", 0x155, 8, bytes(8))
        host = decode_record("left", packet, metadata(), socket.MSG_DONTROUTE)
        bus = decode_record("left", packet, metadata(), 0)
        self.assertEqual(host["origin"], "host_echo")
        self.assertEqual(bus["origin"], "bus_rx")
        self.assertEqual(bus["kernel_realtime_ns"], 123000000456)
        with self.assertRaisesRegex(RuntimeError, "timestamp"):
            decode_record("left", packet, [], 0)

    def test_capture_has_no_transmit_api_and_closes_sockets(self):
        class ReadOnlySocket:
            closed = False
            def setsockopt(self, *args): pass
            def bind(self, address): self.address = address
            def setblocking(self, value): pass
            def recvmsg(self, *args):
                return struct.pack("=IB3x8s", 0x470, 8, b"\xfa" + bytes(7)), metadata(), socket.MSG_DONTROUTE, None
            def close(self): self.closed = True

        sock = ReadOnlySocket()  # Deliberately has no send/sendto/sendmsg method.
        pending = [sock]
        def ready(*args):
            result = pending.copy()
            pending.clear()
            return result, [], []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            with patch("agilex_deploy.can_trace.socket.socket", return_value=sock), \
                 patch("agilex_deploy.can_trace.select.select", side_effect=ready):
                result = capture({"left": "fake-can"}, path, .01)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(result["frames"], 1)
            self.assertEqual(rows[1]["origin"], "host_echo")
            self.assertEqual(rows[-1]["event"], "end")
            self.assertTrue(sock.closed)
            with self.assertRaises(FileExistsError):
                capture({}, path, .01)

    def test_truncated_frames_and_dropped_frame_counts_remain_visible(self):
        packet = struct.pack("=IB3x8s", 0x2A1, 8, bytes(8))
        with self.assertRaisesRegex(RuntimeError, "Truncated"):
            decode_record("right", packet, metadata(), socket.MSG_CTRUNC)
        ancillary = metadata() + [(socket.SOL_SOCKET, SO_RXQ_OVFL, struct.pack("=I", 3))]
        self.assertEqual(decode_record("right", packet, ancillary, 0)["dropped_frames"], 3)


if __name__ == "__main__":
    unittest.main()
