"""CAN queue and shutdown failure regressions; all transports are offline."""
import errno
import unittest
from unittest.mock import Mock, patch

from agilex_deploy import hardware_session
from agilex_deploy.backends import SimBackend
from agilex_deploy.nero_transport import NeroTransport
from agilex_deploy.runtime import Robot


class CanRecoveryTests(unittest.TestCase):
    def test_transient_queue_full_retries_same_frame_without_duplicates(self):
        transport = NeroTransport.__new__(NeroTransport)
        transport.writer = Mock()
        transport.writer.send.side_effect = [16, OSError(errno.ENOBUFS, "queue full"), 16]
        transport.last_send = 0
        with patch("agilex_deploy.nero_transport.time.sleep") as sleep:
            transport._send([(0x155, bytes(8)), (0x156, bytes(8))])
        calls = transport.writer.send.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1], calls[2])
        self.assertNotEqual(calls[0], calls[1])
        sleep.assert_called_once()
        self.assertGreater(transport.last_send, 0)

    def test_persistent_queue_full_respects_batch_deadline(self):
        transport = NeroTransport.__new__(NeroTransport)
        transport.writer = Mock()
        transport.writer.send.side_effect = OSError(errno.ENOBUFS, "queue full")
        transport.last_send = 123
        with patch("agilex_deploy.nero_transport.time.monotonic", side_effect=[1., 1.001, 1.021]), \
             patch("agilex_deploy.nero_transport.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "deadline exceeded on frame 0x155"):
                transport._send([(0x155, bytes(8)), (0x156, bytes(8))])
        self.assertEqual(transport.writer.send.call_count, 2)
        self.assertEqual(transport.last_send, 123)

    def test_disconnected_device_is_not_retried(self):
        transport = NeroTransport.__new__(NeroTransport)
        transport.writer = Mock()
        transport.writer.send.side_effect = OSError(errno.ENETDOWN, "network down")
        with patch("agilex_deploy.nero_transport.time.sleep") as sleep:
            with self.assertRaises(OSError):
                transport._send([(0x155, bytes(8))])
        sleep.assert_not_called()
        self.assertEqual(transport.writer.send.call_count, 1)

    def test_hold_failure_preserves_original_runtime_fault(self):
        with Robot(SimBackend()) as robot:
            with patch.object(robot.backend, "write", side_effect=OSError(errno.ENOBUFS, "queue full")), \
                 patch.object(robot.backend, "hold", side_effect=RuntimeError("feedback stale")):
                with self.assertRaises(OSError) as caught:
                    robot.tick()
            self.assertEqual(caught.exception.errno, errno.ENOBUFS)
            self.assertEqual(robot.mode, "fault")
            self.assertIn("queue full", robot.reason)
            self.assertIn("HOLD failed: feedback stale", robot.reason)

    def test_worker_close_exits_and_reports_failure_even_if_hold_fails(self):
        transport = hardware_session.SimulatedCommissioningTransport("left")
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.return_value = (1, "close", None)
        with patch.object(hardware_session, "SimulatedCommissioningTransport", return_value=transport), \
             patch.object(transport, "hold", side_effect=RuntimeError("HOLD delivery failed")), \
             patch.object(hardware_session.signal, "signal"):
            hardware_session.control_worker(connection, "left", "unused", "unused", True, None)
        self.assertTrue(transport.closed)
        connection.close.assert_called_once()
        self.assertEqual(connection.recv.call_count, 1)
        response = connection.send.call_args_list[-1].args[0]
        self.assertEqual(response[:2], (1, False))
        self.assertIn("final mode unconfirmed", response[2])


if __name__ == "__main__":
    unittest.main()
