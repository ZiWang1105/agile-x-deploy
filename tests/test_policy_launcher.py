"""Single-command policy startup using real HTTP/MCP and simulated hardware."""

import argparse
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from agilex_deploy.model import ROOT
from agilex_deploy.policy_launcher import add_launch_arguments, launch_policy
from agilex_deploy.workshop_http import WorkshopHTTP


def post(server, name, token=None, body=None):
    request = Request(server.endpoint + "/operator/" + name, data=json.dumps(body or {}).encode(),
                      headers={"Content-Type": "application/json",
                               "X-Operator-Token": token or server.operator_token})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


class LauncherTests(unittest.TestCase):
    def test_fault_after_round_keeps_dashboard_visible_until_close(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(config=None, output=Path(directory)/"report.json",
                backend="real", task="place", seed=0, calibration=None, camera_serial=None,
                prompt="Inspect", model="fixture", port=0, no_browser=True, exit_after_policy=False)
            service = Mock(backend="real", instruction="Inspect", lock=threading.RLock(),
                           enabled=True, stopped=False, fault=None)
            server = Mock(endpoint="http://127.0.0.1:12345", operator_token="test-only",
                          close_requested=threading.Event())
            failures = []

            def run():
                service.fault = "Camera failure: fixture"
                return {"status": "finished"}

            def launch():
                try:
                    launch_policy(args, "mcp", run)
                except Exception as error:
                    failures.append(error)

            with patch("agilex_deploy.workshop_service.WorkshopService", return_value=service), \
                 patch("agilex_deploy.workshop_http.WorkshopHTTP", return_value=server), \
                 patch("builtins.print"):
                thread = threading.Thread(target=launch)
                thread.start()
                try:
                    deadline = time.monotonic()+3
                    while (not isinstance(server.launcher, dict) or server.launcher["state"] != "failed") and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertEqual(server.launcher["state"], "failed")
                    self.assertIn("Camera failure", server.launcher["error"])
                    self.assertTrue(thread.is_alive())
                finally:
                    server.close_requested.set()
                    thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIn("Camera failure", str(failures[0]))

    def test_post_cleanup_verification_rejects_cached_frames(self):
        sys.path.insert(0, str(ROOT / "examples/workshop"))
        try:
            module = importlib.import_module("generated_code_policy")
        finally:
            sys.path.pop(0)
        robot = Mock()
        robot.observe.return_value = {"session_enabled": True}
        robot.capture.side_effect = [{"timestamp_s": 9}, {"timestamp_s": 11}]
        with patch.object(module.time, "sleep"):
            self.assertEqual(module.capture_after_cleanup(robot, 10), {"timestamp_s": 11})
        self.assertEqual(robot.capture.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "final state is unverified"):
            module.capture_after_cleanup(robot, 10, timeout_s=0)

    def test_real_launch_never_auto_enables_and_closes_owned_resources(self):
        args = argparse.Namespace(config=None, output=None, backend="real", task="place", seed=0,
                                  calibration=None, camera_serial="fixture", prompt="Inspect",
                                  model="fixture", port=0, no_browser=False, exit_after_policy=True)
        service = Mock(backend="real", instruction="Inspect", lock=threading.RLock(),
                       enabled=False, stopped=False, fault=None)
        server = Mock(endpoint="http://127.0.0.1:12345", operator_token="test-only",
                      close_requested=threading.Event())
        callback = Mock()
        with patch("agilex_deploy.workshop_service.WorkshopService", return_value=service) as factory, \
             patch("agilex_deploy.workshop_http.WorkshopHTTP", return_value=server), \
             patch("agilex_deploy.policy_launcher.webbrowser.open",
                   side_effect=lambda *a, **kw: server.close_requested.set()), patch("builtins.print"):
            launch_policy(args, "mcp", callback)
        self.assertEqual(factory.call_args.kwargs["backend"], "real")
        self.assertEqual(factory.call_args.kwargs["camera_serial"], "fixture")
        self.assertFalse(factory.call_args.kwargs["gravity_comp"])
        service.enable.assert_not_called()
        callback.assert_not_called()
        service.close.assert_called_once()
        server.close.assert_called_once()

    def test_occupied_port_closes_service_without_enabling_or_running_policy(self):
        args = argparse.Namespace(config=None, output=None, backend="sim", task="place", seed=0,
                                  calibration=None, camera_serial=None, prompt="Inspect",
                                  model="fixture", port=8780, no_browser=True, exit_after_policy=True)
        service, callback = Mock(), Mock()
        with patch("agilex_deploy.workshop_service.WorkshopService", return_value=service), \
             patch("agilex_deploy.workshop_http.WorkshopHTTP", side_effect=OSError("Address already in use")):
            with self.assertRaisesRegex(OSError, "Address already in use"):
                launch_policy(args, "code", callback)
        service.enable.assert_not_called()
        callback.assert_not_called()
        service.close.assert_called_once()

    def test_attach_does_not_construct_service_or_open_browser(self):
        args = argparse.Namespace(config=Path("existing/client.json"), output=None)
        with patch("agilex_deploy.workshop_service.WorkshopService") as service, \
             patch("agilex_deploy.policy_launcher.webbrowser.open") as browser:
            calls = []
            launch_policy(args, "code", lambda: calls.append("ran"))
            self.assertEqual(calls, ["ran"])
            service.assert_not_called()
            browser.assert_not_called()

    def test_backend_and_external_config_are_mutually_exclusive(self):
        parser = argparse.ArgumentParser()
        add_launch_arguments(parser)
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--backend", "real", "--config", "elsewhere.json"])

    def _exercise(self, mode, cancel=False, close_before_enable=False, empty_reply=False,
                  repeated_empty=False, request_failure=False, multi_round=False):
        sys.path.insert(0, str(ROOT / "examples/workshop"))
        try:
            module = importlib.import_module("generated_code_policy" if mode == "code" else "agent_policy")
        finally:
            sys.path.pop(0)
        ready = threading.Event()
        servers, failures, calls = [], [], []

        def http_factory(*args, **kwargs):
            server = WorkshopHTTP(*args, **kwargs)
            servers.append(server)
            return server

        def browser(url, **kwargs):
            self.assertEqual(url, servers[0].endpoint + "/#" + servers[0].operator_token)
            ready.set()
            return True

        def model(request, trace, check_active):
            check_active()
            self.assertTrue(servers[0].service.enabled)
            calls.append(json.loads(request.data))
            if multi_round:
                with self.assertRaises(HTTPError):
                    post(servers[0], "prompt", body={"prompt": "Must not replace an active round"})
            if request_failure and len(calls) == 2:
                raise RuntimeError("Provider request failed")
            if (empty_reply and len(calls) == 2) or (repeated_empty and len(calls) >= 2):
                return {"choices": [{"message": {"role": "assistant", "content": None},
                                     "finish_reason": "length"}],
                        "usage": {"completion_tokens": 4096}}
            if len(calls) == 1 or (multi_round and len(calls) == 3):
                if mode == "code":
                    message = {"role": "assistant", "content":
                               '```python\ndef run(robot):\n    state = robot.observe()\n    print("Measured mode:", state["mode"])\n    return {"observed": True}\n```'}
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "observe-once", "type": "function", "function":
                         {"name": "robot_observe", "arguments": "{}"}}]}
            else:
                message = {"role": "assistant", "content": "Observed robot feedback; no task-success claim."}
            return {"choices": [{"message": message}]}

        def main():
            try:
                module.main()
            except BaseException as error:
                failures.append(error)

        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key.txt"
            key.write_text("test-only-placeholder")
            report = Path(directory) / "report.json"
            argv = [module.__file__, "--backend", "sim", "--port", "0", "--key-file", str(key),
                    "--output", str(report), "--prompt", "Inspect the camera and report measured state."]
            with patch("sys.argv", argv), patch.object(module, "request_model", side_effect=model), \
                 patch("agilex_deploy.workshop_http.WorkshopHTTP", side_effect=http_factory), \
                 patch("agilex_deploy.policy_launcher.webbrowser.open", side_effect=browser), \
                 patch("builtins.print"), \
                 patch("agilex_deploy.hardware_session.HardwareSession", side_effect=AssertionError("No CAN in tests")):
                thread = threading.Thread(target=main, name="test-policy-launcher", daemon=False)
                thread.start()
                try:
                    self.assertTrue(ready.wait(20), failures)
                    server = servers[0]
                    with urlopen(server.endpoint + "/state") as response:
                        initial = json.load(response)
                    self.assertFalse(initial["enabled"])
                    self.assertEqual(initial["launcher"]["state"], "waiting_for_enable")
                    self.assertEqual(initial["launcher"]["mode"], mode)
                    self.assertEqual(calls, [])
                    with self.assertRaises(HTTPError):
                        post(server, "close", "not-the-operator")
                    self.assertFalse(server.close_requested.is_set())
                    post(server, "close" if close_before_enable else "stop" if cancel else "enable")
                    deadline = time.monotonic() + 20
                    while server.launcher["state"] not in ("finished", "cancelled", "failed") and time.monotonic() < deadline:
                        time.sleep(.05)
                    expected = "cancelled" if cancel or close_before_enable else "failed" if repeated_empty or request_failure else "finished"
                    self.assertEqual(server.launcher["state"], expected, failures)
                    if cancel or close_before_enable:
                        self.assertEqual(calls, [])
                    else:
                        self.assertEqual(len(calls), 4 if repeated_empty else 3 if empty_reply else 2)
                        self.assertEqual(json.loads(report.read_text())["status"], expected)
                        self.assertEqual(server.service.observe()["mode"], "hold")
                        kinds = [e["kind"] for e in server.service.policy_events]
                        self.assertIn("code_generated" if mode == "code" else "tool_call", kinds)
                        if empty_reply or repeated_empty:
                            self.assertIn("model_retry", kinds)
                            self.assertGreater(calls[-1]["max_tokens"], calls[-2]["max_tokens"])
                        self.assertTrue(thread.is_alive(), "Result dashboard should remain open")
                        with self.assertRaises(HTTPError):
                            post(server, "enable")
                    if not close_before_enable:
                        if multi_round:
                            first_robot = server.service.robot
                            with self.assertRaises(HTTPError):
                                post(server, "prompt", "not-the-operator", {"prompt": "Unauthorized"})
                            with self.assertRaises(HTTPError):
                                post(server, "prompt", body={"prompt": "  "})
                            followup = "Now inspect the other arm using fresh feedback."
                            post(server, "prompt", body={"prompt": followup})
                            deadline = time.monotonic() + 20
                            while (server.launcher.get("round") != 2 or server.launcher["state"] != "finished") and time.monotonic() < deadline:
                                time.sleep(.05)
                            self.assertEqual(server.launcher["round"], 2)
                            self.assertEqual(server.launcher["state"], "finished", failures)
                            self.assertEqual(len(calls), 4)
                            self.assertIs(server.service.robot, first_robot)
                            self.assertEqual(sum(e["event"] == "enabled" for e in server.service.events), 1)
                            self.assertIsNone(server.service.owner)
                            self.assertEqual(server.service.observe()["mode"], "hold")
                            second = report.with_name("report_round_002.json")
                            self.assertEqual(json.loads(second.read_text())["status"], "finished")
                            context = json.dumps(calls[2]["messages"])
                            self.assertIn("Inspect the camera and report measured state.", context)
                            self.assertIn("Observed robot feedback; no task-success claim.", context)
                            self.assertIn(followup, context)
                        if cancel:
                            with self.assertRaises(HTTPError):
                                post(server, "prompt", body={"prompt": "Do not resume a stopped session"})
                        post(server, "close")
                finally:
                    if servers:
                        servers[0].close_requested.set()
                    thread.join(20)
                self.assertFalse(thread.is_alive())
                if repeated_empty or request_failure:
                    self.assertEqual(len(failures), 1)
                    self.assertIsInstance(failures[0], RuntimeError)
                    self.assertIn("Provider request failed" if request_failure else "task is not complete", str(failures[0]))
                    self.assertIn("run_failed", kinds)
                    self.assertNotIn("run_finished", kinds)
                else:
                    self.assertFalse(failures, failures)
                self.assertFalse(servers[0].thread.is_alive())
                self.assertTrue(servers[0].service.closed)

    def test_generated_python_launch_enable_execute_review_and_close(self):
        self._exercise("code")

    def test_mcp_launch_enable_execute_review_and_close(self):
        self._exercise("mcp")

    def test_mcp_two_prompts_share_one_enabled_session_and_context(self):
        self._exercise("mcp", multi_round=True)

    def test_code_two_prompts_share_one_enabled_session_and_context(self):
        self._exercise("code", multi_round=True)

    def test_mcp_empty_reply_retries_instead_of_finishing(self):
        self._exercise("mcp", empty_reply=True)

    def test_code_empty_reply_after_execution_does_not_finish(self):
        self._exercise("code", empty_reply=True)

    def test_repeated_empty_mcp_replies_fail_dashboard_and_report(self):
        self._exercise("mcp", repeated_empty=True)

    def test_model_request_failure_is_saved_in_report(self):
        self._exercise("mcp", request_failure=True)

    def test_stop_before_enable_never_calls_model(self):
        self._exercise("code", cancel=True)

    def test_close_before_enable_never_calls_model(self):
        self._exercise("mcp", close_before_enable=True)


if __name__ == "__main__":
    unittest.main()
