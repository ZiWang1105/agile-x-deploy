"""Generated Python isolation and cancellation, without CAN or model requests."""

import time
import unittest
from agilex_deploy.code_execution import execute_code
from agilex_deploy.policy_trace import scrub


class Trace:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **fields):
        self.rows.append(dict(kind=kind, **fields))


class Robot:
    def __init__(self, stop_after=None):
        self.held = 0
        self.calls = []
        self.started = time.monotonic()
        self.stop_after = stop_after

    def observe(self):
        return {
            "session_enabled": self.stop_after is None
            or time.monotonic() - self.started < self.stop_after,
            "session_fault": None,
        }

    def call(self, method, **params):
        self.calls.append((method, params))
        if method == "robot_observe":
            return self.observe()
        if method == "gripper_open":
            return {"operation_id": "left-open"}
        if method == "operation_get":
            return {"state": "succeeded"}
        raise ValueError("Unknown method")

    def hold(self):
        self.held += 1


class CodeTests(unittest.TestCase):
    def test_generated_code_runs_sdk_and_returns_output(self):
        robot, trace = Robot(), Trace()
        result, _ = execute_code(
            """def run(robot):
    result=robot.gripper_open("left").wait()
    print("left gripper completed")
    return result
""",
            robot,
            trace,
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(robot.calls[0][0], "gripper_open")
        self.assertEqual(robot.calls[0][1]["arm"], "left")
        self.assertIn("left gripper completed", result["stdout"])
        self.assertEqual(robot.held, 1)
        self.assertIn("tool_result", [r["kind"] for r in trace.rows])

    def test_kernel_blocks_keys_network_and_subprocess(self):
        result, _ = execute_code(
            """def run(robot):
    import socket, subprocess
    result={}
    for name, action in [("key",lambda:open("/home/ziwang/projects/agilex-deploy/openrouter_key.txt")),
                         ("network",lambda:socket.socket()),
                         ("process",lambda:subprocess.run(["/bin/true"]))]:
        try:
            action()
            result[name]="UNEXPECTED"
        except PermissionError:
            result[name]="blocked"
    return result
""",
            Robot(),
            Trace(),
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            result["result"], dict(key="blocked", network="blocked", process="blocked")
        )

    def test_hung_code_is_killed_and_holds(self):
        robot = Robot()
        result, _ = execute_code(
            "def run(robot):\n    while True: pass", robot, Trace(), timeout_s=0.8
        )
        self.assertFalse(result["ok"])
        self.assertIn("execution budget", result["error"])
        self.assertEqual(robot.held, 1)

    def test_stop_kills_active_python(self):
        robot = Robot(stop_after=0.5)
        started = time.monotonic()
        result, _ = execute_code(
            "def run(robot):\n    while True: pass", robot, Trace(), timeout_s=10
        )
        self.assertFalse(result["ok"])
        self.assertIn("cancelled", result["error"])
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(robot.held, 1)

    def test_code_cannot_access_operator_methods(self):
        robot = Robot()
        result, _ = execute_code(
            'def run(robot):\n    robot.call("enable")', robot, Trace()
        )
        self.assertFalse(result["ok"])
        self.assertIn("not allowed", result["error"])
        self.assertEqual(robot.calls, [])

    def test_trace_excludes_credentials_and_image_payloads(self):
        value = scrub(
            {
                "token": "sensitive",
                "rgb_png_base64": "large",
                "text": "key=sk-or-v1-" + ("a" * 64),
                "frame_id": "123",
            }
        )
        self.assertEqual(value, {"text": "key=[REDACTED]", "frame_id": "123"})
