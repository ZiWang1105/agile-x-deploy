"""Browser control authentication and cancellation at the command-owner boundary."""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agilex_deploy.agent_dashboard import AgentDashboard
from agilex_deploy.agent_execution import AgentExecutor, AgentStopped
from agilex_deploy.contract import Action, ArmTarget, Pose
from agilex_deploy.model import ROOT


class DashboardTests(unittest.TestCase):
    def test_real_launcher_requires_start_and_fresh_capture_before_enable(self):
        spec = importlib.util.spec_from_file_location(
            "agent_launcher", ROOT / "scripts/run_agent.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        for outcome in ("start", "cancel", "stale_camera"):
            with self.subTest(outcome=outcome):
                self._check_real_launcher(launcher, outcome)

    def _check_real_launcher(self, launcher, outcome):
        with tempfile.TemporaryDirectory() as directory:
            events = []

            class Hardware:
                def __init__(self, *args, **kwargs):
                    self_test.assertEqual(args[0], "both")

                def observe(self):
                    return SimpleNamespace(mode="unarmed")

                def arm(self):
                    events.append("arm")

                def start(self):
                    events.append("start")

                def hold(self):
                    events.append("hold")

                def close(self):
                    events.append("close")

            class Camera:
                def __init__(self, robot, *_, **kwargs):
                    self.robot = robot

                def capture(self):
                    events.append("capture")
                    if outcome == "stale_camera" and events.count("capture") == 2:
                        raise RuntimeError("ZED acquisition is stale")
                    return self.robot.observe(), SimpleNamespace(calibration={})

                def close(self):
                    pass

            class Dashboard:
                def __init__(self, *_):
                    self.lock = threading.RLock()
                    self.cancelled = threading.Event()
                    self.exited = threading.Event()
                    self.exited.set()

                def update(self, **kwargs):
                    pass

                def observation(self, *_):
                    pass

                def watch_real(self, *_):
                    pass

                def wait_start(self):
                    events.append("review")
                    self_test.assertNotIn("arm", events)
                    if outcome == "cancel":
                        raise AgentStopped("Cancelled before Start")
                    return "User task"

                def check_stop(self):
                    pass

                def event(self, *_):
                    pass

                def freeze(self):
                    pass

                def close(self):
                    pass

            self_test = self
            policy = SimpleNamespace(calls=0, cost=0)
            def act(*args):
                self_test.assertEqual(args[-1], ("left", "right"))
                raise AgentStopped("End mock episode")
            policy.act = act
            arguments = [
                "run_agent.py",
                "--backend",
                "real",
                "--execute-real",
                "--output",
                str(Path(directory) / "episode"),
            ]
            with (
                patch("sys.argv", arguments),
                patch("agilex_deploy.hardware_session.HardwareSession", Hardware),
                patch("agilex_deploy.agent_zed.ZedStereoCamera", Camera),
                patch("agilex_deploy.agent_dashboard.AgentDashboard", Dashboard),
                patch.object(launcher, "OpenRouterPolicy", return_value=policy),
            ):
                if outcome == "stale_camera":
                    with self.assertRaisesRegex(RuntimeError, "stale"):
                        launcher.main()
                else:
                    launcher.main()
            if outcome == "start":
                self.assertEqual(
                    events[:5], ["capture", "review", "capture", "arm", "start"]
                )
            else:
                self.assertNotIn("arm", events)
                self.assertNotIn("start", events)

    def test_prompt_start_authentication_and_stop_latch(self):
        dashboard = AgentDashboard(0, "sim", "Original prompt", "test-model")
        holds = []
        dashboard.halt = lambda: holds.append(True)
        dashboard.update(phase="ready", can_start=True)

        def post(path, payload, token):
            request = Request(
                f"http://127.0.0.1:{dashboard.port}{path}",
                data=json.dumps(payload).encode(),
                headers={"X-Agent-Token": token, "Content-Type": "application/json"},
            )
            return urlopen(request, timeout=2)

        try:
            with self.assertRaises(HTTPError) as caught:
                post("/start", {"instruction": "Wrong sender"}, "incorrect")
            self.assertEqual(caught.exception.code, 403)
            self.assertFalse(dashboard.started.is_set())
            with post("/start", {"instruction": "Pick up the cube"}, dashboard.token):
                pass
            self.assertTrue(dashboard.started.is_set())
            self.assertEqual(dashboard.state["instruction"], "Pick up the cube")
            with post("/stop", {}, dashboard.token):
                pass
            self.assertEqual(holds, [True])
            with self.assertRaises(AgentStopped):
                dashboard.check_stop()
            with self.assertRaises(HTTPError):
                post("/start", {"instruction": "Restart"}, dashboard.token)
        finally:
            dashboard.close()
        self.assertFalse(dashboard.thread.is_alive())

    def test_stop_during_motion_prevents_late_decision_and_renewal(self):
        pose = Pose([0.3, 0.2, 0.2], [0, 0, 0, 1])
        observed = SimpleNamespace(
            left_tcp=pose, right_tcp=pose, mode="hold", observation_id=1
        )

        class Robot:
            writes = 0
            holds = 0

            def start(self):
                pass

            def observe(self):
                return observed

            def submit_target(self, *_):
                self.writes += 1

            def hold(self):
                self.holds += 1

            def enter_hand_guiding(self):
                self.guiding = True

        robot = Robot()
        executor = AgentExecutor(robot).start()
        decision = SimpleNamespace(
            action=Action(ArmTarget(pose, None), ArmTarget(pose, None)),
            duration_s=6,
            grounding=None,
        )
        errors = []

        def execute():
            try:
                executor.execute(decision)
            except AgentStopped:
                errors.append("cancelled")

        thread = threading.Thread(target=execute)
        try:
            thread.start()
            until = time.monotonic() + 1
            while not robot.writes and time.monotonic() < until:
                time.sleep(0.01)
            self.assertGreater(robot.writes, 0)
            executor.halt()
            after_stop = robot.writes
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["cancelled"])
            with self.assertRaises(AgentStopped):
                executor.execute(decision)
            time.sleep(0.1)
            self.assertEqual(robot.writes, after_stop)
            self.assertEqual(robot.holds, 1)
            self.assertTrue(robot.guiding)
        finally:
            executor.close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
