"""Fresh-process regressions for the documented plain Python launch commands."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def clean_environment():
    environment = os.environ.copy()
    for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.pop(name, None)
    return environment


class PolicyStartupTests(unittest.TestCase):
    def test_public_exports_are_lazy_and_keep_class_identity(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
import agilex_deploy
assert 'mujoco' not in sys.modules
assert 'numpy' not in sys.modules
from agilex_deploy.policy_launcher import launch_policy
import mujoco
assert mujoco.GLContext.__module__ == 'mujoco.egl'
from agilex_deploy import Action, ArmTarget, Pose, Observation, RobotInterface, Robot
from agilex_deploy.runtime import Robot as RuntimeRobot
from agilex_deploy.contract import Pose as ContractPose
assert Robot is RuntimeRobot
assert Pose is ContractPose
"""], cwd=ROOT, env=clean_environment(), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_plain_commands_render_before_enable(self):
        for filename in ("agent_policy.py", "generated_code_policy.py"):
            with self.subTest(runner=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                key = directory / "dummy_key.txt"
                key.write_text("startup-test-not-a-real-key")
                log_path = directory / "launch.log"
                with log_path.open("w") as log:
                    process = subprocess.Popen(
                        [sys.executable, str(ROOT / "examples/workshop" / filename),
                         "--backend", "sim", "--port", "0", "--no-browser",
                         "--key-file", str(key), "--prompt", "Inspect the camera."],
                        cwd=ROOT, env=clean_environment(), stdout=log, stderr=log,
                    )
                    try:
                        deadline = time.monotonic() + 45
                        match = None
                        while time.monotonic() < deadline:
                            output = log_path.read_text()
                            match = re.search(r"Organizer dashboard: (http://[^\s#]+)/#([^\s]+)", output)
                            if match or process.poll() is not None:
                                break
                            time.sleep(.1)
                        self.assertIsNotNone(match, "Dashboard did not start: " + output)
                        endpoint, token = match.groups()
                        with urlopen(endpoint + "/state", timeout=10) as response:
                            state = json.load(response)
                        self.assertFalse(state["enabled"])
                        self.assertFalse(state["fault"])
                        self.assertEqual(state["launcher"]["state"], "waiting_for_enable")
                        self.assertFalse(state["policy_events"])
                        for path, signature in (("/camera.png", b"\x89PNG"), ("/scene.jpg", b"\xff\xd8")):
                            with urlopen(endpoint + path, timeout=10) as response:
                                self.assertTrue(response.read().startswith(signature))
                        request = Request(endpoint + "/operator/close", data=b"{}", headers={
                            "Content-Type": "application/json", "X-Operator-Token": token})
                        with urlopen(request, timeout=10):
                            pass
                        self.assertEqual(process.wait(timeout=15), 0)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=15)


if __name__ == "__main__":
    unittest.main()
