"""Check both one-command simulator launchers in Chromium, with offline model fixtures.

Run: uv run --no-project --with playwright python scripts/check_policy_launcher.py
No real CAN connection, OpenRouter requests, or private API key is used.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


def launch_environment():
    # Exercise the documented plain Python command. Supplying EGL here would
    # hide package import-order bugs in the launcher's own environment setup.
    environment = os.environ.copy()
    for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.pop(name, None)
    return environment


HARNESS = r'''
import importlib, json, sys
from pathlib import Path
from unittest.mock import patch
mode, url_file, key_file, report = sys.argv[1:]
sys.path.insert(0, str(Path.cwd() / "examples/workshop"))
module = importlib.import_module("agent_policy" if mode == "mcp" else "generated_code_policy")
calls = []
def model(request, trace, check_active):
    check_active()
    calls.append(1)
    if len(calls) in (1, 3):
        if mode == "code":
            message = {"role":"assistant", "content":'```python\ndef run(robot):\n    state = robot.observe()\n    print("Measured joints:", len(state["joint_positions_rad"]))\n    robot.capture()\n    return {"inspection": "complete"}\n```'}
        else:
            message = {"role":"assistant", "content":None, "tool_calls":[{"id":"head-camera", "type":"function", "function":{"name":"camera_capture", "arguments":"{}"}}]}
    else:
        message = {"role":"assistant", "content":"Offline UI fixture completed. No manipulation was attempted."}
    return {"choices":[{"message":message}]}
def open_browser(url, **kwargs):
    Path(url_file).write_text(url)
    return True
sys.argv = [module.__file__, "--backend", "sim", "--port", "0", "--key-file", key_file,
            "--model", "offline-ui-fixture", "--output", report, "--prompt", "Inspect the head camera and measured robot state."]
with patch.object(module, "request_model", side_effect=model), patch("agilex_deploy.policy_launcher.webbrowser.open", side_effect=open_browser):
    module.main()
'''


async def main():
    output = ROOT / "logs/workshop/launcher_ui_validation"
    output.mkdir(parents=True, exist_ok=True)
    results = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=str(sorted(
            (Path.home() / ".cache/ms-playwright").glob(
                "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"))[-1]), headless=True)
        try:
            for mode in ("code", "mcp"):
                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    key = directory / "fake_key.txt"
                    key.write_text("offline-ui-fixture-not-a-key")
                    url = directory / "operator_url.txt"
                    report = output / (mode + "_report.json")
                    with (directory / "process.log").open("w") as log:
                        process = subprocess.Popen([str(ROOT / ".venv-official/bin/python"), "-c", HARNESS,
                                                    mode, str(url), str(key), str(report)], cwd=ROOT,
                                                   stdout=log, stderr=log,
                                                   env=launch_environment())
                        try:
                            for _ in range(200):
                                if url.exists():
                                    break
                                if process.poll() is not None:
                                    raise RuntimeError("Launcher exited before dashboard startup")
                                await asyncio.sleep(.1)
                            assert url.exists(), "Dashboard startup timed out"
                            page = await browser.new_page(viewport={"width":1440,"height":1100})
                            errors = []
                            page.on("pageerror", lambda error: errors.append(str(error)))
                            await page.goto(url.read_text())
                            await page.wait_for_function("document.querySelector('#launcherStatus').textContent.includes('waiting for enable')")
                            assert await page.locator("#enable").inner_text() == "Enable & start policy"
                            state = await (await page.request.get(str((await page.evaluate('location.origin'))) + "/state")).json()
                            assert not state["enabled"] and not state["policy_events"]
                            await page.screenshot(path=str(output / (mode + "_waiting.png")), full_page=True)
                            await page.locator("#enable").click()
                            await page.wait_for_function("document.querySelector('#launcherStatus').textContent.includes('finished')", timeout=30000)
                            assert "HOLD" in await page.locator("#status").inner_text()
                            assert "camera_capture" in await page.locator("#policyCalls").inner_text()
                            if mode == "code":
                                assert "def run(robot)" in await page.locator("#policyCode").inner_text()
                                assert "Measured joints:" in await page.locator("#policyOutput").inner_text()
                            await page.screenshot(path=str(output / (mode + "_finished.png")), full_page=True)
                            await page.locator("#promptInput").fill("Inspect the head camera again using fresh feedback.")
                            await page.locator("#sendPrompt").click()
                            await page.wait_for_function("document.querySelector('#launcherStatus').textContent.includes('round 2') && document.querySelector('#launcherStatus').textContent.includes('finished')", timeout=30000)
                            state = await (await page.request.get(str((await page.evaluate('location.origin'))) + "/state")).json()
                            assert sum(e["event"] == "enabled" for e in state["events"]) == 1
                            assert state["launcher"]["round"] == 2
                            assert await page.locator("#policyPrompt").inner_text() == "Inspect the head camera again using fresh feedback."
                            await page.screenshot(path=str(output / (mode + "_round_2.png")), full_page=True)
                            # Leader-only firmware reports unknown motor and jaw
                            # status. The shared dashboard must not show these as
                            # enabled motors or a measured 0% jaw opening.
                            state = await (await page.request.get(str((await page.evaluate('location.origin'))) + "/state")).json()
                            state["robot"]["feedback_status"] = [{"motor_enabled": None}] * 2
                            state["robot"]["mode"] = "hand_guiding"
                            state["robot"]["gripper_position"] = [None, None]
                            async def unknown_feedback(route):
                                await route.fulfill(json=state)
                            await page.route("**/state", unknown_feedback)
                            await page.wait_for_function("document.querySelector('#status').textContent.includes('MOTOR STATUS UNAVAILABLE')")
                            assert (await page.locator("#robot").inner_text()).count("Gripper: unavailable") == 2
                            await page.screenshot(path=str(output / (mode + "_unknown_feedback.png")), full_page=True)
                            await page.unroute("**/state", unknown_feedback)
                            await page.set_viewport_size({"width":390,"height":844})
                            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                            assert not errors, errors
                            await page.locator("#closeSession").click()
                            await asyncio.to_thread(process.wait, timeout=15)
                            assert process.returncode == 0
                            await page.close()
                            results.append(dict(mode=mode, waiting_gate=True, enable_and_start=True,
                                                trace_visible=True, position_hold_default=True, multi_round=True,
                                                unknown_feedback_visible=True,
                                                close_exited=True, javascript_errors=errors))
                        finally:
                            if process.poll() is None:
                                process.terminate()
                                await asyncio.to_thread(process.wait, timeout=15)
        finally:
            await browser.close()
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
