"""Browser regression checks using a temporary simulator, never real hardware.

Run with: uv run --no-project --with playwright python scripts/check_policy_dashboard.py
Requires a Playwright Chromium headless shell installed on this host.
"""

import asyncio, json, os, subprocess, sys
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agilex_deploy.policy_trace import PolicyTrace


async def main():
    directory = ROOT / "logs/workshop/ui_contract"
    directory.mkdir(exist_ok=True)
    config = directory / "client.json"
    if config.exists():
        config.unlink()
    log = open(directory / "service.log", "w")
    proc = subprocess.Popen(
        [
            str(ROOT / ".venv-official/bin/python"),
            str(ROOT / "scripts/run_workshop.py"),
            "--backend",
            "sim",
            "--port",
            "0",
            "--auto-start-sim",
            "--client-config",
            str(config),
            "--output",
            str(directory),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=log,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MUJOCO_GL": "egl",
        },
    )
    try:
        for _ in range(100):
            if config.exists():
                break
            await asyncio.sleep(0.1)
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=str(
                    sorted(
                        (Path.home() / ".cache/ms-playwright").glob(
                            "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
                        )
                    )[-1]
                ),
                headless=True,
            )
            page = await browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto((directory / "operator_url.txt").read_text().strip())
            trace = PolicyTrace(
                config,
                "code",
                "UI contract test",
                "Render executable-looking text safely",
            )
            trace.emit("model_request", turn=1)
            await page.wait_for_function(
                "document.querySelector('#policyState').textContent.includes('Waiting for model')"
            )
            trace.emit("model_response", turn=1, duration_s=2.5)
            trace.emit(
                "code_generated",
                turn=1,
                code='def run(robot):\n    print("<img src=x onerror=window.injected=1>")',
            )
            trace.emit("code_started", timeout_s=90)
            trace.emit("code_output", text="worker output visible")
            trace.emit(
                "tool_call",
                call_id="test",
                tool="kinematics_ik",
                arguments={"arm": "left"},
            )
            trace.emit(
                "tool_result",
                call_id="test",
                tool="kinematics_ik",
                duration_s=0.2,
                result={"success": True},
            )
            await page.wait_for_function(
                "document.querySelector('#policyCode').textContent.includes('onerror')"
            )
            assert await page.evaluate("window.injected===undefined")
            assert (
                await page.locator("#policyCalls")
                .get_by_text("kinematics_ik", exact=False)
                .count()
                > 0
            )
            assert (
                "worker output visible"
                in await page.locator("#policyOutput").inner_text()
            )
            for mode in ("code", "mcp"):
                if mode == "mcp":
                    trace = PolicyTrace(
                        config, "mcp", "UI contract test", "MCP selection"
                    )
                    trace.emit(
                        "tool_call",
                        call_id="mcp-test",
                        tool="camera_capture",
                        arguments={},
                    )
                    trace.emit(
                        "tool_result",
                        call_id="mcp-test",
                        tool="camera_capture",
                        duration_s=0.1,
                        result={"frame_id": "42"},
                    )
                    await page.wait_for_function(
                        "document.querySelector('#policyRun').options.length===2"
                    )
                    await page.locator("#policyRun").select_option(trace.run_id)
                    await page.wait_for_function(
                        "document.querySelector('#policyMode').textContent==='MCP AGENT'"
                    )
                await page.set_viewport_size({"width": 390, "height": 844})
                assert not await page.evaluate(
                    "document.documentElement.scrollWidth > innerWidth"
                ), mode
                await page.screenshot(
                    path=str(directory / f"{mode}_mobile.png"), full_page=True
                )
                await page.set_viewport_size({"width": 1440, "height": 1100})
            await page.locator("#stop").click()
            await page.wait_for_function(
                "document.querySelector('#status').textContent==='STOPPED'"
            )
            assert not errors, errors
            await browser.close()
            report = {
                "page_errors": errors,
                "mobile_code_no_overflow": True,
                "mobile_mcp_no_overflow": True,
                "model_wait_visible": True,
                "code_and_stdout_visible": True,
                "mcp_arguments_results_visible": True,
                "script_injection_blocked": True,
                "stop_latched": True,
            }
            (directory / "browser_validation.json").write_text(
                json.dumps(report, indent=2)
            )
            print(json.dumps(report))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


asyncio.run(main())
