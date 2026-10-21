"""Supervise isolated generated code and forward only public participant API calls."""

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .policy_trace import scrub

ALLOWED = {
    "robot_capabilities",
    "robot_observe",
    "camera_capture",
    "camera_deproject",
    "kinematics_ik",
    "kinematics_fk",
    "motion_move_tcp",
    "motion_move_both",
    "motion_move_joints",
    "motion_follow_waypoints",
    "gripper_open",
    "gripper_close",
    "operation_get",
    "operation_wait",
    "operation_cancel",
    "robot_hold",
}


def execute_code(code, robot, trace, timeout_s=90):
    """Execute one fence. The model key and HTTP credentials stay in the parent."""
    events = queue.Queue(maxsize=256)
    frames, output = [], []
    deadline = time.monotonic() + timeout_s
    source_root = str(Path(__file__).resolve().parents[1])
    env = {
        "PATH": os.defpath,
        "PYTHONPATH": source_root,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    outcome = None
    trace.emit("code_started", timeout_s=timeout_s)
    with tempfile.TemporaryDirectory(prefix="agilex-policy-") as directory:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agilex_deploy.code_worker"],
            cwd=directory,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        stopped = threading.Event()

        def read():
            try:
                for line in proc.stdout:
                    if len(line) > 4000000:
                        raise RuntimeError("Worker output limit exceeded")
                    while not stopped.is_set():
                        try:
                            events.put(line, timeout=0.1)
                            break
                        except queue.Full:
                            pass
            finally:
                if not stopped.is_set():
                    try:
                        events.put(None, timeout=1)
                    except queue.Full:
                        pass

        reader = threading.Thread(target=read, daemon=False)
        reader.start()
        try:
            proc.stdin.write(json.dumps({"code": code}) + "\n")
            proc.stdin.flush()
            last_check = 0
            while time.monotonic() < deadline:
                if time.monotonic() - last_check > 0.3:
                    observed = robot.observe()
                    if not observed["session_enabled"] or observed["session_fault"]:
                        raise RuntimeError(
                            "Session stopped or faulted; generated code cancelled"
                        )
                    last_check = time.monotonic()
                try:
                    line = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                value = json.loads(line)
                kind = value["type"]
                if kind == "call":
                    method, params = value["method"], value["params"]
                    call_id = uuid.uuid4().hex
                    if method != "operation_get":
                        trace.emit(
                            "tool_call", call_id=call_id, tool=method, arguments=params
                        )
                    started = time.monotonic()
                    try:
                        if method not in ALLOWED:
                            raise ValueError("API method not allowed")
                        result = robot.call(method, **params)
                        if method == "camera_capture":
                            frames.append(result)
                            del frames[:-2]
                        response = {"result": result}
                        logged = scrub(result)
                    except Exception as error:
                        response = {"error": str(error)}
                        logged = response
                    if method != "operation_get":
                        trace.emit(
                            "tool_result",
                            call_id=call_id,
                            tool=method,
                            duration_s=time.monotonic() - started,
                            result=logged,
                        )
                    proc.stdin.write(json.dumps(response, allow_nan=False) + "\n")
                    proc.stdin.flush()
                elif kind == "output":
                    output.append(value["text"])
                    if sum(map(len, output)) > 40000:
                        raise RuntimeError("Output budget exceeded")
                    trace.emit("code_output", text=value["text"])
                elif kind in ("result", "error"):
                    outcome = value
                    break
            if outcome is None:
                raise TimeoutError(
                    "Generated code exceeded execution budget or exited without a result"
                )
        except Exception as error:
            outcome = {"type": "error", "error": str(error)}
        finally:
            stopped.set()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            proc.stdin.close()
            proc.stdout.close()
            reader.join(5)
            # A returned fence never leaves an asynchronous command pursuing a goal.
            robot.hold()
    result = dict(
        ok=outcome["type"] == "result",
        result=outcome.get("result"),
        error=outcome.get("error"),
        stdout="".join(output),
    )
    trace.emit("code_result", result=result)
    return result, frames[-2:]
