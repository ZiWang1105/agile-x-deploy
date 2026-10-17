"""Participant SDK. Standard library only; no CAN, simulator, or vendor imports."""

import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


class APIError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


class Operation:
    def __init__(self, client, result):
        self.client = client
        self.id = result["operation_id"]

    def status(self):
        return self.client.call("operation_get", operation_id=self.id)

    def cancel(self):
        return self.client.call("operation_cancel", operation_id=self.id)

    def wait(self, timeout_s=310):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.status()
            if result["state"] in ("succeeded", "failed", "cancelled"):
                return result
            time.sleep(0.1)
        raise TimeoutError(
            "Client wait expired; inspect/cancel the operation explicitly"
        )


class RobotClient:
    def __init__(self, endpoint, token, client_id=None):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.client_id = client_id or uuid.uuid4().hex
        self.claimed = False
        self.lease_token = ""

    @classmethod
    def from_config(cls, path=None, **kwargs):
        value = json.loads(
            Path(
                path
                or os.environ.get("AGILEX_WORKSHOP_CONFIG", "logs/workshop/client.json")
            ).read_text()
        )
        return cls(value["endpoint"], value["token"], **kwargs)

    def call(self, method, **params):
        request = Request(
            self.endpoint + "/api",
            data=json.dumps(
                dict(method=method, params=params, client_id=self.client_id),
                allow_nan=False,
            ).encode(),
            headers={
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/json",
                "X-Workshop-Lease": self.lease_token,
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
        except HTTPError as error:
            result = json.load(error)
        if "error" in result:
            raise APIError(result["error"]["code"], result["error"]["message"])
        return result["result"]

    def claim(self):
        result = self.call("session_claim")
        self.claimed = True
        self.lease_token = result.get("lease_token", "")
        return result

    def close(self):
        if self.claimed:
            try:
                self.call("session_release")
            finally:
                self.claimed = False

    def capabilities(self):
        return self.call("robot_capabilities")

    def observe(self):
        return self.call("robot_observe")

    def capture(self):
        return self.call("camera_capture")

    def deproject(self, frame_id, pixel_uv):
        return self.call("camera_deproject", frame_id=frame_id, pixel_uv=pixel_uv)

    def ik(self, arm, target, seed_rad=None, tool_frame="tcp"):
        return self.call(
            "kinematics_ik",
            arm=arm,
            target=target,
            seed_rad=seed_rad,
            tool_frame=tool_frame,
        )

    def fk(self, arm, joints_rad, tool_frame="tcp"):
        return self.call(
            "kinematics_fk", arm=arm, joints_rad=joints_rad, tool_frame=tool_frame
        )

    def move_tcp(self, arm, target, tool_frame="tcp", timeout_s=10, request_id=None):
        return Operation(
            self,
            self.call(
                "motion_move_tcp",
                arm=arm,
                target=target,
                tool_frame=tool_frame,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def move_joints(self, arm, joints_rad, timeout_s=10, request_id=None):
        return Operation(
            self,
            self.call(
                "motion_move_joints",
                arm=arm,
                joints_rad=joints_rad,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def move_both(self, targets, tool_frame="tcp", timeout_s=10, request_id=None):
        return Operation(
            self,
            self.call(
                "motion_move_both",
                targets=targets,
                tool_frame=tool_frame,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def follow_waypoints(
        self, arm, targets, tool_frame="tcp", timeout_s=30, request_id=None
    ):
        return Operation(
            self,
            self.call(
                "motion_follow_waypoints",
                arm=arm,
                targets=targets,
                tool_frame=tool_frame,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def gripper_open(self, arm, timeout_s=5, request_id=None):
        return Operation(
            self,
            self.call(
                "gripper_open",
                arm=arm,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def gripper_close(self, arm, timeout_s=5, request_id=None):
        return Operation(
            self,
            self.call(
                "gripper_close",
                arm=arm,
                timeout_s=timeout_s,
                request_id=request_id or uuid.uuid4().hex,
            ),
        )

    def hold(self):
        return self.call("robot_hold")

    def __enter__(self):
        self.claim()
        return self

    def __exit__(self, *_):
        self.close()
