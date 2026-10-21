"""Bounded policy telemetry. This channel cannot execute robot commands."""

import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import uuid


def scrub(value):
    if isinstance(value, dict):
        return {
            k: scrub(v)
            for k, v in value.items()
            if not k.endswith("_base64")
            and k.lower()
            not in {"token", "lease_token", "authorization", "api_key", "key"}
        }
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value[:200]]
    if isinstance(value, str):
        return re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED]", value)[:40000]
    return value


class PolicyTrace:
    def __init__(self, config, mode, model, instruction):
        self.config = json.loads(Path(config).read_text())
        self.run_id = uuid.uuid4().hex
        self.mode = mode
        self.started = time.monotonic()
        self.emit("run_started", model=model, instruction=instruction)

    def emit(self, kind, **fields):
        event = scrub(
            dict(
                run_id=self.run_id,
                mode=self.mode,
                kind=kind,
                elapsed_s=time.monotonic() - self.started,
                **fields,
            )
        )
        request = Request(
            self.config["endpoint"] + "/telemetry",
            data=json.dumps(event, allow_nan=False).encode(),
            headers={
                "Authorization": "Bearer " + self.config["token"],
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            json.load(response)
