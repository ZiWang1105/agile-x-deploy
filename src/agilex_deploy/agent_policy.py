"""Vision policy boundary: RGB-D and measured state in, validated poses out.

The client has no simulator handle, scene file, evaluator, shell, or tools.
The API key is read only while constructing HTTPS authorization and is never
included in saved payloads, exception messages, or model history.
"""

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .agent_sensors import ground_pixel, quantize_depth, save_frame
from .contract import SIDES, Action, ArmTarget, Pose

# Nominal pinch point 15 mm behind the official fingertip TCP, between distal
# finger inner faces. The small rear black CAD pieces are not the grasp tips.
# Nominal tool geometry, not a physical hand-eye/TCP calibration.
PINCH_IN_TCP_M = np.array([-0.015, 0.0, 0.0])
TOP_DOWN_XYZW = (
    Rotation.from_matrix([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).as_quat().tolist()
)


def pinch_pose(tcp):
    return Pose(
        np.asarray(tcp.position_m)
        + Rotation.from_quat(tcp.quaternion_xyzw).apply(PINCH_IN_TCP_M),
        tcp.quaternion_xyzw,
    )


def tcp_pose(pinch):
    return Pose(
        np.asarray(pinch.position_m)
        - Rotation.from_quat(pinch.quaternion_xyzw).apply(PINCH_IN_TCP_M),
        pinch.quaternion_xyzw,
    )


def vector_schema(n, kind="number"):
    return {"type": "array", "items": {"type": kind}, "minItems": n, "maxItems": n}


ARM_SCHEMA = {
    "type": "object",
    "properties": {
        "position_m": vector_schema(3),
        "quaternion_xyzw": vector_schema(4),
        "gripper_open": {"type": ["integer", "null"], "enum": [0, 1, None]},
    },
    "required": ["position_m", "quaternion_xyzw", "gripper_open"],
    "additionalProperties": False,
}
GROUND_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "arm": {"type": "string", "enum": list(SIDES)},
        "pixel_uv_1000": vector_schema(2, "integer"),
        "offset_base_m": vector_schema(3),
        "phase": {"type": "string", "enum": ["inspect", "target"]},
    },
    "required": ["arm", "pixel_uv_1000", "offset_base_m", "phase"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "left": ARM_SCHEMA,
        "right": ARM_SCHEMA,
        "grounding": GROUND_SCHEMA,
        "duration_s": {"type": "number", "minimum": 0.2, "maximum": 6},
        "done": {"type": "boolean"},
        "decision_summary": {"type": "string"},
    },
    "required": [
        "left",
        "right",
        "grounding",
        "duration_s",
        "done",
        "decision_summary",
    ],
    "additionalProperties": False,
}

SYSTEM = """You control a fixed dual NERO robot through measured RGB-D observations.
You have one fixed stand-mounted head camera and no wrist cameras.
Return only one JSON object conforming to the provided schema. Never output code.
Each left/right pose is the desired DISTAL-FINGER MIDPOINT (pinch frame), in base_link meters,
with XYZW unit quaternion. Local +X points along the hand toward the fingers;
local Z is the jaw separation axis. The adapter converts pinch poses to the official
NERO TCP before IK. Both arms are policy-controlled; select either available arm.
Base axes: X forward, Y left, Z up. Left arm starts at positive Y; right at negative Y.
Image left/right are not arm names. Select the hand using grounded base-frame
geometry and measured poses, avoiding unnecessary cross-body reaching.
Copy the measured pinch pose and use gripper_open=null for an arm you leave unchanged.
Always use the CURRENT observation's measured pose for this, not an old history pose.
gripper_open=1 opens, 0 closes and maintains closure; null preserves its prior command.
The feedback gripper_position is normalized travel, not a boolean grasp sensor.
You receive unannotated RGB, optical-Z depth visualization, uniform depth samples,
intrinsics and camera-to-base transform, measured joints and FK pinch/TCP poses.
Only your selected pixel may be resolved to a 3D point. No object detector or
simulator object coordinates are provided. Select a visible interior surface pixel
using grounding={arm,pixel_uv_1000,offset_base_m,phase}. phase='inspect' returns
the surface point WITHOUT moving either arm; phase='target' executes the pose.
When locating a new object, inspect a visible pixel first, then select the arm
and grasp approach using that measured geometry. pixel_uv_1000 is [horizontal,
vertical] normalized to integers 0..1000 across the full RGB image. [0,0] is
TOP LEFT and [1000,1000] is BOTTOM RIGHT. The host converts to sensor pixels.
Its measured 3D surface plus YOUR base-frame offset replaces that arm's pinch
position. Identify objects in the FIRST (RGB) image; the second image is only
a grayscale depth encoding. Use null grounding to command explicit positions
from your own earlier observations and geometry, especially when the hand
occludes the object. Do not ground an occluding hand as if it were the object.
You must infer
object dimensions and appropriate clearance from images/depth, not assume a table
height or cube pose. Use the supplied camera geometry and depth samples to reason.
For a horizontal top face, a top-down grasp center is below its visible top surface
by about half its visually estimated height. Reobserve before closing. A rotation
can sweep the wrist or fingers: gain clearance before rotating or translating near
the table. Reachability is not collision clearance. Reported stalled closing alone
does not establish a grasp; verify visually before lifting and after lifting.
duration_s is the duration of this single pose command, not an automatic task phase.
Allow 4-6 seconds for substantial translations or rotations. If execution.arrived
is false, motion did not finish: inspect the measured errors, and repeat or revise
the target before progressing. Do not close or claim a lift while preceding pose
errors remain large. Consider another reachable orientation/arm if repeated
commands do not converge. A timeout does not mean the hand arrived.
After the interval the executor holds measured arm poses and preserves gripper
requests during your next decision. Use intermediate poses when necessary.
done=true requests final observation/retention verification; it does not assert
evaluator success. Describe only a brief observable decision in decision_summary.
"""


@dataclass(frozen=True)
class Decision:
    action: Action | None
    duration_s: float
    done: bool
    summary: str
    grounding: dict | None


def parse_decision(value, frame):
    if not isinstance(value, dict) or set(value) != set(OUTPUT_SCHEMA["required"]):
        raise ValueError("Missing or unexpected policy output fields")
    duration = value["duration_s"]
    if (
        type(duration) not in (int, float)
        or not np.isfinite(duration)
        or not 0.2 <= duration <= 6
    ):
        raise ValueError("Invalid action duration")
    if type(value["done"]) is not bool or not isinstance(
        value["decision_summary"], str
    ):
        raise ValueError("Invalid decision status/summary")
    pinch = Action.from_dict({s: value[s] for s in SIDES})
    arms = {s: getattr(pinch, s) for s in SIDES}
    grounding = value["grounding"]
    resolved = None
    if grounding is not None:
        if (
            not isinstance(grounding, dict)
            or set(grounding) != {"arm", "pixel_uv_1000", "offset_base_m", "phase"}
            or grounding["arm"] not in SIDES
            or grounding["phase"] not in ("inspect", "target")
        ):
            raise ValueError("Invalid grounding fields")
        uv = grounding["pixel_uv_1000"]
        if len(uv) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in uv):
            raise ValueError(
                "Expected normalized integer pixel coordinates in [0,1000]"
            )
        h, w = frame.depth_m.shape
        pixel = [round(uv[0] * (w - 1) / 1000), round(uv[1] * (h - 1) / 1000)]
        position, surface = ground_pixel(frame, pixel, grounding["offset_base_m"])
        side = grounding["arm"]
        arms[side] = ArmTarget(
            Pose(position, arms[side].pose.quaternion_xyzw), arms[side].gripper_open
        )
        resolved = {
            **grounding,
            "pixel_uv": pixel,
            "surface_base_m": surface.tolist(),
            "pinch_base_m": position.tolist(),
        }
        if grounding["phase"] == "inspect":
            return Decision(
                None, float(duration), False, value["decision_summary"], resolved
            )
    action = Action(
        *(ArmTarget(tcp_pose(arms[s].pose), arms[s].gripper_open) for s in SIDES)
    )
    return Decision(
        action, float(duration), value["done"], value["decision_summary"], resolved
    )


def packet(observation, frame, instruction, execution, available_arms):
    frame.validate()
    if abs(observation.timestamp_s - frame.timestamp_s) > 0.1:
        raise ValueError("Camera/encoder acquisition skew exceeds 100 ms")
    robot = observation.to_dict()
    # Explicit boundary allowlist: no arbitrary backend metadata reaches the model.
    fields = (
        "observation_id",
        "timestamp_s",
        "joint_names",
        "joint_positions_rad",
        "joint_velocities_rad_s",
        "gripper_position",
        "gripper_states",
        "left_tcp",
        "right_tcp",
        "mode",
        "reason",
    )
    measured = {k: robot[k] for k in fields}
    pinches = {
        s: {
            "position_m": pinch_pose(getattr(observation, f"{s}_tcp")).position_m,
            "quaternion_xyzw": getattr(observation, f"{s}_tcp").quaternion_xyzw,
        }
        for s in SIDES
    }
    depth = quantize_depth(frame.depth_m).astype(float) / 1000
    samples = [
        [u, v, float(depth[v, u])]
        for v in range(10, depth.shape[0], 30)
        for u in range(10, depth.shape[1], 30)
    ]
    return {
        "instruction": instruction,
        "robot": measured,
        "pinch_poses": pinches,
        "camera": frame.metadata(),
        "depth_samples_uv_m": samples,
        "execution": execution,
        "available_arms": list(available_arms),
        "tool_geometry": {
            "pinch_in_tcp_m": PINCH_IN_TCP_M.tolist(),
            "status": "nominal_CAD",
            "top_down_quaternion_xyzw": TOP_DOWN_XYZW,
        },
    }


def image_part(rgb):
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Image encoding failed")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(encoded).decode()
        },
    }


class OpenRouterPolicy:
    def __init__(
        self,
        directory,
        key_path,
        model="google/gemini-3.8-flash",
        max_calls=24,
        max_cost=5.0,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.key_path, self.model = Path(key_path).expanduser(), model
        self.max_calls, self.max_cost = max_calls, max_cost
        self.history, self.calls, self.cost = [], 0, 0.0

    def act(self, observation, frame, instruction, execution, available_arms):
        if self.calls >= self.max_calls or self.cost >= self.max_cost:
            raise RuntimeError("Policy request/cost budget exhausted")
        metadata = packet(observation, frame, instruction, execution, available_arms)
        save_frame(frame, self.directory, self.calls)
        depth_view = cv2.cvtColor(
            np.uint8(np.clip(frame.depth_m / 1.5, 0, 1) * 255), cv2.COLOR_GRAY2BGR
        )
        depth_view[frame.depth_m <= 0] = 0
        content = [
            {"type": "text", "text": json.dumps(metadata)},
            image_part(frame.rgb),
            {
                "type": "text",
                "text": "Depth visualization: fixed scale 0 to 1.5 m; use numeric depth samples for metric values.",
            },
            image_part(cv2.cvtColor(depth_view, cv2.COLOR_BGR2RGB)),
        ]
        messages = [
            {
                "role": "system",
                "content": SYSTEM + "\nJSON schema: " + json.dumps(OUTPUT_SCHEMA),
            }
        ]
        messages += self.history[-16:] + [{"role": "user", "content": content}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 8192,
            "temperature": 0.1,
            "reasoning": {"effort": "low"},
            "response_format": {"type": "json_object"},
        }
        # Archive exact model input, excluding HTTP headers and credentials.
        index = self.calls
        (self.directory / f"request_{index:03}.json").write_text(
            json.dumps(payload) + "\n"
        )
        self.calls += 1
        key = self.key_path.read_text().strip()
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        del key
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"OpenRouter HTTP {error.code}; response body omitted"
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError("OpenRouter connection/timeout failure") from None
        # Do not persist hidden reasoning fields or transport response headers.
        answer = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        self.cost += float(usage.get("cost", 0) or 0)
        record = {
            "model": result.get("model"),
            "id": result.get("id"),
            "content": answer,
            "usage": usage,
        }
        (self.directory / f"response_{index:03}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        # History contains the model's own outputs and sensor metadata, not truth.
        self.history += [
            {"role": "user", "content": json.dumps({"previous_execution": execution})},
            {"role": "assistant", "content": answer},
        ]
        return parse_decision(json.loads(answer), frame)
