"""RGB-D color baseline using ONLY the public participant SDK.

This intentionally simple baseline assumes the workshop's 50 mm colored cubes
and horizontal shallow trays. It reads no simulator state or scene files.
"""

import argparse
import base64
import json
import time

import cv2
import numpy as np

from agilex_deploy.workshop_client import RobotClient

TOP_DOWN = [0.5, -0.5, -0.5, -0.5]


def decode(frame):
    rgb = cv2.cvtColor(
        cv2.imdecode(
            np.frombuffer(base64.b64decode(frame["rgb_png_base64"]), np.uint8), 1
        ),
        cv2.COLOR_BGR2RGB,
    )
    depth = (
        cv2.imdecode(
            np.frombuffer(base64.b64decode(frame["depth_png_base64"]), np.uint8), -1
        )
        * frame["depth_png_scale_m"]
    )
    h, w = depth.shape
    v, u = np.indices((h, w))
    rays = np.linalg.solve(
        np.asarray(frame["K"]), np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
    )
    optical = rays * depth.ravel()
    transform = np.asarray(frame["T_base_from_optical"])
    points = (transform[:3, :3] @ optical + transform[:3, 3, None]).T.reshape(h, w, 3)
    return rgb, depth, points


def locate(frame, color, kind):
    rgb, depth, points = decode(frame)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    masks = {
        "red": (hue < 12) | (hue > 170),
        "blue": (hue > 100) & (hue < 135),
        "green": (hue > 35) & (hue < 90),
    }
    mask = masks[color] & (saturation > 100) & (value > 50) & (depth > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    regions = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 12:
            continue
        cloud = points[labels == i]
        regions.append(cloud)
    if not regions:
        raise RuntimeError(f"No visible {color} {kind}; capture a clear head view")
    if kind == "cube":
        cloud = max(regions, key=lambda c: np.percentile(c[:, 2], 80))
        top_z = np.percentile(cloud[:, 2], 85)
        surface = cloud[abs(cloud[:, 2] - top_z) < 0.003]
    else:
        cloud = max(regions, key=len)
        floor_z = np.percentile(cloud[:, 2], 20)
        surface = cloud[abs(cloud[:, 2] - floor_z) < 0.003]
    if len(surface) < 5:
        raise RuntimeError(f"Insufficient visible surface for {color} {kind}")
    return np.median(surface, axis=0)


def complete(operation):
    result = operation.wait()
    print(json.dumps(result), flush=True)
    if result["state"] != "succeeded":
        raise RuntimeError(f"Operation failed: {result}")
    return result


def run_policy(robot, instruction):
    assignments = (
        [("red", "red"), ("blue", "blue")]
        if "red tray" in instruction
        else [("blue" if "blue cube" in instruction else "red", "green")]
    )
    frame = robot.capture()
    targets = [
        (cube, locate(frame, cube, "cube"), locate(frame, tray, "tray"))
        for cube, tray in assignments
    ]
    for color, top, tray in targets:
        arm = "left" if top[1] >= 0 else "right"
        center = top.copy()
        center[2] -= 0.025
        print(
            json.dumps(
                dict(
                    color=color,
                    arm=arm,
                    visual_top_m=top.tolist(),
                    visual_tray_m=tray.tolist(),
                )
            ),
            flush=True,
        )

        def move(point):
            complete(
                robot.move_tcp(
                    arm,
                    dict(position_m=list(map(float, point)), quaternion_xyzw=TOP_DOWN),
                    tool_frame="pinch",
                    timeout_s=12,
                )
            )

        complete(robot.gripper_open(arm))
        move(center + [0, 0, 0.12])
        move(center)
        complete(robot.gripper_close(arm))
        move(center + [0, 0, 0.12])
        # Reobserve the object after lifting. The baseline leaves visual grasp
        # verification to the episode record/scorer, not the gripper stall flag.
        robot.capture()
        destination = np.array([tray[0], tray[1], tray[2] + 0.025 + 0.006])
        move(destination + [0, 0, 0.12])
        move(destination)
        complete(robot.gripper_open(arm))
        move(destination + [0, 0, 0.12])
    # Keep observing while physical contacts settle; no simulator stepping API.
    until = time.monotonic() + 2
    while time.monotonic() < until:
        robot.observe()
        time.sleep(0.1)
    robot.capture()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="logs/workshop/client.json")
    args = parser.parse_args()
    with RobotClient.from_config(args.config) as robot:
        run_policy(robot, robot.capabilities()["instruction"])


if __name__ == "__main__":
    main()
