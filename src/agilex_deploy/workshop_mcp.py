"""Thin MCP adapter over the exact participant HTTP SDK; never opens hardware."""

import argparse
import json
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent

from .workshop_client import RobotClient


def build_server(client):
    mcp = FastMCP(
        "AgileX Workshop",
        instructions=(
            "Use one head RGB-D camera, official IK, and independent left/right grippers. "
            "Claim the session first. Operator enables physical motion in the dashboard. "
            "Every motion requires a unique request_id; retry the SAME request with the SAME ID. "
            "Poll operation_get until terminal. A settled gripper is not proof of a grasp. "
            "All positions are base_link meters and XYZW quaternions. No collision planner is provided."
        ),
    )

    @mcp.tool()
    def session_claim() -> dict:
        """Claim exclusive participant ownership. Does not enable physical motors."""
        return client.claim()

    @mcp.tool()
    def session_release() -> dict:
        """Release ownership and request HOLD. Motors remain enabled."""
        client.close()
        return {"released": True}

    @mcp.tool()
    def robot_capabilities() -> dict:
        """Read API conventions, available arms/camera, and task instruction."""
        return client.capabilities()

    @mcp.tool()
    def robot_observe() -> dict:
        """Read measured robot state, separate gripper states, and TCP/pinch poses."""
        return client.observe()

    @mcp.tool()
    def camera_capture() -> CallToolResult:
        """Capture the single head camera. RGB image plus registered depth/calibration and immutable frame_id."""
        frame = client.capture()
        metadata = {k: v for k, v in frame.items() if not k.endswith("_base64")}
        return CallToolResult(
            content=[
                ImageContent(
                    type="image", data=frame["rgb_png_base64"], mimeType="image/png"
                ),
                TextContent(type="text", text=json.dumps(metadata)),
            ],
            structuredContent=frame,
        )

    @mcp.tool()
    def camera_deproject(frame_id: str, pixel_uv: list[int]) -> dict:
        """Resolve [u,v] in the specified RGB-D frame to base_link meters. Returns acquisition age."""
        return client.deproject(frame_id, pixel_uv)

    @mcp.tool()
    def kinematics_fk(
        arm: Literal["left", "right"],
        joints_rad: list[float],
        tool_frame: Literal["tcp", "pinch"] = "tcp",
    ) -> dict:
        """Compute FK for seven joint angles, without moving the robot."""
        return client.fk(arm, joints_rad, tool_frame)

    @mcp.tool()
    def kinematics_ik(
        arm: Literal["left", "right"],
        target: dict,
        seed_rad: list[float] | None = None,
        tool_frame: Literal["tcp", "pinch"] = "tcp",
    ) -> dict:
        """Official AgileX IK only: target {position_m:[x,y,z], quaternion_xyzw:[x,y,z,w]}. Returns residuals; does not move."""
        return client.ik(arm, target, seed_rad, tool_frame)

    @mcp.tool()
    def motion_move_tcp(
        arm: Literal["left", "right"],
        target: dict,
        request_id: str,
        tool_frame: Literal["tcp", "pinch"] = "tcp",
        timeout_s: float = 10,
    ) -> dict:
        """Move one arm toward a TCP/pinch pose. Other arm and gripper targets are unchanged. Poll returned operation_id."""
        return client.call(
            "motion_move_tcp",
            arm=arm,
            target=target,
            request_id=request_id,
            tool_frame=tool_frame,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def motion_move_both(
        targets: dict,
        request_id: str,
        tool_frame: Literal["tcp", "pinch"] = "tcp",
        timeout_s: float = 10,
    ) -> dict:
        """Explicit paired move: targets has left/right pose objects. Does not change either gripper."""
        return client.call(
            "motion_move_both",
            targets=targets,
            request_id=request_id,
            tool_frame=tool_frame,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def motion_move_joints(
        arm: Literal["left", "right"],
        joints_rad: list[float],
        request_id: str,
        timeout_s: float = 10,
    ) -> dict:
        """Command seven joint angles on one arm with shared joint limits. Other arm and grippers retain targets."""
        return client.call(
            "motion_move_joints",
            arm=arm,
            joints_rad=joints_rad,
            request_id=request_id,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def motion_follow_waypoints(
        arm: Literal["left", "right"],
        targets: list[dict],
        request_id: str,
        tool_frame: Literal["tcp", "pinch"] = "tcp",
        timeout_s: float = 30,
    ) -> dict:
        """Execute ordered poses, awaiting each endpoint. timeout_s is total time. No collision planning."""
        return client.call(
            "motion_follow_waypoints",
            arm=arm,
            targets=targets,
            request_id=request_id,
            tool_frame=tool_frame,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    def gripper_open(
        arm: Literal["left", "right"], request_id: str, timeout_s: float = 5
    ) -> dict:
        """Open ONLY the named gripper. The other jaw retains its target. Poll operation_get."""
        return client.call(
            "gripper_open", arm=arm, request_id=request_id, timeout_s=timeout_s
        )

    @mcp.tool()
    def gripper_close(
        arm: Literal["left", "right"], request_id: str, timeout_s: float = 5
    ) -> dict:
        """Close ONLY the named gripper with configured official force setting. Settling is not grasp confirmation."""
        return client.call(
            "gripper_close", arm=arm, request_id=request_id, timeout_s=timeout_s
        )

    @mcp.tool()
    def operation_get(operation_id: str) -> dict:
        """Read running/succeeded/failed/cancelled, residuals, and measured gripper progress."""
        return client.call("operation_get", operation_id=operation_id)

    @mcp.tool()
    def operation_wait(operation_id: str, timeout_s: float = 20) -> dict:
        """Wait up to 20 seconds for an operation. Returns running if the wait expires; never claims success from time alone."""
        return client.call(
            "operation_wait", operation_id=operation_id, timeout_s=timeout_s
        )

    @mcp.tool()
    def operation_cancel(operation_id: str) -> dict:
        """Cancel this operation and request HOLD; no motor disable."""
        return client.call("operation_cancel", operation_id=operation_id)

    @mcp.tool()
    def robot_hold() -> dict:
        """Cancel motion and request HOLD on both arms/grippers. No motor disable."""
        return client.hold()

    @mcp.resource("workshop://api")
    def api_reference() -> str:
        return json.dumps(client.capabilities())

    @mcp.prompt()
    def workshop_task(instruction: str) -> str:
        return (
            "Complete this task using observed head RGB-D and robot feedback: "
            + instruction
            + " Inspect before moving. Await operations; verify pickup visually; release into the target and retract. "
            "Cubes are 50 mm wide. Use pinch frame for fingertip midpoint targets. "
            "Change only the intended gripper. Never claim success solely from a completed close command."
        )

    return mcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="logs/workshop/client.json")
    args = parser.parse_args()
    client = RobotClient.from_config(args.config)
    try:
        build_server(client).run(transport="stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
