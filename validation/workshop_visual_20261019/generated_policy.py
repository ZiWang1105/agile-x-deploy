import cv2
import numpy as np


def run(robot):
    # Capture fresh observation and camera frame
    cap = robot.capture()
    frame_id = cap["frame_id"]

    # Decode RGB image to locate the red cube and green tray
    import base64

    rgb_bytes = base64.b64decode(cap["rgb_png_base64"])
    nparr = np.frombuffer(rgb_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red mask (red wraps around 0/180)
    mask_r1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(
        hsv, np.array([160, 100, 100]), np.array([180, 255, 255])
    )
    mask_red = mask_r1 | mask_r2

    # Green mask
    mask_green = cv2.inRange(
        hsv, np.array([35, 100, 100]), np.array([85, 255, 255])
    )

    # Find centroids
    m_red = cv2.moments(mask_red)
    u_red = int(m_red["m10"] / (m_red["m00"] + 1e-6))
    v_red = int(m_red["m01"] / (m_red["m00"] + 1e-6))

    m_green = cv2.moments(mask_green)
    u_green = int(m_green["m10"] / (m_green["m00"] + 1e-6))
    v_green = int(m_green["m01"] / (m_green["m00"] + 1e-6))

    print(
        f"Red cube pixel: ({u_red}, {v_red}), Green tray pixel: ({u_green}, {v_green})"
    )

    # Deproject
    dep_red = robot.deproject(frame_id, [u_red, v_red])
    dep_green = robot.deproject(frame_id, [u_green, v_green])

    print("Deprojected red:", dep_red)
    print("Deprojected green:", dep_green)

    # Positions in base frame
    # dep_red["position_m"] or similar structure
    p_red = dep_red["position_m"]
    p_green = dep_green["position_m"]

    # Choose arm based on cube base Y (or average Y)
    # left arm for positive Y, right arm for negative Y
    arm = "left" if p_red[1] >= 0 else "right"
    print(f"Selected arm: {arm} based on Y={p_red[1]}")

    # Top-down pinch quaternion
    # Standard top-down pinch pose: [0.5, -0.5, -0.5, -0.5]
    top_down_quat = [0.5, -0.5, -0.5, -0.5]

    # Target heights:
    # A cube pinch center is about 25 mm below its top face.
    # A tray placement pinch center is about 31 mm above its inner floor.
    # Vertical clearance: ~120 mm (0.12 m)
    cube_pinch_z = p_red[2] - 0.025
    tray_pinch_z = p_green[2] + 0.031
    clearance_z = 0.12

    # Poses
    pre_grasp_pose = {
        "position_m": [p_red[0], p_red[1], cube_pinch_z + clearance_z],
        "quaternion_xyzw": top_down_quat,
    }
    grasp_pose = {
        "position_m": [p_red[0], p_red[1], cube_pinch_z],
        "quaternion_xyzw": top_down_quat,
    }
    lift_pose = {
        "position_m": [p_red[0], p_red[1], cube_pinch_z + clearance_z],
        "quaternion_xyzw": top_down_quat,
    }
    pre_place_pose = {
        "position_m": [p_green[0], p_green[1], tray_pinch_z + clearance_z],
        "quaternion_xyzw": top_down_quat,
    }
    place_pose = {
        "position_m": [p_green[0], p_green[1], tray_pinch_z],
        "quaternion_xyzw": top_down_quat,
    }
    post_place_pose = {
        "position_m": [p_green[0], p_green[1], tray_pinch_z + clearance_z],
        "quaternion_xyzw": top_down_quat,
    }

    # 1. Open gripper
    print("Opening gripper...")
    op = robot.gripper_open(arm)
    res = op.wait()
    print("Gripper open result:", res["state"])

    # 2. Move above cube
    print("Moving above cube...")
    op = robot.move_tcp(arm, pre_grasp_pose, tool_frame="pinch")
    res = op.wait()
    print("Move pre-grasp:", res["state"])

    # 3. Descend to cube
    print("Descending to grasp...")
    op = robot.move_tcp(arm, grasp_pose, tool_frame="pinch")
    res = op.wait()
    print("Descend grasp:", res["state"])

    # 4. Close gripper
    print("Closing gripper...")
    op = robot.gripper_close(arm)
    res = op.wait()
    print("Gripper close result:", res["state"])

    # 5. Lift cube
    print("Lifting cube...")
    op = robot.move_tcp(arm, lift_pose, tool_frame="pinch")
    res = op.wait()
    print("Lift result:", res["state"])

    # 6. Transfer above tray
    print("Transferring above tray...")
    op = robot.move_tcp(arm, pre_place_pose, tool_frame="pinch")
    res = op.wait()
    print("Transfer result:", res["state"])

    # 7. Lower to tray
    print("Lowering to place...")
    op = robot.move_tcp(arm, place_pose, tool_frame="pinch")
    res = op.wait()
    print("Lower result:", res["state"])

    # 8. Open gripper
    print("Opening gripper to release...")
    op = robot.gripper_open(arm)
    res = op.wait()
    print("Gripper open result:", res["state"])

    # 9. Retract
    print("Retracting arm...")
    op = robot.move_tcp(arm, post_place_pose, tool_frame="pinch")
    res = op.wait()
    print("Retract result:", res["state"])

    # Also move back toward home/standby to fully retract
    obs = robot.observe()
    initial_pinch = obs["pinch_poses"][arm]
    # Retract further away
    retract_home = {
        "position_m": [
            0.36,
            0.30 if arm == "left" else -0.30,
            post_place_pose["position_m"][2],
        ],
        "quaternion_xyzw": top_down_quat,
    }
    op = robot.move_tcp(arm, retract_home, tool_frame="pinch")
    res = op.wait()
    print("Final retract result:", res["state"])

    # Capture final image to show the result
    robot.capture()
    robot.hold()
    return "completed"
