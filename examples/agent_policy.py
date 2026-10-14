"""A minimal policy using only the shared robot contract.

Run from the repository: .venv/bin/python examples/agent_policy.py
Replace the target selection with an agent later; keep the robot calls intact.
"""
import time

from agilex_deploy import Action, ArmTarget, Pose, RobotInterface


def move_both(robot: RobotInterface, seconds=3.):
    observation = robot.observe()
    targets = {}
    for side in ("left", "right"):
        pose = getattr(observation, f"{side}_tcp")
        position = list(pose.position_m)
        position[0] += .04
        position[2] += .02
        targets[side] = ArmTarget(Pose(position, pose.quaternion_xyzw), 1)
    action = Action(**targets)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            observation = robot.observe()
            robot.submit_target(action, observation.observation_id)
            time.sleep(.1)  # Refresh the 0.5 s target lease while policy owns motion.
        return robot.observe()
    finally:
        robot.hold()


if __name__ == "__main__":
    from agilex_deploy import Robot
    from agilex_deploy.backends import SimBackend
    with Robot(SimBackend()).start() as robot:
        print(move_both(robot).to_dict())
