"""Replay the recorded gripper/IK/motion sequence through the public MCP API.

No CAN devices, physical robot, model API, or API key are used. The recorded
posture predates the failed control tick; this is not an exact incident replay.
Contact is disabled because the physical table geometry was not recorded.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

from agilex_deploy.agent_sim import make_scene, render_video
from agilex_deploy.backends import SimBackend
from agilex_deploy.contract import Pose
from agilex_deploy.controller import pose_error
from agilex_deploy.model import ROOT
from agilex_deploy.agent_policy import pinch_pose
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService
from agilex_deploy.workshop_tasks import scene_for

sys.path.insert(0, str(ROOT / "examples/workshop"))
from mcp_baseline import MCPRobotClient


class ReplayPlant(SimBackend):
    def __init__(self, q):
        self.scene_config = scene_for()
        self.scene_config["objects"] = []
        self.scene_config["trays"] = []
        self.scene_config["table"]["center_m"][2] = -.04
        model, _ = make_scene(self.scene_config)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        super().__init__(q, model=model)
        self.truth = []
        self.video_frames = []
        self.initial_q = np.array(q)
        self.max_right_drift = 0.
        self.ticks = 0

    def read(self):
        # NERO angle frames are quantized in 0.001 degrees. Apply that same
        # observation resolution, without changing simulated plant positions.
        from dataclasses import replace
        sample = super().read()
        step = np.deg2rad(.001)
        return replace(sample, positions=tuple(np.round(np.array(sample.positions) / step) * step))

    def advance(self):
        super().advance()
        self.ticks += 1
        self.max_right_drift = max(self.max_right_drift,
            float(np.max(abs(self.data.qpos[self.qids][7:] - self.initial_q[7:]))))
        if self.ticks % 10 == 0:
            self.video_frames.append((float(self.data.time), self.data.qpos.copy()))

    def evaluate(self):
        return dict(simulation_only=True, contacts_enabled=False,
                    max_right_joint_drift_rad=self.max_right_drift)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/validation/ik_deployment')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads((ROOT / 'tests/fixtures/ik_start_20261020.json').read_text())
    plant = ReplayPlant(fixture['joint_positions_rad'])
    service = server = None
    results = []
    started = time.monotonic()
    try:
        with patch('agilex_deploy.workshop_tasks.WorkshopSim', return_value=plant):
            service = WorkshopService(output=args.output)
        server = WorkshopHTTP(service, 0, args.output / 'client.json')
        service.enable()
        with MCPRobotClient(args.output / 'client.json') as robot:
            frame = robot.capture()
            before = robot.observe()
            target = fixture['pinch_target']
            result = robot.gripper_open('left').wait()
            assert result['state'] == 'succeeded', result
            results.append(dict(command='gripper_open', result=result))
            ik = robot.ik('left', target, tool_frame='pinch')
            assert ik['success'], ik
            results.append(dict(command='kinematics_ik', result=ik))
            result = robot.move_tcp('left', target, tool_frame='pinch', timeout_s=12).wait()
            assert result['state'] == 'succeeded', result
            results.append(dict(command='motion_move_tcp', result=result))
            after = robot.observe()
            error = pose_error(pinch_pose(Pose(**after['left_tcp'])), Pose(**target))
            assert np.linalg.norm(error[:3]) < .005, error
            assert np.linalg.norm(error[3:]) < .05, error
            assert plant.max_right_drift < .001, plant.max_right_drift
            assert abs(after['gripper_position'][1] - before['gripper_position'][1]) < .001
            report = dict(simulation_only=True, transport='stdio_mcp',
                recorded_start=True, exact_failed_tick_available=False,
                contacts_enabled=False, frame_id=frame['frame_id'],
                wall_duration_s=time.monotonic()-started,
                final_position_error_m=float(np.linalg.norm(error[:3])),
                final_orientation_error_rad=float(np.linalg.norm(error[3:])),
                max_right_joint_drift_rad=plant.max_right_drift,
                final_grippers=after['gripper_position'], operations=results)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k != 'operations'}), flush=True)
    finally:
        if server:
            server.close()
        if service:
            service.close()
    if args.video:
        render_video(plant, args.output / 'recorded_start_mcp.mp4')


if __name__ == '__main__':
    main()
