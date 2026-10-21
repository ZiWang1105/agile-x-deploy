"""Replay the 20:22 code-policy approach through Python and MCP, without CAN I/O.

Use the recorded pre-motion posture and exact requested endpoint, not invented
initial conditions. Simulate millidegree encoder/wire resolution and enforce the
real transport's position/speed/acceleration guards. Contacts are disabled because
the physical table and surrounding people were not recorded; this validates
control and API behavior, not collision clearance or physical deployment safety.
"""
import argparse
import json
from pathlib import Path
import struct
import sys
import time
from unittest.mock import patch

import numpy as np

from agilex_deploy.contract import Pose
from agilex_deploy.controller import pose_error
from agilex_deploy.dual_transport import DualMotionGuard
from agilex_deploy.model import ROOT
from agilex_deploy.nero_transport import position_frames
from agilex_deploy.workshop_client import RobotClient
from agilex_deploy.workshop_http import WorkshopHTTP
from agilex_deploy.workshop_service import WorkshopService
from agilex_deploy.agent_policy import pinch_pose
from validate_ik_deployment import ReplayPlant

sys.path.insert(0, str(ROOT / 'examples/workshop'))
from mcp_baseline import MCPRobotClient


class GuardedReplayPlant(ReplayPlant):
    def __init__(self, q):
        self.guard = None
        self.command_count = 0
        self.max_velocity_fraction = self.max_acceleration_fraction = 0.
        super().__init__(q)
        self.guard = DualMotionGuard.simulation(self.read().positions, self.kinematics.limits)

    def write(self, q, grippers):
        if self.guard is not None:
            for guard in self.guard.guards.values():
                velocity = (np.asarray(q) - guard.previous) / .01
                selected = guard.indices
                self.max_velocity_fraction = max(self.max_velocity_fraction,
                    float(np.max(abs(velocity[selected]) / guard.motion_limits.velocity)))
                self.max_acceleration_fraction = max(self.max_acceleration_fraction,
                    float(np.max(abs(velocity[selected] - guard.velocity[selected])
                                 / .01 / guard.motion_limits.acceleration)))
            self.guard.command(q, self.read().positions)
            wire = []
            for i in (0, 7):
                frames = position_frames(q[i:i+7], self.kinematics.limits[i:i+7])
                millidegrees = [v for _, data in frames for v in struct.unpack('>ii', data)][:7]
                wire.extend(np.radians(np.asarray(millidegrees) * .001))
            q = np.asarray(wire)
            assert np.all(q >= self.kinematics.limits[:, 0])
            assert np.all(q <= self.kinematics.limits[:, 1])
            self.command_count += 1
        super().write(q, grippers)

    def hold(self, *, preserve_grippers=False):
        # HOLD rebases both production guards and the controller from measured
        # feedback; it intentionally stops pursuing the old trajectory.
        if self.guard is not None:
            self.guard.hold(self.read().positions)
        super().hold(preserve_grippers=preserve_grippers)


def run(transport, fixture, output):
    plant = GuardedReplayPlant(fixture['joint_positions_rad'])
    service = server = None
    try:
        with patch('agilex_deploy.workshop_tasks.WorkshopSim', return_value=plant):
            service = WorkshopService(output=output)
        server = WorkshopHTTP(service, 0, output / 'client.json')
        service.enable()
        client = (RobotClient(server.endpoint, server.token) if transport == 'python'
                  else MCPRobotClient(output / 'client.json'))
        with client as robot:
            before = robot.observe()
            opened = robot.gripper_open('left').wait()
            assert opened['state'] == 'succeeded', opened
            target = fixture['pinch_target']
            ik = robot.ik('left', target, tool_frame='pinch')
            assert ik['solver_success'] and not ik['success'], ik
            started = time.monotonic()
            moved = robot.move_tcp('left', target, tool_frame='pinch', timeout_s=12).wait()
            elapsed = time.monotonic() - started
            assert moved['state'] == 'succeeded', moved
            assert moved['planning']['arms']['left']['selected_arc'] == 'alternate', moved
            after = robot.observe()
            residual = pose_error(pinch_pose(Pose(**after['left_tcp'])), Pose(**target))
            assert np.linalg.norm(residual[:3]) <= .005, residual
            assert np.linalg.norm(residual[3:]) <= .05, residual
            assert plant.max_right_drift < .001, plant.max_right_drift
            assert abs(after['gripper_position'][1] - before['gripper_position'][1]) < .001
            # The incident failed when falling back to HOLD. Check idle writes
            # continue after a completed operation and explicit participant HOLD.
            robot.hold()
            time.sleep(.3)
            assert robot.observe()['mode'] != 'fault'
            assert service.fault is None
            report = dict(transport=transport, simulation_only=True, contacts_enabled=False,
                state=moved['state'], selected_arc='alternate', operation_wall_s=elapsed,
                position_error_m=float(np.linalg.norm(residual[:3])),
                orientation_error_rad=float(np.linalg.norm(residual[3:])),
                max_right_joint_drift_rad=plant.max_right_drift,
                max_velocity_fraction=plant.max_velocity_fraction,
                max_acceleration_fraction=plant.max_acceleration_fraction,
                checked_commands=plant.command_count, one_shot_ik=ik, operation=moved)
            (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k:v for k,v in report.items() if k not in ('one_shot_ik','operation')}), flush=True)
            return report
    finally:
        if server:
            server.close()
        if service:
            service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/validation/bound_incident')
    args = parser.parse_args()
    fixture = json.loads((ROOT / 'tests/fixtures/bound_start_20261020.json').read_text())
    results = []
    for transport in ('python', 'mcp'):
        results.append(run(transport, fixture, args.output / transport))
    (args.output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
