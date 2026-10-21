"""Offline NERO firmware/CAN model coupled to MuJoCo, with no real sockets.

Models the observed 1.21 stream behavior, not the physical controller's unknown
gravity-compensation accuracy. Uses the production CAN decoder and transports.
"""
import struct
import time

import numpy as np

from agilex_deploy.backends import SimBackend
from agilex_deploy.can_feedback import FeedbackAssembler, ANGLE_IDS, LEADER_ANGLE_IDS
from agilex_deploy.contract import SIDES
from agilex_deploy.motion_limits import JointLimits
from test_dual_control import BusPlant


class FirmwareSim(BusPlant):
    @staticmethod
    def motion_limits(channel, model):
        # Both physical controllers reported these bounds in the incident log.
        return JointLimits(model, np.array([3.14]*4+[3.92]*3), np.array([2., 2.]+[2.5]*5))

    def __init__(self):
        super().__init__()
        self.plant = SimBackend()
        self.push = dict.fromkeys(SIDES, True)
        self.assembler = FeedbackAssembler(motor_feedback=True, gripper_feedback=SIDES,
                                           discover_leaders=True)
        self.targets = self.q.copy()
        self.timeline = []
        self.drop_leader = set()
        self.drop_driver = set()
        self.ignored_disabled_frames = []
        # Fault injection separates protocol mode from physical servo behavior.
        self.leader_servo_active = set()
        self.leader_gravity_factor = dict.fromkeys(SIDES, 1.0)
        self.last_step = time.monotonic()

    def set_angle_source(self, side, source, since):
        super().set_angle_source(side, source, since)
        self.assembler.set_angle_source(side, source, since)

    def snapshot(self):
        if time.monotonic() - self.last_step >= .01:
            for i, side in enumerate(SIDES):
                ids = self.plant.arm_actuators[i*7:i*7+7]
                active = (self.device_modes[side] == "follower" or side in self.leader_servo_active) and all(self.enabled[i*7:i*7+7])
                self.plant.model.actuator_gainprm[ids] = self.plant.arm_gains[i*7:i*7+7] if active else 0
                self.plant.model.actuator_biasprm[ids] = self.plant.arm_bias[i*7:i*7+7] if active else 0
                self.plant.data.ctrl[ids] = self.targets[i*7:i*7+7]
                factor = self.leader_gravity_factor[side] if self.device_modes[side] == "leader" else 1.0
                for body in range(self.plant.model.nbody):
                    name = self.plant.model.body(body).name or ""
                    if name.startswith(side + "_"):
                        self.plant.model.body_gravcomp[body] = factor
            self.plant.advance()
            self.last_step = time.monotonic()
        self.q = np.asarray(self.plant.read().positions)
        now = time.monotonic()
        for i, side in enumerate(SIDES):
            q = np.rint(np.degrees(self.q[i*7:i*7+7])*1000).astype(int)
            leader = self.device_modes[side] == "leader"
            # Observed leader mode publishes joint packets but suppresses all
            # normal motor, status, and gripper telemetry, even after push=1.
            if leader and side not in self.drop_leader:
                for j, cid in enumerate(LEADER_ANGLE_IDS):
                    values = q[j*2:j*2+2].tolist()
                    self.assembler.feed(side, cid, struct.pack(">ii", *(values + [0])[:2]), now)
            if not leader and self.push[side]:
                if side not in self.drop_driver:
                    for j in range(7):
                        self.assembler.feed(side, 0x261+j, bytes(5)+bytes([self.enabled[i*7+j]])+bytes(2), now)
                for j in range(7):
                    velocity = self.plant.data.qvel[self.plant.dids[i*7+j]]
                    self.assembler.feed(side, 0x251+j, struct.pack(">h", round(velocity*1000))+bytes(6), now)
                self.assembler.feed(side, 0x2A1, bytes([1,0,1])+bytes(5), now)
                self.assembler.feed(side, 0x2A8, struct.pack(">ihBB", 20000 if i == 0 else 40000,0,0x40,0), now)
                for j, cid in enumerate(ANGLE_IDS):
                    values = q[j*2:j*2+2].tolist()
                    self.assembler.feed(side, cid, struct.pack(">ii", *(values + [0])[:2]), now)
        self.timeline.append({"time_s": now, "modes": dict(self.device_modes),
                              "q": self.q.tolist(), "push": dict(self.push)})
        return self.assembler.snapshot(now)

    def read(self, timeout=0, **kwargs):
        deadline = time.monotonic()+timeout
        while True:
            try:
                return self.snapshot()
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.002)

    def socket(self, *args):
        writer = super().socket(*args)
        original = writer.send

        def send(packet):
            cid, _, data = struct.unpack("=IB3x8s", packet)
            side = writer.channel
            i = SIDES.index(side)
            before = self.enabled.copy()
            # The official SDK documents that control/configuration commands
            # do not take effect while disabled. Model the conservative case
            # that they are ignored, not accepted as active targets. The real
            # firmware's buffering behavior is still unverified.
            if cid in (*LEADER_ANGLE_IDS, 0x151, 0x470) and not all(self.enabled[i*7:i*7+7]):
                self.frames.append((side, cid, data))
                self.ignored_disabled_frames.append((side, cid, data))
                return 16
            result = original(packet)
            if cid == 0x151 and data[6] in (1, 2):
                self.push[side] = data[6] == 1
            if cid == 0x471 and self.device_modes[side] == "leader":
                self.enabled = before  # Official disable can be ignored here.
            if cid in LEADER_ANGLE_IDS:
                j = LEADER_ANGLE_IDS.index(cid)*2
                values = struct.unpack(">ii", data)[:min(2, 7-j)]
                self.targets[i*7+j:i*7+j+len(values)] = np.radians(np.array(values)*.001)
            return result
        writer.send = send
        return writer
