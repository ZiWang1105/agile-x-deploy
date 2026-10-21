"""RobotInterface client with an owned control/watchdog process.

The viewer never owns a CAN writer. The child process observes feedback and
enforces a 200 ms operator heartbeat independently of the viewer's Python GIL.
Normal close holds position and joins the child. Hand-guiding is opt-in with
gravity_comp=True. Fault cleanup retains the last confirmed mode; neither path
disables motors. Supported motor disable is a separate explicit operation.
"""
from dataclasses import replace
import multiprocessing as mp
import signal
import time

import numpy as np

from .backends import RealBackend, SimBackend
from .contract import SIDES
from .model import CONTROL_DT, ROOT, Kinematics, initial_positions
from .official_ik import OfficialDualIK
from .nero_transport import NeroTransport, MotionGuard, validate_workspace_radius
from .runtime import Robot
from .dual_transport import DualNeroTransport, DualMotionGuard, selected_sides
from .hardware_interlock import require_qualified_hand_guiding

OPERATOR_TIMEOUT = .2


class SimulatedCommissioningTransport:
    """Explicit test plant for commissioning UI and process/watchdog tests."""
    def __init__(self, side, workspace_radius_m=None, gripper_control=False, **_):
        self.side = side
        self.sides = selected_sides(side)
        self.gripper_control = gripper_control
        self.gripper_active = set()
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.plant = SimBackend()
        self.armed = False
        self.guard = None
        self.closed = False
        self.control_mode = "follower"

    def read_encoders(self):
        sample = self.plant.read()
        return replace(sample, grippers=tuple(g if self.gripper_control and s in self.sides else None
                                             for s, g in zip(SIDES, sample.grippers)))

    def arm(self):
        sample = self.read_encoders()
        options = dict(workspace_radius_m=self.workspace_radius_m, gripper_control=self.gripper_control)
        self.guard = (DualMotionGuard.simulation(sample.positions, self.plant.kinematics.limits, **options)
                      if self.side == "both" else
                      MotionGuard(self.side, sample.positions, self.plant.kinematics.limits, **options))
        self.armed = True

    def activate_gripper(self, action):
        for side in self.sides:
            if getattr(action, side).gripper_open is not None:
                self.gripper_active.add(side)

    def send_positions(self, q, grippers):
        if self.armed:
            sample = self.read_encoders()
            self.guard.command(q, sample.positions)
            command = np.array(sample.positions)
            command[self.guard.indices] = q[self.guard.indices]
            grip = [float(self.plant.data.ctrl[self.plant.model.actuator(s + "_gripper").id] / .05)
                    for s in SIDES]
            for side in self.gripper_active:
                grip[SIDES.index(side)] = grippers[SIDES.index(side)]
            self.plant.write(command, grip)
            self.plant.advance()

    def hold(self, *, preserve_grippers=False):
        if self.guard:
            self.guard.hold(self.read_encoders().positions)
        self.plant.hold(preserve_grippers=preserve_grippers)

    def enter_hand_guiding(self):
        if self.armed:
            self.plant.enter_hand_guiding()
            self.control_mode = "hand_guiding"
            self.gripper_active.clear()

    def begin_control(self):
        self.plant.begin_control()
        self.guard.hold(self.read_encoders().positions)
        self.control_mode = "follower"

    def close(self):
        self.closed = True
        self.plant.close()


def control_worker(connection, side, mapping, commissioning, simulation, workspace_radius_m,
                   gripper_control=False, gripper_force_n=1.0, gravity_comp=False):
    # Terminal Ctrl+C goes to both processes. Let the viewer request HOLD and
    # close over IPC instead of interrupting the owner during a CAN operation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    transport = robot = None
    heartbeat = None
    interlock = None
    next_tick = time.monotonic()
    try:
        # Warm up before opening CAN, so solver construction cannot accumulate
        # a feedback backlog. The nominal warm-up pose is never sent to motors.
        ik = OfficialDualIK(Kinematics(), initial_positions())
        factory = SimulatedCommissioningTransport if simulation else (DualNeroTransport if side == "both" else NeroTransport)
        transport = factory(side=side, mapping=mapping, commissioning=commissioning,
                            workspace_radius_m=workspace_radius_m,
                            gripper_control=gripper_control, gripper_force_n=gripper_force_n)
        robot = Robot(RealBackend(transport), ik=ik)
        if gripper_control:
            robot.controller.gripper_rate = None
        connection.send((0, True, "ready"))
        while True:
            now = time.monotonic()
            if transport.armed and robot.mode != "hand_guiding" and heartbeat is not None and now - heartbeat > OPERATOR_TIMEOUT and interlock is None:
                interlock = "Operator heartbeat expired; HOLD latched. Restart required."
                try:
                    robot.hold()
                except Exception as error:
                    interlock += f" HOLD failed: {error}"
            if now >= next_tick:
                if robot.mode != "fault" and interlock is None:
                    try:
                        robot.tick()
                    except Exception as error:
                        interlock = robot.reason if robot.mode == "fault" else str(error)
                next_tick = time.monotonic() + CONTROL_DT  # Never catch up with command bursts.
            if not connection.poll(max(0., min(CONTROL_DT, next_tick - time.monotonic()))):
                continue
            request_id, operation, payload = connection.recv()
            try:
                if operation == "close":
                    # Close must terminate the owner even when HOLD fails.
                    # Report that failure; never claim a confirmed stop.
                    try:
                        if gravity_comp and payload is not False and transport.armed and interlock is None and robot.mode != "fault":
                            robot.enter_hand_guiding()
                        robot.close()
                    except Exception as error:
                        connection.send((request_id, False, f"Close failed; final mode unconfirmed: {error}"))
                    else:
                        connection.send((request_id, True, None))
                    break
                if operation == "observe":
                    heartbeat = time.monotonic()
                    if interlock is not None:
                        # Continue reading for display when possible, without
                        # clearing the fault or restarting setpoint generation.
                        try:
                            robot._observe(robot._read())
                        except Exception:
                            pass
                    observed = robot.observe()
                    if interlock:
                        observed = replace(observed, mode="fault", reason=interlock)
                    elif not transport.armed:
                        idle = "hand-guiding" if gravity_comp else "position hold"
                        observed = replace(observed, mode="unarmed", reason=f"{side} selected; E enables {idle}; targets start follower control")
                    result = observed
                elif operation == "arm":
                    if interlock:
                        raise RuntimeError(interlock)
                    if gravity_comp and not simulation:
                        # Reject experimental idle mode BEFORE any motor enable.
                        require_qualified_hand_guiding()
                    transport.arm()
                    selected = transport.guard.indices
                    limits = transport.guard.motion_limits
                    velocity = robot.controller.max_velocity.copy()
                    acceleration = robot.controller.max_acceleration.copy()
                    velocity[selected], acceleration[selected] = limits.velocity, limits.acceleration
                    robot.controller.set_motion_limits(velocity, acceleration)
                    robot.kinematics.limits[selected] = limits.position
                    robot.hold()
                    if gravity_comp:
                        robot.enter_hand_guiding()
                    # Physical arming and HOLD may take time; rebase every
                    # reference from the fresh sample, never from preview.
                    heartbeat = time.monotonic()
                    next_tick = heartbeat + CONTROL_DT
                    result = transport.guard.lift_target()
                elif operation == "motion_limits":
                    if not transport.armed:
                        raise RuntimeError("Motion limits are available after arming readback")
                    result = {name: getattr(transport.guard.motion_limits, name).tolist()
                              for name in ("position", "velocity", "acceleration")}
                    heartbeat = time.monotonic()
                elif operation == "submit_target":
                    if not transport.armed or interlock:
                        raise RuntimeError(interlock or "Hardware session is unarmed")
                    action, observed_id = payload
                    transport.guard.action(action)
                    result = robot.submit_target(action, observed_id)
                    if gripper_control:
                        transport.activate_gripper(action)
                    robot.target_deadline = time.monotonic() + .2
                    heartbeat = time.monotonic()
                elif operation == "hold":
                    robot.hold(preserve_grippers=bool(payload))
                    result = None
                elif operation in ("enter_hand_guiding", "begin_control"):
                    if operation == "enter_hand_guiding" and not gravity_comp:
                        raise RuntimeError("Hand-guiding is disabled (gravity_comp=false)")
                    if interlock or robot.mode == "fault":
                        raise RuntimeError(interlock or robot.reason)
                    if transport.armed:
                        getattr(robot, operation)()
                    heartbeat = time.monotonic()
                    next_tick = heartbeat + CONTROL_DT
                    result = robot.observe()
                else:
                    raise ValueError("Unknown hardware session operation")
                connection.send((request_id, True, result))
            except Exception as error:
                if operation in ("arm", "enter_hand_guiding", "begin_control"):
                    interlock = str(error)
                elif operation == "submit_target":
                    try:
                        robot.hold()
                    except Exception as hold_error:
                        interlock = f"Target rejected: {error}; HOLD failed: {hold_error}"
                        error = RuntimeError(interlock)
                connection.send((request_id, False, f"{type(error).__name__}: {error}"))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send((0, False, str(error)))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        try:
            if robot is not None:
                robot.close()
            elif transport is not None:
                transport.close()
        finally:
            connection.close()


class HardwareSession:
    def __init__(self, side, mapping=ROOT / "configs/hardware_mapping.json",
                 commissioning=ROOT / "configs/commissioning.json", simulation=False,
                 workspace_radius_m=None, gripper_control=False, gripper_force_n=1.0,
                 gravity_comp=False):
        if type(gravity_comp) is not bool:
            raise ValueError("gravity_comp must be a boolean")
        self.gravity_comp = gravity_comp
        self.side = side
        self.sides = selected_sides(side)
        self.gripper_control = gripper_control
        self.gripper_force_n = gripper_force_n
        if gripper_control:
            from .gripper import validate_force
            validate_force(gripper_force_n)
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.simulation = simulation
        self.request_id = 0
        self.closed = False
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=control_worker,
            args=(child, side, str(mapping), str(commissioning), simulation, self.workspace_radius_m,
                  gripper_control, gripper_force_n, gravity_comp),
            name="nero-control-owner", daemon=False)
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(10):
                raise RuntimeError("Hardware session startup timed out")
            _, okay, value = self.connection.recv()
            if not okay:
                raise RuntimeError(value)
        except BaseException:
            self.connection.close()
            self.process.join(timeout=3.)
            raise

    def _rpc(self, operation, payload=None, timeout=.5):
        if self.closed:
            raise RuntimeError("Hardware session closed")
        self.request_id += 1
        self.connection.send((self.request_id, operation, payload))
        if not self.connection.poll(timeout):
            # Closing IPC ensures the control owner receives EOF and HOLDs.
            self.connection.close()
            self.closed = True
            self.process.join(timeout=3.)
            raise RuntimeError("Hardware RPC timed out; control process requested to HOLD")
        received_id, okay, value = self.connection.recv()
        if received_id != self.request_id or not okay:
            raise RuntimeError(value)
        return value

    def observe(self):
        return self._rpc("observe")

    def submit_target(self, action, observation_id):
        return self._rpc("submit_target", (action, observation_id), timeout=6.)

    def arm(self):
        return self._rpc("arm", timeout=12. if self.side == "both" else 6.)

    def motion_limits(self):
        """Return effective limits already read during arming; no controller writes."""
        return self._rpc("motion_limits")

    def hold(self, *, preserve_grippers=False):
        return self._rpc("hold", preserve_grippers)

    def enter_hand_guiding(self):
        if not self.gravity_comp:
            raise RuntimeError("Hand-guiding is disabled (gravity_comp=false)")
        return self._rpc("enter_hand_guiding", timeout=8.)

    def begin_control(self):
        return self._rpc("begin_control", timeout=8.)

    def start(self):
        pass  # The child process owns control timing from construction onward.

    def close(self, *, hand_guiding=None):
        if self.closed:
            return
        try:
            self._rpc("close", self.gravity_comp and hand_guiding is not False, timeout=8.)
        finally:
            self.closed = True
            self.connection.close()
            self.process.join(timeout=3.)
            if self.process.is_alive():
                raise RuntimeError("Control process did not exit; check the physical robot")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
