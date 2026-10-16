"""RobotInterface client with an owned control/watchdog process.

The viewer never owns a CAN writer. The child process observes feedback and
enforces a 200 ms operator heartbeat independently of the viewer's Python GIL.
Closing the session requests HOLD and joins the child; it does not disable motors.
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

OPERATOR_TIMEOUT = .2


class SimulatedCommissioningTransport:
    """Explicit test plant for commissioning UI and process/watchdog tests."""
    def __init__(self, side, workspace_radius_m=None, **_):
        self.side = side
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.plant = SimBackend()
        self.armed = False
        self.guard = None
        self.closed = False

    def read_encoders(self):
        return replace(self.plant.read(), grippers=(None, None))

    def arm(self):
        sample = self.read_encoders()
        self.guard = MotionGuard(self.side, sample.positions, self.plant.kinematics.limits,
                                 workspace_radius_m=self.workspace_radius_m)
        self.armed = True

    def send_positions(self, q, grippers):
        if self.armed:
            sample = self.read_encoders()
            self.guard.command(q, sample.positions)
            command = np.array(sample.positions)
            command[self.guard.indices] = q[self.guard.indices]
            self.plant.write(command, [.36, .36])
            self.plant.advance()

    def hold(self):
        if self.guard:
            self.guard.hold(self.read_encoders().positions)
        self.plant.hold()

    def close(self):
        self.closed = True
        self.plant.close()


def control_worker(connection, side, mapping, commissioning, simulation, workspace_radius_m):
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
        factory = SimulatedCommissioningTransport if simulation else NeroTransport
        transport = factory(side=side, mapping=mapping, commissioning=commissioning,
                            workspace_radius_m=workspace_radius_m)
        robot = Robot(RealBackend(transport), ik=ik)
        connection.send((0, True, "ready"))
        while True:
            now = time.monotonic()
            if transport.armed and heartbeat is not None and now - heartbeat > OPERATOR_TIMEOUT and interlock is None:
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
                        robot.close()
                    except Exception as error:
                        connection.send((request_id, False, f"Close completed; HOLD unconfirmed: {error}"))
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
                        observed = replace(observed, mode="unarmed", reason=f"{side} selected; E enables holding at current posture")
                    result = observed
                elif operation == "arm":
                    if interlock:
                        raise RuntimeError(interlock)
                    transport.arm()
                    selected = transport.guard.indices
                    limits = transport.guard.motion_limits
                    velocity = robot.controller.max_velocity.copy()
                    acceleration = robot.controller.max_acceleration.copy()
                    velocity[selected], acceleration[selected] = limits.velocity, limits.acceleration
                    robot.controller.set_motion_limits(velocity, acceleration)
                    robot.kinematics.limits[selected] = limits.position
                    robot.hold()
                    # Physical arming and HOLD may take time; rebase every
                    # reference from the fresh sample, never from preview.
                    heartbeat = time.monotonic()
                    next_tick = heartbeat + CONTROL_DT
                    result = transport.guard.lift_target()
                elif operation == "submit_target":
                    if not transport.armed or interlock:
                        raise RuntimeError(interlock or "Hardware session is unarmed")
                    action, observed_id = payload
                    transport.guard.action(action)
                    result = robot.submit_target(action, observed_id)
                    robot.target_deadline = time.monotonic() + .2
                    heartbeat = time.monotonic()
                elif operation == "hold":
                    robot.hold()
                    result = None
                else:
                    raise ValueError("Unknown hardware session operation")
                connection.send((request_id, True, result))
            except Exception as error:
                if operation == "arm":
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
                 workspace_radius_m=None):
        self.side = side
        self.workspace_radius_m = validate_workspace_radius(workspace_radius_m)
        self.simulation = simulation
        self.request_id = 0
        self.closed = False
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=control_worker,
            args=(child, side, str(mapping), str(commissioning), simulation, self.workspace_radius_m),
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
        return self._rpc("submit_target", (action, observation_id))

    def arm(self):
        return self._rpc("arm", timeout=6.)

    def hold(self):
        return self._rpc("hold")

    def start(self):
        pass  # The child process owns control timing from construction onward.

    def close(self):
        if self.closed:
            return
        try:
            self._rpc("close")
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
