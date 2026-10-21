"""Owned offline planner process for workshop pose operations.

CasADi/Pinocchio planning must not monopolize the control or camera thread's
Python interpreter. Each service owns one spawned process with private solver
state. No robot transport or controller is constructed in this process.
"""
from .agent_policy import tcp_pose
from .cartesian_validation import plan_cartesian_path
from .contract import Pose
from .model import Kinematics
from .motion_limits import bounded_hold_reference
from .official_ik import OfficialDualIK

_kinematics = None
_ik = None
_generation = None


def initialize_planner(initial_joints, generation):
    global _kinematics, _ik, _generation
    _kinematics = Kinematics()
    _ik = OfficialDualIK(_kinematics, initial_joints)
    _generation = generation


def planner_ready():
    return _ik is not None


def close_planner(pool, timeout_s=2.):
    """Join owned workers, terminating only this non-hardware planner if stuck.

    Python 3.10 has no public terminate_workers API. Capture its worker handles
    before shutdown clears them, so a stalled native solver cannot prevent the
    service from closing. No controller process is a member of this pool.
    """
    processes = list((pool._processes or {}).values())
    manager = pool._executor_manager_thread
    pool.shutdown(wait=False, cancel_futures=True)
    if manager is None:
        return
    # The executor manager owns waitpid/join for its workers. Joining worker
    # objects concurrently races with that manager and can leave exitcode
    # unset even after the OS process exited. Join the manager instead.
    manager.join(timeout=timeout_s)
    if manager.is_alive():
        for process in processes:
            if process.is_alive():
                process.terminate()
        manager.join(timeout=timeout_s)
    if manager.is_alive():
        for process in processes:
            if process.is_alive():
                process.kill()
        manager.join(timeout=timeout_s)
    if manager.is_alive():
        raise RuntimeError("Cartesian planner manager did not exit")



def plan_step(generation, step, joint_positions, position_limits, position_tolerance, orientation_tolerance):
    """Validate the operation's geometry; return reports without sending motion."""
    def cancelled():
        return _generation.value != generation

    q = bounded_hold_reference(joint_positions, position_limits)
    reports = {}
    for side, value in step['poses'].items():
        target = Pose(**value)
        if step.get('tool_frame') == 'pinch':
            target = tcp_pose(target)
        _, reports[side] = plan_cartesian_path(
            _kinematics, _ik, side, target, q,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
            position_limits=position_limits, cancelled=cancelled)
    return reports
