# Interactive IK runtime, contract version 3

## Ownership and data flow

This prototype follows the existing `agentic-mobile-manipulator` / `hexfellow-deploy` ownership split without importing either project or assuming its robot dimensions, lift, joint count, TCP definition, or hardware control mode.

```text
Operator viewer / agent policy
        RobotInterface: observe, submit_target, hold, close
                    |
       Robot runtime: one reference owner, 100 Hz
       admission, feedback FK, IK, integration, fault/HOLD
                    |
              Backend protocol
          /                       \
  SimBackend                  RealBackend(transport)
  MuJoCo encoders             timestamped physical encoders
  position actuators          paired joint/gripper setpoints
  500 Hz dynamics             commissioned transport watchdog
```

All pose targets are in the fixed stand's `base_link`. Both arms use the supplied mounting transforms, including the source's rounded 1.57 rad values. The root is fixed, so the simulation's world frame coincides with `base_link`; policies are not given simulator world state. No mobile base, lift, synthetic RGB-D, or object ground truth is added.

The renderer uses a separate model reconstructed from `Observation`. It cannot access the plant `MjData`. Observation FK uses MuJoCo; the official IK uses two seven-axis Pinocchio models synchronized from the measured 14 arm encoders. Both models share the official TCP, and their FK is checked under the stand mounting transforms. Exactly one controller integrates joint references. Only the backend advances physics or writes a hardware setpoint. MuJoCo is a kinematics dependency of this first shared controller, including when used with the hardware adapter; no simulated plant is stepped by the real backend.

For simulation, the owned control loop runs in a joined thread while the native viewer renders separately. Physical commissioning runs the same Robot/controller/RealBackend stack in an owned child process, accessed through a `HardwareSession` implementing `RobotInterface`. The child enforces viewer heartbeat expiry independently of the renderer's Python GIL. Neither arrangement is hard real time. Future deployments can place the same contracts on the existing message transport.

## Contract

`src/agilex_deploy/contract.py` defines immutable `Pose`, `ArmTarget`, `Action`, and `Observation` types. JSON actions use:

```json
{
  "left": {
    "position_m": [0.43, 0.42, 0.15],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "gripper_open": 1
  },
  "right": {
    "position_m": [0.43, -0.42, 0.15],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "gripper_open": 1
  }
}
```

This is a schema example, not a recommended startup command. Use `observation.as_action()` as the basis for a target, then modify the desired pose.

Both arms are always specified together. Missing arms, extra fields, non-finite poses, non-unit quaternions, and gripper values other than integer 0/1 or null are rejected. Version 2 adds `gripper_open: null` to leave a gripper unchanged; this is also the default when constructing `ArmTarget`. An unavailable calibrated gripper measurement is `null` in `Observation.gripper_position`, never an invented zero. A non-null gripper command is rejected when that feedback is unavailable. Sim and real use the same version 3 types. Version 3 changes the TCP definition to the official NERO `ee`; method signatures and action fields are unchanged. Do not replay version 2 TCP targets. The 2 m coordinate envelope catches gross unit mistakes; it is not a reachable workspace or collision check.

Encoder order is explicit: `left_joint1` through `left_joint7`, then `right_joint1` through `right_joint7`. Arm units are radians and radians/second. Gripper feedback is normalized physical travel, from 0 closed to 1 open; simulation maps this to 0–0.05 m travel of the first finger and couples the second finger with opposite displacement. It is not force feedback or an asserted calibrated jaw aperture. Real transport calibration must implement the same semantic quantity.

TCPs are named `left_tcp` and `right_tcp`. The official NERO YAML defines `[0.1755, 0, -0.0235]` m translation and zero RPY relative to each `joint7` frame (the corresponding `link7` body in MuJoCo). The same definition drives observations, red handles, official IK targets and physical control. World/stand-frame targets are transformed into each arm's fixed base frame before solving. This definition replaces the old gripper-base finger-root TCP and has not been physically calibrated.

Observation timestamps use host monotonic acquisition time. Backend delivery must not retimestamp old hardware feedback. A paired hardware sample advances its sequence only after both arms have fresh measurements; the transport must bound left/right acquisition skew. The runtime rejects backwards time/sequence, changed content under an unchanged sequence, stale/future timestamps, bad dimensions, and wrong joint order. Cameras are an empty tuple until a separately timestamped sensor contract is implemented.

## Control and failure behavior

- Control period: 0.01 s; simulated physics period: 0.002 s.
- IK: two original AgileX `ArmIK` instances using Pinocchio/CasADi/IPOPT, synchronized to measured joints before solving. The NERO YAML supplies defaults; `configs/ik.yaml` overrides only `w_pos` to `40` and `w_ori` to `4`. All other objective weights and solver settings retain their official values. No local Jacobian fallback or coupled collision optimization remains. A downstream reference generator executes the joint solution within the selected speed/acceleration limits.
- One joint-reference integrator; position limits from the supplied URDF intersected with live selected-controller bounds on hardware, and per-joint controller-reported speed/acceleration limits during tracking. Simulation uses the documented October 14 left-controller readback fixture for both arms. Discrete braking margins reduce speed near a limit. These are requested-motion limits, not measured-dynamics certification. No jerk bound is claimed.
- Gripper references ramp at normalized travel rate 0.6/s (0.03 m/s per simulated finger).
- No arbitrary joint tracking-error cutoff or Cartesian velocity clip is applied. Firmware faults still latch HOLD.
- Target admission requires an observation younger than 0.2 s with valid sequence and acquisition timestamps.
- A target must be refreshed within 0.5 s. The viewer and example policy refresh every 0.1 s. A stopped policy loses motion ownership through lease expiry.
- HOLD discards the Cartesian goal and captures the measured joint/gripper position. Servos decelerate toward that position. This intentionally overrides tracking continuity; it does not invoke a hardware emergency stop.
- Failure to make meaningful progress for 3 wall-clock seconds with aggregate pose error above the threshold enters HOLD and reports an unreachable/stalled goal. This is a heuristic, not a proof of reachability. A fresh user request may resume.
- Stale or invalid encoders, firmware faults, and control overruns latch a fault. Targets remain rejected after an operator HOLD; inspect and restart to clear it. The transport's HOLD is invoked when valid feedback cannot be read, rather than transmitting stale joint positions.

Runtime state is `hold`, `tracking`, or `fault`, with a reason in every observation. Rejected malformed/stale requests are exceptions and do not silently replace the active target. A native display Reset enters HOLD while preserving the plant state. Selecting/dragging robot geometry is ignored by the command path.

## Simulation model

`build/dual_nero.preview.xml` remains a geometry-only artifact. `model.build_sim_model()` derives `build/dual_nero.sim.xml` with arm/gripper position actuators, finger coupling constraints, nominal TCP sites, a fixed top-camera bracket, gravity, and ideal body gravity compensation. At initialization only, the plant receives the existing forward-facing display pose. Subsequent plant motion is exclusively through actuator control and `mj_step`.

Contact geoms remain disabled because supplied collision transforms/inertias require review. Consequently arms can intersect each other, the stand, and the floor. The simulator does not test grasp forces, collision avoidance, impact, torque realism, or hardware tracking. Simulation position-servo gains are not hardware gains.

Native rendering follows the [MuJoCo passive-viewer API](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). MuJoCo 3.10 starts a daemon rendering thread and `Handle.close()` only requests exit. This environment exhibited a GLFW teardown race; the viewer wrapper joins the renderer thread it just created before interpreter shutdown. Native launch/render/close was retested after the fix.

## Hardware boundary and remaining work

`RealBackend` adapts a `HardwareTransport` to the same runtime. The generic adapter does not itself open CAN or auto-enable motors. `NeroTransport` provides the explicit single-arm NERO 1.21 commissioning implementation, hosted by `HardwareSession` in a separate process. Its physical observation includes timestamped joint angles and firmware-reported velocities; gripper travel is null. Its joint-position wire commands require explicit operator arming. `--backend real-commissioning` exposes that path, while `--backend commissioning-sim` exercises it against a simulated plant. The `sim` and receive-only `real-preview` modes remain separate and never enable hardware. See [hardware commissioning](HARDWARE_COMMISSIONING.md) for physical qualification status and limits.

Remaining qualification and extensions:

1. Confirm each arm's firmware/SDK pairing, CAN identity, joint signs/offsets, encoder freshness, gripper travel calibration, limits, and supported command mode. Firmware profiles differ in velocity reporting; unavailable velocity must be addressed explicitly rather than invented.
2. The single-arm transport and independent 0.2 s operator watchdog are implemented. Before adding simultaneous physical dual-arm control, extend ownership and fault handling so a failure of either arm holds both; qualify this on hardware.
3. Calibrate mount/TCP transforms and camera frames. Validate source collision geometry and introduce whole-assembly command admission, including self, inter-arm, stand, and workspace collisions.
4. The official solver is now integrated, with direct upstream solve parity and shared-model FK tests. Qualify its measured tracking, timing and stop behavior on hardware separately. See [official IK](OFFICIAL_IK.md).
5. Commission conservative single-arm motion, then paired motion, with measured tracking/stop tests. Map the shared joint-reference semantics to the selected SDK mode; do not assume Hexfellow's MIT mode or gains apply to NERO.
6. Add timestamped RGB-D observations and task/perception components above the robot contract. Introduce process isolation and execution logs before deploying long-running agent workflows.

## Agent integration

`examples/agent_policy.py` demonstrates the exact robot-facing surface an agent should use. An agent chooses a paired target from an observation and refreshes that target while it retains motion ownership. All IK and actuator timing remain in the robot runtime. The example's composition root constructs a simulated robot; its `move_both(RobotInterface)` policy is backend-independent.

The next task layer can implement observe, propose, validate, execute, and verify as finite states while preserving this contract. No external model API, credential, prompt service, or autonomous physical action is included in this first interactive version.
