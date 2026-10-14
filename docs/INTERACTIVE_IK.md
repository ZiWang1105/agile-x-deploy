# Interactive IK runtime, version 0.1

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

The renderer uses a separate model reconstructed from `Observation`. It cannot access the plant `MjData`. The IK also uses its own FK data built from the measured 14 arm encoders. Exactly one controller integrates joint references. Only the backend advances physics or writes a hardware setpoint. MuJoCo is a kinematics dependency of this first shared controller, including when used with the hardware adapter; no simulated plant is stepped by the real backend.

For this local prototype, the owned control loop runs in a joined thread while the native viewer renders separately. This is not a hard real-time or process-isolated deployment. The next deployed version can place these same contracts on the existing message transport. The viewer and agent API need not change.

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

Both arms are always specified together. Missing arms, extra fields, non-finite values, non-unit quaternions, and gripper values other than integer 0/1 are rejected. The 2 m coordinate envelope catches gross unit mistakes; it is not a reachable workspace or collision check.

Encoder order is explicit: `left_joint1` through `left_joint7`, then `right_joint1` through `right_joint7`. Arm units are radians and radians/second. Gripper feedback is normalized physical travel, from 0 closed to 1 open; simulation maps this to 0–0.05 m travel of the first finger and couples the second finger with opposite displacement. It is not force feedback or an asserted calibrated jaw aperture. Real transport calibration must implement the same semantic quantity.

TCPs are named `left_tcp` and `right_tcp`, each translated `[0, 0, 0.1358]` m from the corresponding `gripper_base`, with the same orientation. This is a declared CAD reference at the finger-root plane, not the official teleoperation `ee` offset and not a calibrated grasp center. Before real deployment, update this single shared model definition with a measured TCP and validate both arms. Do not compensate with hidden offsets inside one backend.

Observation timestamps use host monotonic acquisition time. Backend delivery must not retimestamp old hardware feedback. A paired hardware sample advances its sequence only after both arms have fresh measurements; the transport must bound left/right acquisition skew. The runtime rejects backwards time/sequence, changed content under an unchanged sequence, stale/future timestamps, bad dimensions, and wrong joint order. Cameras are an empty tuple until a separately timestamped sensor contract is implemented.

## Control and failure behavior

- Control period: 0.01 s; simulated physics period: 0.002 s.
- IK: two seven-axis damped least-squares Jacobian tasks, both evaluated against measured FK; soft null-space posture preference. No coupled collision optimization.
- One joint-reference integrator; position limits from the supplied URDF, 0.6 rad/s target speed, and 2 rad/s² target acceleration during tracking. Discrete braking margins reduce speed near a limit. These are requested-motion limits, not measured-dynamics certification. No jerk bound is claimed.
- Gripper references ramp at normalized travel rate 0.6/s (0.03 m/s per simulated finger).
- Joint reference versus measured tracking error above 0.2 rad faults.
- Target admission requires an observation younger than 0.2 s and no joint displacement above 0.15 rad since that observation.
- A target must be refreshed within 0.5 s. The viewer and example policy refresh every 0.1 s. A stopped policy loses motion ownership through lease expiry.
- HOLD discards the Cartesian goal and captures the measured joint/gripper position. Servos decelerate toward that position. This intentionally overrides tracking continuity; it does not invoke a hardware emergency stop.
- Failure to make meaningful progress for 3 wall-clock seconds with aggregate pose error above the threshold enters HOLD and reports an unreachable/stalled goal. This is a heuristic, not a proof of reachability. A fresh user request may resume.
- Stale or invalid encoders, tracking failures, and control overruns latch a fault. Targets remain rejected after an operator HOLD; inspect and restart to clear it. The transport's HOLD is invoked when valid feedback cannot be read, rather than transmitting stale joint positions.

Runtime state is `hold`, `tracking`, or `fault`, with a reason in every observation. Rejected malformed/stale requests are exceptions and do not silently replace the active target. A native display Reset enters HOLD while preserving the plant state. Selecting/dragging robot geometry is ignored by the command path.

## Simulation model

`build/dual_nero.preview.xml` remains a geometry-only artifact. `model.build_sim_model()` derives `build/dual_nero.sim.xml` with arm/gripper position actuators, finger coupling constraints, nominal TCP sites, a fixed top-camera bracket, gravity, and ideal body gravity compensation. At initialization only, the plant receives the existing forward-facing display pose. Subsequent plant motion is exclusively through actuator control and `mj_step`.

Contact geoms remain disabled because supplied collision transforms/inertias require review. Consequently arms can intersect each other, the stand, and the floor. The simulator does not test grasp forces, collision avoidance, impact, torque realism, or hardware tracking. Simulation position-servo gains are not hardware gains.

Native rendering follows the [MuJoCo passive-viewer API](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer). MuJoCo 3.10 starts a daemon rendering thread and `Handle.close()` only requests exit. This environment exhibited a GLFW teardown race; the viewer wrapper joins the renderer thread it just created before interpreter shutdown. Native launch/render/close was retested after the fix.

## Hardware boundary and remaining work

`RealBackend` adapts a `HardwareTransport` to the same runtime. It does not open CAN, import pyAgxArm, auto-enable motors, or pretend simulation feedback is hardware data. The CLI deliberately exposes only `--backend sim`. Tests use a clearly named loopback fixture to prove matching contract and controller behavior; this provides no physical commissioning evidence.

Implement and qualify the following before enabling real execution:

1. Confirm each arm's firmware/SDK pairing, CAN identity, joint signs/offsets, encoder freshness, gripper travel calibration, limits, and supported command mode. Firmware profiles differ in velocity reporting; unavailable velocity must be addressed explicitly rather than invented.
2. Implement `HardwareTransport.read_encoders`, `send_positions`, `hold`, and `close`. Preserve source acquisition time, ensure bounded I/O and paired-arm fault handling, and enforce an independent command watchdog of at most 0.2 s. If one arm fails, hold both.
3. Calibrate mount/TCP transforms and camera frames. Validate source collision geometry and introduce whole-assembly command admission, including self, inter-arm, stand, and workspace collisions.
4. Benchmark the official AgileX CasADi/IPOPT solver offline against the local Jacobian controller with identical frames and targets. The official solver has not been ported in this version.
5. Commission conservative single-arm motion, then paired motion, with measured tracking/stop tests. Map the shared joint-reference semantics to the selected SDK mode; do not assume Hexfellow's MIT mode or gains apply to NERO.
6. Add timestamped RGB-D observations and task/perception components above the robot contract. Introduce process isolation and execution logs before deploying long-running agent workflows.

## Agent integration

`examples/agent_policy.py` demonstrates the exact robot-facing surface an agent should use. An agent chooses a paired target from an observation and refreshes that target while it retains motion ownership. All IK and actuator timing remain in the robot runtime. The example's composition root constructs a simulated robot; its `move_both(RobotInterface)` policy is backend-independent.

The next task layer can implement observe, propose, validate, execute, and verify as finite states while preserving this contract. No external model API, credential, prompt service, or autonomous physical action is included in this first interactive version.
