# NERO dual-arm interactive IK

This workspace assembles the two user-supplied archives and provides an actuated MuJoCo simulation, a native interactive IK viewer, and a backend-independent robot API. The simulation works locally. The real-backend transport boundary is implemented and tested with a loopback fixture; a commissioned AgileX hardware driver is not included, and no hardware was connected or commanded.

## Demo

![Dual NERO simulation demonstrating dual-arm translation and rotation, independent gripper control, HOLD, and resume](previews/dual_nero_ik_validation.gif)

32-second simulation playback at 1× speed, with live TCP error readouts. Scripted targets use the same robot API as the interactive viewer; this is not a mouse recording. Contacts are disabled. [Watch the full-resolution MP4](previews/dual_nero_ik_validation.mp4).

## Run interactive IK

Run in a desktop terminal:

```bash
cd ~/projects/agilex-deploy
uv venv --python 3.10 .venv  # First checkout only; reuse an existing environment.
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python scripts/build_assembly.py
.venv/bin/agilex-interactive
```

Clone with `git clone https://github.com/ZiWang1105/agile-x-deploy.git agilex-deploy` to use the directory name above. The repository includes the source assets required by this assembly; the original upload archives are not needed to build it. Generated files in `build/` are ignored. `build/dual_nero.sim.xml` is generated automatically from the geometry model when absent.

- Double-click a red sphere to select a TCP target.
- Hold **Ctrl + right mouse button** and drag to translate the target.
- Hold **Ctrl + left mouse button** and drag to rotate the target. Shift changes the manipulation plane/axis using MuJoCo's native controls.
- **1 / 2** toggle the left/right gripper target.
- **Space / R** enter HOLD and reset handles to the measured TCPs. Drag again to resume.
- **Q** or window close exits and releases the owned control thread.

Red spheres/axes are targets; small green markers are encoder-derived TCPs. The display reports position/orientation error and runtime state. The renderer has its own model; dragging robot geometry or using native Reset cannot reset or force the simulated plant. Native Reset requests HOLD.

The CLI uses the active desktop's `DISPLAY`. On this machine, `DISPLAY=:1 .venv/bin/agilex-interactive` was tested successfully. Prefer the display variable supplied by your desktop session. The command stays in the foreground; no server or detached process is installed.

## Same interface for sim, real, and agents

```python
observation = robot.observe()
request_id = robot.submit_target(action, observation.observation_id)
robot.hold()
robot.close()
```

`Action` always contains both arms: a base-frame TCP position in meters, an XYZW unit quaternion, and integer `gripper_open` (0 or 1) per arm. `Observation` contains the 14 named arm encoders, measured velocities, normalized gripper travel, encoder-derived TCPs, acquisition timestamp, observation ID, and runtime state. Camera observations are explicitly unavailable in this release.

`Robot(SimBackend())` and `Robot(RealBackend(transport))` run the exact same contract, IK, target admission, and joint-reference controller. A policy or viewer receives `RobotInterface`; it does not branch on the backend. Only the composition root selects the backend. See [the architecture and commissioning boundary](docs/INTERACTIVE_IK.md) and [a minimal agent-style policy](examples/agent_policy.py).

The viewer uses the shared request path and does not command joints. The simulation uses position actuators and `mj_step`, with a 100 Hz control loop and 500 Hz physics. Setting simulator `qpos` is restricted to initialization; the separate FK/display models are reconstructed from encoder observations.

## Scope and validation

This first controller uses damped Jacobian IK, not the official CasADi/IPOPT solver. Joint targets have position limits, a 0.6 rad/s speed bound, and a 2 rad/s² acceleration bound during tracking. HOLD cancels the target and captures the current position; it is a servo hold, not a physical emergency stop. Targets expire after 0.5 s without refresh. Stale feedback, bad joint ordering, tracking faults, and control overruns cause a latched fault and backend HOLD.

The simulator uses ideal gravity compensation. **Contacts and collision admission are not implemented.** It is suitable for free-space interactive IK, not grasp/contact validation or hardware execution. Source inertias and limits are unqualified. The nominal TCP is `[0, 0, 0.1358]` meters in each `*_gripper_base`, at the CAD finger-root plane; it is not a physically calibrated pinch frame. Top-camera orientation is preserved, and its bracket joint is fixed in the runtime model.

```bash
.venv/bin/python -m unittest discover -s tests -v
MUJOCO_GL=egl .venv/bin/python scripts/smoke_ik.py
.venv/bin/python scripts/validate_assembly.py
.venv/bin/python examples/agent_policy.py
```

The automated suite covers paired 6D tracking, command limits, actuator-only motion, HOLD/resume, command expiry, stale feedback, invalid joint order, unreachable-target handling, gripper coupling, camera lock, Jacobian finite differences, drag/reset routing, thread cleanup, and identical sim/real-adapter trajectories through a loopback transport. This is interface evidence, not a real-robot test.

`previews/interactive_ik.png` and `previews/interactive_ik_report.json` show a repeatable two-arm translation/rotation command. The native desktop window also passed a timed launch/render/close test. Actual human mouse manipulation has not been exercised by automation; handle event conversion is covered separately.

The [32-second validation video](previews/dual_nero_ik_validation.mp4) shows translation, orientation tracking, independent gripper commands, HOLD during motion, and resume. Its target sequence uses the same robot API as the viewer. It is simulation-time playback, not a mouse recording. Reproduce it with `MUJOCO_GL=egl .venv/bin/python scripts/record_ik_validation.py`.

The remainder documents the original geometry assembly and its provenance.

## Current result

- The supplied `wbcd_urdf/urdf/dual_nero.xacro` already defines the complete assembly. Its mounting transforms are used unchanged.
- The assembly contains 37 links and 36 joints: 14 arm revolute joints, 4 gripper prismatic joints, 1 camera-bracket revolute joint, and 17 fixed joints.
- There are two seven-axis NERO arms, two grippers, two wrist D435 camera models, and one top D435 camera model.
- This is a fixed stand. The supplied model has no mobile chassis or lift joint. The camera-bracket joint does not establish that the physical bracket is motorized.
- All 28 unique source mesh paths resolve. Source files under `vendor/` are unchanged.
- MuJoCo and Pinocchio both load the assembled kinematic model. Independent URDF FK and MuJoCo FK agree over 22 configurations and all 37 link frames; detailed results are in `build/validation.json`.

## Files

- `build/dual_nero.urdf`: expanded URDF with relative mesh paths. Keep the sibling `meshes/` directory with it. Configure a consuming loader to resolve mesh paths relative to this URDF's directory if it does not do so automatically.
- `build/dual_nero.preview.xml`: MuJoCo geometry inspector model. Keep the sibling `preview_meshes/` directory with it.
- `build/assembly_report.json`: source mesh hashes, joint definitions, and known limitations.
- `previews/assembly_overview.png`: perspective, front, side, and source-zero-pose views.
- `previews/display_pose.json`: example joint values used for the previews. These are visualization values, not a hardware startup pose.
- `previews/camera_diagnostic.png`: close-up views of the top camera's lens face with an orange forward-axis arrow.

### Preview orientation correction

The first preview used left J3 = +1.57 rad and right J3 = -1.57 rad, bending the forearms toward the stand's rear (-X). This made the arms face the opposite direction from the top camera. The corrected display pose uses left J3 = -1.57 rad and right J3 = +1.57 rad, placing the grippers toward the robot front (+X). Source camera and arm mounting transforms are unchanged.

The top camera's forward axis at the source bracket zero pose is approximately `[0.518671, 0, -0.854974]` in the stand frame: forward and downward, about 58.8 degrees below horizontal. Its untextured CAD shell and steep pitch make the housing's top/back more visible from above; the diagnostic shows the actual lens face from below. This orientation is the source model, not a measured physical-camera calibration. MuJoCo free-camera azimuth 180 degrees places the observer on +X, not -X; this sign was initially misinterpreted during troubleshooting and was checked against the rendered camera position before the final correction. Validation now checks that the top camera looks forward/down and that both preview grippers lie in front of the stand.

The source mounting origins, in meters and radians, are:

| Mount | Parent | XYZ | RPY |
|---|---|---|---|
| Left arm | `base_link` | `-0.002 0.1 0.59` | `-1.57 -1.57 0` |
| Right arm | `base_link` | `-0.002 -0.1 0.59` | `1.57 -1.57 0` |

The approximately 90-degree values are retained as supplied, rather than silently replaced with exact pi/2 values. The images show example folded-arm poses and the unmodified zero pose. They do not attempt to infer exact joint angles from the exhibition photo.

## Reproduce

Python 3.10 was used. From this directory:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/build_assembly.py
.venv/bin/python scripts/validate_assembly.py
MUJOCO_GL=egl .venv/bin/python scripts/preview_assembly.py
```

For the standard local MuJoCo model inspector on a desktop with a display:

```bash
.venv/bin/python scripts/preview_assembly.py --viewer
```

The inspector uses MuJoCo's standard UI, including joint sliders. Only offscreen rendering and model/FK validation were exercised here; a desktop GUI session was not tested. Do not treat the preview as a dynamics simulation. Gravity and all contacts are disabled, and no actuators are defined.

## Source limitations that remain visible

1. The stand DAE has no material colors, so the render uses the supplied URDF color. The physical stand's black/silver finish in the reference photo is not fully represented. Arm DAE material colors are retained.
2. The wrist camera-stand visual and collision origins differ in the supplied model. The URDF preserves both; collision geometry needs review before planning or execution.
3. The two fingers of each gripper are independent prismatic joints in the source, without a mimic constraint. Both values are explicitly set in the preview. Real gripper coupling must be modeled when adding control.
4. Some camera inertia values are explicitly marked unreliable in the source. Arm limits and effort values also require hardware/firmware confirmation. No dynamics or torque qualification is claimed.
5. Thin decorative mesh components use MuJoCo shell mesh inertia preprocessing to permit preview compilation. The model uses the supplied explicit body inertias; visual geoms have zero mass and no contact participation.
6. The wrist-camera descriptions do not provide the same full optical-frame tree as the top camera. Sensor integration must define and validate each optical frame and physical calibration.

## Official IK options inspected

The supplied `agx_arm_sim/Moveit2/nero_gripper_moveit_config/config/kinematics.yaml` selects `kdl_kinematics_plugin/KDLKinematicsPlugin`. This is the MoveIt IK path.

AgileX's [QuestArmTeleop IK implementation](https://github.com/agilexrobotics/QuestArmTeleop/blob/ros2/src/oculus_reader/scripts/arm_ik_pose_node.py) uses Pinocchio, CasADi, and IPOPT. It optimizes end-effector pose error, regularization, and change from the previous joint vector, subject to joint position limits.

The [dual-NERO launch](https://github.com/agilexrobotics/QuestArmTeleop/blob/ros2/src/oculus_reader/launch/teleop_double_nero.launch.py) runs one solver per arm. It does not implement one coupled 14-axis solver with stand and inter-arm collision constraints. In the inspected IK node, optional collision detection occurs after publishing the joint target; it is not an execution-blocking collision check. A smoothing objective also does not impose hard velocity, acceleration, or jerk bounds. The launch enables hardware automatically and selects fast mode; it was inspected only and was not run.

The interactive prototype uses a separate damped Jacobian solver to establish the shared runtime without adding ROS/CasADi dependencies. The official optimization solver remains an offline comparison task; it has not been ported or benchmarked in this runtime. Whole-assembly collision admission, physical TCP calibration, and a commissioned driver remain prerequisites for hardware execution.

## Provenance

Inputs are the user-supplied `wbcd_urdf-20261014T025717Z-1-001.zip` and `agx_arm_sim-20261014T025717Z-1-001.zip`. Archive hashes are recorded in `source_manifest.json`.

Visual reference: [AgileX exhibition photo](https://x.com/AgilexRobotics/status/2089985024515002714/photo/1). The image was retrieved from the post's public X syndication media URL. It shows the same general stand-and-dual-arm configuration; no exact model/pose identity is asserted.

Preserve the upstream license notices included with the assets. The supplied stand archive does not include a standalone license file; this workspace does not assign it a new license or establish redistribution rights.
