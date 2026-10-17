# Dual NERO control and agentic manipulation

This workspace assembles the supplied dual-NERO stand and runs **AgileX's official QuestArmTeleop `ArmIK` solver with NERO defaults and an explicit `w_pos=40, w_ori=4` override**. Simulation and the explicitly armed single- or dual-arm hardware adapter share the same solver, TCP definition and robot interface. The official IK path has been tested in simulation; physical motion with this solver has not been tested. Earlier left-arm hardware motion used the previous local Jacobian solver. See [official IK provenance and validation](docs/OFFICIAL_IK.md).

## Workshop API: Python and MCP

Participants can use the same **Python SDK or 18 MCP tools** for simulation and
physical deployment: official IK/FK, head RGB-D, joint/TCP motion, both arms,
**independent left/right grippers**, operation feedback, and cancellation.

```bash
uv pip install --python .venv-official/bin/python -r requirements-workshop.txt
./scripts/run_workshop.sh --backend sim --task place --auto-start-sim
# Second terminal:
.venv-official/bin/python examples/workshop/code_policy.py --config logs/workshop/client.json
```

Open **http://127.0.0.1:8780**. Tasks: `place`, `select`, and `sort`.
[Workshop plan, participant API, MCP setup, and physical rollout](docs/WORKSHOP.md).

The simulator passed placement, distractor-preserving selection, and two-arm
sorting with the RGB-D baseline. A live model-driven MCP policy also placed the
cube, lifting it **121.4 mm** before release. The full regression suite passed
**143 tests**. The real adapter is implemented; physical workshop tasks and
camera/tool calibration still require qualification. No real motion was run
for this validation.

[Placement video](previews/workshop_place.mp4) ·
[Two-arm sorting video](previews/workshop_sort.mp4) ·
[Validation results](previews/workshop_validation.json)

![Workshop dashboard with head camera, simulator view, independent arm/gripper feedback, and operation events](previews/workshop_dashboard.png)

## Agentic cube pickup

The OpenRouter policy now picks up a free cube using rendered RGB-D, measured
robot state, model-selected pixel grounding, official NERO IK, and physical
finger contacts. The final seven-call run lifted the cube **152.2 mm** and retained
it for **32.6 seconds**, with no arm/environment collisions. The policy received
no simulator object coordinates or contact truth. The same policy and sensor/
action contract are wired to the real NERO/ZED adapter.

[Watch the complete 59-second validation video](previews/agent_cube_pickup.mp4).
[Validation report](previews/agent_policy_validation.json) ·
[Archived sensor/API evidence](validation/agent_cube_20261015) ·
[Setup, architecture and physical deployment](docs/AGENT_POLICY.md).

```bash
uv pip install --python .venv-official/bin/python -r requirements-agent.txt
./scripts/run_agent.sh --backend sim --max-calls 16 --video \
  --prompt 'Pick up the red cube, lift it at least 10 cm, and hold it for at least 10 seconds.'
./scripts/run_agent.sh --backend real --side both  # Read-only RGB-D/robot preflight.
```

Open **http://127.0.0.1:8770**, edit the task, and click **Start simulation**.
The live dashboard shows the simulator, the fixed head-camera view, the latest
policy input image, decisions, execution feedback, and results. **Stop / HOLD**
ends policy commands, including decisions that return after Stop. **Close session**
exits after cleanup. Use `--no-web` to run immediately without the dashboard.
The physical preflight command is a separate session; close the simulator first
or choose another `--port`.

For the **real robot dashboard**, close any existing agent session, then run:

```bash
./scripts/run_agent.sh --backend real --side both --execute-real --camera-serial 35330610 \
  --prompt 'Pick up the cube using the left arm and hold it above the table.'
```

At **http://127.0.0.1:8770**, the **PHYSICAL ROBOT** page streams the real ZED and
measured arm/gripper feedback. It stays unarmed until **Start real policy** is
clicked. That button enables the selected physical arm and runs the task.
Camera selection verifies the
serial directly on the video device; the live panel displays its serial and path. Its calibration remains a
candidate; autonomous physical pickup has not been validated.

The external key defaults to `~/projects/openrouter_key.txt`. The full suite
passed 130 tests, including dashboard authentication, cancellation, camera identity,
paired arm/gripper control, and the combined gripper-opening command. Physical read-only preflight passed; autonomous physical
grasping has not been executed. The physical owner supports both arms in one session; camera/tool calibration
remains provisional. Paired physical motion has not yet been qualified.

The policy uses only one stand-mounted head RGB-D view (ZED 2i on hardware),
with no wrist cameras. `--prompt` / `--instruction` supplies the task. The agent
scene now removes the source URDF's wrist-camera assemblies; the archived video
and pickup metrics above predate this geometry correction. Those CAD parts were
never additional policy camera feeds.

## Historical demo (previous IK)

![Dual NERO simulation demonstrating dual-arm translation and rotation, independent gripper control, HOLD, and resume](previews/dual_nero_ik_validation.gif)

This existing GIF/MP4 predates the official IK migration and uses the old TCP and solver. It is not validation of the current solver. 32-second simulation playback at 1× speed, with live TCP error readouts. Scripted targets use the same robot API as the interactive viewer; this is not a mouse recording. Contacts are disabled. [Watch the full-resolution MP4](previews/dual_nero_ik_validation.mp4).

## ZED 2i live camera stream

Connect the camera over USB 3, then run:

```bash
cd ~/projects/agilex-deploy
./scripts/setup_zed_stream.sh  # Once per checkout; installs into .venv-camera.
./scripts/zed_stream.sh start
```

Open **http://127.0.0.1:8766** on the connected desktop. The page switches between
left, right, and side-by-side stereo views. Capture is 2560 × 720 at 30 fps,
giving 1280 × 720 per eye. This is raw UVC color; it does not require the ZED SDK
and does not provide depth or calibrated/rectified imagery.

```bash
./scripts/zed_stream.sh status
./scripts/zed_stream.sh logs
./scripts/zed_stream.sh stop
./scripts/zed_stream.sh restart
```

`start` creates an observable `agilex-zed-stream` systemd user service that stays
running after the terminal closes. It is not installed for automatic startup at
boot. Starting an already active service preserves it; use `restart` to apply
code or option changes. For a foreground process, stop the service and run
`./scripts/zed_stream.sh run`; Ctrl+C releases the camera.

The stream automatically selects the only ZED capture device in `/dev/v4l/by-id`.
If multiple cameras are present, select one explicitly:

```bash
./scripts/zed_stream.sh restart --device /dev/v4l/by-id/YOUR_ZED-video-index0 --port 8766
```

Endpoints on the same local server:

| Endpoint | Result |
|---|---|
| `/` | Browser viewer |
| `/stream.mjpg?view=left` | Live multipart JPEG stream |
| `/snapshot.jpg?view=left` | Latest fresh JPEG frame |
| `/status` | Capture health, frame age, sequence, measured fps, resolution, and device |

Use `view=right` or `view=stereo` for the other images. No frames are recorded to
disk. The server binds to loopback only. For remote viewing, forward the port
with `ssh -N -L 8766:127.0.0.1:8766 USER@ROBOT_DESKTOP`, then open the same URL
on the SSH client. Stop this stream before opening the camera in another program.
If capture fails or frames become stale, image endpoints refuse stale snapshots;
inspect `/status` and the logs, check the USB connection, then restart.

Dependencies are pinned separately in `requirements-camera.txt`. The scripts
use neither the IK environment nor robot control interfaces. Camera frames are
not yet integrated into the robot observation API. The connected ZED 2i was
verified at approximately 30 fps with all three image views and consecutive
decodable MJPEG frames.

## Calibrate the head camera

The ZED calibration workflow plans a nine- or fifteen-pose single-arm route with official IK,
records stationary images with measured joints, lets you select fixed 3D arm
landmarks and their image pixels, then fits a candidate camera transform with
held-out-pose validation. See [the calibration guide](docs/CAMERA_CALIBRATION.md).

```bash
./scripts/setup_camera_calibration.sh
.venv-official/bin/python scripts/calibrate_camera.py demo --output logs/calibration/demo
.venv-official/bin/python scripts/calibrate_camera.py annotate --dataset logs/calibration/demo/dataset.json
```

This opens an explicitly synthetic rehearsal dataset at http://127.0.0.1:8772.
Real captures need this ZED's matching raw intrinsics. Physical route execution is
separate and explicit; fitting never applies calibration or commands the arms.

The [physical calibration archive](calibration/2026-10-15_zed35330610/README.md)
includes raw images, measured joints, assistant-selected landmarks and calculated
extrinsics for ZED serial `35330610`. The latest raised sequence has 15 captures
and 132 observations: 4.521 px all-point RMS and 7.010 px worst held-out-pose RMS.
**These extrinsics are candidates, not validated or applied to robot control.**
The archive includes earlier results, annotation corrections and reproduction
instructions. The [camera mount design](hardware/zed2i_mount/README.txt) includes
printable STL, editable STEP and installation illustrations.

## Run interactive IK

Run in a desktop terminal:

```bash
cd ~/projects/agilex-deploy
./scripts/setup_official_ik.sh  # First checkout: isolated Pinocchio/CasADi environment.
./scripts/run_interactive.sh --backend sim
```

Clone with `git clone https://github.com/ZiWang1105/agile-x-deploy.git agilex-deploy` to use the directory name above. The repository includes the source assets required by this assembly; the original upload archives are not needed to build it. The setup uses `.venv-official` because standard pip Pinocchio wheels do not provide the required CasADi binding. The old `.venv` is preserved; missing official dependencies cause an explicit error, never a fallback solver. Generated files in `build/` are ignored. `build/dual_nero.sim.xml` is generated automatically from the geometry model when absent.

- Double-click a red sphere to select a TCP target.
- Hold **Ctrl + right mouse button** and drag to translate the target.
- Hold **Ctrl + left mouse button** and drag to rotate the target. Shift changes the manipulation plane/axis using MuJoCo's native controls.
- **1 / 2** toggle the left/right gripper target.
- **Space / R** enter HOLD and reset handles to the measured TCPs. Drag again to resume.
- **Q** or window close exits and releases the owned control thread.

Red spheres/axes are targets; small green markers are encoder-derived TCPs. The display reports position/orientation error and runtime state. The renderer has its own model; dragging robot geometry or using native Reset cannot reset or force the simulated plant. Native Reset requests HOLD.

The CLI uses the active desktop's `DISPLAY`. On this machine, `DISPLAY=:1 ./scripts/run_interactive.sh` was tested successfully. Prefer the display variable supplied by your desktop session. The command stays in the foreground; no server or detached process is installed.

## Live hardware mirror and IK preview

On the commissioned desktop, with both CAN interfaces already configured at 1 Mbps:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-preview
```

Blue arms show physical joint feedback. Translucent amber arms show simulated IK, initialized from the physical joint angles when the application starts. Drag the red EEF targets with the same controls as simulation. **Targets command only the preview simulator; this mode cannot enable or move hardware.** Press **M** to hide/show the amber preview and inspect the blue measured posture. Space/R stops only the simulation preview. Gripper controls are disabled and fingers are hidden because physical travel is not calibrated.

The reader resolves each arm by the confirmed USB serial in `configs/hardware_mapping.json`, even if Linux renumbers `can0`/`can1`. It refuses missing/ambiguous adapters, stale feedback, receive overflow, and excessive paired-frame skew. A hardware feedback fault freezes the last measured display, cancels preview tracking, and requires restart. This is a receive-only commissioning tool, not the `RealBackend` motion transport. It does not fabricate physical velocity or gripper measurements to satisfy the full robot contract.

```bash
# Read a paired joint snapshot; sends no CAN data frames.
.venv-official/bin/python -m agilex_deploy.can_feedback --output logs/commissioning/live_snapshot.json

# Test simulated IK from live joints and render a comparison image/report.
MUJOCO_GL=egl .venv-official/bin/python scripts/validate_live_preview.py
```

See [hardware commissioning](docs/HARDWARE_COMMISSIONING.md) for the remaining steps before physical following.

## Single-arm physical commissioning

For full-workspace mouse control, start with both arms supported and disabled, clear the arm's movement area, then run:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-commissioning --side both --workspace full --gripper
```

This starts **UNARMED**. E enables both selected arms holding their current postures. Double-click either red sphere, Ctrl+right-drag to translate, and Ctrl+left-drag to rotate. Full mode removes the radius, height floor, orientation cap and joint excursion cap relative to the arming pose. Full workspace is now the default. Joint position limits, live controller speed/acceleration limits, pair-wide ownership and communication watchdogs remain. The old 0.03 rad/s, 0.1 rad/s², 5% firmware speed and arbitrary tracking-error caps are removed. It is not a collision-aware planner: the table, stand, opposite arm and other obstacles are not protected. An IK target may be unreachable even though the handle can be dragged there. Space requests HOLD; Q closes the controller and leaves motors enabled. To use the identical full-workspace controls without hardware access, replace `real-commissioning` with `commissioning-sim`.

On **E**, the transport selects CAN joint-position mode at **100% of the controller's configured speed**, reads that selected controller's seven position/speed/acceleration limits while disabled, seeds current angles, then enables its joints. Incomplete readback prevents enable; settings from the other arm are never substituted. The speed reference respects the lower of the manual maximum and controller readback. Joint position bounds use the intersection of the supplied URDF and controller settings, without additional margins.

The October 14 left-arm readback was `[3.14, 3.14, 3.14, 3.14, 3.92, 3.92, 3.92]` rad/s and `[2, 2, 2.5, 2.5, 2.5, 2.5, 2.5]` rad/s². These are ceilings, not constant commanded speeds. Simulation uses that readback as an explicit offline fixture for both arms; real sessions read their own selected controller anew. The right-arm read-only probe did not return limits in its observed mode.

Torque/current protection remains in the firmware, unchanged by this host. A numeric hardware torque ceiling has not been verified: URDF effort `100` and MIT torque encoding `±16 Nm` are not evidence of a safe joint torque limit. The simulator's force range is not a validated hardware torque model. See [limit provenance](docs/HARDWARE_COMMISSIONING.md#motion-limit-sources).

**G** requests a fixed nominal-TCP target 30 mm above the arming pose; repeated presses do not accumulate movement. Grippers are not enabled or commanded by default. Add `--gripper` to expose selected-side feedback and explicit open/close keys; **E** and pose dragging alone still send no gripper commands. See [gripper control](docs/GRIPPER_CONTROL.md). An optional legacy envelope remains available only with explicit `--workspace bounded`; it is never selected by default.

To open both physical jaws and finish with both arms disabled, close active
controllers, mechanically support both arms, and run:

```bash
.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported --open-grippers
```

This temporarily holds each arm at its measured joint positions while opening
its jaw, verifies opening from feedback, then disables the arm again.
Omit `--open-grippers` for disable-only behavior.

Gripper motion uses AgileX's official width/force command path. The shared `Observation.gripper_states` reports measured endpoint progress, stationary shortfall, and available width/force/status feedback; the viewer displays those states. A `stalled` report is not a confirmed grasp and does not alter the firmware target. This is feedback supervision, not a port of Hexfellow's host-side motor-effort regulator. The simulator reports force as unavailable, and physical force behavior remains untested.

**Space/R** requests servo HOLD. **Q/window close requests HOLD and leaves motors enabled**; it does not release the arm or guarantee physical emergency stopping. A CAN or power failure can prevent HOLD. Full-range physical following and failure-stop behavior remain unqualified. Use `--side both` (the default) to control both arms and both grippers in one owned session; `--side left` and `--side right` retain single-arm selection. Both buses share admission, watchdog, and fault handling. A fault on either side requests HOLD on both.

Practice the identical arming/UI workflow without opening CAN:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend commissioning-sim --side both --gripper
.venv-official/bin/python scripts/check_real_session.py  # Reads real feedback; never arms.
```

The hardware control loop runs in an owned, joined child process. A viewer heartbeat gap over 200 ms latches HOLD independently of the viewer's Python GIL. This covers viewer failure; it is not a certified controller-level watchdog or proof of stopping after the control process, host, CAN link, or power fails.

## Same interface for sim, real, and agents

```python
observation = robot.observe()
request_id = robot.submit_target(action, observation.observation_id)
robot.hold()
robot.close()
```

`Action` always contains both arms: a base-frame TCP position in meters, an XYZW unit quaternion, and `gripper_open` (integer 0/1, or null to leave unchanged) per arm. `Observation` contains the 14 named arm encoders, measured velocities, normalized gripper travel, encoder-derived TCPs, acquisition timestamp, observation ID, and runtime state. Camera observations are explicitly unavailable in this release.

`Robot(SimBackend())` and `Robot(RealBackend(transport))` run the exact same contract, IK, target admission, and joint-reference controller. A policy or viewer receives `RobotInterface`; it does not branch on the backend. Only the composition root selects the backend. See [the architecture and commissioning boundary](docs/INTERACTIVE_IK.md) and [a minimal agent-style policy](examples/agent_policy.py).

The viewer uses the shared request path and does not command joints. The simulation uses position actuators and `mj_step`, with a 100 Hz control loop and 500 Hz physics. Setting simulator `qpos` is restricted to initialization; the separate FK/display models are reconstructed from encoder observations.

## Scope and validation

The controller loads the unmodified official `ArmIK` class (Pinocchio + CasADi + IPOPT) and the pinned NERO YAML. Its soft objective trades pose error against posture regularization and smoothing; submillimeter convergence is not promised. Joint targets use source position bounds and per-joint speed/acceleration limits described above. Cartesian velocity clipping and fixed tracking-error thresholds are removed; the optimization code is upstream, with `w_pos` changed from `2` to `40` and `w_ori` from `2` to `4` in `configs/ik.yaml` to reduce the persistent zero-posture bias. Official `w_reg=0.01`, `w_smooth=0.1`, joint bounds and solver settings are retained. HOLD cancels the target and captures the current position; it is a servo hold, not a physical emergency stop. Targets expire after 0.5 s without refresh. Stale feedback, bad joint ordering, controller faults, and control overruns cause a latched fault and backend HOLD.

The simulator uses ideal gravity compensation. **Contacts and collision admission are not implemented.** It is suitable for free-space interactive IK, not grasp/contact validation or hardware execution. Source inertias and limits are unqualified. The TCP follows the official NERO `ee`: `[0.1755, 0, -0.0235]` meters with zero RPY relative to each `joint7` child frame. Observation schema version 3 identifies this change from the previous TCP. Recreate targets from fresh observations; do not replay old TCP targets. This is a nominal official definition, not a measured tool calibration. Top-camera orientation is preserved, and its bracket joint is fixed in the runtime model.

```bash
./scripts/setup_camera_calibration.sh  # Image-processing dependency for calibration tests.
./scripts/setup_gripper.sh             # Pinned official gripper codec for protocol tests.
.venv-official/bin/python -m unittest discover -s tests -v
MUJOCO_GL=egl .venv-official/bin/python scripts/smoke_ik.py
.venv-official/bin/python scripts/validate_assembly.py
.venv-official/bin/python examples/agent_policy.py
```

The automated suite covers paired 6D tracking, command limits, actuator-only motion, HOLD/resume, command expiry, stale feedback, invalid joint order, unreachable-target handling, gripper coupling, camera lock, Jacobian finite differences, drag/reset routing, thread/process cleanup, receive-only parsing, diagnostic request whitelists, startup setpoints before enable, nullable gripper capabilities, commissioning admission, and the independent viewer heartbeat watchdog. Command-path evidence uses simulated or in-memory transports; it does not establish physical stopping or collision clearance.

`previews/official_ik.png` and `previews/official_ik_report.json` show the current official solver with local tuning: submillimeter position residual in the seven-second simulation. `previews/official_defaults_ik_report.json` preserves the original-default comparison (17.20 / 16.09 mm) using the same TCP and targets. These are modeled errors, not physical accuracy measurements. The older `interactive_ik` artifacts are historical. The native desktop window also passed a timed launch/render/close test. The tuned solver also passed a 165-second native desktop viewer check using X11 mouse/keyboard events: both handles translated, orientation dragging changed the target by about 14 degrees, and HOLD/resume and clean exit worked without a controller fault. Settled sampled position errors were 0.049–0.072 mm. See `previews/interactive_desktop_report.json`. This was automated desktop input, not a human mouse session or physical-robot test.

The historical [32-second validation video](previews/dual_nero_ik_validation.mp4) shows translation, orientation tracking, independent gripper commands, HOLD during motion, and resume. Its target sequence uses the same robot API as the viewer. It is simulation-time playback, not a mouse recording. It is retained as evidence of the previous solver; use `scripts/smoke_ik.py` for current official-solver validation.

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
./scripts/setup_official_ik.sh
.venv-official/bin/python scripts/build_assembly.py
.venv-official/bin/python scripts/validate_assembly.py
MUJOCO_GL=egl .venv-official/bin/python scripts/preview_assembly.py
```

For the standard local MuJoCo model inspector on a desktop with a display:

```bash
.venv-official/bin/python scripts/preview_assembly.py --viewer
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

The active runtime now uses the official optimization solver directly, without starting its ROS node or launch files. Installation uses Pinocchio 3.2.0 with CasADi support. Source hashes, exact parameters, the virtual-gripper compatibility fix and validation results are documented in [OFFICIAL_IK.md](docs/OFFICIAL_IK.md).

## Provenance

Inputs are the user-supplied `wbcd_urdf-20261014T025717Z-1-001.zip` and `agx_arm_sim-20261014T025717Z-1-001.zip`. Archive hashes are recorded in `source_manifest.json`.

Visual reference: [AgileX exhibition photo](https://x.com/AgilexRobotics/status/2089985024515002714/photo/1). The image was retrieved from the post's public X syndication media URL. It shows the same general stand-and-dual-arm configuration; no exact model/pose identity is asserted.

Preserve the upstream license notices included with the assets. The supplied stand archive does not include a standalone license file; this workspace does not assign it a new license or establish redistribution rights.
