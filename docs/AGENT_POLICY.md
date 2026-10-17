# RGB-D agentic cube pickup

The policy uses one sensor/action contract across simulation and the physical
NERO adapter. It receives unannotated head RGB, registered metric optical-Z depth,
camera intrinsics/extrinsics, measured joints, encoder-derived TCP/pinch poses,
and measured gripper state. It returns paired Cartesian poses, binary gripper
commands, an optional image-pixel grounding request, and an execution duration.

The installed sensor is a single stand-mounted ZED 2i. There are no wrist-camera
observations. The agent scene removes the wrist-camera assemblies inherited from
the supplied URDF, including their mass and collision geometry. The supplied
source model remains unchanged. The existing archived validation video predates
this geometry correction; its visible wrist cameras were CAD parts, never policy
sensors. Its results apply to that archived model, not a new pickup trial.

There are no host object detections, scene coordinates, scripted pickup phases,
object welds, or model-generated Python execution. Arm selection, approach,
closure, lift, recovery and termination are model decisions. The implementation
follows the observation/decision/execution pattern of `agentic-mobile-manipulator`
without importing that application or its Hexfellow-specific controller.

## Run simulation

```bash
cd ~/projects/agilex-deploy
uv pip install --python .venv-official/bin/python -r requirements-agent.txt
./scripts/run_agent.sh --backend sim --max-calls 16 --video \
  --prompt 'Pick up the red cube, lift it at least 10 cm, and hold it for at least 10 seconds.'
```

Open **http://127.0.0.1:8770** on the robot desktop. The dashboard starts by default.
Edit the task prompt and click **Start simulation**. No model calls or simulated
motion occur while it waits for Start. The page shows:

- A live external simulator view with orbit, elevation, and distance controls.
- The live fixed head-camera RGB view and the exact RGB frame used for the latest decision.
- Model request status, decision summaries, targets, measured execution feedback,
  gripper state, API calls/cost, and the final episode evaluation.

The external observer view is never supplied to the policy. Rendering copies
plant state under the controller lock, then releases it before drawing.
There are no wrist-camera feeds. In physical mode the head panel continuously displays the real ZED stream from
the same camera owner used for policy RGB-D. The separate policy panel retains
the exact frame used for the latest request.

**Stop / HOLD** latches cancellation at the execution owner. A pending network
request can finish, but its returned decision cannot execute. HOLD does not
disable physical motors. **Close session** stops the episode and closes the
server after pending work and cleanup finish. Each process runs one episode;
restart the command for a fresh scene. The final page remains open for review.
Ctrl+C closes the process. Use `--no-web` for immediate headless execution, or
`--port PORT` to select another loopback port. Physical enable requires
`--backend real --execute-real` and an explicit **Start real policy** click.
Without `--execute-real`, the website is a live monitor with Start disabled.
With `--no-web --execute-real`, the command itself starts physical execution.

This requires the existing official IK environment. The default model is
`google/gemini-3.8-flash`. The default external key file is
`~/projects/openrouter_key.txt`; override it with `--key-file PATH`. The client
reads the key only for HTTPS authorization. Headers and key contents are never
written to episode logs. `--max-calls` and `--max-cost` bound requests and reported
spending; the cost threshold is checked between calls, so a last request can
cross it. The default cost threshold is $5. API calls incur provider charges.

`--prompt` (alias `--instruction`) sets the task sent on every policy observation.
If omitted, the launcher uses its default red-cube pickup instruction.
`--scene` selects plant geometry and initial state.
Task instructions do not belong in scene files. `--output` must name a new
directory; otherwise a timestamped directory under `logs/agent` is created.
The simulator starts from a generic raised posture, not a scripted approach.

## Shared contract and execution

- `agent_sensors.RGBDFrame` defines uint8 RGB, registered optical-Z depth in meters,
  pinhole K, optical-to-base T and a host acquisition timestamp. Invalid depth is
  zero; archived depth is uint16 millimeters. No RGB-only fallback is used.
- `agent_policy.packet()` explicitly allowlists sensor fields. Rendered and real
  images pass through the same code. Camera/encoder receipt skew is at most 100 ms;
  this is not hardware exposure synchronization.
- Policy poses refer to a nominal pinch point 15 mm behind the official NERO TCP,
  between the distal finger faces. The adapter converts them to the existing
  `Action`/`RobotInterface` TCP convention before official IK. Tool geometry is
  CAD-derived and not physically calibrated. Local +X points along the hand;
  local Z is the jaw-separation axis. Coordinates are in stand `base_link`.
- The model supplies normalized `pixel_uv_1000` coordinates, converted explicitly
  to sensor pixels. Only that pixel's archived depth is back-projected. A 3x3
  neighborhood rejects invalid/discontinuous depth; it does not select a feature
  or fill holes. `phase=inspect` returns geometry without moving either arm.
- A separate execution owner maintains robot updates while HTTPS waits. Each
  policy interval ends in measured arm HOLD with the current binary gripper
  request preserved. This preserves closure while the model deliberates.
  Execution returns measured pose errors and an `arrived` diagnostic; elapsed
  duration alone is not arrival. No Cartesian speed or workspace cap is added.
- Ctrl+C/error/exit requests the existing full robot HOLD and joins owned workers.
  Motor disable is not implicit. Existing hardware feedback, ownership, joint
  limits, command leases and the independent 200 ms viewer/client watchdog remain.
  A 120-second idle policy-session lease prevents indefinite abandoned execution.

Simulator and hardware have identical packet fields and action semantics.
Both simulation and hardware expose **both arms in one session** with
`--side both` (the default). `--side left` or `--side right` restricts the policy
on either backend. The policy receives `available_arms`, and attempts to command
an unavailable arm are rejected. A shared action contains both arm poses and
independent gripper targets. One physical control owner validates both targets
before dispatch and requests HOLD on both arms if either side faults.
The two CAN adapters are written sequentially within a control tick; this is
coordinated control, not hardware-triggered synchronization. Paired physical
motion still requires measured qualification; this change was tested offline.

## Physical deployment

First run the non-actuating preflight:

```bash
./scripts/run_agent.sh --backend real --side both
```

It starts the existing unarmed hardware owner, captures robot feedback and ZED
RGB-D, saves the observation, and keeps the live browser monitor open. Add
`--no-web` for a one-shot preflight that exits. It performs firmware/range read queries;
it sends no arm-enable, gripper-motion or arm-position commands. It requires the
existing disabled-arm startup state. It makes no model API calls.

To prepare the real policy dashboard (the arm remains unarmed until Start):

```bash
./scripts/run_agent.sh --backend real --side both --execute-real \
  --instruction 'Pick up the cube with the available arm and hold it above the table.'
```

Open **http://127.0.0.1:8770**, confirm the badge says **PHYSICAL ROBOT**, review
the live ZED image and measured feedback, edit the prompt, then click
**Start real policy**. This enables the selected arms at their measured starting
posture after reacquiring fresh RGB-D/feedback. Start cancellation and stale
camera failures prevent enable. Use `--side right` for the other arm. The launcher
uses the existing official NERO joint-position/IK path and official gripper
width/force commands; it does not use Hexfellow MIT actuation.

The camera implementation uses the already connected ZED's raw UVC stereo stream,
factory calibration, OpenCV rectification and StereoSGBM. The ZED SDK is not
installed in this environment. This is an explicit CPU stereo implementation,
not a claim of ZED neural-depth quality. The raw-left extrinsic is rotated by
the inverse left rectification rotation before grounding rectified depth. The
saved calibration remains a candidate; using it in a packet does not promote it
to physically validated calibration. Camera selection now reads the serial directly from each ZED video device
using the UVC identity-read protocol documented in Stereolabs zed-open-capture.
It requires exactly one match to the calibration camera serial (35330610 here),
checks again after opening capture, and rejects missing, mismatched, duplicate,
or unreadable identities. It does not infer identity from the generic OV0001
USB label or a separate HID device. `--camera-serial` can state the expected
serial explicitly; it must still match the extrinsic calibration. An optional
`--camera-device /dev/videoN` restricts discovery but does not bypass verification.
The live panel displays the measured serial and device path. Disconnects stop
the session; frozen frames are labeled as stopped, not live.

Read-only physical preflight succeeded with a 640x360 RGB-D frame, about 34.5%
valid depth pixels and 11.7 ms camera/encoder receipt skew. The arm remained
unarmed. Live depth accuracy, camera mounting consistency, physical grasping,
force behavior and autonomous collision clearance remain unverified. Review the
saved live view and candidate geometry before using the actuation launcher.

## Simulation and evidence

`agent_sim.py` enables source mesh/environment contacts, excludes planar meshes
from convex-hull collision generation, and adds distal inner-face contact proxies
inside the finger envelope. Finger mechanical stops are stiffened without
clipping encoder feedback. The cube is a free rigid body with mass and gravity;
friction/contact alone supports a grasp. Arm self-collision is not modeled by
this contact mask. Arm servos use ideal gravity compensation; jaw forces and
friction are not calibrated hardware dynamics.

The evaluator runs only after policy termination. Success requires at least
100 mm lift, opposing finger contact, no table/floor support, at least 10 seconds
of uninterrupted retention through the final verification, and no arm/table or
arm/floor collision during the episode. The model cannot read evaluator state.
The final verification lasts 11 wall-clock seconds, independent of whether the
model claims success; the evaluator measures actual simulated elapsed time.

The final default-policy validation used seven live OpenRouter calls. It lifted
the cube **152.2 mm**, retained it for **32.6 seconds**, and recorded zero arm/
environment collision frames. Reported API cost was **$0.06927**. A preceding
14-call run recovered from a missed grasp and also passed. These runs do not
establish general grasp success rates or sim-to-real transfer.

The complete regression suite passed **108 tests** in 135.195 seconds, including
rendered depth projection, tool-frame round trips, normalized pixel handling,
inspect-only behavior, rectification-frame composition, evaluator retention,
credential exclusion and model-input field admission. The input audit reproduced
all seven archived requests and the selected-pixel geometry. See the
[archived evidence](../validation/agent_cube_20261015/README.md) and
[complete video](../previews/agent_cube_pickup.mp4).

Episode artifacts include exact credential-free requests, model outputs, RGB and
metric depth frames, camera geometry, resolved actions, source snapshots/hashes,
and a separate evaluator trace. Video renders the complete measured simulator
trajectory at 10 fps, including waiting, recovery and the final verification;
it is not a mouse recording or a scripted pickup replay.

```bash
.venv-official/bin/python scripts/audit_agent_run.py logs/agent/cube_agent_08
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-official/bin/python -m unittest discover -s tests -p test_agent.py -v
```

Earlier failed runs remain in `logs/agent/cube_agent_01` through `_07`, alongside
the successful `_06`; they exposed pixel-convention, joint-stop and nominal
grasp-frame errors. The archived successful evidence identifies its exact source
and configuration. No earlier failed or collision-assisted run is counted as a
successful pickup.

References: [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs),
[Stereolabs native OpenCV capture](https://github.com/stereolabs/zed-opencv-native),
[Stereolabs OpenCapture depth example](https://github.com/stereolabs/zed-open-capture/blob/master/examples/zed_oc_depth_example.cpp).

### Managed desktop dashboard

The current desktop dashboard is a transient user service named
`agilex-agent-dashboard.service`, bound to `127.0.0.1:8770`. It now connects to
the **real** ZED and robot, selects the left arm, and waits unarmed for
**Start real policy**. Closing the session ends that service. It does not start
at login. The separate `agilex-zed-stream.service` was stopped to transfer the
single ZED device to this dashboard; camera streaming is now on port 8770.

```bash
systemctl --user status agilex-agent-dashboard.service
journalctl --user -u agilex-agent-dashboard.service -n 40 --no-pager
systemctl --user stop agilex-agent-dashboard.service
```

After closing it, run the physical deployment command above to create another
real session. The simulator remains available with `--backend sim`.

Dashboard validation passed 110 full-suite tests in 140.082 seconds. Browser
checks covered desktop/mobile rendering, prompt editing, Start, viewpoint changes,
and Stop during a live OpenRouter request. No browser JavaScript errors were
observed. The dashboard changes have not been used for a new full pickup trial.

Real-dashboard verification: nine focused tests passed, including real-launcher
Start gating, cancellation before enable, and stale-camera rejection. A browser
check confirmed continuous live ZED frames and changing robot feedback IDs,
with the physical arm unarmed, zero model calls, and no JavaScript errors. The
real Start button was not clicked during verification. The initial dashboard
showed a room view before the camera disconnected. After reconnecting and fixing
identity selection, the directly verified camera 35330610 shows both grippers
and the worktable, visually matching the calibration images. This identifies
the camera; it does not independently validate the fitted extrinsics.

Camera-selection regression checks passed four identity tests plus nine agent
and dashboard tests. The protocol reference is
[Stereolabs zed-open-capture, pinned source](https://github.com/stereolabs/zed-open-capture/blob/93739154903a66c112ee52c3c6fbfd83b7218617/src/videocapture.cpp).
Only the identity flash-read command is implemented; there are no reset or
flash-programming commands. The video device returned serial 35330610 directly.
