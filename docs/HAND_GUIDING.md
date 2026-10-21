# Optional hand-guiding (disabled by default)

This guide covers optional hand-guiding for the
[Workshop on Agentic Robotics](https://agentic-robotics-workshop.github.io/)
demo platform.

`gravity_comp=False` is the default for `WorkshopService` and `HardwareSession`.
Both [workshop](https://agentic-robotics-workshop.github.io/) policy runners, the service launcher, the interactive commissioning
viewer, and the legacy agent runner accept `--no-gravity-comp` explicitly; omitting
the option has the same effect. `--gravity-comp` opts into the experimental
hand-guiding lifecycle below in simulation. Physical hand-guiding remains blocked.

With the default setting, Enable remains in follower position HOLD. Policy
completion, session release, STOP, Ctrl-C, and Close never request leader mode.
They retain position HOLD and do not automatically disable motors. To remove
holding torque, first support both arms, close the controller, then explicitly
run `.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported`.
The tool confirms all 14 motors disabled. No compensation is sent afterward;
disabled arms require mechanical support. The simulator's internal ideal servo
gravity feedforward is distinct from this optional hand-guiding lifecycle.

**Ordinary physical position control is restored at the operator's request.**
The former blanket enable block is replaced with fresh disabled-follower,
healthy-status, stationary-feedback admission checks. Controller-limit,
measured-position seeding, arming-displacement, and watchdog checks remain.
These checks do not prove firmware startup is motion-free or resolve the original
swing. The separate `PHYSICAL_HAND_GUIDING_BLOCKED` guard rejects physical
gravity-compensation startup before any enable command, and blocks direct leader
transition requests. It has no CLI/configuration override. The opt-in lifecycle
below remains available in simulation. Explicit supported shutdown is unchanged.

## Supported gravity-compensation diagnostic

**Current incident status (2026-10-20, 16:07 run): do not repeat physical
commissioning yet.** The operator reported falling and could not distinguish
descent during leader mode from descent after the diagnostic's damped stop.
Leader encoder reception is not proof of weight support. Both arms were
subsequently disabled with operator-confirmed support; shutdown feedback showed
0/7 motors enabled on each side. The commands below document the diagnostic;
they are not a recommendation to retry this unresolved incident.

`scripts/test_gravity_comp.sh` is a separate commissioning command. It does not
patch or bypass the policy transport's interlock. It has a closed command
allow-list and no policy, IK, joint target, MIT, gripper, mounting, reset or
motor-disable command path. Physical enable is available only with the explicit
`--arms-supported` confirmation; it is an experiment, not qualified deployment.

```bash
cd ~/projects/agilex-deploy
./scripts/test_gravity_comp.sh --side left --check-only
./scripts/test_gravity_comp.sh --side left --arms-supported
```

Keep both arms mechanically supported against falling or swinging throughout
the test, keep the sweep clear of people and objects, and keep the emergency stop
accessible. Close other robot controllers and leave the official Web UI controls
untouched. The tool takes the same pair-wide ownership lock as the policy and
supported shutdown. It resolves CAN interfaces by the commissioned USB serials
and requires fresh firmware 1.21 responses, both arms disabled, healthy CAN joint
mode, stationary encoders, and measured joints within the model limits.
`--check-only` sends two firmware read requests but no control commands.

During the three-second cancel window, the command continuously drains and
validates fresh feedback. `--check-only` exercises the same three-second receive
window. Sleeping with receive sockets open caused the first operator run to
overflow before enable; the regression now models an independently arriving
stream instead of only generating frames on demand. Overflow detection remains
strict; dropped feedback is never ignored or accepted as fresh.

The second operator run exposed a separate timing bug: immediately after enable,
the reader returned its latest still-fresh cached snapshot before the next CAN
cycle arrived. The diagnostic incorrectly treated that repeated sequence as a
fault and requested emergency stop before sending leader-mode commands. It now
waits up to 50 ms for acquisition progress, validating timestamps throughout.
Repeated samples never count as acknowledgement; stale samples, backwards
sequences, or no progress within the deadline still fail. Regression tests cover
both a 20 ms cached snapshot and a genuinely frozen stream. All 15 diagnostic
tests passed after this fix. These tests do not verify physical gravity support.

After the operator-authorized shutdown, live feedback confirmed all 14 motors
disabled, but the left controller still reported emergency-stop status. The
read-only preflight correctly refused another test; disabling motors does not
itself establish that the controller emergency-stop state has cleared.

After that receive window, the physical command invokes the wire
encoding produced by the pinned SDK's `enable(255)` for the selected arm. Once
fresh driver feedback confirms all seven selected motors enabled and healthy,
it sends the pinned SDK's `set_leader_mode()` messages. It never sends a position
seed before enable or replays a target afterward. Firmware may activate an
unknown retained target during enable; the host monitor cannot guarantee no
movement before stopping. Mechanical support and clearance are required.

The command checks two seconds of stillness after receiving leader encoders,
then announces a 30-second supported dragging window (`--seconds`, 1–120).
The other arm remains disabled. Leader encoders confirm the feedback stream;
motor status and physical torque support remain unknown. Check for gentle manual
motion without strong pulling or stiffness, keeping supports in place. Do not
force a resisting joint or release support to test whether the arm falls.

During this diagnostic only, a 2-degree startup/untested-arm excursion, a
15-degree selected-arm excursion during dragging, or a selected joint speed
above 0.35 rad/s causes a stop request. These are diagnostic stop thresholds,
not firmware speed settings or new production IK constraints. Missing/stale
feedback, unhealthy drivers, unexpected other-arm enable, and wrong modes also
fail the test. No automatic fault clearing or follower/position-mode recovery
is attempted.

**Ctrl-C, SIGTERM, a fault, or the deadline requests the SDK's damped emergency
stop three times. This may permit descent; keep the arms supported.** A stop
frame reaching SocketCAN is not proof of physical stopping. The tool does not
disable motors or claim the robot can be powered off. If movement persists,
use the physical emergency stop. A killed process, disconnected CAN, controller
failure, or lost power cannot be handled reliably by this host-only monitor.

To explicitly disable after the test, with BOTH arms fully supported, use the
existing `python -m agilex_deploy.hardware_power --arms-supported` command in
`.venv-official`. Leader exit and shutdown remain separate, acknowledged steps;
never assume Ctrl-C disabled the robot. A latched controller emergency/fault
requires review before another enable; this diagnostic intentionally does not
reset it automatically. Repeat with `--side right` only after the first arm is
supported and confirmed disabled and both controllers pass preflight again.

Logs under `logs/gravity_comp/<timestamp>/` contain command attempts/delivery,
timestamped paired feedback, post-stop feedback when available, and a report.
The report deliberately keeps `physical_gravity_compensation_verified=false`
and `stop_confirmed=false`: an operator's physical assessment is still needed.

Offline validation runs this exact diagnostic state machine against the existing
MuJoCo/firmware simulator with real sockets forbidden, including independent
left/right entry, retained-target motion, missing leader feedback, interruption,
and stop delivery despite failed logging. Simulated stop delivery does not model
or qualify real firmware damping. Run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python -m unittest discover -s tests -p test_gravity_comp_test.py
```

## Opt-in policy lifecycle (`gravity_comp=True`)

Normal policy completion, participant session release, dashboard STOP, and normal
hardware-session close enter hand-guiding with the motors enabled. No additional
terminal command is required. Enabling a session also enters this idle mode until
a policy submits a target. Both arms switch when the session owns both arms;
left and right grippers remain independent.

The implementation calls the pinned official `pyAgxArm` NERO v121 driver's
`set_leader_mode()` and `set_follower_mode()` methods through an encoder-only
adapter. The adapter uses the official codec and the existing exclusive CAN
writer; it does not start another SDK connection or command thread. It preserves
the configured mounting orientation and 100% firmware speed setting. It does not
implement host-side gravity torques or change motor torque limits.

Official reference: [NERO leader/follower API](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero_api.md#set-leader-mode--set_leader_mode).
The documented behavior is zero-force dragging. The adequacy of gravity support
for these side-mounted arms and attached tools still requires a supported physical
test; software mode acknowledgement is not a measurement of gravity compensation.

## Opt-in lifecycle

| Event | Behavior |
| --- | --- |
| Startup | Observe only. No constructor enables motors. |
| Organizer enables | Enable at measured positions, stop, then enter hand-guiding. |
| First policy target after hand-guiding | Wait for stationary feedback, seed current positions, switch to follower, refresh controller/guard references, then execute. |
| LLM waiting, Python fence returns, operation completes | Keep position HOLD while the policy still owns the session. Do not release a carried object between steps. |
| Policy releases session | Cancel pending targets, stop, and enter hand-guiding automatically. |
| Operator STOP | Cancel and latch the policy, stop, then enter hand-guiding. Later model responses cannot execute. |
| Normal service/process close | Enter hand-guiding, close the host connection, and leave motors enabled. Firmware owns the resulting mode; host monitoring ends. |
| Hardware fault, stale feedback, heartbeat loss while tracking | Latch the fault. Attempt the existing mode-appropriate HOLD. Never disable or claim a successful hand-guiding transition. |

An individual `robot_hold` or `operation_cancel` remains a position HOLD during
controlled execution. It does not end the policy session. HOLD while already
hand-guiding does not silently recapture the arm. `session_release` ends the
policy in both the Python client and MCP wrapper; their normal cleanup calls it.

Normal `robot_hold`, successful operation completion, and session release retain
each gripper's last actuator target and configured force. They must not replace a
closing target with the measured aperture: an object blocks closure, and using
that aperture as a new target removes the squeeze. Use `gripper_open(arm)` to
release an object explicitly. Cancellation, operator STOP, and fault handling
retain their measured-aperture stopping behavior; they do not guarantee grasp
retention. Physical grasp retention across leader/follower transitions still
requires hardware validation.

An already enabled robot can be reconnected only when fresh physical leader and
motor-status feedback establish its mode and enabled state. Firmware 1.21's
observed leader-only stream cannot meet that requirement: a new session refuses
to adopt that unknown enabled state. The supported shutdown procedure must restore
follower feedback and confirm disable before reconnecting. Within a running
session, leader-to-follower transitions use the current encoder posture and require
fresh follower status before policy targets resume. The pair-wide ownership lock
still applies. An enabled follower controller is not adopted.

## Feedback and failure handling

NERO 1.21 leader feedback uses `0x155`, `0x156`, `0x157`, and `0x170`. Those IDs
also carry outgoing joint commands. The receive path discards SocketCAN local
loopback (`MSG_DONTROUTE`) so outgoing targets cannot masquerade as encoders.
Each source change invalidates cached angle snapshots and rejects packets acquired
before the mode request. All seven angles must remain fresh for each arm. On the
observed firmware 1.21, leader mode suppresses normal driver, motor, status, and
gripper reports. The reader represents those reports as unavailable, including
`motor_enabled: null`; it does not reuse the last enabled bit as current evidence.
Leader velocities are differences of consecutive complete encoder samples,
explicitly labeled `angle_difference` in `feedback_status`. A confirmed leader
stream establishes the protocol mode only; it does not establish sufficient
gravity compensation. Follower mode still requires fresh driver, motor, status,
and configured gripper reports, and waits for the leader stream to stop. No
missing physical measurement is fabricated.

The SDK mode methods include a CAN-push disable frame. The follower return path
explicitly restores normal CAN reporting using the official codec before waiting
for follower confirmation. It does not change mounting configuration or firmware
speed limits. Suppression of normal leader-mode telemetry no longer causes a
false timeout. Missing leader angles or missing follower telemetry still latch
a fault.

Both arms are checked for stationary feedback before either transition request.
Both mode requests are sent before waiting for either acknowledgement. Mode writes
remain sequential across the two CAN adapters, not hardware atomic.
A partial change latches a fault. Cleanup does not send stale follower targets to
an arm whose mode is leader or uncertain, and does not automatically switch it
back or disable it. Inspect the reported mode and support the arms before recovery.

The existing supported-power-off tool can leave a detected leader mode before
requesting disable, as required by the official SDK. It remains a separate,
explicitly supported shutdown operation, never a policy-stop action.

## Validation and first physical use

**Current status (2026-10-20): physical hand-guiding is not qualified.** The real
leader transition lost expected status feedback and the user reported a near
collision. The offline firmware emulator now reproduces the communication failure
and exercises the corrected transitions and supported shutdown. It uses ideal
MuJoCo gravity compensation; it does not reproduce or establish the cause of the
reported physical swing. Do not repeat leader-mode tests until the transition
and gravity support have been separately qualified. Default position-control
policies no longer request that transition.

Run offline checks without touching CAN:

```bash
cd ~/projects/agilex-deploy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python -m unittest discover -s tests -p 'test_hand_guiding.py'

# Real transport + production CAN decoder + observed firmware stream behavior
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python -m unittest discover -s tests -p 'test_firmware_hand_guiding.py'
.venv-official/bin/python scripts/validate_hand_guiding.py
```

Tests exercise official encoded mode messages, paired transitions, missing
acknowledgements, partial failures, fresh-pose reseeding, gripper preservation,
process ownership, and [workshop](https://agentic-robotics-workshop.github.io/) policy release/STOP. MuJoCo removes only arm
position-servo gains while retaining the existing ideal gravity compensation and
jaw servos. This validates lifecycle behavior, not real motor torque support.

Do not use the existing launch/enable flow as a physical hand-guiding test yet.
The dashboard's `HAND-GUIDING` state confirms the expected encoder stream, not
that the firmware removed position stiffness or supplies correct gravity torque.
The user further described the incident as a sideways swing immediately after
Enable, followed by stiff pose holding that resisted manual movement. This shifts
the investigation toward unintended position control or an incomplete physical
mode transition; it does not establish a specific firmware or target error.

The startup code first enables joint-position control and only then requests
leader mode. The official SDK documentation says control commands do not take
effect while disabled, so sending a posture before enable does not establish
that the controller accepted it as its active target. The test firmware previously
accepted those targets while disabled. It now exercises the conservative case
where disabled commands are ignored. The real controller's target-latching
behavior during enable remains unverified. Likewise, the stored commissioning
record contains the user's earlier "side mounting" statement, not an independent
readback of the controller configuration. Do not infer NERO mounting values from
the Piper-derived left/right installation enum in the SDK message class.

On 2026-10-20, the user identified `http://127.0.0.1:8780/` as the page used.
The user then confirmed using only that dashboard, never the separate AgileX
settings page. That address is this project's [workshop](https://agentic-robotics-workshop.github.io/) dashboard. It does not expose or read
the manufacturer's mounting configuration. The earlier statement therefore
does not establish the actual controller setting; the stored statement is kept
as history, and software verification remains unavailable.

The official SDK discussion [#51](https://github.com/agilexrobotics/pyAgxArm/discussions/51)
contains a similar report: NERO V1.12 moved in one direction and resisted dragging
after entering leader mode. The reporter resolved that case by selecting the
correct mounting configuration after a firmware upgrade had left it unset.
This is a specific diagnostic lead, not proof of this robot's fault or a reason
to write guessed installation bytes. The same discussion confirms that normal
arm-status reporting is unavailable in leader mode.

The [NERO manual](https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/vx3iqycffq0pg8af)
also identifies zero calibration orientation, tool mass/center of gravity, and
legacy gripper ID compatibility as inputs to correct gravity compensation.
None has been diagnosed as the incident cause. Do not automatically recalibrate
zero, change gripper IDs, flash firmware, or change mounting settings as a test.
The linked CAN protocol workbook required login when fetched; its contents have
not been verified. No alternative teaching command has been inferred from it.

The manual section 4.1 documents a Wi-Fi route to the manufacturer's interface:
connect to the selected arm's `agx-7ax-xx` hotspot (default password `12345678`),
then open `http://192.168.31.1/` (default login `admin`, password `123456`).
Each arm uses the same default IP on its own hotspot; identify the controller
before assigning left/right labels. A fresh desktop scan on 2026-10-20 found two
matching hotspots. This does not establish their physical side mapping or any
mounting value. Inspect System Settings / Mounting Position first, without
enabling, changing modes, changing settings, or starting zero calibration.
Using a separate phone or laptop preserves the desktop's existing connection.

Later on 2026-10-20, the user provided official status-page screenshots after
changing the mounting setting. The 15:39:40 screenshot identifies Wi-Fi
`agx-7ax-4t4`; the 15:40:43 screenshot identifies `agx-7ax-m2t`. Both status
pages display `2-侧装`, CAN control, follower linkage mode, and gripper tool type.
The actual firmware UI offers `1-水平正装`, `2-侧装`, and `3-倒装`; it does not
offer separate left-side and right-side mounting values. This supersedes the
earlier absence of controller-setting evidence. These screenshots do not prove
motor enable state or physical gravity compensation. The physical hand-guiding
interlock remains in force pending supported validation of that transition.

The incident session saved only a `ready` event, not a continuous joint trace or
raw CAN command/feedback capture. Consequently, the exact time of motion relative
to enable, leader entry and error cleanup cannot be reconstructed from that log.
A future physical diagnosis needs those records and a separately reviewed,
mechanically supported procedure. Do not infer physical qualification from the
offline tests or the displayed mode name.

An offline sensitivity experiment is available:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/investigate_leader_swing.py
```

It contrasts ideal compensation, artificial torque errors, and a deliberately
injected old position target with the left servo still active. That last case
moves toward the injected target and resists a simulated push. These are
hypothesis tests using a preview posture, not reconstructions of the real swing.
Results are in `validation/leader_swing_investigation_20261020/sensitivity.json`.

## Offline official torque calculation

AgileX also provides a separate host-side implementation:
[`nero/main_gc.py`](https://github.com/agilexrobotics/agilex-arm-gravity-compensation/blob/8e2545c00ec9ec7590414946ab5f913274c7b643/nero/main_gc.py).
It calls Pinocchio inverse dynamics at a nominal 200 Hz and sends MIT commands
with zero position/velocity gains and feedforward torque. This is distinct from
the firmware-internal leader mode used in the failed physical diagnostic.

`official_gravity.py` imports the checksum-pinned, unmodified AgileX numerical
helper. It extracts both arms from the assembled URDF, uses the stand's fixed
mounting rotations, includes the grippers, and removes the absent wrist-camera
assemblies. Joints are mapped by name. The gripper opening must be supplied;
the preview assumes stationary jaws and no grasped object. It has no physical
writer, no enable API, and no mode-switch code. The upstream example's hardcoded
CAN channel, automatic enable, upright orientation, and position-control cleanup
are not adopted.

Run the entirely offline comparison using the recorded initial joint posture:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_official_gravity.py \
  --incident logs/gravity_comp/20261020_160715_191871/events.jsonl
```

The validation forbids sockets. MuJoCo has no position actuators or ideal gravity
compensation in these experiments. Pinocchio's static torques agree with MuJoCo
to numerical roundoff; both arms remain stationary for the 0.5-second test using
the 200 Hz torque calculation. Missing compensation and an incorrect upright
orientation produce substantial motion. MuJoCo static URDF fusion is explicitly
disabled: retaining fixed-link inertias is necessary for cross-engine agreement
with this model and installed importer.

These are mathematical checks of the same URDF, not a physical identification
or a reconstruction of the fall. The URDF mass/center-of-mass parameters, torque
scale and sign, object payload, startup transition, and host-loss behavior still
require qualification before a physical torque loop can be added. The existing
physical hand-guiding interlock remains in force. Do not execute the upstream
`main_gc.py` directly on this robot.

`validation/official_gravity_20261020/report.json` contains the simulations and
limitations. `incident_analysis.json` records what the actual test establishes:
leader feedback lasted 9.253 seconds; estimated left joint 4 speed reached
0.361291 rad/s against the diagnostic's 0.35 rad/s threshold; the right arm did
not move in the saved samples. Thirteen old post-stop log rows repeated one
snapshot. They do not establish continued post-stop acquisition. The diagnostic
now records its phase and first stop-attempt timestamp and excludes pre-stop or
duplicate snapshots from post-stop evidence. No stop threshold was relaxed.

## Receive-only incident recording

The following records existing traffic on both mapped interfaces. It sends no
CAN frames, does not enable either arm, and does not alter interface settings:

```bash
cd ~/projects/agilex-deploy
.venv-official/bin/python -m agilex_deploy.can_trace --duration 60
```

The output is a new JSONL file under `logs/diagnostics/`. Each frame retains its
kernel timestamp, CAN ID, bytes, side and origin. `host_echo` marks locally sent
commands; `bus_rx` marks incoming bus traffic. An echo does not prove controller
acceptance. Missing timestamps, truncated records and queue overflow make the
capture fail explicitly; existing recordings are never overwritten. Recording
alone does not authorize or initiate a physical test.

The new regression suite verifies that the real worker rejects Enable and later
control requests with zero outgoing command frames, even when an old controller
target could differ from the measured pose. Separate fault injections demonstrate
that leader encoder packets can coexist with a retained position servo. Those
tests must not be described as a physical repair or compliance verification.

The original implementation was tested without physical mode switching. The
2026-10-20 user run exposed the mismatch above. The regression fix and validation
commands here perform no physical I/O. Results are saved under
`validation/hand_guiding_fix_20261020/`, including the original failing test,
corrected replay metrics, joint trace, and simulated CAN commands.
