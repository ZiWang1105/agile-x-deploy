# Dual NERO commissioning

## Current solver revision

The active IK is now the official AgileX solver with NERO YAML defaults and the explicit local `w_pos=40, w_ori=4` override in `configs/ik.yaml`. TCPs and observations use schema version 3. The hardware observations, motion sessions and CAN fault below are historical evidence from before this solver migration; no physical motion has been performed with the new solver. See [official IK provenance and tests](OFFICIAL_IK.md). Use `.venv-official` through `scripts/run_interactive.sh`; the old environment has no fallback solver.

## Current state

The fixed stand's robot-left arm maps to USB serial `0030002D4648570A20303731` (observed as `can1`); robot-right maps to `0040002C4648570220393831` (observed as `can0`). Robot left is the observer's right when facing the front. The user confirmed the mapping by moving only robot left while both channels were monitored. The user also confirmed mounting and support.

The receive-only viewer resolves current interface names from those USB identities. It does not configure networking, change controller modes, enable motors, release brakes, change zeros, or send CAN data frames. Normal bus ACK behavior is determined by the existing SocketCAN interface configuration.

A separate `real-commissioning` mode implements explicitly armed physical joint-position control through the same `RobotInterface` as simulation. It supports `--side both` (default), `--side left`, or `--side right`. Both-arm mode shares one owner and validates paired targets before sending either bus. Its command path has been tested with simulated/in-memory transports, and the real feedback path has been tested while unarmed. Subsequent operator sessions enabled and commanded the left arm; saved encoder readings were consistent with a 30 mm lift. A current-position HOLD was sent on the operator's explicit stop request. This is not qualification of full-range following, collision clearance, or failure-stop behavior. General dual-arm physical dragging remains unqualified.

The user subsequently confirmed: "Blue posture matches both arms." This records static visual agreement, not a full-range sign check or TCP/collision calibration. Both controllers replied to firmware queries with **1.21**, corresponding to official SDK profile **v121**. These observations are recorded in `configs/commissioning.json`; that file does not enable motion.

Earlier read-only configuration probes sent one `0x4AF` firmware request per arm and seven `0x472` joint-limit requests per arm. Firmware queries succeeded; joint-limit queries received no `0x473` replies in the observed controller mode. No motion, enable, zeroing, or configuration-write commands were sent. Do not substitute SDK defaults for an unanswered hardware query. The reviewed official SDK commit is `1ae40ad4755ca25ad32bece3ab42325698185fc7`.

## Motion limit sources

Full workspace is the default. The old host limits of 0.03 rad/s and 0.1 rad/s², firmware speed scaling of 5%, Cartesian velocity clips, 0.025/0.2 rad tracking-error cutoffs, 0.15 rad observation-displacement cutoff, and extra 0.01/0.02 rad joint-position margins have been removed. Full mode imposes no radius, height floor, orientation cap or joint excursion cap relative to the arming pose. Bounded mode is a legacy envelope selected only by an explicit `--workspace bounded` option.

The [NERO manual](https://agilexsupport.yuque.com/staff-hso6mo/alxgtf/vx3iqycffq0pg8af?singleDoc) specifies maximum joint speeds of 180°/s for J1–J3 and 225°/s for J4–J7. A read-only probe on October 14 returned these left-controller settings:

| Joints | Configured maximum speed (rad/s) | Configured maximum acceleration (rad/s²) |
| --- | ---: | ---: |
| J1–J2 | 3.14 | 2.0 |
| J3–J4 | 3.14 | 2.5 |
| J5–J7 | 3.92 | 2.5 |

J4's configured speed is lower than the manual's maximum; the host honors the controller setting. Acceleration is decoded from `0x47C` as unsigned big-endian uint16 times 0.001 rad/s², matching the [official NERO parser at the reviewed commit](https://github.com/agilexrobotics/pyAgxArm/blob/1ae40ad4755ca25ad32bece3ab42325698185fc7/pyAgxArm/protocols/can_protocol/drivers/nero/default/parser.py). The reviewed v120 setter uses a different scaling factor (`1e2`) despite documenting `1e-3` precision. This implementation follows the official read decoder and does not invoke the setter; acceleration values are SDK-decoded readback, not independently calibrated physical limits. The right controller returned no limits in its observed mode. Raw query results are retained locally in `logs/commissioning/official_limit_readback.json`; no motion or configuration writes occurred during this probe.

Each explicit E arming request selects CAN J mode at 100% of configured firmware speed, then queries the selected controller with `0x472` search types 1 and 2 while motors remain disabled. All seven position, speed and acceleration replies must be valid before current-position commands and enable requests are sent. No acceleration, angle-limit, motor-current, torque-limit or collision-sensitivity settings are written. Effective position bounds intersect source-model and firmware limits. Effective speeds take the lower of manual and firmware ceilings. Acceleration uses the actual readback. The shared IK and wire admission layer enforce those same per-joint reference limits. These bounds constrain host references; measured physical dynamics are not certified by the host.

Simulation uses the left readback as an explicit offline fixture for both arms. It is not a claim about the unqueried right controller. Real arming never falls back to that fixture. No numeric torque ceiling was found in the reviewed manual/API. The SDK's MIT `t_ff` encoding range and URDF effort values do not establish a joint torque safety limit. Joint-position mode keeps drive torque/current protection under firmware control without modifying it. The simulation retains its existing, unqualified actuator force model; no verified sim/real torque equivalence is claimed.

Fresh feedback, firmware fault checks, pair-wide ownership, command/heartbeat expiry and operator HOLD remain. The official IK objective and the unreachable/stalled-target diagnostic are numerical control behavior, not additional vendor motion limits. No collision avoidance is implemented. Changes take effect in a newly started process; running controllers are not modified automatically.

## CAN queue-fault recovery

The October 14 full-workspace session latched a fault after approximately ten seconds of tracking with `ENOBUFS` (`No buffer space available`). Feedback then stopped advancing, and the original exit path raised a stale-feedback error during HOLD. The physical stop could not be established from that frozen log. A subsequent receive-only check found both arms disabled and stationary. Both firmware queries later succeeded, and `can1` returned from ERROR-WARNING to ERROR-ACTIVE. The underlying cause of the bus disturbance has not been established.

The sender now handles `ENOBUFS` and `EAGAIN` with 1 ms backoff inside the existing 20 ms deadline for the complete frame batch. It retries only the frame that failed to enqueue. It does not grow the transmit queue, replay previously accepted frames, enable motors, or alter joint limits. Persistent congestion still faults. A successful socket send confirms local enqueue, not physical receipt or stopping.

The control owner now exits on a close request even if HOLD fails, closes its transport, and reports the unconfirmed HOLD. The runtime preserves the original failure along with any HOLD failure. The viewer reports an exit-time HOLD failure without masking Ctrl+C. No automatic recovery resumes motion after a latched fault. Offline fault-injection tests cover transient congestion, persistent congestion, disconnection, failed HOLD and owner cleanup. Physical tracking under the repaired path remains untested.

## Inspect the current posture

From the repository root:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-preview
```

The blue geometry is reconstructed from measured joint angles. Amber geometry is a separate simulated plant controlled by the shared IK/runtime interface. Both use the same supplied URDF mounting transforms. Press M to hide the amber geometry and compare shoulder, elbow, wrist, and flange orientations with the physical robot. Red target handles move only the simulation. The fixed stand and camera geometry are nominal CAD, not sensed geometry.

The preview is seeded once from the real joint angles at startup. It does not continuously reset the simulator while solving IK. Restart after manually repositioning the real arms if a new preview starting posture is desired. An initial joint outside the source model limits is rejected rather than clipped. During operation, a fresh out-of-limit reading cancels preview tracking and latches a display fault.

Physical finger travel is not calibrated, so both sets of finger meshes are hidden and gripper keys are disabled. The simulator still owns its internal gripper state; this is not a hardware measurement. Physical velocities are unavailable in this partial feedback interface.

The separate commissioning transport requires firmware 1.21, receives actual high-speed velocity frames `0x251`–`0x257` and arm status `0x2A1`, and includes velocity acquisition times in its paired-sample age/skew checks. By default gripper travel remains unavailable (`null`) and only `gripper_open: null` is accepted. With explicit `--gripper`, the selected side also receives timestamped `0x2A8` width feedback and requires controller range readback. Its normalized travel is exposed through the same version 3 contract; only an explicit gripper target starts gripper command dispatch. Physical zero/stroke and actual open/close following remain unverified. See [gripper control](GRIPPER_CONTROL.md).

The receiver decodes the standard NERO joint feedback IDs `0x2A5`, `0x2A6`, `0x2A7`, and `0x2A9`, with signed big-endian angles in 0.001 degrees, plus driver status IDs `0x261`–`0x267`. It uses Linux kernel receive timestamps, publishes a new pair only after all eight joint frames advance, enforces 200 ms age and 30 ms intra/inter-arm skew, and rejects socket overflow or clock discontinuity. CAN frames do not expose a shared hardware sampling clock; these checks bound host reception timing rather than claiming atomic simultaneous sampling.

On feedback failure, the UI freezes the last available hardware posture, labels it frozen, and stops the simulation preview. This is not a physical HOLD or emergency stop. Another process must not be assumed stopped by closing this viewer.

## Preparation pose versus encoder zero

A useful preparation pose bends the elbows, avoids table/stand/other-arm interference, and leaves joint-limit margin. It must specify all 14 joint angles. A visible 90-degree elbow bend is not a complete joint configuration or an instruction to set J4 to 90 degrees.

Do not use `calibrate_joint()` to record a preparation pose: that API changes encoder zero. Keep the factory zeros. A future approved preparation pose should be saved as a named joint target with verified model/firmware identity; it must not become an automatic power-on command.

## Gates before connecting physical commands

1. Static blue-model posture agreement is user-confirmed. Still verify joint direction through motion before qualifying full-range control. Current source limits differ slightly from the manual/SDK examples; qualify limits against the actual firmware and model revision rather than broadening them silently.
2. Both firmware versions are confirmed as 1.21 (`v121`), and the user confirmed side mounting in both controllers' official Web UI. Still verify supported continuous joint command mode, feedback units, velocity availability, and gripper calibration. Older NERO SDK profiles force reported velocity to zero; that value must not be presented as measured velocity. The current joint-only reader intentionally does not expose velocity yet.
3. The single-arm real transport now implements bounded I/O, paired feedback, explicit unsupported gripper handling, and controller-reported joint reference bounds. Its independent viewer-heartbeat watchdog is tested in a separate process, but physical HOLD behavior and controller/host/CAN failure behavior still require qualification. Disabling motors is not a substitute for HOLD because unsupported arms may descend.
4. Calibrate the TCP and verify collision geometry. Add admission for the table, stand, self-collision, and inter-arm collisions. Current preview has no collision admission, and visually plausible IK is not permission to execute a path.
5. Start a commissioned motion session from a fresh real joint sample; seed both the IK reference and initial command from that sample. First test one arm over a small displacement at conservative speed, then validate tracking, operator stop, command expiry, feedback loss, and process loss. Only then commission paired following and a preparation-pose trajectory.
6. Keep viewer/agent targets on `RobotInterface`. Hardware-specific enable, firmware selection, and fault recovery belong in the transport/session lifecycle. Agent task execution follows only after these physical tests pass.

## First supervised physical session

This is a limited commissioning procedure, not collision-qualified general teleoperation. Keep mechanical support in place, clear the selected arm's workspace, and stop other arm-control applications. Model limits are enforced but firmware-limit readback remains unavailable; do not interpret this first test as full-range qualification.

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-commissioning --side left
```

The program starts unarmed and refuses an already-enabled arm system. Robot left is the observer's right when facing the stand. Before arming it rechecks the actual firmware, USB mapping, paired feedback, stationary joints, healthy driver flags, and model-limit margin.

1. Press **E** to explicitly enable the selected arm at its current measured joint pose. The transport sends CAN joint-position mode at 100% of configured firmware speed, reads that controller’s joint limits while disabled, seeds all seven current joint targets, and then enables joint indices 1–7 individually. It never enables index 8 or sends gripper commands. It waits for enabled-driver and healthy CAN joint-mode feedback, and rejects an unexpected posture shift.
2. Inspect the physical hold before requesting movement. Press **G** for a nominal-TCP target 30 mm above the arming pose; repeated presses do not accumulate lifts. Mouse translation and rotation use full workspace by default. Per-joint position, speed and acceleration limits follow the sources above. There is no artificial slow-speed profile or arming-relative travel cap in full mode.
3. Press **Space/R** to cancel the goal and request servo HOLD. **Q/window close also requests HOLD and leaves motors enabled.** This prevents an automatic torque release but means the arm remains powered after exit. Safely support and disable it through the robot's established controls before switching sides or restarting; the program will not silently take over an already-enabled controller.

The selected arm's CAN writer lives only in an owned control process. The viewer sends `observe`, `submit_target`, `hold`, and an explicit arming request through IPC. Viewer heartbeat or target expiry stops accepting new goals after 200 ms. If fresh encoders are available, HOLD uses the latest measured pose; otherwise it retains the last bounded joint setpoint. A missing CAN connection can prevent even that request from reaching the controller. Closing the viewer does not disable the motors, and software HOLD is not a physical emergency stop. Partial arming failure may leave some motors enabled; preserve support and inspect the robot before recovery.

The process watchdog covers viewer/GIL stalls and IPC loss. A killed control process, failed host, severed CAN link, and power loss require separate controller-level/fail-safe qualification; no such guarantee is claimed. No physical arming or motion was performed by the implementation tests.

The G lift remains a fixed 30 mm target. A running viewer must be closed and restarted to load code changes. Closing leaves motors enabled; support and disable them before restarting. No automatic motor release occurs.

### Disable motors before restarting

Closing the viewer leaves motors enabled. To disable them, first mechanically support both arms against falling or swinging and close all arm-control applications. Keep power and USB-C connected while running:

```bash
cd ~/projects/agilex-deploy
.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported
```

The flag explicitly confirms physical support; the software cannot inspect it. The command takes the same controller ownership lock, resolves the adapters by their recorded USB serials, and sends only individual NERO joint-disable requests (`0x471`, indices 1–7, flag 1). It does not enable motors, switch modes, command positions, or address grippers. It reports success only after feedback confirms no enabled arm joints; otherwise keep support in place and inspect the reported failure. The ownership lock does not cover unrelated vendor applications, which must also be closed.

After `DISABLED confirmed: left 0/7 enabled; right 0/7 enabled.`, you can start a new unarmed commissioning session. Leave mechanical support in place until the arm's hold has been checked again.

### Dragging and rejected targets

Full workspace is the default; the explicit equivalent is:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-commissioning --side left --workspace full
```

Use `commissioning-sim` instead of `real-commissioning` to practice without CAN access. Full mode starts unarmed and still controls only the selected arm. It removes all arming-relative Cartesian radius, minimum height, orientation and joint excursion limits. Ctrl+right-drag translates; Ctrl+left-drag rotates. The starting pose is retained only for the fixed G test target and display offsets. The source joint position bounds, selected-controller speed/acceleration limits, feedback freshness and heartbeat/command watchdogs apply. The common contract also rejects coordinates outside its existing 2 m envelope, which exceeds the modeled arm's reach.

Full mode has no collision avoidance: moving downward or sideways can contact the table, stand, opposite arm, or other objects. Clear the complete movement area and supervise every movement. Do not interpret a movable red handle as proof of IK reachability or collision clearance. Joint-limit violations and firmware faults request HOLD; the numerical IK can stop on unreachable/stalled goals. Offline full-mode checks from the recorded posture reached a 100 mm upward target with 0.064 mm position error and a rotated FK-derived target with 0.038 mm error, including joint travel beyond 0.45 rad. No full-mode physical motion was performed during these checks.

The optional bounded workspace has a 3 cm radius. It can be explicitly configured between 3 and 20 cm with `--workspace bounded --workspace-radius-cm`; the same value is enforced by the viewer and the control process in both commissioning backends. For example, to practice a 10 cm workspace without CAN access:

```bash
DISPLAY=:1 ./scripts/run_interactive.sh --backend commissioning-sim --side left --workspace bounded --workspace-radius-cm 10
```

For a supervised physical session, use `real-commissioning` with the same option after clearing the arm's movement area and starting with supported, disabled arms. The larger radius does not enable motors automatically or change speed, acceleration, model joint limits, the 0.45 rad joint excursion bound, the height floor, or communication watchdog checks. G still requests only the fixed 3 cm lift. The radius is a target allowance, not a guarantee that every point is reachable. Offline tests from the recorded October 14 posture reached 10 cm in each horizontal direction for the left arm; an upward request encountered the joint excursion guard near 6.7 cm. These results do not establish physical clearance or stopping performance.

The viewer shows target and measured offsets, remaining radius, and an explicit AT LIMIT indicator. Session logs retain feedback at 5 Hz and raw versus constrained mouse targets so target admission can be distinguished from physical tracking.

Only the selected arm has an active red handle; the other handle is gray and cannot submit a target. Double-click the red sphere, then Ctrl+right-drag to translate it. In bounded mode, mouse input preserves the arming orientation. The handle is constrained to the configured sphere around the arming position and the space at or above its starting height. At the top of the sphere, dragging sideways also lowers the constrained target slightly to remain inside that sphere. The UI reports when this constraint is active. These arming-relative restrictions do not apply to full-workspace or ordinary simulation mode.

Previously, G placed the handle at the top of the sphere and a sideways drag immediately exceeded its radius: even a 1 mm sideways change was rejected. Both handles still appeared active, and rejected targets remained visible. This made the display look as if an accepted command was not being followed. The corrected viewer constrains mouse targets before submission, while the control process independently checks every target. A rejected submission now cancels the previous goal and requests HOLD; the viewer resets the handle to feedback and prints the rejection. The bounds have not been enlarged by this correction.

Commissioning sessions automatically write timestamped JSONL logs under `logs/commissioning/`, including accepted targets, rejections, arming, and controller mode transitions. The path is printed at startup. Older sessions started without `--log` did not retain enough information to identify their exact rejection or fault after exit.

Terminal Ctrl+C is handled by the viewer. The owned control process ignores the terminal's duplicate SIGINT so that the viewer can request HOLD and join it over IPC. This fixes the observed simultaneous interruption and broken-pipe shutdown error. Q, window close, and Ctrl+C leave motors enabled; none replaces a physical emergency stop or guarantees stopping after host/CAN failure.

To practice this exact interface without any CAN access, use `--backend commissioning-sim --side left`. To verify real observations and process cleanup without arming, run `.venv-official/bin/python scripts/check_real_session.py`.

## Repeatable checks

```bash
.venv-official/bin/python -m unittest discover -s tests -v
.venv-official/bin/python -m agilex_deploy.can_feedback --output logs/commissioning/live_snapshot.json
# These two commands transmit only fixed read requests, unlike the passive viewer.
.venv-official/bin/python -m agilex_deploy.firmware_probe --output logs/commissioning/firmware.json
.venv-official/bin/python -m agilex_deploy.limits_probe --output logs/commissioning/firmware_limits.json
MUJOCO_GL=egl .venv-official/bin/python scripts/validate_live_preview.py
DISPLAY=:1 ./scripts/run_interactive.sh --backend real-preview --duration 5
```

`validate_live_preview.py` reads live joints, seeds the simulator, commands a 20 mm upward target to the simulated arms through `RobotInterface`, records convergence and CAN net-device TX deltas, and writes `logs/commissioning/live_preview.png` and `live_preview_report.json`. This is a scripted simulation validation with physical input, not a mouse interaction or physical tracking test. Logs are intentionally ignored by Git.

Protocol references reviewed on 2026-10-14:

- [Official NERO joint feedback definition](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/protocols/can_protocol/msgs/nero/default/feedback/arm_feedback_joint_states.py)
- [Official NERO API](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/nero_api.md)
- [Official NERO firmware behavior matrix](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/nero/firmware_reference.md)

## Controller mounting configuration

The NERO user manual section 6.8.4 exposes mounting position in the controller Web UI. The model's correct geometry does not prove that this controller setting matches the physical side mounting. The SDK's NERO API does not expose the Piper-specific `set_installation_pos` method; do not assume the Piper command semantics apply. No mounting setting has been written by this project.

The user confirmed both controllers are configured for side mounting in the official Web UI. This is recorded as user-provided evidence, not software readback. No additional Ethernet connection is required for that check or for the planned CAN control path. The desktop's USB-C connection exposes two working USB-CAN interfaces; the absence of a wired Ethernet link does not prevent CAN communication. No network addresses have been changed by the commissioning tools.

## Coordinated dual-arm sessions

```bash
# Same control owner and official IK, simulated plant:
./scripts/run_interactive.sh --backend commissioning-sim --side both --gripper
# Physical paired session, initially unarmed:
./scripts/run_interactive.sh --backend real-commissioning --side both --gripper
```

**E** enables both arms at their measured joint positions. Both controller limit
readbacks and both starting setpoints are prepared before either arm is enabled.
Both arms must acknowledge joint-position mode and all seven motor enables.
A partial arming failure is latched and attempts HOLD on both; it never claims
success or automatically releases holding torque. **G** targets a 30 mm lift on
both arms. Both red handles accept dragging; **1/2** command the left/right jaws.
**Space/R** HOLDs both; closing leaves enabled motors enabled.

The paired owner uses one feedback snapshot per tick and applies each arm's own
position, velocity, acceleration and gripper-range readback. Target validation
finishes for both arms before either CAN writer sends a new reference. If a bus
write fails after the other has succeeded, both HOLD attempts still run and the
session latches the failure. Independent adapters cannot provide atomic wire
transmission or a guaranteed physical stop after communication loss.

Paired official-IK tracking, independent grippers, HOLD, heartbeat expiry,
per-bus routing, partial enable failure, and partial write failure are verified
with simulated or in-memory transports. These checks do not establish physical
collision clearance, synchronization accuracy, or failure-stop performance.
No physical enable or motion was issued while implementing this change.

Validation on 2026-10-16: the complete offline suite passed 130 tests in
148.175 seconds. A native `commissioning-sim --side both --gripper` viewer
completed a three-second unarmed launch/render/close check. The IPC test moved
both TCPs upward 30 mm through official IK, verified each final position within
3 mm, independently opened/closed the jaws, and verified paired HOLD and
heartbeat fault latching. The viewer input test exercised E, both gripper keys,
and HOLD through the actual control process using a simulated plant.
