# AgxGripper control

The shared `Action` keeps the existing `gripper_open` convention: integer `0`
closes, integer `1` opens, and `None` leaves the gripper target unchanged. Physical
control is optional and selected per arm, matching the current single-arm NERO
commissioning session. The other arm and gripper cannot be commanded by that
session. Standalone simulation continues to control both grippers.

## Install and inspect

```bash
./scripts/setup_gripper.sh
.venv-official/bin/python -m agilex_deploy.gripper --query-parameters
```

The inspection command receives paired arm/gripper feedback and sends only the
official gripper-parameter read request (`0x477`, enquiry 4). It does not enable,
open, close, reset, recalibrate or change operating mode. Omit
`--query-parameters` for strictly receive-only inspection.

The driver uses the original `pyAgxArm` gripper codec, pinned to commit
`d90e234ba8e408cfa964b38aa9120b3131fc9f85` in `requirements-gripper.txt`. No SDK arm
instance, automatic connection routine or vendor demonstration is executed.

## Open both jaws and disable both arms in one command

Close any active robot controller, mechanically support both arms, and clear the
jaws. Run:

```bash
cd ~/projects/agilex-deploy
.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported --open-grippers
```

This first verifies both arms are disabled, disabling them if needed. It then
opens each gripper in turn while temporarily enabling that arm at its measured
joint positions. Opening is verified from feedback within 1 mm of configured
travel for 100 ms. Each arm is disabled again after opening, including on an
opening error. A failure aborts the sequence; it does not report both jaws open.
The command does not home the arms or change gripper zero calibration.

Without `--open-grippers`, the command retains its disable-only behavior and
never enables motors or moves jaws. The standalone `scripts/open_grippers.py`
uses the same opening implementation but requires already-disabled arms.

Physical opening was verified on 2026-10-15: left 99.473 mm and right 99.704 mm,
with configured travel of 100 mm. All 14 arm motors were verified disabled
afterward. The combined command's orchestration is tested offline.

## Viewer control

Start with supported, stationary arms disabled, as required by the existing
commissioning workflow. Keep the jaws clear before issuing an open/close target.

```bash
./scripts/run_interactive.sh --backend real-commissioning --side left --gripper
```

1. Press **E** to enable the selected arm at its current posture. This sends no
   gripper command.
2. Press **1** to toggle the left gripper. Its first explicit target also enables
   the gripper. Pose dragging alone never enables or commands it.
3. Press **Space** or **R** to discard the target and request HOLD at measured
   joint and gripper positions. **Q** requests HOLD and closes the session;
   enabled motors remain enabled.

For the right side, use `--side right --gripper` and press **2**. Close the previous
owner before starting another session; the existing paired ownership lock and
disabled-arm startup requirement still apply. This does not introduce concurrent
dual-arm physical control. Avoid disabling unsupported arms when switching.

Exercise the same selected-side interface without hardware:

```bash
./scripts/run_interactive.sh --backend commissioning-sim --side left --gripper
```

The force command defaults to `1.0`, matching the official `move_gripper_m` API;
`--gripper-force-n` accepts the documented 0–3 N range. This is the SDK's force
setting, not an independently measured physical jaw-force guarantee. Physical
gripper targets go directly to firmware, without the simulator's 0.6/s reference
ramp. The host does not invent a gripper speed/acceleration limit or overwrite
controller parameters.

## Feedback, units and failure behavior

### Official control and host supervision

The official API already accepts a target width and target gripping force via
`move_gripper_m(value, force)` and returns measured width, force and driver status
via `get_gripper_status()`. The reviewed Python driver encodes and sends those
targets; it does not implement a host-side effort regulator. The official ROS 2
driver exposes the same concept through `gripper` position/effort commands and
`/feedback/gripper_status`; it is another interface, not evidence of a different
gripper servo. Firmware gains, internal loop design and calibrated fingertip
force accuracy are not established by these public API documents.

This integration uses that official width/force path. It does not copy the
Hexfellow GP80 motor-torque regulation law or its Nm gains into AgileX's N-valued
interface. No extra host force integrator, motion ramp, stall timeout or automatic
force escalation has been added. Existing feedback/enable/watchdog faults still
request HOLD.

`Observation.gripper_states` now provides one measured tracking report per hand,
with the same shape in standalone simulation and hardware:

- `phase`: `idle`, `opening`, `closing`, `reached`, `stalled`, `hold`, `disabled`,
  `unavailable`, `stale`, or `fault`.
- `requested_open`, normalized `position_error`, and measured
  `velocity_per_s` (finite difference of distinct acquisition samples).
- `width_m`, signed `force_n`, `enabled`, and acquisition `timestamp_s` when
  those signals are available. The simulation has no calibrated force sensor;
  its raw width/force/status fields remain `None`, not invented measurements.

The viewer displays phase and reported force, and commissioning JSONL feedback
records include these reports. Scripted agents read them from `robot.observe()`.
`reached` means the measured opening stayed within 1% of configured travel for
100 ms. `stalled` means it remained outside that endpoint tolerance while moving
no more than 0.2% of travel over 500 ms, using at least three distinct samples.
For 100 mm travel these position criteria are 1 mm and 0.2 mm respectively.
These are local, configurable diagnostic criteria in `TrackingSettings`, not
official motion constraints. They do not change commands, stop closing, or
declare contact/grasp success. A gripper holding an object may correctly report
`stalled`; the selected official width/force target continues while its lease is
renewed. Reported force is exposed for inspection without assuming its sign
proves contact. Physical force behavior still needs a measured bench test.

Repeated requests preserve the tracking window. `None` preserves the previous
gripper request, matching the existing action contract; explicit HOLD cancels
it. Duplicate feedback cannot accumulate settling time, stale/disabled reports
cannot declare completion, and resumed motion clears a stationary diagnostic.
The existing schema version 3 gains the additive `gripper_states` field; action
fields and units remain unchanged.

References:
[official Python API](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/effector/agx_gripper/agx_gripper_api.md),
[official Python implementation](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/driver.py),
[official ROS 2 interface](https://github.com/agilexrobotics/agx_arm_ros/blob/ros2/README_EN.md).

### Transport checks

`0x2A8` width-mode feedback is decoded by the official codec: signed micrometers,
signed millinewtons, driver status and mode. Angle mode is rejected rather than
interpreted as meters. The controller's `0x47E` configured range is required;
supported values are 70 or 100 mm. Missing/invalid range readback has no fallback.
Normalized travel is measured width divided by that configured range. No zero
position is inferred or written, and configured travel is not a physical
calibration of the finger geometry.

Selected gripper frames participate in the same timestamp freshness and paired
feedback checks as the arm. Missing/stale feedback, a driver fault, or missing
enable acknowledgement stops target dispatch and latches the existing session
fault path. The first command permits 0.5 s for enable acknowledgement; this is
a communications timeout, not a motion limit. The viewer's separate-process
200 ms heartbeat watchdog also requests gripper HOLD.

HOLD sends an enabled, healthy gripper its measured width; it never enables a
disabled gripper. With stale feedback it attempts the last observed width rather
than the previous distant target. Actual stopping cannot be confirmed after
communication loss. A bad gripper value does not suppress the arm's HOLD request.
No reset/clear-error bit, zeroing command or all-motor enable index is used.

NERO requires compatible gripper firmware and **CAN node ID10**, as explained in
the [official node-ID announcement](https://github.com/agilexrobotics/pyAgxArm/discussions/40).
Receiving controller-forwarded `0x2A8` frames does not independently prove the
motor node ID or physical following. This integration does not rewrite node IDs
or flash firmware. The official [API reference](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/docs/effector/agx_gripper/agx_gripper_api.md)
and [codec](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/parser.py)
define the reviewed wire format and API units.

## Validation status

Receive-only inspection on 2026-10-15 found both mapped controllers forwarding
gripper status at approximately 200 Hz. Both parameter queries returned 100 mm
configured travel. At inspection, both grippers reported disabled, zeroed=false,
and no driver-fault bits; reported widths were 0 mm left and 0.895 mm right.
These are controller reports, not physical zero or stroke verification.

Offline tests cover official wire bytes, signed feedback, units and modes,
selected-side routing, absent/stale feedback, no implicit startup/drag actuation,
HOLD behavior, shared sim/real actions, and the process heartbeat watchdog.
Physical open/close execution has not yet been tested by this change.

The regression suite passed 93 tests. The final focused run passed all 15 gripper
tests, including an additional viewer-key test for both selected sides through
the actual IPC owner and a simulated plant (94 distinct tests in total).
Both physical transports also passed unarmed startup with gripper feedback;
their motion writers remained unopened. Results are recorded in
[`previews/gripper_validation.json`](../previews/gripper_validation.json).

After adding shared feedback supervision, the full suite passed **102 tests**
in 129.512 seconds; the focused gripper suite passed 23 tests. Coverage includes
measured endpoint dwell, stationary shortfall, command renewal, HOLD, acquisition
gaps, duplicate samples, missing simulated force and real transport force-field
propagation. The native commissioning-sim viewer also completed a three-second
unarmed startup/shutdown and logged 15 observations with the new fields. Viewer
key routing still passes through the actual IPC owner with a simulated plant.
These checks sent no hardware motion commands and do not validate physical
gripping force or object retention.
