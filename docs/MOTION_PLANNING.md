# Cartesian motion planning

This guide describes Cartesian motion on the
[Workshop on Agentic Robotics](https://agentic-robotics-workshop.github.io/)
demo platform.

TCP actions now run through an online Cartesian waypoint tracker before the
official AgileX IK solver. This applies to the interactive viewer, Python SDK,
generated code policies, and MCP calls because all use the shared `Controller`.
The hardware process uses that same controller. No new participant command or
backend-specific action format is required.

## What CaP-X does

The reference inspected was
[`capgym/cap-x` at `53e9966d7a8e2fa7494676772bccc35280f5c0ed`](https://github.com/capgym/cap-x/tree/53e9966d7a8e2fa7494676772bccc35280f5c0ed).
It provides two distinct mechanisms:

* [`plan_trajectory_linear_ik`](https://github.com/capgym/cap-x/blob/53e9966d7a8e2fa7494676772bccc35280f5c0ed/capx/serving/launch_pyroki_server.py)
  interpolates translation linearly, interpolates orientation with SLERP, and
  solves each waypoint using PyRoKi with a soft velocity-limit penalty relative
  to the previous configuration. The helper defaults to 25 waypoints; the
  HTTP planning request defaults to 20. This function does not check obstacles.
* The separate
  [cuRobo server](https://github.com/capgym/cap-x/blob/53e9966d7a8e2fa7494676772bccc35280f5c0ed/capx/serving/launch_curobo_server.py)
  uses `MotionGen`, robot collision geometry, world obstacles, trajectory
  optimization, and graph search. Its default robot configuration is Franka.
  This cannot be used unchanged for a side-mounted dual NERO assembly.

This implementation adapts the first architecture. It does not vendor CaP-X
code or replace the official AgileX solver with PyRoKi. Differential IK, which
computes local joint velocity updates from Cartesian error, is another distinct
algorithm. Neither geometric waypoint planning nor collision planning requires
that particular IK algorithm.

## Shared execution

1. A changed TCP target creates a path from the arm's current measured TCP.
   Position follows a straight line. [Workshop](https://agentic-robotics-workshop.github.io/) pose operations first validate
   shortest-arc orientation waypoints with the unchanged official IK. If that
   candidate fails joint bounds or the operation's pose residual criteria,
   they test the complementary rotation arc to the same endpoint. The chosen
   start pose is frozen through execution, so encoder noise near a half-turn
   cannot reverse the selected direction. The interactive drag viewer keeps
   its existing feedback-rebased shortest-arc path.
2. The planner samples at at most 10 mm translation and 0.05 rad orientation
   between waypoints. These are numerical path resolutions, not Cartesian
   speed limits or workspace restrictions.
3. Measured feedback anchors the waypoint lookahead. The stationary switching
   region is half the sample spacing, with floors of 1 mm and 0.005 rad for the
   official solver's soft-objective residual. During motion, the controller
   computes the joint stopping horizon `max(abs(qdot_reference) / acceleration)`
   and maps reference velocity to TCP twist using the model Jacobian. It adds
   `norm(linear_twist) * horizon` and `norm(angular_twist) * horizon` to the
   respective lookahead distances. The selected target can pass multiple
   intermediate waypoints in one tick. This avoids braking toward each nearby
   sample when the path continues, without raising joint speed or acceleration
   limits. The final endpoint is unchanged.
   Only changing components gate intermediate progress:
   a pure translation does not wait for orientation to settle at every sample.
   Both position and orientation remain IK objectives and final success checks.
   These are not operation success tolerances.
4. At each control tick, the existing official IK solver receives the active
   waypoint and measured joint configuration. Existing joint position,
   velocity, acceleration, watchdog, and firmware protections remain in place.
   This is feedback-anchored execution with adaptive lookahead, not a
   precomputed timed joint trajectory or time-optimal trajectory optimization.
5. A replacement target rebases that arm's path on its measured pose while the
   joint reference retains its acceleration-limited continuity. HOLD, a fault,
   or lease expiry clears both paths. Jaw changes do not restart either path.
   Explicit `JointAction` targets bypass Cartesian planning and clear paths.
   [Workshop](https://agentic-robotics-workshop.github.io/) jaw-only operations and post-operation maintenance use joint holds
   so they cannot accidentally restart arm IK.
6. The final target remains active for IK refinement. Existing operation
   completion checks still use measured error to the final goal. A complementary
   arc must also reach its final waypoint before the operation can complete.
   Advancing its waypoints counts as progress for the existing stall watchdog:
   this route can initially increase the shortest distance to the endpoint.

Left and right paths advance independently. A dual-arm request does not promise
simultaneous arrival or inter-arm collision avoidance. Each gripper remains
independently controlled. The nominal host period remains 10 ms; solver cost and
physical transport latency can reduce the achieved rate.

## API and inspection

Existing `robot.move_tcp(...)` and `robot.move_both(...)` calls now use the path
tracker automatically, including the corresponding MCP tools. Read-only
`ik_solve` still solves a single endpoint; it does not command or plan motion.
Its `success` now requires position error at most 5 mm and orientation error at
most 0.05 rad. `solver_success` reports optimizer convergence separately. A
local one-seed solve failing those criteria does not prove global unreachability.
Capabilities report `motion_planner="online_cartesian_waypoints"`,
`differential_ik=false`, and `motion_is_collision_planned=false`.

The public observation's `motion_paths` reports each arm's target, selected arc,
path revision, and active waypoint. Each operation's `planning` field reports
validation state, candidate residuals, and rejected waypoints. Locally,
`robot.controller.planner.status()` provides the same progress.
`CartesianPath.waypoints` exposes geometry without connecting to hardware.

## Limits of this implementation

[Workshop](https://agentic-robotics-workshop.github.io/) pose operations validate sampled IK solutions before dispatch, in an
owned spawned process with private solver state. This keeps planning out of
the control heartbeat and camera interpreter. Both sim and real use the same
preflight. Real preflight uses the effective limits already read during arming.
Cancellation invalidates outstanding plans; a cancelled plan cannot dispatch.
If neither candidate passes, the operation returns `PLANNING_FAILED`, retains
HOLD, and leaves the session available for a corrected target. During execution,
lack of progress uses the existing stalled-target HOLD and fails the operation
as `PATH_STALLED`; it cannot silently resubmit the stale path. Control solver
exceptions still latch a fault and attempt HOLD. There is no fallback to direct
endpoint motion.

Sampled feasibility does not certify the continuous path. Execution seeds IK
from measured feedback and can diverge from the preflight branch under tracking
error. A straight Cartesian path may be unreachable even with a reachable
endpoint. The interactive drag viewer does not prevalidate all future samples.

Joint interpolation and servo lag can depart from the nominal Cartesian line.
The planner does not model the table, objects, camera mount, cables, held
objects, or the other arm as collision obstacles. A path is not a collision
clearance certificate. Adding CaP-X's cuRobo route requires an independently
validated NERO collision model, scene registration, attached-object handling,
and whole-assembly checks.

This change is tested in simulation and hardware-style loopback only. It does
not establish physical stopping performance, collision clearance, or hardware
qualification. Gravity compensation mode is separate work.

### Encoder rounding and the October 20 20:22 failure

The recorded approach reached a joint boundary while the shortest orientation
arc stalled. The controller's continuous upper bound for joint 2 is 1.74 rad.
Rounding that value to NERO's 0.001-degree wire resolution produced a decoded
angle of 1.7400059977757467 rad. Joint 7's truncated model bound has the same
round-trip issue at 90 degrees. HOLD copied measured feedback into the next
outgoing reference, which the strict command guard then rejected as
`Command violates model-limit bounds`.

HOLD now clips admitted feedback into the unchanged outgoing bounds, and CAN
encoding rounds inward at an effective bound. Invalid feedback outside the
existing admission envelope is still rejected. Initial arming admission and
joint speed/acceleration limits are unchanged.

The same recorded start and target complete through the complementary rotation
arc. To replay gripper-open, the read-only IK query, the approach, and subsequent
HOLD through both public interfaces, with real transport guards and millidegree
wire/feedback quantization but no physical I/O:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_bound_incident.py
```

Both Python and stdio MCP replays completed in about 2.9 seconds with 1.3 mm
position error, 0.010 rad orientation error, and zero unselected-arm joint drift.
Contacts are disabled because the physical surroundings were not recorded.
This is a control regression, not a physical clearance or grasp test.

A subsequent live generated-code policy (`google/gemini-3.8-flash`) completed
the contact-enabled `place` task: 125.90 mm lift, release in the tray, 3.46 s of
stable tray support at scoring, and zero scored robot/environment collision
frames. The independent scorer returned success; the model's own completion
claim was not used as evidence. Reports and full-speed simulator video are in
`logs/validation/bounds_code_policy/`. A compact result record is retained in
`validation/bound_incident_20261020/report.json`.

### Numerical IK failures

The October 20 physical run reached its gripper-open target, then IPOPT reported
`Invalid_Number_Detected` about 55 ms after the first TCP operation started.
The exact failed control tick was not recorded, so the original incident cannot
be replayed exactly.

A deterministic numerical regression now reproduces that solver failure with
valid, in-limit joints and a valid target rotation. In the pinned Pinocchio 3.2 /
CasADi 3.6.7 stack, a near-identity rotation can make an inactive `log6` branch
contain a negative square-root argument at floating-point precision. CasADi's
reverse automatic differentiation propagates NaNs from that branch into the
objective gradient, even though the pose error and objective value are finite.
The reproducer makes the uncorrected official solver fail at iteration zero.

The adapter sets `solver.error.change_option("ad_weight", 0)` before IPOPT
constructs derivatives. This selects forward automatic differentiation for the
same official error function. The objective, weights, model, joint limits,
IPOPT tolerance, and iteration limit are unchanged; the vendored source is
unchanged. The reproducer now returns `Solve_Succeeded`. Regression tests check
identical objective values, finite gradient/Hessian, agreement with finite
differences, and 1,024 exact/near-FK targets. A separate 20,000-case probe also
returned finite derivatives. This proves the numerical defect and correction;
it does not establish which arm or iterate failed in the unrecorded incident.

The speculative initial-guess retries have been removed. Solver failures still
latch HOLD; no failed iterate is dispatched. The adapter saves the arm, exact
measured joints, target matrix, initial guess, and available failed iterate under
`logs/ik_failures/`.

A separate arm-selection bug is also corrected. A single-arm [workshop](https://agentic-robotics-workshop.github.io/) command
previously represented the other arm as a Cartesian target derived from fresh
feedback. Tiny encoder changes could then cause that unselected arm to enter IK.
[Workshop](https://agentic-robotics-workshop.github.io/) pose operations now use `MixedAction`: selected arms use Cartesian IK,
while other arms retain explicit captured joint targets. The same action crosses
the hardware process boundary. Both-arm operations and independent jaw commands
remain available through the same public Python and MCP methods.

To replay the recorded starting posture and gripper-open / IK-query / TCP-move
sequence through an actual stdio MCP client and the shared [workshop](https://agentic-robotics-workshop.github.io/) service:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_ik_deployment.py --video
```

This replay uses 0.001-degree encoder quantization. It is simulation only, with
contact disabled because the physical table geometry was not recorded; it is
not a grasp or collision test. The recorded target completed with 1.09 mm
position error, 0.00372 rad orientation error, and zero unselected-arm joint
drift. Physical motion has not been rerun for these fixes.

A subsequent live OpenRouter MCP policy (`google/gemini-3.8-flash`) completed the
contact-enabled simulator pick-and-place task through the public tool interface.
All 10 operations succeeded. The independent simulator scorer measured a
126.10 mm lift, release inside the tray, 13.69 seconds of stable tray support,
and zero scored arm/environment collision frames. The run used 39 model calls
and 62 tool calls, taking 139.39 seconds wall time. A rejected depth pixel was
reselected successfully; there were no IK or motion failures. The generated
video retains the full 131.6 seconds of simulator time, including inference
waits. Reports and video are under `logs/validation/ik_policy_sim/`.

## Reproduce the simulation comparison

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_cartesian_path.py --video
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_workshop.py \
  --task sort --output logs/workshop/cartesian_path_sort --video
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_workshop.py \
  --task select --transport mcp --output logs/workshop/cartesian_path_mcp_select
```

The six-second dual-arm comparison uses the same official solver, limits, and
actuator simulation across direct endpoint execution, the original short
waypoint lookahead, and adaptive lookahead. Only waypoint selection changes.
Left jaws open while right jaws close.

The original short-lookahead result below is historical. It reduced path
deviation but unintentionally limited speed by repeatedly braking toward nearby
intermediate goals. Adaptive lookahead fixes that pacing without raising joint
velocity or acceleration limits:

| Mode | Settled within 5 mm / 0.05 rad | Settled within 1 mm / 0.05 rad | Peak deviation from straight line |
| --- | --- | --- | --- |
| Direct endpoint IK | 0.93 s | approximately 2.55 s | 16.82 mm |
| Original waypoint pacing | 3.86 s | 3.99 s | 6.58 mm |
| Adaptive lookahead | 0.86 s | 2.55 s | 11.58 mm |

Settling means the measured error remains within the stated tolerance for the
rest of the six-second run. Final position error stays below 0.05 mm. The
faster pacing has a real path-accuracy tradeoff; it is not an exact straight-line
tracking guarantee or a claim of time-optimal control.

See the [video](../previews/cartesian_path_validation.mp4) and
[numeric comparison](../validation/cartesian_path_20261020/path_comparison.json).
An animated [GIF](../previews/cartesian_path_validation.gif) is also available.
These three links preserve the original slower implementation. The updated
[before/after MP4](../previews/cartesian_pacing_comparison.mp4),
[GIF](../previews/cartesian_pacing_comparison.gif), and
[report](../validation/cartesian_pacing_20261020/comparison.json) show adaptive
lookahead. Both panels play at the same simulation-time rate; playback is not
accelerated.
The video is rendered at 1x simulation time after execution, not recorded
wall-clock control timing. Yellow marks the geometric path; green marks measured
TCP positions. These results are specific trials, not general accuracy bounds.

The scripted Python RGB-D baseline also passed two-arm sorting: both cubes
lifted approximately 123 mm, then released and settled inside their assigned
trays, with zero scored arm/environment collision frames. This result is a
task test, not evidence of collision avoidance.
The scripted MCP RGB-D baseline passed right-arm selection and placement,
lifting the blue cube 122.2 mm while leaving the red cube within the task's
10 mm distractor-displacement tolerance. It also recorded zero scored
arm/environment collision frames. Reports:
[Python sorting](../validation/cartesian_path_20261020/python_sort.json) and
[MCP selection](../validation/cartesian_path_20261020/mcp_select.json).
These runs exercised the APIs without making new OpenRouter calls.

The initial full regression run passed 159 tests, including simulated hardware
IPC, sim/real loopback parity, both-arm tracking, joint reference limits,
retargeting, cancellation, solver failure, and independent grippers. The
[test log](../validation/cartesian_path_20261020/tests.log) preserves that run.
After adding adaptive lookahead, the full suite passed 160 tests. The
[updated test log](../validation/cartesian_pacing_20261020/tests.log) preserves
the current run.
