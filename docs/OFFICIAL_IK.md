# Official NERO IK integration

The active controller uses AgileX's original `ArmIK` class from QuestArmTeleop. The original NERO YAML supplies defaults; `configs/ik.yaml` explicitly overrides only `w_pos` to `40.0` and `w_ori` to `4.0`. The previous local damped-Jacobian solver is removed from the execution path. There is no fallback if the official dependencies are missing.

## Source and installation

- QuestArmTeleop: `145b80360cf2c5f2e63179086d0aab2cb563826c`, branch `ros2`.
- agx_arm_urdf: `ee534cea40b915cd70bd93d240ebf62110ae78d3`, branch `flattened`.
- Runtime: Pinocchio 3.2.0 with its CasADi Python bindings; CasADi 3.6.7 and IPOPT.
- Original source, NERO YAML, package license declaration, URDF license and asset hashes are preserved under `third_party/questarmteleop`.

Run `./scripts/setup_official_ik.sh` once. This installs an isolated `.venv-official`, builds the model and warms up the solver without opening CAN. The previous `.venv` is preserved. `./scripts/run_interactive.sh --backend sim` launches both simulated arms. Hardware selection remains explicit: `--backend real-commissioning --side left` or `right`. Simultaneous physical dual-arm control is not added by this migration.

## Parameters and model adaptation

| Parameter | Original NERO YAML value |
| --- | --- |
| `w_pos` | 2.0 |
| `w_ori` | 2.0 |
| `w_reg` | 0.01 |
| `w_smooth` | 0.1 |
| `ipopt_max_iter` | 50 |
| `ipopt_tol` | 1e-4 |
| `ee_parent_joint` / `ee_frame_name` | `joint7` / `ee` |
| `tool_pre_rot_rpy` | `[0, 0, 0]` rad |
| `tool_translation_xyz` | `[0.1755, 0, -0.0235]` m |
| `enable_collision_check` | false |

The active overrides are `w_pos: 40.0` and `w_ori: 4.0`; all other entries above retain their official values. These weights multiply the residuals before squaring. This prioritizes TCP tracking while retaining official zero-posture regularization and smoothing. Both backends use the same merged configuration, recorded in hardware session logs and simulation reports. Restart a session after editing the configuration. An empty `overrides: {}` restores upstream defaults; no source edit is required. Only solver tuning keys are accepted, not geometry or joint-limit overrides.

The old controller used damped differential Jacobian IK: damping `0.025`, task gain `4`, angular task scaling `0.4`, and a projected posture gain `0.08` toward its captured home pose. These are not interchangeable with the official nonlinear objective weights. In particular, the old posture term was projected into the approximate Jacobian null space, while the official `w_reg * ||q||²` directly competes with pose accuracy. This integration reuses the official implementation and tunes its own parameter interface; it does not claim the old parameter set has been transplanted. The selected weights reduce the relative influence of the zero-posture term while retaining it. Smoothness favors nearby configurations but does not guarantee globally optimal IK, collision avoidance or joint-limit clearance.

The original source file and YAML are unmodified. A loader selects only the `ArmIK` class and `xyzrpy_to_mat` helper AST definitions, using the original method bodies. ROS imports, node callbacks, publishers and automatic enabling launch files are not executed. The adapter resolves mesh paths and converts fixed-stand targets into the appropriate arm base frame. Measured joint feedback calls the official `sync_state()` before `solve()`.

One model compatibility correction is explicit: the pinned current URDF contains an additional virtual `gripper` axis not present in the older YAML lock list. The adapter locks that named gripper axis in addition to the YAML's gripper joints, retaining exactly `joint1` through `joint7` as the active coordinates. It does not lock any arm joint or change any optimization weight. All seven arm joint transforms and position bounds match the supplied stand assembly. The official URDF is retained separately for IK; this does not replace the user's stand or camera geometry.

Both simulation and hardware observations now use the official TCP relative to `link7`. Observation schema version 3 distinguishes this from the previous gripper-base TCP. API method signatures and action fields are unchanged. Old logs, GIFs and TCP targets describe the previous frame; recreate targets from fresh observations. The official nominal TCP is not a physical calibration.

## Execution and failure semantics

The same official solver is used by both backends. Its joint solution goes through the existing per-joint speed/acceleration reference generator; those limits are not changed by this migration. The original physical joint-position transport is retained. The hardware solver is warmed up before opening CAN, preventing solver initialization from accumulating a receive backlog; all warm-up solutions are discarded. Initial observations are refreshed after warm-up.

An arm whose target remains its HOLD pose stays at HOLD; it is not repeatedly optimized toward the regularizer's neutral posture. Actual changed targets use the original objective. Solver failure, non-finite output or an out-of-range solution is rejected rather than replaced with local IK. In particular, an unsuccessful `solve_limited()` iterate is not treated as a successful hardware command. Faults stay latched; no automatic rearm occurs.

Official pose errors are soft costs, not exact equality constraints. The cost weights position and orientation residuals before squaring, then adds joint regularization and smoothing. With the original `w_reg=0.01`, a reachable goal can retain centimeter-scale residuals, and a very small Cartesian request can be outweighed by posture regularization. An offline repeated-solve comparison using the same targets measured 17.20 / 16.09 mm with upstream defaults, 17.19 / 16.08 mm with only `ipopt_tol=1e-8`, and 0.0069 / 0.0066 mm with only `w_reg=0`. This isolates zero-posture regularization as the main bias for those targets. The initial zero/small-regularization candidates subsequently encountered IPOPT `Invalid_Number_Detected` in broader tests. They are not the shipped configuration. The selected `w_pos=40, w_ori=4` retains `w_reg=0.01` and passed the seven-second tracking test, 32-second sequence and recorded-posture lift test. This is empirical tuning, not a general numerical-stability fix; the upstream solver and error-on-failure behavior remain unchanged. Reported errors are measured, not suppressed. Collision checking remains disabled in the official profile and no stand/inter-arm collision admission is implemented.

## Validation

`tests/test_official_ik.py` checks pinned provenance, the explicit override, rejection of invalid tuning, exact agreement with a direct upstream solve given identical parameters/target/feedback, seven-axis output ordering, failed-iterate rejection, 5 mm target convergence, and Pinocchio/MuJoCo FK agreement for both mounts across random joint configurations. Existing tests exercise shared simulation/real-adapter behavior, joint-reference limits, drag conversion, inactive-arm holding, explicit enable and communication failures without moving hardware. The paired actuator-tracking test requires less than 1 mm position error with the active configuration.

Run the current end-to-end simulation report with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/smoke_ik.py
```

It writes `previews/official_ik_report.json` and `previews/official_ik.png`, including measured pose residuals, command rates and tick timing. These validate software and the modeled plant only. Subsequent physical camera-calibration routes used the migrated solver; their measured joint and capture evidence is preserved in the [calibration archive](../calibration/2026-10-15_zed35330610/README.md). Those executions do not establish general physical tracking accuracy or collision clearance. The older README GIF/MP4 remains labeled historical rather than presented as new-solver evidence.

All 62 tests passed with the active pose weights, including shared sim/real-adapter behavior, small target convergence, command limits and control-owner process cleanup. No physical arm was enabled or moved during validation.

Validation measurements are recorded in `previews/official_ik_report.json`. The upstream-default run (17.20 / 16.09 mm) is preserved in `previews/official_defaults_ik_report.json`. Both use the same official TCP, target sequence and simulated plant. These modeled errors are not claims of physical accuracy. The historical local-Jacobian report uses a different TCP and is not an identical-condition comparison.

With the active pose weights, the same seven-second target sequence measured 0.04575 / 0.04291 mm position error and 0.04253° / 0.04224° orientation error (left/right). Median control tick time was 4.41 ms; p95 was 7.46 ms and maximum was 10.86 ms in that run, not a hard real-time guarantee. The 32-second sequence passed translation, rotation, independent gripper operation, HOLD/resume and joint-reference speed/acceleration checks; its largest sampled phase-end position residual was 0.06750 mm.

The active configuration also passed a 165-second check in the actual native MuJoCo window on desktop display `:1`, driven with X11 mouse and keyboard events. Target translations were 132.7 mm upward for the left arm and 93.4 mm for the right, followed by HOLD/resume and a 14.0-degree orientation drag. Settled sampled position errors were 0.049–0.072 mm; orientation errors were at most 0.074 degrees. There were no controller faults, and the viewer exited cleanly after HOLD. `previews/interactive_desktop_report.json` records the parameters and measured checkpoints. No physical control backend was opened.
