# [Workshop on Agentic Robotics](https://agentic-robotics-workshop.github.io/): code-as-policy and agent-as-policy

The [Workshop on Agentic Robotics](https://agentic-robotics-workshop.github.io/)
service gives participants one API for both NERO arms, independent
left/right grippers, official AgileX IK/FK, and the stand-mounted head RGB-D
camera. Python and MCP clients call the same service. Changing the service from
`sim` to `real` does not change participant method names, coordinate conventions,
or operation results. There are no wrist cameras.

## Implementation and rollout plan

1. **Shared API — implemented.** One controller owner, measured observations,
   asynchronous operations, request deduplication, cancellation, and independent
   gripper commands. Python SDK and official-SDK MCP wrapper share this API.
2. **Simulation [workshop](https://agentic-robotics-workshop.github.io/) — implemented and tested.** Three tasks, seeded layouts,
   organizer-only scoring, RGB-D baselines, a model-driven MCP example, and a live
   dashboard. The tasks use contact physics, not object attachments.
3. **Physical adapter — implemented and exercised.** The organizer confirmed
   code-as-policy and agent-as-policy working on the installed robot on October
   20, 2026. The service uses `HardwareSession("both")` and the serial-checked
   ZED adapter. This records successful policy demonstrations, not qualification
   of every task, object, calibration change, or failure condition.
4. **[Workshop](https://agentic-robotics-workshop.github.io/) release.** Freeze a tested revision, record task reset procedures,
   distribute participant configuration and examples, and rehearse participant
   turnover with an organizer operating the real robot.

## Start an agent and its dashboard with one command

Both agent runners own the robot service and open its organizer dashboard by
default. The simulator and real robot use the same page, enable/start button,
STOP, Close session, camera/robot APIs, and code/tool trace panels. Only the
backend connection and the explicit SIMULATION / PHYSICAL ROBOT badge differ.

```bash
cd ~/projects/agilex-deploy
# MCP agent; starts the dashboard and waits for its enable/start button.
.venv-official/bin/python examples/workshop/agent_policy.py --backend sim \
  --prompt 'Put the red cube in the green tray using the left arm.'

# Generated Python policy, with the same dashboard and start sequence.
.venv-official/bin/python examples/workshop/generated_code_policy.py --backend sim \
  --prompt 'Put the red cube in the green tray using the left arm.'
```

Use `--backend real` in either command for the physical robot. Close an existing
IK viewer with Q first so it releases CAN ownership. There is no separate
`run_workshop.sh` or `xdg-open` step. Review the camera and displayed prompt, then
click **Enable & start policy**. Until that click, the runner does not call the
model or enable motors. Enable/start also requires the existing private organizer
token; participant and MCP tools cannot enable the robot.

Gravity compensation/hand-guiding defaults to **false** in both backends.
Normal completion releases the participant lease and retains position HOLD.
The website stays open for result review and follow-up prompts. Edit **Policy
prompt** before enabling, or click **Run next prompt** after a round finishes.
MCP and generated-code policies reuse the same enabled robot and camera session;
follow-up rounds do not re-arm, reset poses, or reset grippers. The last ten rounds'
prompts and result summaries provide conversation context, while each round must
inspect current feedback and images. A new prompt cannot interrupt an active
round or clear STOP/a hardware fault. Each round has its own report; later reports
use `_round_002.json`, `_round_003.json`, and so on.

**STOP** cancels the policy and holds
position; **Close session** or Ctrl-C also closes the service. These actions do
not disable motors. With both arms supported and the controller closed, use
`.venv-official/bin/python -m agilex_deploy.hardware_power --arms-supported`
to remove motor holding torque. `--no-gravity-comp` makes the default explicit;
`--gravity-comp` opts into the experimental [hand-guiding lifecycle](HAND_GUIDING.md).
Ordinary real position control is available with the default setting. Physical
`--gravity-comp` requests are blocked before motor enable. Both disabled arms,
fresh healthy feedback, controller limits, and arming-displacement checks remain
required; faults are never reset automatically.

The key is read from the repository's ignored `openrouter_key.txt`. Per-run logs,
client configuration, organizer URL, and policy report live under
`logs/workshop/policy_<timestamp>_<id>/`. Use `--no-browser` if you want to open the
printed organizer URL yourself, `--port 0` to select an available port, or
`--exit-after-policy` to close automatically after execution. The default backend
is sim. `--task place|select|sort` selects the simulator task layout; `--prompt`
supplies the agent's instruction.

For an organizer-managed service, the existing `--config PATH` mode still
attaches without starting another server or opening a browser. It cannot be
combined with `--backend`. The separate-service examples below use that mode.

## Start a simulator

Use the prepared robot environment. For a fresh Linux x86_64 checkout, first run
`./scripts/setup_official_ik.sh` and `./scripts/setup_gripper.sh`, then install
the camera and [workshop](https://agentic-robotics-workshop.github.io/) dependencies:

```bash
cd ~/projects/agilex-deploy
.venv-official/bin/python -m pip install -r requirements-agent.txt -r requirements-workshop.txt
./scripts/run_workshop.sh --backend sim --task place --seed 0 --auto-start-sim
```

Open **http://127.0.0.1:8780**. The dashboard shows the head camera, an external
simulator view, measured arm/gripper state, operations, and events. The external
view is for observers; policies receive only the head camera.

The server writes participant credentials to `logs/workshop/client.json` and a
private organizer URL to `logs/workshop/operator_url.txt` (both mode 0600). Open
the organizer URL for Enable and STOP controls. Keep that URL private. STOP
cancels pursuit, latches policy cancellation, and retains position HOLD by default.
Restart the service to clear operator STOP. A normal policy exit releases its
participant lease and retains HOLD; another policy can claim that same service.
No leader-mode transition occurs unless hand-guiding was explicitly opted into.
Support the arms before disabling motors or disconnecting power.

The service runs in the foreground. Ctrl-C closes it. Use a different `--port`
and `--client-config` for another simulator. To reset, stop the service and start
a new episode with the desired task and seed; participant APIs cannot reset it.

| Task | Instruction | Baseline arm use |
| --- | --- | --- |
| `place` | Red cube into green tray | Left |
| `select` | Blue cube into green tray; red cube stays untouched | Right |
| `sort` | Red cube into red tray and blue cube into blue tray | Both, sequential picks |

Cubes are 50 mm, 40 g. Seed zero uses the nominal layout; other seeds vary cube
XY by up to 8 mm. These are bounded tabletop tasks, not an arbitrary scene
generator or a guaranteed collision-free workspace.

## Python: code-as-policy

Run the supplied visual baseline in a second terminal:

```bash
cd ~/projects/agilex-deploy
.venv-official/bin/python examples/workshop/code_policy.py \
  --config logs/workshop/client.json
```

The baseline segments colored objects in RGB and uses registered depth and the
camera transform. It does not import simulator state or query object truth.
It is a scripted baseline, not an LLM policy.

The SDK itself uses only the Python standard library:

```python
from agilex_deploy.workshop_client import RobotClient

with RobotClient.from_config("logs/workshop/client.json") as robot:
    state = robot.observe()
    frame = robot.capture()
    target = state["left_tcp"]
    solution = robot.ik("left", target)  # Read-only; does not execute.

    # Only the left jaw changes. The right jaw keeps its previous target.
    result = robot.gripper_open("left").wait()
    if result["state"] != "succeeded":
        raise RuntimeError(result)

    # Example endpoint command: hold the currently measured left TCP pose.
    result = robot.move_tcp("left", target).wait()
    if result["state"] != "succeeded":
        raise RuntimeError(result)
```

Use `move_both({"left": left_pose, "right": right_pose})` for an explicit paired
arm request. It does not open or close either gripper. CAN transmissions are
sequential, not hardware-atomic. One operation runs at a time; another motion
request returns `BUSY`. Both arms are available in one session.

### Execute model-generated Python

The scripted baseline above is separate from the generated-code policy. To have
OpenRouter write and execute Python against the SDK, run:

```bash
.venv-official/bin/python examples/workshop/generated_code_policy.py \
  --config logs/workshop/client.json \
  --prompt 'Put the red cube in the green tray using the left arm, release it, and retract.'
```

The model receives the task, measured robot state, camera images/geometry, and
SDK documentation. It returns one Python fence defining `run(robot)`. An isolated
worker executes that code, forwarding SDK calls through the owning host process.
Results, captured images, stdout, and errors return to the model for its next
response. The runner does not substitute the scripted baseline for generated code.

The worker uses Linux Landlock (ABI >=3) and libseccomp; it fails closed if these
are unavailable. It has read access to the Python runtime/system libraries and
write access to a temporary scratch directory, but no repository key, direct
network access, subprocess creation, or CAN access. Credentials stay in the host.
Each fence has a 90-second wall-clock budget and bounded CPU/output resources.
The host polls session status, cancels the worker on STOP, and requests HOLD when
the fence returns or fails. A new policy must claim the session after the previous
one exits. Session release, individual fences, and operation endpoints retain
position HOLD by default. Only the opt-in hand-guiding setting calls the official
NERO leader/follower methods. Simulation and protocol tests do not verify physical
gravity compensation.

Normal HOLD, successful operation completion, and session release preserve each
gripper's last actuator target and configured force. A measured aperture can
indicate an object blocking closure; substituting it for the closing target
would remove grip force. Explicit `gripper_open(arm)` releases the object.
Cancellation, operator STOP, and faults retain measured-aperture stopping behavior.

After every generated fence, the host captures a fresh camera image after worker
cleanup and sends it to the model as final-state evidence. Earlier in-code
captures are labeled separately because cleanup can change the state. A finished
model response is not an independent task-success check: the code-policy report
keeps `task_success` null unless separately evaluated.

## MCP: agent-as-policy

Launch the stdio MCP server from an MCP-capable client using this configuration
(adjust the checkout and configuration paths on another machine):

```json
{
  "mcpServers": {
    "agilex-workshop": {
      "command": "/home/ziwang/projects/agilex-deploy/scripts/run_workshop_mcp.sh",
      "args": ["--config", "/home/ziwang/projects/agilex-deploy/logs/workshop/client.json"]
    }
  }
}
```

The wrapper uses the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk),
pinned in `requirements-workshop.txt`. Claim the session before observation or
execution and release it afterward. `workshop://api` documents conventions;
the `workshop_task` prompt accepts the participant's instruction.

| Python SDK | MCP tool |
| --- | --- |
| `claim()` / `close()` | `session_claim` / `session_release` |
| `capabilities()` / `observe()` | `robot_capabilities` / `robot_observe` |
| `capture()` / `deproject()` | `camera_capture` / `camera_deproject` |
| `ik()` / `fk()` | `kinematics_ik` / `kinematics_fk` |
| `move_tcp()` / `move_both()` | `motion_move_tcp` / `motion_move_both` |
| `move_joints()` / `follow_waypoints()` | `motion_move_joints` / `motion_follow_waypoints` |
| `gripper_open(arm)` / `gripper_close(arm)` | `gripper_open` / `gripper_close` |
| `Operation.status()` / `.wait()` / `.cancel()` | `operation_get` / `operation_wait` / `operation_cancel` |
| `hold()` | `robot_hold` |

Policy camera captures retain their original RGB-D data for the last 32 distinct
captured frames, independently of live dashboard updates and model latency.
`deproject()` always uses the requested frame, never a newer substitute. Its
`age_s` reports acquisition age; capture again after the robot or objects move.
Live dashboard polling does not consume these retained-frame slots.

The real head camera streams continuously: its UVC thread updates the latest
stereo image, and a separate RGB-D worker publishes registered frames without
robot RPCs or the arm-control lock. Dashboard camera refresh is independent of
robot-state polling. `capture()` and MCP `camera_capture` pull the latest buffered
RGB-D frame and match it to fresh robot feedback; they do not trigger camera
acquisition. Frames older than 0.5 seconds are rejected, and policy captures
require robot/image timestamp skew of at most 100 ms. An unmatched frame is
never relabeled with a newer timestamp. Preview does not require robot feedback.

Both model runners reject empty and token-truncated responses. They retry within
the configured model-call budget, increase the output token allowance up to
16,384, and fail after three consecutive unusable responses. Partial tools or code
are never executed. Response traces include finish reasons and token usage;
an empty response cannot mark the policy finished. A nonempty model conclusion
still represents the model's claim, not an independent task-success measurement.

The runners keep `google/gemini-3.8-flash` and its default reasoning effort.
`--provider-sort latency` prioritizes faster OpenRouter providers; same-model
provider fallbacks stay enabled. This cannot guarantee availability or task
success. Use `--provider-sort default` to restore OpenRouter's default ordering.
`--reasoning-effort low`, `medium`, or `high` explicitly overrides model reasoning.
The low-effort live trial repeated pixel probes and did not complete the task,
so reduced reasoning is an experimental option, not the default or a verified fix.
See OpenRouter's [reasoning options](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

Transient model requests get at most three transport attempts, with exponential
backoff and jitter. `Retry-After` is honored up to 60 seconds; longer requested
delays end the run instead of retrying early. Authentication/configuration errors
fail immediately. HTTP 200 bodies containing provider errors are also checked.
Only the unanswered model request is retried: conversation history and completed
tool results are preserved, and robot commands are not replayed by the retry loop.
STOP is checked during backoff and after a response before it can trigger actions.
The dashboard and `policy_events.jsonl` show sanitized provider errors and retry
delays. An HTTP 504 from the server can arrive before the 45-second client timeout;
raising that client timeout alone does not resolve a server-generated timeout.

Test the scripted baseline through actual MCP stdio:

```bash
.venv-official/bin/python examples/workshop/mcp_baseline.py \
  --config logs/workshop/client.json
```

Run a model that chooses MCP tools from the task prompt:

```bash
.venv-official/bin/python examples/workshop/agent_policy.py \
  --config logs/workshop/client.json \
  --prompt 'Put the red cube in the green tray, release it, and retract your hand.'
```

Place your OpenRouter key in `openrouter_key.txt` at the repository root
(`~/projects/agilex-deploy/openrouter_key.txt` on this desktop). Put only the key
in the file, without quotes or an environment-variable assignment. The default
path is resolved from the checkout, regardless of the current working directory.
The file is ignored by Git and must not be committed or distributed to participants;
each participant supplies their own key. Run `chmod 600 openrouter_key.txt` from
the repository root to restrict access. Use `--key-file`, `--model`, and
`--max-calls` to override. Both generated-code and MCP policies default to
100 model calls per prompt, with a fresh budget for each dashboard round. Restart
an already running policy process to load a changed default. Only one
policy may own the session; stop/release the previous client before starting
another. Model completion text is not the task score.

## Live policy visualization

Both model-driven runners publish authenticated telemetry to the same dashboard
at **http://127.0.0.1:8780**. The **Agent activity** panel shows:

- The policy format, model, task prompt, and current model wait.
- Generated Python exactly as submitted for execution, plus stdout and errors.
- Python SDK or MCP call arguments, returned results, and call durations.
- Model response count and accumulated request time, including retry delays.

The run selector switches between recent runs. Returned asynchronous calls are
distinct from completed operations; inspect `operation_wait` results and the
Operations panel for motion completion. Internal Python status polling is omitted
from the activity list to keep code and meaningful calls visible. The server
retains the latest 200 policy events for the live page and the full stream in
`policy_events.jsonl`. Telemetry cannot enable or command the robot; it is
participant-reported evidence, not an independent success score. Text is rendered
as text, not executable HTML. API credentials and raw base64 image payloads are
excluded from the trace.

Use `--agent --policy-mode code` or `--agent --policy-mode mcp` with
`scripts/validate_workshop.py` to run isolated model-driven simulator trials.
On 2026-10-19, both completed placement: generated Python lifted 124.9 mm and MCP
lifted 130.5 mm, with release into the tray and no scored arm/environment
collisions. An initial code trial exposed a native-output/RPC framing bug; an
initial MCP trial encountered an upstream HTTP 504. The framing bug was fixed
and bounded model-request retries were added before the successful runs. These are individual trials, not success-rate estimates.

[Code dashboard recording](../previews/policy_code_dashboard.mp4) ·
[MCP dashboard recording](../previews/policy_mcp_dashboard.mp4) ·
[Validation details](../previews/policy_visual_validation.json) ·
[Executed generated Python](../validation/workshop_visual_20261019/generated_policy.py) ·
[Archived policy traces](../validation/workshop_visual_20261019).
The full regression suite passed 150 tests; browser checks reported no JavaScript
errors and no horizontal overflow for either policy view at mobile width.
These recordings show actual browser time, including model waits, rather than
re-rendered simulation trajectories. The browser interval may exclude startup or
shutdown outside the recording. Physical hardware was not accessed in these runs.

Repeat the browser checks with:

```bash
uv run --no-project --with playwright python scripts/check_policy_dashboard.py
```

This starts and closes a temporary simulator, checks both layouts at desktop and
mobile widths, checks safe text rendering, and verifies the STOP control. It
requires an installed Playwright Chromium headless shell.

## Coordinate and execution contract

- Poses use `base_link`, meters, and normalized quaternions in **xyzw** order.
  A pose is `{"position_m": [x,y,z], "quaternion_xyzw": [x,y,z,w]}`.
- Joint targets are seven radians for the named arm. Observation ordering is
  left joints 1–7, then right joints 1–7. Joint commands do not perform an FK/IK
  round trip.
- `tool_frame="tcp"` uses the official TCP; `"pinch"` uses the modeled grasp
  center. The baseline's top-down quaternion is `[0.5,-0.5,-0.5,-0.5]`.
- Camera results include RGB PNG, registered uint16 depth PNG (0.001 m per unit,
  zero invalid), intrinsics `K`, `T_base_from_optical`, a frame ID, and timestamps.
  Optical axes are X right, Y down, Z forward; depth is optical Z. Deprojection
  uses the requested cached frame, not a newer image.
- Gripper state is separate for each arm. A settled close can mean contact or an
  endpoint; it does **not** prove an object was grasped. Observe the object.
- Operations report `running`, `succeeded`, `failed`, or `cancelled`. Default
  pose completion tolerances are 5 mm and 0.05 rad; these are completion checks,
  not added speed/workspace constraints. Always check the terminal state.
- Reuse a `request_id` to retry an identical command without executing twice.
  Reusing it with changed arguments is rejected. Operations have execution
  timeouts; a client-side wait timeout alone is not cancellation.
- Official IK and the existing joint speed/acceleration limits remain in the
  control stack. IK is not a collision planner. No arbitrary participant torque
  control or firmware-limit modification is exposed.
- TCP actions use [online Cartesian waypoint planning](MOTION_PLANNING.md):
  linear translation and measured-feedback progression. [Workshop](https://agentic-robotics-workshop.github.io/) operations
  validate sampled shortest-arc IK first, then the complementary rotation arc
  if needed, without changing the endpoint. Python and MCP share this planner;
  the interactive viewer retains its continuously updated shortest-arc path.
  Joint targets bypass Cartesian planning. No obstacle avoidance is claimed.
- An authenticated owner lease expires after 120 seconds without participant
  calls, causing latched HOLD. Polling operation status renews the lease. The
  lower-level control watchdog remains separate.

## Real deployment

Close other controllers that own the arms or camera. Start the same service
with the physical backend:

```bash
./scripts/run_workshop.sh --backend real --task place --camera-serial 35330610
```

It starts unarmed. Inspect real camera/state in the dashboard. The organizer
uses the private URL to enable the arms after confirming the physical setup.
Participant tools cannot enable hardware. Run the same Python or MCP policy
against the generated configuration; it now commands the physical robot.

The existing camera extrinsics are a **candidate**, not a qualified [workshop](https://agentic-robotics-workshop.github.io/)
calibration. The simulated table height, cube/tray dimensions, grasp-center
offset, and nominal poses must match the physical setup before baseline use.
The task argument supplies an instruction; it does not arrange physical objects.
Physical reset and scoring currently require an organizer. Both model-driven
policy formats have been exercised on the installed robot, as confirmed by the
organizer. The [README demonstrations](../README.md#simulator-results) and
[machine-readable results](../previews/workshop_policy_results.json) are
independently scored simulator trials; they are not physical benchmark scores.

For remote participants, run clients on a separate machine/account and forward
the localhost service with SSH. Share only `client.json`, not the private
organizer URL, controller checkout access, CAN devices, or organizer logs. The
API is not a sandbox for arbitrary code running under the controller's own OS
user. The SDK can be installed with `pip install --no-deps -e .`; the scripted
vision example additionally needs NumPy and OpenCV, and MCP clients need the
[workshop](https://agentic-robotics-workshop.github.io/) requirements.

## Validation and records

The organizer can run an isolated sim episode and score it:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-official/bin/python scripts/validate_workshop.py \
  --task sort --seed 0 --video --output logs/workshop/my_sort_trial
```

Add `--transport mcp` for the scripted MCP baseline or `--agent` for the live
model-driven MCP policy. Use a fresh output directory for each trial.

Logs include API/execution events, requested RGB-D frames, and organizer-only
simulation truth and scores. Participant methods expose neither scoring nor
object coordinates. The scorer requires a lift of at least 25 mm, the full cube
footprint inside its assigned tray, release from the fingers, tray support,
and at least one second of low-velocity stability. It rejects scored penetrating
arm/environment contacts. `select` additionally limits distractor translation
to less than 10 mm throughout the episode. This contact metric is not a proof
of universal collision avoidance or self-collision checking.

Recorded successful trials on 2026-10-16:

| Policy / transport | Task / seed | Measured lift | Result |
| --- | --- | --- | --- |
| Scripted RGB-D / Python | `place` / 0 | 126.5 mm | Released into tray; 2.81 s stable |
| Scripted RGB-D / Python | `sort` / 0 | Red 126.2 mm; blue 125.1 mm | Both released into assigned trays |
| Scripted RGB-D / MCP | `select` / 2 | Blue 126.3 mm | Placed; maximum red displacement 1.6 mm |
| Live Gemini / MCP | `place` / 0 | 121.4 mm | Released into tray; 24.83 s stable |

All listed runs recorded zero scored arm/environment collision steps. These are
individual successful trials, not a statistical success-rate benchmark. The
model run used 34 model calls. Baseline sorting uses both arms sequentially;
it does not demonstrate a simultaneous two-object grasp.

See [validation summary](../previews/workshop_validation.json) and
[dashboard screenshot](../previews/workshop_dashboard.png).
