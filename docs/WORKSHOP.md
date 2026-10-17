# Workshop: code-as-policy and agent-as-policy

The workshop service gives participants one API for both NERO arms, independent
left/right grippers, official AgileX IK/FK, and the stand-mounted head RGB-D
camera. Python and MCP clients call the same service. Changing the service from
`sim` to `real` does not change participant method names, coordinate conventions,
or operation results. There are no wrist cameras.

## Implementation and rollout plan

1. **Shared API — implemented.** One controller owner, measured observations,
   asynchronous operations, request deduplication, cancellation, and independent
   gripper commands. Python SDK and official-SDK MCP wrapper share this API.
2. **Simulation workshop — implemented and tested.** Three tasks, seeded layouts,
   organizer-only scoring, RGB-D baselines, a model-driven MCP example, and a live
   dashboard. The tasks use contact physics, not object attachments.
3. **Physical adapter — implemented; qualification remains.** The service uses
   `HardwareSession("both")` and the serial-checked ZED adapter. Mocked hardware
   contract tests do not establish physical grasp performance. Before participant
   access, validate camera/tool calibration, both-arm motion, each gripper, STOP,
   disconnect behavior, and each task on the installed robot.
4. **Workshop release.** Freeze a tested revision, record task reset procedures,
   distribute participant configuration and examples, and rehearse participant
   turnover with an organizer operating the real robot.

## Start a simulator

Use the existing robot environment, then install the additional MCP dependency:

```bash
cd ~/projects/agilex-deploy
uv pip install --python .venv-official/bin/python -r requirements-workshop.txt
./scripts/run_workshop.sh --backend sim --task place --seed 0 --auto-start-sim
```

Open **http://127.0.0.1:8780**. The dashboard shows the head camera, an external
simulator view, measured arm/gripper state, operations, and events. The external
view is for observers; policies receive only the head camera.

The server writes participant credentials to `logs/workshop/client.json` and a
private organizer URL to `logs/workshop/operator_url.txt` (both mode 0600). Open
the organizer URL for Enable and STOP controls. Keep that URL private. STOP
cancels pursuit and latches HOLD; restarting the service is required to resume.
HOLD does not disable motors or make power disconnection safe.

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

The model example reads its OpenRouter key from `~/projects/openrouter_key.txt`
by default. Use `--key-file`, `--model`, and `--max-calls` to override. Only one
policy may own the session; stop/release the previous client before starting
another. Model completion text is not the task score.

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

The existing camera extrinsics are a **candidate**, not a qualified workshop
calibration. The simulated table height, cube/tray dimensions, grasp-center
offset, and nominal poses must match the physical setup before baseline use.
The task argument supplies an instruction; it does not arrange physical objects.
Physical reset and scoring currently require an organizer. No physical motion
was executed while implementing or testing this workshop service.

For remote participants, run clients on a separate machine/account and forward
the localhost service with SSH. Share only `client.json`, not the private
organizer URL, controller checkout access, CAN devices, or organizer logs. The
API is not a sandbox for arbitrary code running under the controller's own OS
user. The SDK can be installed with `pip install --no-deps -e .`; the scripted
vision example additionally needs NumPy and OpenCV, and MCP clients need the
workshop requirements.

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

See [validation summary](../previews/workshop_validation.json),
[placement video](../previews/workshop_place.mp4),
[dual-arm sorting video](../previews/workshop_sort.mp4), and
[dashboard screenshot](../previews/workshop_dashboard.png).
