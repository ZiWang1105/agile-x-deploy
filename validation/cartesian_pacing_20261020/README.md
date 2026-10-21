# Waypoint pacing correction — 2026-10-20

The original tracker gave the joint controller a nearby IK goal and waited
before advancing. Its braking law consequently slowed down toward intermediate
samples even though the Cartesian path continued.

Adaptive lookahead uses the current joint reference velocity, acceleration
limits, and model Jacobian to choose a farther target over the joint stopping
horizon. It does not change the joint limits, final endpoint, IK weights,
physical firmware configuration, or playback rate.

`comparison.json` records the same simultaneous dual-arm motion with direct
endpoint IK, original waypoint pacing, and adaptive lookahead. For both arms:

| Metric | Original pacing | Adaptive lookahead |
| --- | --- | --- |
| Settled within 5 mm / 0.05 rad | 3.86 s | 0.86 s |
| Settled within 1 mm / 0.05 rad | 3.99 s | 2.55 s |
| Maximum distance from the planned straight line | 6.58 mm | 11.58 mm |
| Final position error | approximately 0.046 mm | approximately 0.046 mm |

Settling means remaining inside the stated tolerance for the rest of the
six-second recording. Both versions meet the joint reference velocity and
acceleration limits. Faster pacing has a measurable path-deviation tradeoff.

`tests.log` records 160 passing regression tests. `python_sort.json` and
`mcp_select.json` record the scripted RGB-D API task checks. These tests use
simulation and do not call OpenRouter or command physical hardware.

The [MP4](../../previews/cartesian_pacing_comparison.mp4) and
[GIF](../../previews/cartesian_pacing_comparison.gif) show before and after at
the same 1x simulation-time rate. Video is rendered after actuator simulation,
not a wall-clock recording. See [motion planning documentation](../../docs/MOTION_PLANNING.md)
for reproduction commands and limitations.
