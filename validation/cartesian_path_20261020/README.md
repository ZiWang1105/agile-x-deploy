# Cartesian waypoint validation — 2026-10-20

All runs used actuator simulation. No physical robot commands or OpenRouter
requests were issued.

* `path_comparison.json`: six-second, simultaneous dual-arm comparison of the
  previous direct-endpoint IK execution against the new Cartesian path tracker.
  Both use official AgileX IK and identical joint limits. Left jaws open while
  right jaws close.
* `python_sort.json`: scripted RGB-D Python SDK policy sorts both cubes using
  both arms, sequentially, with independent grippers.
* `mcp_select.json`: scripted RGB-D MCP policy selects and places the blue cube
  with the right arm while preserving the red distractor.
* `tests.log`: final full regression run; 159 tests passed.

See [reproduction commands and limitations](../../docs/MOTION_PLANNING.md),
[GIF](../../previews/cartesian_path_validation.gif), and
[MP4](../../previews/cartesian_path_validation.mp4).

The recording shows 1x simulation-time playback, not wall-clock timing. The
tracker does not check collisions or guarantee reachability of future
waypoints. These trials are not a general accuracy or physical safety claim.
