# Automatic hand-guiding validation — 2026-10-20

No physical CAN writes or robot mode switches were performed. No OpenRouter
requests were made and no API key is included.

- Full regression suite: 170 tests passed in 202.444 seconds.
- Focused hand-guiding suite: 11 tests passed in 16.084 seconds. This includes
  an additional supported-power-off regression added after full-suite discovery.
- Workshop API regression after the camera transition freshness change: 12 tests passed.
- Interactive viewer regression after idle-handle refresh: 8 tests passed.
- Python API RGB-D baseline: two-arm sorting passed; red lift 126.24 mm,
  blue lift 126.36 mm; zero scored arm/environment collision frames.
- MCP RGB-D baseline: blue-object selection passed; lift 126.22 mm; red distractor
  preserved; zero scored arm/environment collision frames.
- Both policy hosts released their sessions and recorded automatic hand-guiding,
  with no fault event. These are scripted interface checks, not new LLM evaluations.

The focused tests use the installed official SDK encoder methods and a mocked CAN
plant, the real child-process session API backed by MuJoCo, and the workshop service.
They check source timestamps, leader/follower transitions, local loopback rejection,
partial failures, no position writes during hand-guiding, fresh-pose reseeding,
reconnection, independent gripper preservation, and supported shutdown.

Physical firmware telemetry behavior, gravity support with side mounting, and
payload compensation remain unverified. See ../../docs/HAND_GUIDING.md for the
first supported physical test and lifecycle semantics.
