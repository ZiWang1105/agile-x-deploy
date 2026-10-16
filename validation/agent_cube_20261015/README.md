# Live OpenRouter cube pickup evidence

This archives `logs/agent/cube_agent_08`, the final default-policy simulator run
on October 15, 2026. It used seven live `google/gemini-3.8-flash` calls and
reported $0.0692655 API cost. No physical robot motion was executed.

- `report.json`: final measured observation and offline evaluation: 152.2 mm
  maximum lift, 32.6 seconds uninterrupted final retention, no arm/environment
  collision frames, success=true.
- `policy/`: exact credential-free model requests, model responses, unannotated
  RGB, millimeter depth PNGs, and per-frame camera geometry.
- `turns.json`: model decisions, selected-pixel resolution, executed actions,
  and measured execution results. The model selected the left arm after an
  inspect-only grounding call; no host pickup sequence was executed.
- `evaluator_truth.json.gz`: separate privileged physics trace, read only after
  the policy/control session ended. It was not available to the model.
- `scene.json`: plant geometry and initial posture; not policy input.
- `source/` and `source_manifest.json`: exact Python implementation at launch.
- `input_audit.json`: all seven actual requests checked against archived RGB
  and depth; selected-pixel geometry reproduced exactly.
- `archive_manifest.json`: SHA-256 hashes of archive files.

The [complete video](../../previews/agent_cube_pickup.mp4) includes the entire
59-second simulated trajectory through final verification. It is reconstructed
from recorded physical simulator joint/object states, not generated target
animation or a scripted replay. The task cube has a free joint; friction and
opposing finger contact support it. No grasp weld or attachment is used.

The [validation summary](../../previews/agent_policy_validation.json) also lists
earlier trials, including failures and the preceding successful recovery run.
This is a demonstrated task instance, not a general success-rate estimate.
Self-collision and calibrated hardware force dynamics are outside this validation.

To repeat the model-input audit from the repository root:

```bash
.venv-official/bin/python scripts/audit_agent_run.py validation/agent_cube_20261015
```
