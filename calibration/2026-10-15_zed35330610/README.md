# ZED 2i physical calibration archive — 2026-10-15

**Status: candidate extrinsics, review required, not applied to robot control.**
Camera serial: `35330610`. Images are the raw, distorted LEFT camera stream at
1280 × 720. The robot reference frame is `base_link`; the camera frame is
`zed_left_optical` (X right, Y down, Z forward). Transform translations are meters.

## Latest result

- [Candidate extrinsics](raised_probe/results_agent_v2/candidate.json): both
  `T_camera_from_base` and its inverse, `T_base_from_camera`.
- [Review](raised_probe/results_agent_v2/REVIEW.md) and
  [machine-readable diagnostics](raised_probe/results_agent_v2/review.json).
- [All 15 annotated images](raised_probe/results_agent_v2/all_15_annotated.jpg)
  and [six enlarged views](raised_probe/results_agent_v2/validation_detail.jpg).
- [Annotated dataset](raised_probe/dataset_agent_v2.json), including measured
  joint samples, camera intrinsics, image hashes, CAD anchors and observations.

The latest sequence contains 15 stationary captures, 11 landmark definitions and
132 image observations. The assistant selected points from photographs; they
were not generated from projected CAD coordinates. All-point RMS is 4.521 px;
112/132 points are within the 6 px inlier threshold, with 3.129 px inlier RMS.
Worst leave-one-pose-out RMS is 7.010 px. These pixel consistency measurements
do not establish independent millimeter accuracy.

Gap-corner occlusion and the wrist barrel screw correspondence remain uncertain.
No active calibration was changed. Neither the full candidate nor the wrist-only
diagnostic is a validated transform for autonomous physical motion.

## Provenance and history

`physical/` preserves the earlier nine-pose dataset, its rejected candidates and
the original factory calibration file. `raised_probe/` preserves the subsequent
15-pose dataset, annotation revisions, CAD picks, result images, review scripts,
the executed continuation plan and its completion evidence. Earlier annotation
versions are retained so corrections can be audited. `archive_manifest.json`
records SHA-256 hashes of the copied files; source logs remain local and unchanged.

`physical/intrinsics.json` uses the factory LEFT HD matrix and all eight
coefficients from `LEFT_DISTO`. The serial was read from the sole connected ZED
HID device; the UVC interface does not expose that serial, so the saved direct
capture-device serial-verification flag remains false. Host image-read timestamps
are not exposure timestamps; `timing_qualified` remains false.

Absolute paths in saved provenance refer to the original capture workstation.
Image paths in the datasets are relative and remain usable from this archive.
The archive retains the exact dataset bytes, preserving the candidate's dataset
hash. Model hashes are checked when reproducing a fit.

## Reproduce the fit offline

From the repository root, after installing the official IK and calibration
environments and building the model as described in the main documentation:

```bash
.venv-official/bin/python scripts/calibrate_camera.py solve \
  --dataset calibration/2026-10-15_zed35330610/raised_probe/dataset_agent_v2.json \
  --output logs/calibration/reproduced_20261015/candidate.json
```

This command reads saved images and joint states; it does not open CAN or move
hardware. Write new results under `logs/` to preserve the archived evidence.
Archived motion plans describe historical starting poses. Do not replay them on
the robot: acquire a fresh stationary seed and generate/review a new approach.
