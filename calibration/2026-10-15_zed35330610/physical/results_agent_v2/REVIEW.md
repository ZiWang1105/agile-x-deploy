# Physical ZED calibration review

Status: **not validated; not applied to the robot**.

All nine captured poses passed the recorded stationarity checks. Original images,
encoder samples, and the original unannotated dataset remain unchanged.

The assistant selected four fixed rail end-cap corners in CAD and raw photos,
then added one independent wrist-housing screw visible in five photos. Two
occluded rail ends were omitted in each of poses 8 and 9. These are visual
annotations, not automatic detections or subpixel measurements. No camera
projection was used as an observed image coordinate. Rounded CAD edge locations
and approximate screw surface location introduce possible systematic error.

## Results

- 37 correspondences across nine poses and five fixed landmarks.
- All-point reprojection RMS: 8.65 pixels.
- RANSAC inlier RMS: 3.22 pixels, using only 24 of 37 matches.
- Worst leave-one-pose-out RMS: 15.05 pixels.
- Corner-only fit: 6.43-pixel all-point RMS. Independent screw predictions miss
  by 11.5–14.4 pixels. This prevents interpreting the small inlier error as
  evidence of accurate extrinsics.
- Static upper-background optical flow has less than 0.43-pixel 90th-percentile
  displacement per image. Large camera motion is not apparent in that region.

`candidate.json` contains a diagnostic transform, not an accepted calibration.
`validation.jpg` shows green observed points, red projected crosses and yellow
residual segments. `review.json` preserves numerical checks.

Factory intrinsics and kinematics were held fixed. The original eight-coefficient
factory distortion model remains in use. Diagnostic use of only five coefficients
and a robust all-pair fit did not resolve the inconsistency and were not adopted.
Image/encoder timestamps remain host-read based, not qualified exposure timing.

The cause is unresolved. Check feature identity and CAD/physical assembly geometry
before adopting a transform or collecting another identical motion sequence.
Do not tune camera intrinsics or kinematics merely to reduce these residuals.
No physical arm commands were sent during annotation, fitting, or review.
