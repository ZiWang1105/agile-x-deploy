# Raised physical camera calibration review

The latest run contains 15 stationary captures and 132 image observations on
11 CAD landmarks. The assistant selected the image points. No robot commands
were sent during this analysis and no active calibration was changed.

The all-observation reprojection RMS is **4.52 px**. Of 132 observations,
112 are within the 6 px RANSAC threshold; their RMS is **3.13 px**. The worst
leave-one-pose-out RMS is **7.01 px**. These are pixel consistency measures,
not independently established metric accuracy.

The initial 132-point review had 7.38 px RMS. Enlarged image inspection found
one background-bracket misidentification and several rail front/side corner
ambiguities. `annotation_revisions_v2.json` records every pixel revision.
The original captures and first annotation version are preserved.

All fixed factory intrinsics, measured joints, geometry, and landmark local
coordinates remain unchanged. The candidate remains **not applied** because
some gap corners depend on occlusion and the wrist barrel screw is less
consistent. The 40-point wrist-only diagnostic has 2.66 px RMS and 5.64 px
worst held-out-pose RMS; that post-review subset is not proof of accuracy.

Next, resolve the gap-corner and barrel-screw CAD correspondences using the
existing images. Additional robot motion is not needed for that investigation.
See `review.json` for per-landmark held-out checks and static-background flow.
