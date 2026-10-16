# ZED head-camera extrinsics from NERO arm landmarks

This workflow estimates the fixed transform `T_base_from_camera` from the ZED's
LEFT optical frame into the stand's `base_link`. It follows the Hexfellow
head-camera landmark procedure: plan diverse poses, capture images with measured
joint configurations, define fixed 3D arm landmarks, click their corresponding
image pixels, and fit/validate one camera transform. It does not require a board.

The implementation is in `src/agilex_deploy/camera_calibration.py` and the
`calibration_*` modules. The command-line entrypoint is
`scripts/calibrate_camera.py`. The browser annotation server has no robot motion
or enable endpoints. Fitting writes a candidate file; it does not update the URDF,
robot observations, controller, or any active calibration.

## 1. Install and rehearse without hardware

From the repository root:

```bash
./scripts/setup_official_ik.sh       # If the official IK environment is not ready.
./scripts/setup_camera_calibration.sh

.venv-official/bin/python scripts/calibrate_camera.py demo \
  --output logs/calibration/demo
.venv-official/bin/python scripts/calibrate_camera.py annotate \
  --dataset logs/calibration/demo/dataset.json
```

Open http://127.0.0.1:8772. The demo contains explicitly synthetic images,
known camera pose, fixed surface anchors and noisy pixel matches. Its ground
truth is used only for offline evaluation, never as a robot observation.
Use a new output directory each time; the demo refuses to overwrite evidence.
Ctrl+C closes the annotation server.

Generate a nominal route and rehearse it through the shared simulation controller:

```bash
.venv-official/bin/python scripts/calibrate_camera.py plan \
  --side left --output logs/calibration/left_plan.json
.venv-official/bin/python scripts/calibrate_camera.py run \
  --plan logs/calibration/left_plan.json --mode sim
```

The plan produces a JSON route and a nine-panel JPG. Defaults sample 60 mm TCP
offsets and 15-degree orientation offsets; `--translation-m` and `--rotation-deg`
change the sampling extent. These are pose choices, not additional speed or
acceleration constraints. For a target that fails IK convergence, planning tries
the opposite X direction, then half and quarter translation/orientation extents.
It retains the 2 mm / 1 degree convergence gates and rejects near-duplicate joint
postures. The plan records every attempted offset and residual, including the
selected alternative. Review the generated route rather than assuming it follows
the nominal offsets. If no candidate works, planning fails without writing a route;
do not run the next command after a planning failure.
Official AgileX IK computes the targets, and rehearsal
uses the existing robot interface, reference controller and simulated plant.
The opposite arm keeps its reference pose. Final measured simulation errors are
saved separately from the kinematic preview.

A nominal simulation seed is deliberately ineligible for physical execution.
The current simulator disables contacts: successful rehearsal does not establish
clearance to the stand, other arm, table, tools, people or cables. The route does
not yet optimize camera visibility, reject self-occluded features, or perform
collision-aware path planning. Review the whole trajectory, not only its endpoints.

### Raised, more varied capture sequence

To improve image detail from a low starting posture, `plan --lift-m 0.20 --poses 15`
raises the sampling centre by 200 mm along base Z and samples 15 distinct poses.
The added poses include a lower depth layer and both roll directions. For example:

```bash
.venv-official/bin/python scripts/calibrate_camera.py plan \
  --seed logs/calibration/start_snapshot.json --side left \
  --lift-m 0.20 --poses 15 --translation-m 0.06 --rotation-deg 20 \
  --output logs/calibration/raised_left_plan.json
```

The route explicitly includes the initial hold and intermediate lift targets
spaced at most 50 mm apart; these approach targets are not calibration captures.
This spacing is a route design choice, not a speed or acceleration limit.
Existing routes without a `capture` field retain their original capture behavior.
There is no automatic descent at completion.

Raising the arm does not establish camera clearance or visibility. Review the
approach and sampling paths and inspect an actual raised image before collecting
the full sequence. Aim for identifiable fixed features spread across the gripper
base, wrist and forearm rather than many points on one small rail. Do not count
occluded, blurred, moving-finger or reflection features merely to meet a point
quota. A camera-view preview from an unvalidated extrinsic is only approximate.

## 2. Obtain intrinsics for this camera and image space

The existing stream serves **raw, unrectified LEFT 1280 × 720** images. It needs
matching raw intrinsics and distortion. Do not substitute rectified SDK intrinsics
or another camera's factory file. Stereolabs documents the factory file and raw/
rectified distinction in its [camera calibration documentation](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration).

Find `SN<SERIAL>.conf` in `/usr/local/zed/settings/`, or download the factory file
from `https://calib.stereolabs.com/?SN=<SERIAL>` using the serial on this camera.
The UVC video interface may expose only `OV0001`. On a ZED 2i, inspect the
separate Stereolabs HID interface (`2b03:f881`) with `lsusb -v -d 2b03:f881`;
its `iSerial` contains the camera serial. Do not associate an arbitrary HID
device with a video device when multiple cameras are connected. The importer
alone does not verify the serial against the capture device.

```bash
.venv-official/bin/python scripts/calibrate_camera.py import-intrinsics \
  --conf /path/to/SN12345678.conf --serial 12345678 \
  --output logs/calibration/zed_left_raw_intrinsics.json
.venv-official/bin/python scripts/calibrate_camera.py init \
  --intrinsics logs/calibration/zed_left_raw_intrinsics.json \
  --dataset logs/calibration/physical/dataset.json
```

The importer selects `LEFT_CAM_HD` for the camera matrix and prefers the separate
`LEFT_DISTO` section for distortion when present, falling back to `LEFT_CAM_HD`
for older files. It preserves OpenCV coefficient order
`k1,k2,p1,p2,k3[,k4,k5,k6,s1,s2,s3,s4]`, and records the file hash. The solver
supports the declared OpenCV Brown-Conrady model; it rejects unspecified/fisheye
models. Intrinsics are fixed during extrinsic fitting. No focal length, robot
kinematics, landmark depth, or mount translation is secretly adjusted to improve
pixel errors.

## 3. Capture stationary physical observations

The camera must stay rigid relative to the stand throughout collection. Changing
the tilt adjustment or remounting the camera invalidates the previous extrinsics.
Capture several distinct arm configurations and landmarks spread across the image
and in depth. Use at least four distinct annotated poses; six to nine give more
opportunities to inspect held-out predictions.

Restart the updated stream once to expose atomic image/acquisition metadata:

```bash
./scripts/zed_stream.sh restart
```

There are two capture modes.

### Read-only, operator-positioned capture

Position the arm with the existing interactive control workflow, then hold it
stationary. This command only listens to CAN; it never enables motors or sends
motion commands:

```bash
.venv-official/bin/python scripts/calibrate_camera.py capture \
  --dataset logs/calibration/physical/dataset.json
```

Repeat at different poses. It uses the physically confirmed left/right USB-to-CAN
mapping. It does not identify arm sides again or infer unmeasured gripper travel.
Only fixed arm housing/flange landmarks are allowed, so finger position is not
needed for the fit.

### Execute a reviewed single-arm route and capture automatically

A nine-pose physical sequence and a subsequent 15-pose raised sequence have
completed. The raised sequence combines one probe capture with 14 continuation
captures. This does not qualify other paths or establish collision clearance.
The [physical calibration archive](../calibration/2026-10-15_zed35330610/README.md)
contains original images, joint feedback, annotations and calculated candidate
extrinsics. The latest 132-observation fit has 4.521 px all-point RMS and 7.010 px
worst leave-one-pose-out RMS. Feature correspondence review is incomplete, and
the candidate has not been applied to robot control.

Before recording the seed, mechanically support the arms, close other controllers,
and ensure both arms are disabled. Record the seed **after** disabling: gravity or
manual support can change joint angles substantially when motors release. Keep
that posture through planning, rehearsal and startup. A successful capture run
leaves motors enabled; do not plan a continuation from its held pose and then
disable afterward. Disable with support first, then acquire fresh feedback and
plan the approach from the resulting pose.

Obtain a receive-only seed, plan from it, rehearse, and inspect the result:

```bash
.venv-official/bin/python -m agilex_deploy.can_feedback \
  --output logs/calibration/start_snapshot.json
.venv-official/bin/python scripts/calibrate_camera.py plan \
  --seed logs/calibration/start_snapshot.json --side left \
  --output logs/calibration/physical_left_plan.json
.venv-official/bin/python scripts/calibrate_camera.py run \
  --plan logs/calibration/physical_left_plan.json --mode sim
```

After reviewing the entire approach and path, with the selected arm supported,
both arms initially disabled, and the workspace ready, the explicit execution
command is:

```bash
.venv-official/bin/python scripts/calibrate_camera.py run \
  --plan logs/calibration/physical_left_plan.json --mode real --reviewed-plan \
  --dataset logs/calibration/physical/dataset.json
```

It requires typing `ARM` before creating the hardware owner. Fresh measured joints
must match the reviewed starting configuration within 0.05 rad per joint. This
prevents an unreviewed approach from an unrelated posture. `HardwareSession` owns
all physical commands; existing feedback, watchdog and controller-readback
speed/acceleration limits remain in force. No calibration-specific speed cap is
added. The sequence runs one arm, waits for measured stability, captures, then
advances. Failed tracking/capture aborts with a HOLD request. There is no automatic
return motion or automatic retry/rearm.

Ctrl+C requests HOLD and closes the owner. As in the existing control stack,
closing leaves motors enabled and does not guarantee a physical emergency stop.
Offline planning, annotation and fitting do not send real motion commands. The
explicit real execution command above does.

### Image/encoder timing and data provenance

Each capture retains the original JPEG and hash, camera headers, all timestamped
measured joint samples, selected measured configuration, and stationarity evidence.
Capture requires a two-second acquisition interval, at least 0.5 seconds of
measured stability before and after the image's host read timestamp, no feedback
gap over 200 ms, a nearby sample within 50 ms, and maximum joint span below
0.003 rad. `--max-span-rad` adjusts the read-only capture quality gate; it does not
change motion limits.

The UVC camera currently exposes **host read-completion time, not exposure time**.
JPEG encoding and HTTP delivery times are not substituted for acquisition time.
Buffered frame/exposure latency is unqualified, so `timing_qualified=false` remains
in captures/results. A stationary interval reduces this sensitivity but does not
prove hardware synchronization. Capture refuses remote stream URLs because host
monotonic clocks are not interchangeable. For future moving-camera or moving-arm
calibration, use qualified SDK/exposure timestamps and an explicit clock mapping.

## 4. Select 3D anchors and matching image pixels

```bash
.venv-official/bin/python scripts/calibrate_camera.py annotate \
  --dataset logs/calibration/physical/dataset.json
```

1. Select a captured pose. The left pane replays that image's **measured** joints.
2. Enter a feature name, such as `left_elbow_screw`. Rotate/zoom the model view,
   then click the precise visible rigid surface feature. MuJoCo ray selection
   records its body name and body-local XYZ in meters.
3. Click the **same physical feature** in the right image. Coordinates are saved
   at the original image resolution, regardless of browser display scaling.
4. At another pose, select the existing feature name and click it in the image.
   Its 3D anchor stays fixed on that body and is transformed by measured FK.
5. Skip hidden/ambiguous features. Use “Remove this pose's match” to exclude a
   correspondence. A new click replaces the same pose/feature match; duplicates
   do not inflate the fit. To redefine a 3D anchor, use a new feature name and
   remove old matches rather than silently changing its depth.

Use screws, housing junctions or other uniquely identifiable CAD features. Avoid
curved silhouettes, cables, movable fingers and points whose identity changes
with viewpoint. 2D-to-2D image matching alone cannot establish metric extrinsics.
The UI's points are geometric surface intersections, not an automatic guarantee
that the chosen CAD and physical features correspond.

Do not capture/write the same dataset while editing it in the annotation server.
Finish collection first, then annotate. Per-capture sidecars preserve original
measurements separately from the editable annotation index.

## 5. Fit, inspect and validate

Click **Fit + held-out validation**, or run:

```bash
.venv-official/bin/python scripts/calibrate_camera.py solve \
  --dataset logs/calibration/physical/dataset.json \
  --output logs/calibration/physical/results/candidate_01.json
```

The fit uses OpenCV RANSAC EPNP initialization (3,000 iterations, 0.999 confidence,
6 px threshold), followed by robust `soft_l1` refinement on RANSAC inliers with a
3 px scale. Projection includes the declared distortion coefficients. See the
[OpenCV PnP conventions](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
At least eight correspondences and non-coplanar 3D spread are required.
Near-duplicate annotated postures (all joints within 0.01 rad) are rejected to
keep the held-out assessment from training on a second capture of the same pose. The report
keeps inlier RMS and **all-point RMS** separate, includes individual residuals and
per-pose errors, and refits with each entire pose held out. Failed held-out fits
are reported as failures, not silently omitted. Residual images use green circles
for clicked points, red crosses for projections and connecting error segments.

Transforms are explicitly named:

```text
p_base   = T_base_from_camera @ p_camera
p_camera = T_camera_from_base @ p_base
```

Coordinates are homogeneous; translation is meters. Camera axes are X right,
Y down, Z forward. These refer to the LEFT optical frame, not the camera housing,
right imager, mounting bracket or nominal URDF camera frame.

Review image residuals, rejected points, held-out errors, pose diversity, camera
serial and stationary timing evidence before using a candidate. Low training
pixel error alone does not establish metric accuracy or correct physical feature
identity. This first implementation intentionally has no automatic promotion or
URDF-update command.

## Reference and validation

Reference inspected read-only:
`~/projects/agentic-mobile-manipulator/third_party/hexfellow-deploy`, commit
`13cf8038f3e4a3d5935e1fda83bbedb7dd97c26e`, particularly
`calibration/docs/camera_calibration.md` and
`calibration/tools/camera/fit_landmark_extrinsics.py`.
The requested `~/projects/hexfellow-deploy` path was not present. This NERO
implementation has no runtime dependency on the reference checkout.

Validation includes numerical raw-distortion/outlier tests, capture-in-motion and
clock-gap rejection, a known-transform synthetic dataset with noisy observations,
full nine-waypoint simulation rehearsal, and browser mouse tests for 3D picking,
original-resolution image coordinates, pose switching and fitting. The live
physical route and a real-camera extrinsic fit remain untested.

The saved verification report is [camera_calibration_validation.json](../previews/camera_calibration_validation.json).
The six-pose synthetic case recovered the known transform within 0.185 mm and
0.0102 degrees, with 0.498 px all-point RMS and 0.398–0.550 px held-out-pose RMS.
These figures measure synthetic recovery only, not physical accuracy. All nine
simulated route waypoints settled; the largest final TCP position error was
0.046 mm. Twelve focused tests passed. A browser screenshot is available at
[camera_calibration_ui.png](../previews/camera_calibration_ui.png).
