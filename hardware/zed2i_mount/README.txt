AgileX dual-arm stand → ZED 2i camera adapter — fit prototype v1
2026-10-14 | Units: mm

DESIGN
One-piece, gusseted L bracket. It attaches to the outward-facing surface of the
existing tilting metal bracket and supports the bottom of the camera. The ZED
lenses face away from the stand. The original pivot and adjustment screws remain.
The supplied print is an adapter only; do not print the assembly preview.

Confirmed by the user:
  Stand mounting face: 68 wide × 20 high.
  Two M3 mounting holes, 45 centre-to-centre.
Confirmed from official Stereolabs drawing / STEP:
  Two bottom M3 × 0.5 camera holes, 36 centre-to-centre.
  Maximum insertion into these camera holes: 10 mm.
Assumptions still requiring a physical check:
  Stand holes are at mid-height of the 20 mm face.
  Stand face is approximately vertical at the desired neutral camera angle.
  Hole type (threaded or clearance) and metal thickness are not confirmed.
  The preview uses a simplified 2 mm thick metal plate, not the entire stand.

Adapter envelope in assembly orientation: 64 wide × 46 forward × 26 high.
Backplate and shelf thickness: 6. Side ribs: 4 thick.
Stand holes: two 5.4 × 3.4 horizontal slots centred 45 apart. These permit
approximately 43–47 mm effective hole spacing; use washers to clamp the slots.
Camera holes: two 3.4 diameter clearance holes, 36 apart, 25 forward of the
stand contact face. Camera contact surface is 16 above the stand-hole row.
A 64 mm width leaves 2 mm on each side of the nominal 68 mm plate; this does
not establish clearance to every part of the actual pivot assembly.

FILES
  output/zed2i_adapter_PRINT.stl       Import this into your slicer, units mm.
  output/stand_fit_coupon_PRINT.stl   Optional fit-check plate; not required.
  output/zed2i_adapter_PRINT.step      Editable solid in print orientation.
  output/zed2i_adapter_assembly_orientation.step  Editable installed solid.
  output/assembly_preview.step        Adapter + official camera + proxy plate.
  output/preview.png                  Actual CAD assembly render.
  output/dimensions.png               Dimensioned reference views.
  output/validation.json              Geometry and clearance results.
  build_mount.py                     CadQuery construction and validation source.

PRINT
Suggested starting point, not a validated manufacturing specification:
  PETG; 0.4 mm nozzle; 0.2 mm layers; 5 perimeters; 6 top/bottom layers;
  40–50% infill. A brim can help the tall side-on print adhere.
  Print at 100% scale. Keep the supplied PRINT STL orientation: one side is
  on the bed, with a 26 × 46 mm footprint and 64 mm height. This places the
  L cross-section within each layer. The small horizontal holes bridge;
  inspect the slicer and printed holes before assembly. No large shelf support
  is needed in this orientation. Clear any hole sag gently; do not force bolts.
  Do not put the assembly-orientation model upright on the bed by mistake.
The coupon prints flat and is only for hole/face fit; it is not a load-bearing
camera mount. If it does not sit flush and line up, revise before the full print.

FASTENERS
Camera side:
  2 × M3 × 12 socket-head screws, 2 × M3 flat washers (~0.5 mm thick).
  With a 6 mm shelf and 0.5 mm washer, nominal camera insertion is 5.5 mm.
  Measure the actual stack. Insertion = screw under-head length − shelf − washer.
  Stay below the official 10 mm maximum. If a screw bottoms out before clamping,
  stop and shorten it. Do not use the centre 1/4-inch hole for this design.
Stand side:
  2 × M3 screws with flat washers. Use the existing metal thread if the plate
  is tapped; use nuts and rear washers if it has clearance holes.
  Select length after checking plate thickness and rear clearance.
  For a 2 mm clearance plate, 6 mm plastic, two 0.5 mm washers and a normal
  M3 nut, M3 × 14 is a reasonable trial length; verify protrusion and tilt clearance.
  A tapped plate needs a different length; the above nut-stack example does
  not apply. Do not force an M3 screw into an unidentified thread.
No printed threads, heat-set inserts or embedded nuts are required in the part.

ASSEMBLY
1. Print the coupon. Place it against the outward face of the existing plate.
   Verify both holes align, screws pass, and the coupon sits flat.
2. Print and inspect the complete adapter. Attach its vertical backplate to the
   metal plate using the 45 mm pair. The shelf points outward from the stand.
   The slot washers and screw heads sit on the front of the printed backplate.
3. Place the ZED 2i on the shelf with lenses facing outward. Insert the two
   camera screws with washers from below, through the 36 mm pair. Support the
   camera while fastening. Tighten by hand until snug; do not crush the print.
4. Connect the rear cable with slack. Support the camera while adjusting the
   existing tilt bracket, then tighten its original adjustment screws.
5. Check rear connector clearance, the desired tilt range, screw protrusion and
   arm clearance on the actual stand. Start with the arms stationary. Check the
   live camera image for obstruction and recheck fasteners after initial use.

VALIDATION AND LIMITS
CAD solid validity, one-solid topology, closed STL and consistent mesh winding
were checked. The imported official camera body has no solid overlap with the
adapter. Four 7 mm diameter tool/washer envelopes clear the printed part.
These checks cover the adapter, camera and simplified plate at one orientation.
They do not establish full stand/cable/pivot clearance, dynamic strength, thermal
creep, print quality or lifetime. This is a fit prototype awaiting physical testing.
The full robot URDF and camera extrinsics have not been modified by this task.
After physical mounting, measure/calibrate the actual camera transform.

REBUILD
Use an isolated Python environment (Python 3.10 or newer):
  pip install cadquery==2.7.0 trimesh matplotlib
  python build_mount.py
The official STEP must be at references/ZED2i.step. The archive includes it as
reference geometry; it is not an original part designed by this project.

OFFICIAL SOURCES
Specifications / thread insertion limits:
https://docs.stereolabs.com/docs/products/cameras/zed/specifications
Datasheet V1.5, drawings on pages 9–10:
https://cdn.sanity.io/files/s18ewfw4/staging/17e28531d47c0c5feab2be4ca304abf06ffa889a.pdf
Official model download page:
https://www.stereolabs.com/3dmodels
Official ZED2i STEP:
https://cdn.sanity.io/files/s18ewfw4/staging/1eae249387d9bde53972abbe470483024c31c6fb.step/ZED2i.step
