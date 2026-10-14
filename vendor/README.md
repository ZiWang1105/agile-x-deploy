# Vendored model sources

This repository retains the source meshes, assembly macros, and attribution files needed to rebuild the dual-NERO stand. Files originate from the two user-supplied archives listed in `../source_manifest.json`. Their contents are preserved; unrelated robot models, generated exports, and archive-local Git metadata are not tracked.

`manifest.json` records the SHA-256 hash and size of each retained upstream file. `scripts/build_assembly.py` resolves its package names against these local directories and builds all generated assets without ROS or the original ZIP files.

The relevant official NERO description and MoveIt kinematics configuration are retained for reference. This is a selected model-source snapshot, not a complete upstream ROS workspace.

The upstream NERO asset license is retained at `agx_arm_sim/agx_arm_description/agx_arm_urdf/LICENSE`. RealSense files retain their embedded copyright notices and package license metadata. The supplied stand archive contains no standalone license; no new license or redistribution grant is assigned to those assets here.
