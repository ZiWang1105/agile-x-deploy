# AgileX gravity calculation source

Unmodified `core/agx_pinocchio.py` from AgileX's
[gravity-compensation repository](https://github.com/agilexrobotics/agilex-arm-gravity-compensation),
commit `8e2545c00ec9ec7590414946ab5f913274c7b643`.
The manifest pins its checksum. No top-level license file was present in that
snapshot; no additional license grant is asserted here.

Only the numerical helper is imported. The upstream `nero/main_gc.py` hardware
entry point is not installed or executed. That example auto-enables a hardcoded
CAN channel, assumes an upright bare arm, and returns to position control on
interrupt. Those behaviors are not adopted by this offline adapter.
