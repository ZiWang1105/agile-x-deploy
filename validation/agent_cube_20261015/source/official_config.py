"""Pinned AgileX NERO configuration, also used for the shared TCP definition."""
from functools import lru_cache
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/questarmteleop"
TUNING_PATH = ROOT / "configs/ik.yaml"


@lru_cache(maxsize=1)
def official_parameters():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Official IK source checksum mismatch: {name}")
    return yaml.safe_load((SOURCE / "arm_ik_pose_node.nero.yaml").read_text())["/**"]["ros__parameters"]


def solver_parameters():
    """Merge explicit local tuning without mutating the official snapshot."""
    parameters = deepcopy(official_parameters())
    document = yaml.safe_load(TUNING_PATH.read_text())
    if not isinstance(document, dict) or set(document) != {"overrides"}:
        raise ValueError("IK tuning requires an overrides mapping")
    overrides = document["overrides"]
    allowed = {"w_pos", "w_ori", "w_reg", "w_smooth", "ipopt_tol", "ipopt_max_iter"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("Unsupported IK tuning parameter")
    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"IK parameter {name} must be finite and numeric")
        if value < 0 or (name not in {"w_reg", "w_smooth"} and value == 0):
            raise ValueError(f"Invalid IK parameter {name}: {value}")
        if name == "ipopt_max_iter" and not isinstance(value, int):
            raise ValueError("ipopt_max_iter must be an integer")
    parameters.update(overrides)
    return parameters
