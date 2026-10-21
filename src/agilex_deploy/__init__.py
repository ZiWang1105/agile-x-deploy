"""Shared robot API for operator interfaces and agent policies.

Load public classes on demand so entry points can configure rendering and
numerical libraries before MuJoCo or NumPy is imported.
"""
from importlib import import_module

__all__ = ["Action", "ArmTarget", "Pose", "Observation", "RobotInterface", "Robot"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".runtime" if name == "Robot" else ".contract", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
