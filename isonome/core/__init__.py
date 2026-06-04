"""Core module — re-exports the primary public API.

Torch-dependent models (state.py) are imported lazily to allow
non-torch environments to use bus, config, and safety modules.
"""
from __future__ import annotations

# Eager imports (no torch dependency)
from isonome.core.config import AppConfig
from isonome.core.safety import AgentMode

# Torch-dependent state model names — imported lazily via __getattr__
_STATE_NAMES = frozenset({
    "RawSensorState", "CanonicalActionChunk", "CorrectedMotorCommand",
    "CortexAdvice", "ExecutionResult", "Discrepancy", "JointLimits",
    "MotorCommandChunk", "MotorCommand", "SafeMotorCommand",
    "SensorState", "LegacyMotorCommand", "Adjustment", "WorldModel",
    "Patch", "PatchType", "ErrorEvent", "JointReading",
    "ContactReading", "IMUReading", "PredictedState",
})


def __getattr__(name):
    """Lazy import: torch-dependent state models and agent/app."""
    if name in _STATE_NAMES:
        from isonome.core import state as _state_mod
        return getattr(_state_mod, name)
    if name == "Agent":
        from isonome.core.agent import Agent
        return Agent
    if name == "IsonomeApp":
        from isonome.core.app import IsonomeApp
        return IsonomeApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RawSensorState", "CanonicalActionChunk", "CorrectedMotorCommand",
    "CortexAdvice", "ExecutionResult", "Discrepancy", "JointLimits",
    "MotorCommandChunk", "MotorCommand", "SafeMotorCommand",
    "SensorState", "LegacyMotorCommand", "Adjustment", "WorldModel",
    "Patch", "PatchType", "ErrorEvent", "JointReading",
    "ContactReading", "IMUReading", "PredictedState",
    "AppConfig", "AgentMode", "Agent", "IsonomeApp",
]
