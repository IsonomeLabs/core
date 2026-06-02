"""Safety module — re-exports with backward compatibility.

v0.2 note: RobotState has been replaced by AgentMode.
"""
from __future__ import annotations

import warnings

from isonome.core.safety import AgentMode, EmergencyStop, SafetyGovernor
from isonome.safety.reflex_watchdog import ReflexWatchdog
from isonome.safety.sandbox import LLMSandbox

# Backward compatibility: RobotState maps to AgentMode
RobotState = AgentMode

warnings.warn(
    "isonome.safety is deprecated in v0.2. Use isonome.core.safety instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SafetyGovernor",
    "AgentMode",
    "RobotState",
    "EmergencyStop",
    "ReflexWatchdog",
    "LLMSandbox",
]
