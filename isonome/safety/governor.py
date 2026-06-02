"""Deprecated — re-exports from isonome.core.safety for backward compatibility.

These symbols have moved to isonome.core.safety in v0.2.
"""
from __future__ import annotations

import warnings

from isonome.core.safety import AgentMode, EmergencyStop, SafetyGovernor

warnings.warn(
    "isonome.safety.governor is deprecated in v0.2. "
    "Use isonome.core.safety instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AgentMode", "EmergencyStop", "SafetyGovernor"]
