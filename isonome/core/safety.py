"""Isonome v0.2 Core Safety.

SafetyGovernor gates execution and kernel loading based on agent mode.
EmergencyStop bypasses all layers.
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from isonome.utils.logging import get_layer_logger

if TYPE_CHECKING:
    from isonome.core.agent import Agent


class AgentMode(str, Enum):
    BOOT = "boot"
    CALIBRATING = "calibrating"
    RUNTIME = "runtime"
    IDLE = "idle"
    SAFE_STOP = "safe_stop"


class EmergencyStop(Exception):
    """Exception that bypasses all layers and stops the robot immediately."""

    pass


class SafetyGovernor:
    """Gates execution and kernel loading based on agent lifecycle mode."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._logger = get_layer_logger("safety.governor")

    def can_execute(self) -> bool:
        """Return True if the agent is allowed to execute motor commands."""
        return self.agent.mode in {AgentMode.RUNTIME, AgentMode.CALIBRATING}

    def can_load_kernel(self) -> bool:
        """Return True if the agent is allowed to load a kernel."""
        return self.agent.mode in {AgentMode.BOOT, AgentMode.IDLE}

    def emergency_stop(self) -> None:
        """Trigger emergency stop. Sets agent mode to SAFE_STOP and kills reflex."""
        self._logger.critical("safety_emergency_stop_triggered")
        self.agent.mode = AgentMode.SAFE_STOP
        self.agent.reflex.kill()
