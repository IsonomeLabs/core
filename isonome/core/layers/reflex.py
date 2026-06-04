"""Reflex Layer — interpolation, safety enforcement, and execution.

Orchestrates: interpolate → enforce → execute at control frequency.
"""
from __future__ import annotations

from typing import Iterator, List

import torch

from isonome.core.layers.base import LayerBase
from isonome.core.state import (
    CorrectedMotorCommand,
    JointLimits,
    MotorCommand,
    MotorCommandChunk,
    SafeMotorCommand,
)
from isonome.utils.logging import get_layer_logger


class ActionInterpolator:
    """Interpolate a low-frequency policy chunk to high-frequency control commands."""

    def __init__(self, control_freq: float = 100.0, policy_freq: float = 1.0) -> None:
        self.ratio = int(control_freq / policy_freq)
        if self.ratio < 1:
            self.ratio = 1

    def interpolate(self, chunk: MotorCommandChunk) -> Iterator[MotorCommand]:
        """Yield intermediate commands via linear interpolation."""
        commands = chunk.commands  # [chunk_size, robot_dof]
        if commands.shape[0] == 1:
            for _ in range(self.ratio):
                yield MotorCommand(command=commands[0])
            return

        for i in range(commands.shape[0] - 1):
            start = commands[i]
            end = commands[i + 1]
            for step in range(self.ratio):
                alpha = step / self.ratio
                interp = start + alpha * (end - start)
                yield MotorCommand(command=interp)


class SafetyEnforcer:
    """Clamp commands to joint limits and check emergency stop."""

    def enforce(
        self, cmd: MotorCommand, joint_limits: JointLimits | None
    ) -> SafeMotorCommand:
        """Clamp to joint limits and return a SafeMotorCommand."""
        command = cmd.command.clone()
        was_clamped = False

        if joint_limits is not None:
            lower = joint_limits.lower
            upper = joint_limits.upper
            if command.shape == lower.shape:
                clamped = torch.clamp(command, lower, upper)
                was_clamped = not torch.equal(command, clamped)
                command = clamped

        return SafeMotorCommand(
            command=command,
            was_clamped=was_clamped,
            emergency_stop=False,
        )


class EmergencyStop(Exception):
    """Bypasses all layers and stops the robot immediately."""

    pass


class ReflexLayer(LayerBase):
    """Hard real-time reactive control. Runs at ~100Hz.

    Orchestrates interpolation → safety enforcement → execution.
    """

    def __init__(
        self,
        frequency_hz: float = 100.0,
        control_freq: float = 100.0,
        policy_freq: float = 1.0,
    ) -> None:
        super().__init__(name="reflex", frequency_hz=frequency_hz)
        self._interpolator = ActionInterpolator(control_freq, policy_freq)
        self._enforcer = SafetyEnforcer()
        self._joint_limits: JointLimits | None = None
        self._emergency_stop = False
        self._logger = get_layer_logger("reflex")

    def set_joint_limits(self, limits: JointLimits) -> None:
        self._joint_limits = limits

    def process(
        self, corrected_chunk: CorrectedMotorCommand
    ) -> List[SafeMotorCommand]:
        """Interpolate, enforce safety, and return safe commands."""
        if self._emergency_stop:
            raise EmergencyStop("Emergency stop is active")

        chunk = MotorCommandChunk(commands=corrected_chunk.commands)
        safe_commands: List[SafeMotorCommand] = []
        for cmd in self._interpolator.interpolate(chunk):
            safe = self._enforcer.enforce(cmd, self._joint_limits)
            if safe.emergency_stop:
                raise EmergencyStop("Safety enforcer triggered emergency stop")
            safe_commands.append(safe)
        return safe_commands

    def kill(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True
        self._logger.critical("reflex_emergency_stop_activated")

    def reset_emergency_stop(self) -> None:
        """Clear emergency stop (requires manual reset)."""
        self._emergency_stop = False
        self._logger.info("reflex_emergency_stop_reset")

    async def on_boot(self) -> None:
        self._logger.info("reflex_layer_booting")

    async def on_tick(self) -> None:
        pass  # tick logic driven externally by agent.py

    async def on_shutdown(self) -> None:
        self._logger.info("reflex_layer_shutdown")
