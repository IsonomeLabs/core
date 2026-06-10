"""BodyBridge adapter for the abstract HardwareBridge.

This allows ``SomaLayer`` to drive a hardware backend that implements the
legacy ``isonome.bridge.hardware.HardwareBridge`` interface.

Because the legacy interface uses v0.1 state models (``SensorState``,
``MotorCommand``), this adapter performs bidirectional translation.
"""
from __future__ import annotations

from pathlib import Path

import torch

from isonome.bridge.hardware import HardwareBridge
from isonome.core.ports.body_bridge import BodyBridge, BridgePortError
from isonome.core.state import (
    CorrectedMotorCommand,
    ExecutionResult,
    RawSensorState,
)


class HardwareBridgeAdapter(BodyBridge):
    """Adapts a v0.1 HardwareBridge to the v0.2 BodyBridge port."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        hardware_bridge: HardwareBridge | None = None,
        joint_count: int | None = None,
    ) -> None:
        super().__init__(urdf_path)
        self._hw = hardware_bridge
        self._explicit_joint_count = joint_count
        self._detected_joint_count = 0

    @property
    def name(self) -> str:
        return "hardware"

    @property
    def joint_count(self) -> int:
        if self._explicit_joint_count is not None:
            return self._explicit_joint_count
        return self._detected_joint_count

    async def _do_boot(self) -> None:
        if self._hw is None:
            from isonome.bridge.hardware import StubHardwareBridge

            self._hw = StubHardwareBridge()
        await self._hw.connect()
        # Try to detect joint count from a sensor read
        sensor_state = await self._hw.read_sensors()
        self._detected_joint_count = len(sensor_state.joints) or (
            self._explicit_joint_count or 0
        )

    async def _do_shutdown(self) -> None:
        if self._hw is not None:
            await self._hw.disconnect()

    async def _do_perceive(self) -> RawSensorState:
        if self._hw is None:
            raise BridgePortError("Hardware bridge not initialized")
        sensor_state = await self._hw.read_sensors()
        positions = torch.tensor(
            [j.position for j in sensor_state.joints], dtype=torch.float32
        )
        velocities = torch.tensor(
            [j.velocity for j in sensor_state.joints], dtype=torch.float32
        )
        proprio = torch.cat([positions, velocities])
        return RawSensorState(
            proprioception=proprio,
            camera_frames=[],
        )

    async def _do_act(self, cmd: CorrectedMotorCommand) -> None:
        if self._hw is None:
            raise BridgePortError("Hardware bridge not initialized")
        commands = cmd.commands
        if commands.ndim == 2:
            commands = commands[0]
        values = commands.tolist()

        # Translate to v0.1 MotorCommand (dict of joint_name -> position)
        from isonome.core.state import LegacyMotorCommand as V01MotorCommand

        joint_positions = {}
        for i, name in enumerate(self._joint_names_from_count()):
            if i < len(values):
                joint_positions[name] = float(values[i])

        motor_cmd = V01MotorCommand(joint_positions=joint_positions)
        await self._hw.write_motors(motor_cmd)

    async def _do_observe_result(self) -> ExecutionResult:
        raw = await self._do_perceive()
        return ExecutionResult(
            final_proprioception=raw.proprioception,
            success=True,
            error_metric=0.0,
        )

    def _joint_names_from_count(self) -> list[str]:
        n = self.joint_count
        return [f"joint_{i}" for i in range(n)]
