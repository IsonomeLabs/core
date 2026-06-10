"""BodyBridge adapter for the PyBullet SimBridge.

Runs PyBullet's synchronous API in a background thread so that the
async ``BodyBridge`` contract is honored.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from isonome.bridge.sim import SimBridge
from isonome.core.config import SimConfig
from isonome.core.ports.body_bridge import BodyBridge, BridgePortError
from isonome.core.state import (
    CorrectedMotorCommand,
    ExecutionResult,
    RawSensorState,
)


class PyBulletBridgeAdapter(BodyBridge):
    """Wraps ``isonome.bridge.sim.SimBridge`` as a BodyBridge."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        sim_config: SimConfig | None = None,
        max_workers: int = 1,
    ) -> None:
        super().__init__(urdf_path)
        self._sim_config = sim_config or SimConfig()
        self._bridge = SimBridge(self._sim_config)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pybullet_bridge"
        )
        self._joint_count = 0

    @property
    def name(self) -> str:
        return "pybullet"

    @property
    def joint_count(self) -> int:
        return self._joint_count

    async def _do_boot(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._bridge.connect)
        robot_id = await loop.run_in_executor(
            self._executor,
            self._bridge.load_urdf,
            str(self._urdf_path),
        )
        self._joint_count = self._bridge._num_joints
        self._logger.info(
            "pybullet_urdf_loaded",
            extra={"robot_id": robot_id, "joints": self._joint_count},
        )

    async def _do_shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._bridge.disconnect)
        self._executor.shutdown(wait=True)

    async def _do_perceive(self) -> RawSensorState:
        loop = asyncio.get_running_loop()
        sensor_state = await loop.run_in_executor(
            self._executor, self._bridge.get_sensor_state
        )
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
        loop = asyncio.get_running_loop()
        commands = cmd.commands
        if commands.ndim == 2:
            commands = commands[0]

        # Build v0.1 MotorCommand (dict joint_name -> position)
        from isonome.core.state import LegacyMotorCommand as V01MotorCommand

        joint_positions = {}
        joint_names = self._bridge._joint_names
        values = commands.tolist()
        for i, name in enumerate(joint_names):
            if i < len(values):
                joint_positions[name] = float(values[i])

        motor_cmd = V01MotorCommand(joint_positions=joint_positions)
        await loop.run_in_executor(
            self._executor, self._bridge.apply_motor_command, motor_cmd
        )
        # Step physics once per act so the next perceive sees the result
        await loop.run_in_executor(self._executor, self._bridge.step)

    async def _do_observe_result(self) -> ExecutionResult:
        loop = asyncio.get_running_loop()
        # Step physics once to propagate the last command
        await loop.run_in_executor(self._executor, self._bridge.step)
        raw = await self._do_perceive()
        return ExecutionResult(
            final_proprioception=raw.proprioception,
            success=True,
            error_metric=0.0,
        )
