"""BodyBridge adapter for MockSimBridge.

Used in frontend development and CI when no real physics engine is
available.  The mock robot runs a simple damped-pendulum simulation.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import torch

from isonome.sim.mock_bridge import MockRobot  # re-use the physics model
from isonome.core.ports.body_bridge import BodyBridge
from isonome.core.state import CorrectedMotorCommand, ExecutionResult, RawSensorState


class MockBridgeAdapter(BodyBridge):
    """Wraps MockRobot as a BodyBridge for SomaLayer."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        physics_hz: float = 60.0,
    ) -> None:
        super().__init__(urdf_path)
        self._physics_hz = physics_hz
        self._dt = 1.0 / physics_hz
        self._robot: Any = None
        self._joint_names: list[str] = []
        self._sim_task: asyncio.Task | None = None
        self._running = False

    @property
    def name(self) -> str:
        return "mock"

    @property
    def joint_count(self) -> int:
        return len(self._joint_names)

    async def _do_boot(self) -> None:
        # Parse URDF for joint names (same logic as MockSimBridge)
        from xml.etree import ElementTree as ET

        tree = ET.parse(self._urdf_path)
        root = tree.getroot()
        joint_names = []
        for joint in root.findall("joint"):
            jtype = joint.get("type", "fixed")
            if jtype != "fixed":
                name = joint.get("name", f"joint_{len(joint_names)}")
                joint_names.append(name)

        self._joint_names = joint_names
        self._robot = MockRobot(joint_names)
        self._running = True
        self._sim_task = asyncio.create_task(self._sim_loop())

    async def _do_shutdown(self) -> None:
        self._running = False
        if self._sim_task is not None:
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass
            self._sim_task = None

    async def _sim_loop(self) -> None:
        while self._running:
            self._robot.step(dt=self._dt)
            await asyncio.sleep(self._dt)

    async def _do_perceive(self) -> RawSensorState:
        state = self._robot.get_state()
        positions = torch.tensor(
            [j["position"] for j in state["joints"]], dtype=torch.float32
        )
        velocities = torch.tensor(
            [j["velocity"] for j in state["joints"]], dtype=torch.float32
        )
        proprio = torch.cat([positions, velocities])
        return RawSensorState(
            proprioception=proprio,
            camera_frames=[],
        )

    async def _do_act(self, cmd: CorrectedMotorCommand) -> None:
        # Apply first row of command tensor as target positions
        commands = cmd.commands
        if commands.ndim == 2:
            commands = commands[0]
        values = commands.tolist()
        for i, name in enumerate(self._joint_names):
            if i < len(values):
                # Simple position override (mock is position-controlled)
                self._robot.positions[i] = float(values[i])

    async def _do_observe_result(self) -> ExecutionResult:
        # After act, observe the current state and measure how close we got
        raw = await self._do_perceive()
        # Simple success heuristic: all joints within reasonable bounds
        positions = raw.proprioception[: self.joint_count]
        in_bounds = torch.all((positions >= -3.14) & (positions <= 3.14))
        return ExecutionResult(
            final_proprioception=raw.proprioception,
            success=bool(in_bounds.item()),
            error_metric=0.0 if in_bounds else 1.0,
        )
