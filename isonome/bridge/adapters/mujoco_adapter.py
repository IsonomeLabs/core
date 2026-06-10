"""BodyBridge adapter for MuJoCoBridge.

This adapter wraps ``isonome.sim.mujoco_bridge.MuJoCoBridge`` and drives it
locally from ``SomaLayer``.  The WebSocket / MJPEG servers are **not**
started by default; they are a dashboard concern.  We only need the
physics engine.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import torch

from isonome.core.ports.body_bridge import BodyBridge, BridgePortError
from isonome.core.state import CorrectedMotorCommand, ExecutionResult, RawSensorState
from isonome.sim.mujoco_bridge import MuJoCoBridge

try:
    import mujoco

    HAS_MUJOCO = True
except Exception:  # pragma: no cover
    HAS_MUJOCO = False
    mujoco = None  # type: ignore[assignment]


class MuJoCoBridgeAdapter(BodyBridge):
    """BodyBridge implementation backed by MuJoCo physics.

    Parameters
    ----------
    urdf_path:
        Path to a MuJoCo-compatible XML/URDF file.
    enable_servers:
        If ``True``, also start the WebSocket and MJPEG servers so the
        dashboard can watch.  Default ``False`` for headless use inside
        ``Agent.tick()``.
    websocket_port, mjpeg_port:
        Server ports (only used when ``enable_servers=True``).
    """

    def __init__(
        self,
        urdf_path: Path,
        *,
        websocket_port: int = 8765,
        mjpeg_port: int = 8766,
    ) -> None:
        if not HAS_MUJOCO:
            raise BridgePortError(
                "MuJoCo is required for MuJoCoBridgeAdapter. "
                "Install with: pip install mujoco"
            )
        super().__init__(urdf_path)
        self._bridge = MuJoCoBridge(
            websocket_port=websocket_port,
            mjpeg_port=mjpeg_port,
        )
        self._sim_task: asyncio.Task | None = None

    @property
    def name(self) -> str:
        return "mujoco"

    @property
    def joint_count(self) -> int:
        return len(getattr(self._bridge, "_joint_names", []))

    async def _do_boot(self) -> None:
        result = self._bridge._cmd_load_urdf(str(self._urdf_path))
        if not result.get("ok"):
            raise BridgePortError(
                f"Failed to load MuJoCo model {self._urdf_path}: "
                f"{result.get('error')}"
            )

        # Start physics loop in background (uses bridge's internal _sim_loop)
        self._sim_task = asyncio.create_task(
            self._bridge._sim_loop(), name="mujoco_physics_loop"
        )

    async def _do_shutdown(self) -> None:
        self._bridge.shutdown()
        if self._sim_task is not None:
            self._sim_task.cancel()
            try:
                await self._sim_task
            except asyncio.CancelledError:
                pass

    async def _do_perceive(self) -> RawSensorState:
        obs = self._bridge.get_observation(intent="")
        proprio = np.asarray(obs.get("proprioception", []), dtype=np.float32)
        images = obs.get("image")
        camera_frames: list[torch.Tensor] = []
        if images is not None:
            if isinstance(images, list):
                camera_frames = [torch.from_numpy(np.asarray(img)) for img in images]
            else:
                camera_frames = [torch.from_numpy(np.asarray(images))]
        return RawSensorState(
            proprioception=torch.from_numpy(proprio),
            camera_frames=camera_frames,
        )

    async def _do_act(self, cmd: CorrectedMotorCommand) -> None:
        commands = cmd.commands
        if commands.ndim == 2:
            commands = commands[0]
        action = commands.detach().cpu().numpy().tolist()
        self._bridge._cmd_apply_action(action)

    async def _do_observe_result(self) -> ExecutionResult:
        raw = await self._do_perceive()
        # Simple success heuristic: proprioception contains finite values
        success = torch.all(torch.isfinite(raw.proprioception)).item()
        return ExecutionResult(
            final_proprioception=raw.proprioception,
            success=bool(success),
            error_metric=0.0 if success else 1.0,
        )
