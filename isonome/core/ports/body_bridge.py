"""BodyBridge — the outbound port between SomaLayer and the physical/simulated body.

This is the critical integration point that ``architecture.md`` labels
``SIM_B`` / ``HW_B``.  ``SomaLayer`` owns the URDF-derived morphology
(NaiveMapper, robot hash, kernel) but delegates all runtime sensor/actuator
I/O to a ``BodyBridge`` implementation.

Design principles
-----------------
1. **Async by default** — the bridge runs in the Agent's asyncio loop.
   Sync backends (PyBullet, MuJoCo) should use a thread-pool executor.
2. **v0.2 state models** — speaks ``RawSensorState``, ``CorrectedMotorCommand``,
   and ``ExecutionResult`` natively. Adapters translate to legacy formats.
3. **Lifecycle explicit** — ``boot()`` loads the URDF into the backend and
   connects; ``shutdown()`` releases resources.  Agent owns the call order.
4. **Joint-count validation** — the bridge reports how many DOF it actually
   sees; SomaLayer validates against the URDF-derived count.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch

from isonome.core.state import (
    CorrectedMotorCommand,
    ExecutionResult,
    JointLimits,
    RawSensorState,
)
from isonome.utils.logging import get_layer_logger


class BridgePortError(RuntimeError):
    """Raised when a BodyBridge cannot satisfy a port contract."""

    pass


class BodyBridge(ABC):
    """Outbound port: runtime body interface used by SomaLayer.

    A BodyBridge is constructed with a URDF path and optional joint limits.
    It is responsible for loading that URDF into the target runtime
    (PyBullet, MuJoCo, Isaac Sim, hardware abstraction, etc.), reading
    RAW sensor state, writing corrected motor commands, and observing the
    result of the last command.

    Implementations must be safe to boot/shutdown repeatedly and must
    release all resources in ``shutdown()``.
    """

    def __init__(
        self,
        urdf_path: Path,
        joint_limits: JointLimits | None = None,
    ) -> None:
        self._urdf_path = Path(urdf_path)
        self._joint_limits = joint_limits
        self._logger = get_layer_logger(f"bridge.{self.name}")
        self._connected = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs (e.g. ``pybullet``, ``mujoco``)."""

    @property
    @abstractmethod
    def joint_count(self) -> int:
        """Number of actuated DOF reported by the backend.

        This is validated against ``SomaLayer.naive_mapper.joint_count``
        during boot.
        """

    @property
    def urdf_path(self) -> Path:
        return self._urdf_path

    @property
    def joint_limits(self) -> JointLimits | None:
        return self._joint_limits

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Idempotent boot: connect, load URDF, report joint count."""
        if self._connected:
            return
        self._logger.info(
            "body_bridge_booting",
            extra={"bridge": self.name, "urdf": str(self._urdf_path)},
        )
        await self._do_boot()
        self._connected = True
        self._logger.info(
            "body_bridge_ready",
            extra={"bridge": self.name, "joints": self.joint_count},
        )

    async def shutdown(self) -> None:
        """Idempotent shutdown: release backend resources."""
        if not self._connected:
            return
        self._logger.info("body_bridge_shutting_down", extra={"bridge": self.name})
        await self._do_shutdown()
        self._connected = False

    @abstractmethod
    async def _do_boot(self) -> None:
        """Backend-specific boot logic."""

    @abstractmethod
    async def _do_shutdown(self) -> None:
        """Backend-specific shutdown logic."""

    # ------------------------------------------------------------------
    # Runtime I/O
    # ------------------------------------------------------------------

    async def perceive(self) -> RawSensorState:
        """Read RAW sensor state from the body.

        Returns uncorrected proprioception and camera frames. This is RAW
        data — no kernel correction, no post-processing.
        """
        self._ensure_connected()
        return await self._do_perceive()

    async def act(self, cmd: CorrectedMotorCommand) -> None:
        """Send corrected motor commands to the body."""
        self._ensure_connected()
        await self._do_act(cmd)

    async def observe_result(self) -> ExecutionResult:
        """Observe the result of the last ``act()`` for discrepancy analysis.

        The default implementation re-perceives and packages an
        ``ExecutionResult`` with ``success=True``.  Backends that can
        provide richer feedback (contact forces, goal detectors, etc.)
        should override this.
        """
        self._ensure_connected()
        return await self._do_observe_result()

    @abstractmethod
    async def _do_perceive(self) -> RawSensorState:
        """Backend-specific perceive implementation."""

    @abstractmethod
    async def _do_act(self, cmd: CorrectedMotorCommand) -> None:
        """Backend-specific act implementation."""

    async def _do_observe_result(self) -> ExecutionResult:
        """Backend-specific observe implementation.

        Default: just re-perceive and package the result.
        """
        raw = await self._do_perceive()
        return ExecutionResult(
            final_proprioception=raw.proprioception,
            success=True,
            error_metric=0.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise BridgePortError(
                f"{self.name} bridge is not connected. Call boot() first."
            )

    def _canonical_to_robot_dof(
        self, canonical: torch.Tensor, robot_dof: int
    ) -> torch.Tensor:
        """Resize a canonical action tensor to the robot's DOF.

        This is a shared fallback used by adapters when the backend does
        not already know how to map canonical 14-DOF intent to the
        robot's joint count.  It truncates or pads with zeros.
        """
        if canonical.numel() == 0:
            return torch.zeros(robot_dof)

        # Flatten to 1-D, then truncate / pad
        flat = canonical.flatten()
        if flat.shape[0] >= robot_dof:
            return flat[:robot_dof].clone()
        padded = torch.zeros(robot_dof, dtype=flat.dtype, device=flat.device)
        padded[: flat.shape[0]] = flat
        return padded

    def _camera_frames_to_tensors(
        self, frames: List[torch.Tensor] | torch.Tensor | None
    ) -> List[torch.Tensor]:
        """Normalize camera frames to a list of torch tensors."""
        if frames is None:
            return []
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 3:
                return [frames]
            return [frames[i] for i in range(frames.shape[0])]
        return [torch.as_tensor(f) for f in frames]
