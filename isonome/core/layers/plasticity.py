"""Plasticity Layer — Runtime-Only Kernel Manager.

The open-source runtime only consumes pre-trained kernels.
It does NOT implement training loops, loss functions, or cloud API calls.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from isonome.core.layers.base import LayerBase
from isonome.core.layers.soma import SomaKernel
from isonome.utils.logging import get_layer_logger


@dataclasses.dataclass
class KernelMetadata:
    """Metadata for a saved kernel."""

    version: str = "0.2.0"
    training_episodes: int = 0
    robot_hash: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KernelMetadata:
        return cls(**d)


class PlasticityLayer(LayerBase):
    """Runtime-only kernel loader and persister.

    - load_kernel(): loads a .pt file into SomaLayer
    - save_runtime_state(): persists current kernel to disk
    - has_kernel_for(): checks if a kernel exists for the current robot
    """

    def __init__(self, kernel_dir: str = "~/.isonome/kernels") -> None:
        super().__init__(name="plasticity", frequency_hz=0.0)
        self._kernel_dir = Path(kernel_dir).expanduser()
        self._kernel_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_layer_logger("plasticity")

    async def on_boot(self) -> None:
        self._logger.info(
            "plasticity_layer_booting",
            extra={"kernel_dir": str(self._kernel_dir)},
        )

    async def on_tick(self) -> None:
        pass

    async def on_shutdown(self) -> None:
        self._logger.info("plasticity_layer_shutdown")

    def load_kernel(self, path: Path) -> SomaKernel:
        """Load a .pt file into a SomaKernel instance."""
        self._logger.info("plasticity_loading_kernel", extra={"path": str(path)})
        if not path.exists():
            raise FileNotFoundError(f"Kernel not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        canonical_dim = data.get("canonical_dim", 14)
        robot_state_dim = data.get("robot_state_dim", 14)
        kernel = SomaKernel(canonical_dim, robot_state_dim)
        kernel.load_state_dict(data["net"])
        kernel.eval()
        for param in kernel.parameters():
            param.requires_grad = False
        self._logger.info("plasticity_kernel_loaded")
        return kernel

    def save_runtime_state(
        self,
        kernel: SomaKernel,
        metadata: KernelMetadata,
        path: Path | None = None,
    ) -> Path:
        """Save current kernel weights and metadata to disk."""
        if path is None:
            path = self._kernel_dir / f"{metadata.robot_hash}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "net": kernel.state_dict(),
            "canonical_dim": kernel.net[0].in_features - metadata.training_episodes,  # heuristic
            "robot_state_dim": metadata.training_episodes,
            "metadata": metadata.to_dict(),
        }
        # Re-compute dims correctly from layer shapes
        canonical_dim = kernel.net[0].in_features - kernel.net[-1].out_features
        robot_state_dim = kernel.net[-1].out_features
        payload["canonical_dim"] = canonical_dim
        payload["robot_state_dim"] = robot_state_dim
        torch.save(payload, path)
        self._logger.info("plasticity_kernel_saved", extra={"path": str(path)})
        return path

    def has_kernel_for(self, robot_hash: str) -> bool:
        """Check if a kernel exists on disk for the given robot hash."""
        return (self._kernel_dir / f"{robot_hash}.pt").exists()

    def kernel_path(self, robot_hash: str) -> Path:
        return self._kernel_dir / f"{robot_hash}.pt"
