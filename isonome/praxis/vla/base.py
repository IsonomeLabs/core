"""Abstract base class for Vision-Language-Action models."""
from __future__ import annotations

import abc
from typing import Any

import numpy as np


class VLABase(abc.ABC):
    """Common interface for VLA policy models.

    A concrete wrapper must implement ``load`` and ``predict``.  Training
    helpers such as ``train_step`` are optional — offline / fine-tuning
    scripts should guard with ``hasattr`` or ``isinstance`` checks.
    """

    @abc.abstractmethod
    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        """Load model weights from *checkpoint_path*."""
        ...

    @abc.abstractmethod
    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """Return an action vector (or action chunk) for *obs*.

        *obs* is the dictionary returned by
        :meth:`isonome.sim.mujoco_bridge.MuJoCoBridge.get_observation`.

        Returns
        -------
        np.ndarray
            Shape ``[n_joints]`` for a single-step action, or
            ``[T, n_joints]`` for an action chunk.
        """
        ...

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Optional training hook.  Raises ``NotImplementedError`` by default."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support training.")
