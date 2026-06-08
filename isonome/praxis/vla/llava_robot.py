"""LLaVA-robot wrapper — lightweight open VLA for fast iteration.

This is a placeholder.  When a community checkpoint is available it can
be dropped in without changing the bridge interface.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from isonome.praxis.vla.base import VLABase

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]


class LLaVARobot(VLABase):
    """LLaVA-robot policy wrapper (placeholder for fast iteration).

    Recommended as the **first model** to validate the full VLA → MuJoCo
    pipeline because it is smaller and faster than OpenVLA / π0.5.
    """

    def __init__(self, action_dim: int = 7, device: str = "cpu") -> None:
        self._action_dim = action_dim
        self._device = device
        self._model: Any = None

    # ------------------------------------------------------------------
    # VLABase interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        """Placeholder — raises with install instructions."""
        raise RuntimeError(
            "LLaVA-robot checkpoint not yet integrated.\n"
            "1. Download a LLaVA-robot checkpoint.\n"
            "2. Implement the tokenizer / vision-tower load logic here.\n"
            "3. Or swap to MockVLABackend for immediate testing."
        )

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # Placeholder — continuous actions would be decoded here
        return np.zeros(self._action_dim, dtype=np.float32)
