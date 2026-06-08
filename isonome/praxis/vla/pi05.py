"""π0.5 wrapper — placeholder for the closed-weights 7B VLA model.

π0.5 (Physical Intelligence) is the highest-performing option listed in
Iteration 029 but requires approved model access.  This module is a
stub that documents the expected interface so the swap is trivial once
weights are available.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from isonome.praxis.vla.base import VLABase


class PiZeroFive(VLABase):
    """π0.5 policy wrapper (placeholder).

    Paper: *π0: A Vision-Language-Action Flow Model for General Robot
    Control* (Physical Intelligence, 2024).
    """

    def __init__(self, action_dim: int = 7) -> None:
        self._action_dim = action_dim
        self._model: Any = None

    # ------------------------------------------------------------------
    # VLABase interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        """Placeholder — π0.5 weights are not publicly downloadable."""
        raise RuntimeError(
            "π0.5 weights are closed.  Access requires approval from "
            "Physical Intelligence.  For fast iteration use MockVLABackend, "
            "LLaVARobot, or OpenVLA instead."
        )

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return np.zeros(self._action_dim, dtype=np.float32)
