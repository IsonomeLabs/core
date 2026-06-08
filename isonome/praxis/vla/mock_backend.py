"""Mock VLA backend for testing without downloading multi-billion-parameter weights.

Outputs deterministic action chunks so examples and integration tests run
instantly and without GPU requirements.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from isonome.praxis.vla.base import VLABase


class MockVLABackend(VLABase):
    """Lightweight mock VLA that outputs deterministic action chunks.

    Simulates a policy that always tries to drive the first three joints
    toward ``target = [0.0, 0.5, -0.3]`` (or a user-supplied override).
    """

    def __init__(self, target: np.ndarray | None = None, action_dim: int = 7) -> None:
        self._action_dim = action_dim
        if target is not None:
            self._target = np.asarray(target, dtype=np.float32)
        else:
            self._target = np.array([0.0, 0.5, -0.3] + [0.0] * (action_dim - 3), dtype=np.float32)
        self._loaded = False

    # ------------------------------------------------------------------
    # VLABase interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        """No-op — mock does not load weights."""
        self._loaded = True

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """Return a P-controller-like action toward the fixed target.

        Parameters
        ----------
        obs:
            Must contain ``"proprioception"`` — a 1-D array of joint
            positions (length >= ``action_dim``).

        Returns
        -------
        np.ndarray
            Shape ``[action_dim]`` — delta positions (target - current).
        """
        proprio = obs.get("proprioception", np.zeros(self._action_dim))
        if proprio is None or proprio.size == 0:
            proprio = np.zeros(self._action_dim)
        current = np.asarray(proprio, dtype=np.float32)[: self._action_dim]
        action = self._target[: self._action_dim] - current
        # gentle scaling so we don't overshoot in one step
        action *= 0.2
        return action
