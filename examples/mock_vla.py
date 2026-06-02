"""Mock VLA backend for testing without downloading 3B parameters.

Outputs deterministic action chunks so examples run instantly.
"""
from __future__ import annotations

import torch

from isonome.core.state import RawSensorState


class MockVLABackend:
    """Lightweight mock VLA that outputs deterministic action chunks.

    Simulates a policy that always tries to reach a fixed target
    [0.50, 0.50, 0.50, 0.0, ...] in canonical space.
    """

    def __init__(self, canonical_dim: int = 14) -> None:
        self.canonical_dim = canonical_dim
        self._target = torch.tensor([0.50, 0.50, 0.50] + [0.0] * (canonical_dim - 3))

    def infer(self, raw_state: RawSensorState, prompt: str) -> torch.Tensor:
        """Return a deterministic action chunk toward the fixed target."""
        proprio = raw_state.proprioception
        # Simple: action = target - current (first 3 DOFs)
        actions = torch.zeros(1, self.canonical_dim)
        current = proprio[:3] if proprio.numel() >= 3 else torch.zeros(3)
        actions[0, :3] = self._target[:3] - current
        return actions
