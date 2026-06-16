"""Bridge adapters — BodyBridge implementations for each backend.

Each adapter wraps an existing bridge class (PyBullet, MuJoCo, MuJoCo MJX,
Isaac Sim, Isaac Lab, Mock, Hardware) and exposes the ``BodyBridge`` port
that ``SomaLayer`` expects.
"""
from __future__ import annotations

from isonome.bridge.adapters.hardware_adapter import HardwareBridgeAdapter
from isonome.bridge.adapters.mock_adapter import MockBridgeAdapter
from isonome.bridge.adapters.pybullet_adapter import PyBulletBridgeAdapter

try:
    from isonome.bridge.adapters.mujoco_adapter import MuJoCoBridgeAdapter
except Exception:  # pragma: no cover
    MuJoCoBridgeAdapter = None  # type: ignore[misc, assignment]

try:
    from isonome.bridge.adapters.mjx_adapter import MJXBridgeAdapter
except Exception:  # pragma: no cover
    MJXBridgeAdapter = None  # type: ignore[misc, assignment]

try:
    from isonome.bridge.adapters.isaac_lab_adapter import IsaacLabBridgeAdapter
except Exception:  # pragma: no cover
    IsaacLabBridgeAdapter = None  # type: ignore[misc, assignment]

__all__ = [
    "HardwareBridgeAdapter",
    "MockBridgeAdapter",
    "PyBulletBridgeAdapter",
    "MuJoCoBridgeAdapter",
    "MJXBridgeAdapter",
    "IsaacLabBridgeAdapter",
]
