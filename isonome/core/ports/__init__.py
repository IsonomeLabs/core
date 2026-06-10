"""Core ports — outbound interfaces that core layers expect from the runtime.

Ports define what core needs from the outside world (simulation bridges,
hardware drivers, network transports). Concrete adapters live outside
core, typically under ``isonome.bridge``.
"""
from __future__ import annotations

from isonome.core.ports.body_bridge import BodyBridge, BridgePortError

__all__ = ["BodyBridge", "BridgePortError"]
