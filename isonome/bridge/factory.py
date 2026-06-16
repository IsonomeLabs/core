"""Factory for constructing the right BodyBridge from AppConfig."""
from __future__ import annotations

from pathlib import Path

from isonome.core.config import AppConfig, BridgeConfig
from isonome.core.ports.body_bridge import BodyBridge
from isonome.utils.logging import get_layer_logger

logger = get_layer_logger("bridge.factory")


def build_body_bridge(config: AppConfig) -> BodyBridge | None:
    """Construct a BodyBridge from the engine declared in ``AppConfig``.

    Returns ``None`` when ``bridge.engine == "none"`` — in that case
    ``SomaLayer`` falls back to no-op perceive/act behavior.
    """
    # Bridge selection is explicit via ``bridge.engine``.  We deliberately do
    # NOT auto-construct a bridge from the legacy ``sim.engine`` field so that
    # existing code and configs that set ``sim.engine: pybullet`` but never
    # connected a bridge continue to work (i.e. they get no-op stub behavior).

    bridge_cfg = config.bridge
    engine = bridge_cfg.engine
    urdf_path = Path(config.soma.urdf_path)

    if engine == "none" or not config.soma.urdf_path:
        return None

    if engine == "mock":
        from isonome.bridge.adapters.mock_adapter import MockBridgeAdapter

        return MockBridgeAdapter(urdf_path, **bridge_cfg.engine_options)

    if engine == "pybullet":
        from isonome.bridge.adapters.pybullet_adapter import PyBulletBridgeAdapter
        from isonome.core.config import SimConfig

        sim_cfg = SimConfig(
            engine="pybullet",
            timestep=config.sim.timestep,
            gravity=config.sim.gravity,
            gui=config.sim.gui,
        )
        return PyBulletBridgeAdapter(
            urdf_path,
            sim_config=sim_cfg,
            **bridge_cfg.engine_options,
        )

    if engine == "mujoco":
        from isonome.bridge.adapters.mujoco_adapter import MuJoCoBridgeAdapter

        return MuJoCoBridgeAdapter(urdf_path, **bridge_cfg.engine_options)

    if engine == "mujoco_mjx":
        from isonome.bridge.adapters.mjx_adapter import MJXBridgeAdapter

        return MJXBridgeAdapter(urdf_path, **bridge_cfg.engine_options)

    if engine == "isaac_lab":
        from isonome.bridge.adapters.isaac_lab_adapter import IsaacLabBridgeAdapter

        return IsaacLabBridgeAdapter(urdf_path, **bridge_cfg.engine_options)

    if engine == "hardware":
        from isonome.bridge.adapters.hardware_adapter import HardwareBridgeAdapter

        return HardwareBridgeAdapter(urdf_path, **bridge_cfg.engine_options)

    raise ValueError(f"Unknown bridge engine: {engine}")
