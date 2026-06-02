from __future__ import annotations
from isonome.presets.base import Preset
from isonome.core.config import AppConfig, ReflexConfig, JEPAConfig, CortexConfig


class PatrolPreset(Preset):
    """Autonomous patrol -- balanced layers, JEPA-heavy navigation."""

    @property
    def name(self) -> str:
        return "patrol"

    @property
    def description(self) -> str:
        return "Autonomous patrol robot with balanced cognition and JEPA-heavy navigation"

    def default_config(self) -> AppConfig:
        return AppConfig(
            agent_name="patrol_robot",
            reflex=ReflexConfig(frequency_hz=100.0),
            jepa=JEPAConfig(frequency_hz=20.0, prediction_horizon_s=2.0),
            cortex=CortexConfig(frequency_hz=0.5),
        )
