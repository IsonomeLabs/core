from __future__ import annotations
from isonome.presets.base import Preset
from isonome.core.config import AppConfig, ReflexConfig, JEPAConfig, CortexConfig


class PetPreset(Preset):
    """Reactive companion robot -- dominant Reflex, minimal Cortex."""

    @property
    def name(self) -> str:
        return "pet"

    @property
    def description(self) -> str:
        return "Reactive companion robot with fast reflexes and minimal deliberation"

    def default_config(self) -> AppConfig:
        return AppConfig(
            agent_name="pet_robot",
            reflex=ReflexConfig(frequency_hz=100.0),
            jepa=JEPAConfig(frequency_hz=5.0, prediction_horizon_s=0.5),
            cortex=CortexConfig(frequency_hz=0.1),
        )
