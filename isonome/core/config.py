from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReflexConfig(BaseModel):
    frequency_hz: float = 100.0
    max_latency_ms: float = 10.0


class JEPAConfig(BaseModel):
    frequency_hz: float = 10.0
    prediction_horizon_s: float = 1.0
    model_path: str | None = None


class CortexConfig(BaseModel):
    frequency_hz: float = 0.5
    anomaly_threshold: float = 0.7
    sandbox_timeout_s: float = 30.0
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"


class PlasticityConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    swarm_size: int = 3
    sandbox_timeout_s: float = 60.0


class SafetyConfig(BaseModel):
    permit_boot_adaptation: bool = False
    error_window_s: float = 300.0  # 5 minutes
    error_repeat_threshold: int = 3
    sim_validation_ticks: int = 100


class SimConfig(BaseModel):
    engine: Literal["pybullet", "godot"] = "pybullet"
    timestep: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    gui: bool = False


class AppConfig(BaseModel):
    agent_name: str = "isonome_agent"
    reflex: ReflexConfig = Field(default_factory=ReflexConfig)
    jepa: JEPAConfig = Field(default_factory=JEPAConfig)
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    plasticity: PlasticityConfig = Field(default_factory=PlasticityConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    preset: str | None = None
    layers_dir: str = "layers"

    @classmethod
    def from_yaml(cls, path: Path) -> AppConfig:
        import yaml

        data = yaml.safe_load(path.read_text())
        return cls(**data)

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        import tomllib

        data = tomllib.loads(path.read_text())
        # Flatten isonome section if present
        if "isonome" in data:
            data = data["isonome"]
        if "tool" in data and "isonome" in data["tool"]:
            data = data["tool"]["isonome"]
        return cls(**data)
