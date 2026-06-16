"""Isonome v0.2 configuration models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReflexConfig(BaseModel):
    frequency_hz: float = 100.0
    max_latency_ms: float = 10.0
    control_freq_hz: float = 100.0
    policy_freq_hz: float = 1.0


class JEPAConfig(BaseModel):
    frequency_hz: float = 10.0
    backend: str = "openvla"
    model_id: str | None = None


class CortexConfig(BaseModel):
    frequency_hz: float = 0.5


class SomaConfig(BaseModel):
    urdf_path: str = ""
    kernel_path: str | None = None


class PlasticityConfig(BaseModel):
    """Runtime-only kernel persistence config."""

    kernel_dir: str = "~/.isonome/kernels"


class SafetyConfig(BaseModel):
    permit_boot_adaptation: bool = False
    error_window_s: float = 300.0
    error_repeat_threshold: int = 3
    sim_validation_ticks: int = 100


class BridgeConfig(BaseModel):
    """Body bridge selection — connects SomaLayer to a sim or hw backend."""

    engine: Literal[
        "none", "mock", "pybullet", "mujoco", "mujoco_mjx", "hardware", "isaac", "isaac_lab"
    ] = "none"
    engine_options: dict = Field(default_factory=dict)


class SimConfig(BaseModel):
    engine: Literal["pybullet", "godot"] = "pybullet"
    timestep: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    gui: bool = False


class CoordinationConfig(BaseModel):
    """Chamber 3 coordination engine config."""

    enabled: bool = False
    strategy: Literal["priority", "weighted_average", "nullspace"] = "priority"
    frequency_hz: float = 200.0


class CalibrationPipelineConfig(BaseModel):
    """Top-level calibration / training pipeline config (gap #3)."""

    enabled: bool = False
    task_type: str = "reach"
    vla_version: str = "openvla-7b-v1"
    output_dir: str = "~/.isonome/calibrations"
    cache_namespace: str = "public"
    coordinator_strategy: Literal["priority", "weighted_average", "nullspace"] = "priority"
    agent_configs: dict[str, Any] = Field(default_factory=dict)
    reflex_gains: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    agent_name: str = "isonome_agent"
    reflex: ReflexConfig = Field(default_factory=ReflexConfig)
    jepa: JEPAConfig = Field(default_factory=JEPAConfig)
    cortex: CortexConfig = Field(default_factory=CortexConfig)
    soma: SomaConfig = Field(default_factory=SomaConfig)
    plasticity: PlasticityConfig = Field(default_factory=PlasticityConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    bridge: BridgeConfig = Field(default_factory=BridgeConfig)
    coordination: CoordinationConfig = Field(default_factory=CoordinationConfig)
    calibration: CalibrationPipelineConfig = Field(default_factory=CalibrationPipelineConfig)
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
        if "isonome" in data:
            data = data["isonome"]
        if "tool" in data and "isonome" in data["tool"]:
            data = data["tool"]["isonome"]
        return cls(**data)
