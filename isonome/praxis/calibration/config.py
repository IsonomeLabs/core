"""Configuration models for the calibration / training pipeline.

Addresses architecture gap #3: the PRD describes a full simulation pipeline
with URDF stripping, domain randomization, CMA-ES optimization, composition
validation, auto-adjustment, and export of a certified policy package (.zip).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class URDFStripperConfig:
    """Which joints / links to preserve when stripping a URDF per agent."""

    keep_joints: list[str] | None = None
    keep_links: list[str] | None = None
    remove_sensors: bool = False
    remove_transmissions: bool = False


@dataclass
class DomainRandomizationConfig:
    """Ranges for domain randomization.  A value of ``None`` disables that axis."""

    mass_scale_range: tuple[float, float] = (0.8, 1.2)
    friction_scale_range: tuple[float, float] | None = (0.5, 1.5)
    damping_scale_range: tuple[float, float] | None = (0.5, 1.5)
    lighting_intensity_range: tuple[float, float] | None = None
    seed: int | None = 42


@dataclass
class CMAESConfig:
    """Hyper-parameters for the built-in CMA-ES optimizer."""

    population_size: int = 16
    initial_sigma: float = 0.5
    max_generations: int = 20
    fitness_target: float | None = None
    seed: int | None = 42


@dataclass
class ValidationConfig:
    """Composition validation settings."""

    episodes: int = 1000
    success_rate_threshold: float = 0.99
    max_steps_per_episode: int = 200
    seed: int | None = 42


@dataclass
class AutoAdjustmentConfig:
    """How the pipeline auto-adjusts when validation fails."""

    enabled: bool = True
    max_iterations: int = 5
    dr_strength_growth: float = 1.2
    episode_growth: int = 200


@dataclass
class CalibrationConfig:
    """Top-level configuration for the calibration / training pipeline."""

    task_type: str = "reach"
    vla_version: str = "openvla-7b-v1"
    output_dir: str | Path = "~/.isonome/calibrations"
    cache_namespace: str = "public"
    stripper: URDFStripperConfig = field(default_factory=URDFStripperConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    optimizer: CMAESConfig = field(default_factory=CMAESConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    auto_adjustment: AutoAdjustmentConfig = field(
        default_factory=AutoAdjustmentConfig
    )
    coordinator_strategy: Literal["priority", "weighted_average", "nullspace"] = (
        "priority"
    )
    agent_configs: dict[str, Any] = field(default_factory=dict)
    reflex_gains: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir).expanduser()
