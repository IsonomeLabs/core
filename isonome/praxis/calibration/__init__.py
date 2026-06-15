"""Calibration / training pipeline for the open-source Isonome runtime.

This package provides the foundational building blocks for architecture gap #3:

* :class:`URDFStripper` — per-agent joint subset extraction.
* :class:`DomainRandomizer` — mass / friction / damping randomization.
* :class:`CMAESOptimizer` — lightweight black-box policy optimization.
* :class:`CompositionValidator` — episode-based success-rate validation.
* :class:`PolicyPackageExporter` — ``.zip`` export of certified packages.
* :class:`CalibrationPipeline` — end-to-end orchestration with auto-adjustment.

The implementation is intentionally decoupled from Isaac Lab / Isaac Sim so it
can run with the existing mock or MuJoCo bridges in the open-source runtime.
Enterprise deployments are expected to swap in GPU-accelerated backends.
"""
from __future__ import annotations

from isonome.praxis.calibration.config import (
    AutoAdjustmentConfig,
    CMAESConfig,
    CalibrationConfig,
    DomainRandomizationConfig,
    URDFStripperConfig,
    ValidationConfig,
)
from isonome.praxis.calibration.domain_randomization import DomainRandomizer
from isonome.praxis.calibration.exporter import (
    PolicyPackageArtifacts,
    PolicyPackageExporter,
)
from isonome.praxis.calibration.optimizer import (
    BlackBoxObjective,
    CMAESOptimizer,
    OptimizationResult,
)
from isonome.praxis.calibration.pipeline import CalibrationPipeline, CalibrationResult
from isonome.praxis.calibration.urdf_stripper import URDFStripper
from isonome.praxis.calibration.validator import (
    CompositionValidator,
    EpisodeResult,
)

__all__ = [
    "AutoAdjustmentConfig",
    "BlackBoxObjective",
    "CMAESConfig",
    "CMAESOptimizer",
    "CalibrationConfig",
    "CalibrationPipeline",
    "CalibrationResult",
    "CompositionValidator",
    "DomainRandomizationConfig",
    "DomainRandomizer",
    "EpisodeResult",
    "OptimizationResult",
    "PolicyPackageArtifacts",
    "PolicyPackageExporter",
    "URDFStripper",
    "URDFStripperConfig",
    "ValidationConfig",
]
