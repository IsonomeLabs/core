"""End-to-end calibration pipeline orchestrator.

Architecture gap #3: orchestrates URDF stripping, domain randomization,
CMA-ES policy optimization, composition validation, auto-adjustment, and
export of a certified policy package (.zip) into the calibration cache.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from isonome.praxis.calibration.config import CalibrationConfig
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
from isonome.praxis.calibration.urdf_stripper import URDFStripper
from isonome.praxis.calibration.validator import (
    CompositionValidator,
    EpisodeResult,
    ValidationReport,
)
from isonome.praxis.calibration_cache import (
    CacheKey,
    CalibrationCache,
    CertifiedPolicyPackage,
)
from isonome.utils.logging import get_layer_logger


class MockPendulumObjective(BlackBoxObjective):
    """Default black-box objective for the open-source runtime.

    Treats the robot as a damped pendulum and optimizes a linear feedback
    policy ``u = W @ state + b``.  The parameter vector is the flattened
    ``[W, b]``.  Higher fitness means the robot reaches and stays near the
    origin.
    """

    def __init__(self, dof: int, seed: int = 42, episodes: int = 5) -> None:
        self._dof = dof
        self._seed = seed
        self._episodes = episodes
        # Linear policy u = W @ [pos, vel] + b; W is (dof, 2*dof), b is dof.
        self._dim = dof * (2 * dof) + dof

    @property
    def dim(self) -> int:
        return self._dim

    def evaluate(self, params: torch.Tensor) -> float:
        rng = random.Random(self._seed)
        total_reward = 0.0
        w = params[: self._dof * (2 * self._dof)].reshape(self._dof, 2 * self._dof)
        b = params[self._dof * (2 * self._dof) :]

        for _ in range(self._episodes):
            pos = torch.tensor(
                [rng.uniform(-0.6, 0.6) for _ in range(self._dof)], dtype=torch.float32
            )
            vel = torch.tensor(
                [rng.uniform(-0.2, 0.2) for _ in range(self._dof)], dtype=torch.float32
            )
            gravity = torch.tensor(
                [4.0 + i * 0.5 for i in range(self._dof)], dtype=torch.float32
            )
            damping = torch.tensor(
                [0.3 + i * 0.05 for i in range(self._dof)], dtype=torch.float32
            )
            mass = torch.ones(self._dof, dtype=torch.float32)
            length = torch.tensor(
                [0.15 + i * 0.02 for i in range(self._dof)], dtype=torch.float32
            )

            dt = 1.0 / 60.0
            episode_reward = 0.0
            for _ in range(200):
                state = torch.cat([pos, vel])
                u = torch.tanh(w @ state + b)
                torque = -gravity * torch.sin(pos) - damping * vel + u
                acc = torque / (mass * length ** 2)
                vel = vel + acc * dt
                pos = pos + vel * dt
                pos = torch.clamp(pos, -math.pi, math.pi)
                episode_reward += float(-pos.norm().item() - 0.1 * vel.norm().item())

            total_reward += episode_reward / 200.0

        return total_reward / self._episodes


class MockEpisodeRunner:
    """Default validation episode runner for the mock pendulum objective."""

    def __init__(
        self,
        policy_params: torch.Tensor,
        dof: int,
        seed: int = 42,
        max_steps: int = 200,
    ) -> None:
        self._params = policy_params
        self._dof = dof
        self._seed = seed
        self._max_steps = max_steps
        self._rng = random.Random(seed)
        self._w = policy_params[: dof * (2 * dof)].reshape(dof, 2 * dof)
        self._b = policy_params[dof * (2 * dof) :]

    def __call__(self) -> EpisodeResult:
        pos = torch.tensor(
            [self._rng.uniform(-0.6, 0.6) for _ in range(self._dof)], dtype=torch.float32
        )
        vel = torch.tensor(
            [self._rng.uniform(-0.2, 0.2) for _ in range(self._dof)], dtype=torch.float32
        )
        gravity = torch.tensor(
            [4.0 + i * 0.5 for i in range(self._dof)], dtype=torch.float32
        )
        damping = torch.tensor(
            [0.3 + i * 0.05 for i in range(self._dof)], dtype=torch.float32
        )
        mass = torch.ones(self._dof, dtype=torch.float32)
        length = torch.tensor(
            [0.15 + i * 0.02 for i in range(self._dof)], dtype=torch.float32
        )

        dt = 1.0 / 60.0
        total_error = 0.0
        success_steps = 0
        for step in range(self._max_steps):
            state = torch.cat([pos, vel])
            u = torch.tanh(self._w @ state + self._b)
            torque = -gravity * torch.sin(pos) - damping * vel + u
            acc = torque / (mass * length ** 2)
            vel = vel + acc * dt
            pos = pos + vel * dt
            pos = torch.clamp(pos, -math.pi, math.pi)

            error = float(pos.norm().item())
            total_error += error
            if error < 0.15:
                success_steps += 1

        mean_error = total_error / self._max_steps
        success = success_steps / self._max_steps >= 0.95
        return EpisodeResult(
            success=success,
            steps=self._max_steps,
            error_metric=mean_error,
        )


@dataclass
class CalibrationResult:
    """Result returned by :class:`CalibrationPipeline`."""

    certified: bool
    package_path: Path | None
    cache_key: CacheKey | None
    cache_entry_dir: Path | None
    optimization: OptimizationResult | None
    validation: ValidationReport | None
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "package_path": str(self.package_path) if self.package_path else None,
            "cache_key": self.cache_key.to_dict() if self.cache_key else None,
            "cache_entry_dir": str(self.cache_entry_dir) if self.cache_entry_dir else None,
            "optimization": {
                "best_fitness": self.optimization.best_fitness,
                "generation": self.optimization.generation,
                "converged": self.optimization.converged,
            }
            if self.optimization
            else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }


class CalibrationPipeline:
    """End-to-end calibration pipeline.

    Parameters
    ----------
    config:
        Pipeline configuration.
    objective:
        Optional custom black-box objective.  If omitted, a
        :class:`MockPendulumObjective` is constructed from ``config`` metadata
        or a default 2-DOF pendulum.
    episode_runner_factory:
        Optional factory that builds an episode runner from optimized params.
    """

    def __init__(
        self,
        config: CalibrationConfig,
        objective: BlackBoxObjective | None = None,
        episode_runner_factory: Callable[[torch.Tensor], Callable[[], EpisodeResult]] | None = None,
    ) -> None:
        self._config = config
        self._objective = objective
        self._episode_runner_factory = episode_runner_factory
        self._logger = get_layer_logger("praxis.calibration.pipeline")
        self._stripper = URDFStripper(config.stripper)
        self._randomizer = DomainRandomizer(config.domain_randomization)
        self._exporter = PolicyPackageExporter()
        self._cache = CalibrationCache()

    def run(
        self,
        base_urdf_path: str | Path,
        topology_hash: str,
        topology_vector: torch.Tensor | None = None,
    ) -> CalibrationResult:
        """Run the full calibration pipeline.

        Parameters
        ----------
        base_urdf_path:
            Source URDF for the robot.
        topology_hash:
            SHA-256 topology hash used for the cache key.
        topology_vector:
            Optional 32-D topology vector for near-match search in the cache.

        Returns
        -------
        :class:`CalibrationResult` summarizing certification, optimization,
        validation, and cache storage.
        """
        base_urdf_path = Path(base_urdf_path)
        if not base_urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {base_urdf_path}")

        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: strip URDF per agent.
        stripped_root = self._stripper.strip(base_urdf_path)
        stripped_path = self._config.output_dir / "stripped.urdf"
        from xml.etree import ElementTree as ET

        ET.ElementTree(stripped_root).write(
            stripped_path, encoding="utf-8", xml_declaration=True
        )

        # Step 2: optional domain randomization.
        randomized_path, lighting = self._randomizer.randomize_to_file(
            stripped_path,
            self._config.output_dir / "randomized.urdf",
            seed=self._config.domain_randomization.seed,
        )

        # Infer DOF from URDF or use metadata override.
        dof = int(
            self._config.metadata.get("dof", len(self._stripper.list_joints(base_urdf_path)))
        )

        objective = self._objective or MockPendulumObjective(
            dof=dof,
            seed=self._config.optimizer.seed or 42,
            episodes=5,
        )

        validation_config = self._config.validation
        best_result: OptimizationResult | None = None
        best_report: ValidationReport | None = None
        certified = False
        iteration = 0

        while iteration < self._config.auto_adjustment.max_iterations:
            iteration += 1
            self._logger.info(
                "calibration_iteration_start",
                extra={"iteration": iteration, "dof": dof},
            )

            # Step 3: optimize with CMA-ES.
            optimizer = CMAESOptimizer(objective, self._config.optimizer)
            best_result = optimizer.optimize()

            # Step 4: validate composition.
            runner_factory = self._episode_runner_factory or (
                lambda params: MockEpisodeRunner(
                    params,
                    dof=dof,
                    seed=validation_config.seed or 42,
                    max_steps=validation_config.max_steps_per_episode,
                )
            )
            validator = CompositionValidator(
                validation_config,
                episode_runner=runner_factory(best_result.best_params),
            )
            best_report = validator.validate()
            certified = best_report.certified

            if certified:
                self._logger.info(
                    "calibration_certified",
                    extra={"iteration": iteration, "success_rate": best_report.success_rate},
                )
                break

            if not self._config.auto_adjustment.enabled:
                break

            # Step 5: auto-adjust.
            self._adjust_for_retry(
                iteration, validation_config, self._config.domain_randomization
            )

        if not certified:
            self._logger.warning(
                "calibration_not_certified",
                extra={
                    "iterations": iteration,
                    "best_success_rate": best_report.success_rate if best_report else 0.0,
                },
            )

        # Step 6: export package.
        package_path: Path | None = None
        cache_entry_dir: Path | None = None
        cache_key: CacheKey | None = None

        if best_result is not None and best_report is not None:
            run_id = f"{topology_hash[:16]}_{self._config.task_type}"
            package_path = self._config.output_dir / f"{run_id}.zip"

            artifacts = PolicyPackageArtifacts(
                manifest={
                    "topology_hash": topology_hash,
                    "task_type": self._config.task_type,
                    "vla_version": self._config.vla_version,
                    "certified": certified,
                    "success_rate": best_report.success_rate,
                    "iterations": iteration,
                },
                agent_configs=self._config.agent_configs,
                coordinator_config={
                    "strategy": self._config.coordinator_strategy,
                },
                reflex_gains=self._config.reflex_gains,
                sim_metrics={
                    "validation": best_report.to_dict(),
                    "lighting": lighting,
                },
                policy_params=best_result.best_params,
                metadata={
                    "optimizer": self._config.optimizer.__dict__,
                    "domain_randomization": self._config.domain_randomization.__dict__,
                },
            )
            package_path = self._exporter.export(artifacts, package_path)

            # Step 7: cache the package.
            cache_key = CacheKey(
                topology_hash=topology_hash,
                task_type=self._config.task_type,
                vla_version=self._config.vla_version,
            )
            pkg = CertifiedPolicyPackage(
                manifest=artifacts.manifest,
                agent_configs=artifacts.agent_configs,
                coordinator_config=artifacts.coordinator_config,
                reflex_gains=artifacts.reflex_gains,
                sim_metrics=artifacts.sim_metrics,
                policy_package_path=str(package_path),
            )
            cache_entry_dir = self._cache.put(
                cache_key,
                pkg,
                namespace=self._config.cache_namespace,
                topology_vector=topology_vector,
            )

        return CalibrationResult(
            certified=certified,
            package_path=package_path,
            cache_key=cache_key,
            cache_entry_dir=cache_entry_dir,
            optimization=best_result,
            validation=best_report,
            iterations=iteration,
            metadata={
                "base_urdf": str(base_urdf_path),
                "stripped_urdf": str(stripped_path),
                "randomized_urdf": str(randomized_path),
                "lighting": lighting,
            },
        )

    def _adjust_for_retry(
        self,
        iteration: int,
        validation_config: ValidationConfig,
        dr_config: Any,
    ) -> None:
        """Mutate configs for the next auto-adjustment iteration."""
        aa = self._config.auto_adjustment
        # Increase domain randomization strength.
        if dr_config.mass_scale_range is not None:
            lo, hi = dr_config.mass_scale_range
            scale = aa.dr_strength_growth
            dr_config.mass_scale_range = (lo / scale, hi * scale)
        # Increase validation budget.
        validation_config.episodes += aa.episode_growth
        self._logger.info(
            "calibration_auto_adjust",
            extra={
                "iteration": iteration,
                "new_episodes": validation_config.episodes,
                "dr_range": dr_config.mass_scale_range,
            },
        )
