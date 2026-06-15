"""Composition validation — run episodes and compute success rate.

Architecture gap #3, Step 2: ``Composition Validation`` runs 1000 episodes and
certifies the policy when success rate exceeds 99%.  The open-source validator
is backend-agnostic: it accepts a callable ``episode_runner`` that returns an
:class:`EpisodeResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from isonome.praxis.calibration.config import ValidationConfig
from isonome.utils.logging import get_layer_logger


@dataclass
class EpisodeResult:
    """Outcome of a single validation episode."""

    success: bool
    steps: int
    error_metric: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregated composition validation report."""

    episodes: int
    successes: int
    success_rate: float
    mean_error: float
    certified: bool
    threshold: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "mean_error": self.mean_error,
            "certified": self.certified,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


class CompositionValidator:
    """Run repeated episodes and certify a policy package.

    Parameters
    ----------
    config:
        Validation settings (episode count, success threshold, etc.).
    episode_runner:
        Callable that runs one episode and returns :class:`EpisodeResult`.
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        episode_runner: Callable[[], EpisodeResult] | None = None,
    ) -> None:
        self._config = config or ValidationConfig()
        self._episode_runner = episode_runner
        self._logger = get_layer_logger("praxis.calibration.validator")

    def validate(
        self,
        episode_runner: Callable[[], EpisodeResult] | None = None,
    ) -> ValidationReport:
        """Run ``config.episodes`` episodes and return a validation report.

        Parameters
        ----------
        episode_runner:
            Optional override for the runner supplied at construction.

        Returns
        -------
        :class:`ValidationReport` with aggregated statistics and certification
        status.
        """
        runner = episode_runner or self._episode_runner
        if runner is None:
            raise ValueError("An episode_runner must be provided")

        successes = 0
        total_error = 0.0
        metadata: dict = {"max_steps": [], "errors": []}

        for episode in range(self._config.episodes):
            result = runner()
            if result.success:
                successes += 1
            total_error += result.error_metric
            metadata["max_steps"].append(result.steps)
            metadata["errors"].append(result.error_metric)

        success_rate = successes / self._config.episodes if self._config.episodes else 0.0
        mean_error = total_error / self._config.episodes if self._config.episodes else 0.0
        certified = success_rate >= self._config.success_rate_threshold

        self._logger.info(
            "validation_complete",
            extra={
                "episodes": self._config.episodes,
                "successes": successes,
                "success_rate": success_rate,
                "certified": certified,
            },
        )

        return ValidationReport(
            episodes=self._config.episodes,
            successes=successes,
            success_rate=success_rate,
            mean_error=mean_error,
            certified=certified,
            threshold=self._config.success_rate_threshold,
            metadata=metadata,
        )
