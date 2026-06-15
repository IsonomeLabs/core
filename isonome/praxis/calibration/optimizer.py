"""Black-box policy optimizer used by the calibration pipeline.

Architecture gap #3, Step 1: ``CMA-ES / Differentiable Sim`` optimizes a
policy parameter vector.  The open-source runtime ships with a lightweight
CMA-ES implementation that does not require external evolutionary-optimization
libraries.  Users can also provide their own :class:`BlackBoxObjective` to
plug in differentiable sim or other optimizers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import torch

from isonome.praxis.calibration.config import CMAESConfig
from isonome.utils.logging import get_layer_logger


class BlackBoxObjective:
    """Interface for objectives that CMA-ES can optimize.

    Subclasses implement :meth:`evaluate` to map a parameter vector to a scalar
    fitness.  Higher fitness is better.
    """

    def evaluate(self, params: torch.Tensor) -> float:
        """Return a scalar fitness for ``params``.

        Parameters
        ----------
        params:
            1-D parameter vector.

        Returns
        -------
        Scalar fitness.  Higher is better.
        """
        raise NotImplementedError

    @property
    def dim(self) -> int:
        """Dimensionality of the parameter vector."""
        raise NotImplementedError


@dataclass
class OptimizationResult:
    """Result of a CMA-ES run."""

    best_params: torch.Tensor
    best_fitness: float
    generation: int
    mean: torch.Tensor
    covariance: torch.Tensor
    sigma: float
    history: list[float] = field(default_factory=list)
    converged: bool = False


class CMAESOptimizer:
    """Lightweight CMA-ES optimizer for low-dimensional policy parameters.

    Parameters
    ----------
    objective:
        A :class:`BlackBoxObjective` instance.
    config:
        CMA-ES hyper-parameters.
    """

    def __init__(
        self,
        objective: BlackBoxObjective,
        config: CMAESConfig | None = None,
    ) -> None:
        self._objective = objective
        self._config = config or CMAESConfig()
        self._logger = get_layer_logger("praxis.calibration.cmaes")

        self._dim = objective.dim
        self._lambda = self._config.population_size
        if self._lambda < 2:
            self._lambda = 2

        # Default: half the population, rounded up.
        self._mu = self._lambda // 2
        if self._mu < 1:
            self._mu = 1

        # Log-linear weights that sum to 1 and favor better samples.
        raw_weights = [math.log(self._mu + 0.5) - math.log(i + 1) for i in range(self._mu)]
        weight_sum = sum(raw_weights)
        self._weights = torch.tensor([w / weight_sum for w in raw_weights], dtype=torch.float64)
        self._mu_eff = 1.0 / sum(w.item() ** 2 for w in self._weights)

        # Default strategy parameters (Hansen 2016, small dimension defaults).
        self._cc = (4.0 + self._mu_eff / self._dim) / (
            self._dim + 4.0 + 2.0 * self._mu_eff / self._dim
        )
        self._cs = (self._mu_eff + 2.0) / (self._dim + self._mu_eff + 5.0)
        self._c1 = 2.0 / ((self._dim + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1.0 - self._c1,
            2.0 * (self._mu_eff - 2.0 + 1.0 / self._mu_eff)
            / ((self._dim + 2.0) ** 2 + self._mu_eff),
        )
        self._damps = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self._mu_eff - 1.0) / (self._dim + 1.0)) - 1)
            + self._cs
        )

        self._expect_norm = math.sqrt(self._dim) * (
            1.0 - 1.0 / (4.0 * self._dim) + 1.0 / (21.0 * self._dim ** 2)
        )

        rng = random.Random(self._config.seed)
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(rng.randint(0, 2 ** 31 - 1))

    def optimize(self) -> OptimizationResult:
        """Run CMA-ES and return the best parameters found."""
        d = self._dim
        dtype = torch.float64

        mean = torch.zeros(d, dtype=dtype)
        sigma = float(self._config.initial_sigma)
        cov = torch.eye(d, dtype=dtype)
        pc = torch.zeros(d, dtype=dtype)
        ps = torch.zeros(d, dtype=dtype)

        best_fitness = -float("inf")
        best_params = mean.clone()
        history: list[float] = []

        for generation in range(1, self._config.max_generations + 1):
            samples, fitnesses = self._sample_and_evaluate(mean, cov, sigma)

            # Sort by fitness descending.
            sorted_idx = torch.argsort(torch.tensor(fitnesses, dtype=dtype), descending=True)

            gen_best_fitness = fitnesses[sorted_idx[0].item()]
            history.append(gen_best_fitness)
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_params = samples[sorted_idx[0].item()].clone()

            old_mean = mean.clone()

            # Update mean.
            selected = samples[sorted_idx[: self._mu]]
            mean = (selected * self._weights.unsqueeze(1)).sum(dim=0)

            # CMA update.
            c_minus_half = torch.linalg.cholesky(torch.linalg.inv(cov + torch.eye(d, dtype=dtype) * 1e-12))
            y = c_minus_half @ (mean - old_mean) / sigma

            ps = (1.0 - self._cs) * ps + math.sqrt(
                self._cs * (2.0 - self._cs) * self._mu_eff
            ) * y

            hsig = (
                torch.norm(ps).item() / math.sqrt(1.0 - (1.0 - self._cs) ** (2.0 * generation))
                < (1.4 + 2.0 / (d + 1.0)) * self._expect_norm
            )

            pc = (1.0 - self._cc) * pc + (
                hsig
                * math.sqrt(self._cc * (2.0 - self._cc) * self._mu_eff)
                * (mean - old_mean)
                / sigma
            )

            # Art: rank-mu update matrix.
            art = torch.zeros(d, d, dtype=dtype)
            for i in range(self._mu):
                z = (selected[i] - old_mean) / sigma
                art += self._weights[i].item() * torch.outer(z, z)

            cov = (
                (1.0 - self._c1 - self._cmu) * cov
                + self._c1 * torch.outer(pc, pc)
                + self._cmu * art
            )

            # Step-size update.
            ps_norm = torch.norm(ps).item()
            sigma = sigma * math.exp(
                (self._cs / self._damps)
                * (ps_norm / self._expect_norm - 1.0)
            )

            # Clamp covariance for numerical stability.
            cov = self._stabilize_covariance(cov)

            self._logger.debug(
                "cmaes_generation",
                extra={
                    "generation": generation,
                    "best": gen_best_fitness,
                    "sigma": sigma,
                    "mean_norm": mean.norm().item(),
                },
            )

            if (
                self._config.fitness_target is not None
                and best_fitness >= self._config.fitness_target
            ):
                self._logger.info(
                    "cmaes_converged_target",
                    extra={"generation": generation, "best": best_fitness},
                )
                return OptimizationResult(
                    best_params=best_params.float(),
                    best_fitness=best_fitness,
                    generation=generation,
                    mean=mean.float(),
                    covariance=cov.float(),
                    sigma=sigma,
                    history=history,
                    converged=True,
                )

        self._logger.info(
            "cmaes_finished",
            extra={
                "generations": self._config.max_generations,
                "best": best_fitness,
            },
        )
        return OptimizationResult(
            best_params=best_params.float(),
            best_fitness=best_fitness,
            generation=self._config.max_generations,
            mean=mean.float(),
            covariance=cov.float(),
            sigma=sigma,
            history=history,
            converged=False,
        )

    def _sample_and_evaluate(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, list[float]]:
        """Sample ``lambda`` individuals from ``N(mean, sigma^2 C)``.

        Uses an eigendecomposition of ``C`` so we avoid the legacy
        ``MultivariateNormal.sample(generator=...)`` API that varies across
        PyTorch versions.
        """
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = torch.clamp(eigvals, min=1e-12)
        b = eigvecs @ torch.diag(torch.sqrt(eigvals))

        samples = torch.zeros(self._lambda, self._dim, dtype=mean.dtype)
        fitnesses: list[float] = []

        for i in range(self._lambda):
            z = torch.randn(self._dim, dtype=mean.dtype, generator=self._torch_rng)
            sample = mean + sigma * (b @ z)
            samples[i] = sample
            fitnesses.append(self._objective.evaluate(sample.float()))

        return samples, fitnesses

    @staticmethod
    def _stabilize_covariance(cov: torch.Tensor) -> torch.Tensor:
        """Force symmetric PSD covariance with small jitter."""
        cov = 0.5 * (cov + cov.T)
        eigvals = torch.linalg.eigvalsh(cov)
        min_eig = eigvals.min().item()
        if min_eig < 1e-12:
            cov = cov + (1e-12 - min_eig) * torch.eye(cov.shape[0], dtype=cov.dtype)
        return cov
