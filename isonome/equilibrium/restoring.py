"""Momentum-Modulated Restoring Force — engine-level homeostatic pull.

Adds a gentle, momentum-aware restoring nudge directly into the
equilibrium engine's position-update loop. Unlike the pillar-level
stress feedback (BasePillar._emit_stress_feedback), which generates
a Feedback signal that re-enters apply_feedback(), this module
applies a small position correction *inside* the engine — avoiding
the feedback loop overhead and providing a more immediate, physically
grounded pull toward homeostasis.

Design:

  restoring_delta = -momentum_weight * position_velocity

Where:
  - position_velocity = (position - default) — the raw displacement
  - momentum_weight is computed from the TensionVelocityTracker's
    momentum score:
    - momentum < 0 (drifting away): weight = base_weight * drifting_boost
      → pull harder toward default
    - momentum > 0 (approaching default): weight = base_weight * approaching_damp
      → let the axis coast (weaker pull)
    - momentum == 0: weight = base_weight (neutral)

The restoring delta is clamped to a small fraction of the displacement
(max_restoring_fraction) to prevent overshooting, and the total
position (including this nudge) is clamped to [-1, 1].

Integration:
  - Opt-in via EquilibriumEngine(enable_momentum_restoring_force=True)
  - Requires velocity tracking (enable_velocity_tracking=True)
  - Applied during apply_feedback() *after* the primary feedback
    adjustment but *before* the position is stored
  - The pillar-level stress feedback remains independent and
    complementary — this is a faster, lower-overhead nudge

Why both?
  - Engine-level restoring: immediate, low-latency, per-tick micro-nudge
  - Pillar-level stress feedback: deliberate, higher-magnitude, enters
    the feedback loop for audit trail and cooldown awareness
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isonome.types import TensionID


class MomentumRestoringForce:
    """Computes a momentum-modulated restoring force toward homeostasis.

    The restoring force is a gentle nudge applied directly to axis
    positions within the engine, modulated by the axis's momentum
    (velocity direction relative to default).

    Attributes:
        base_weight: The neutral restoring strength (applied when
            momentum is zero or velocity tracking is off).
        drifting_boost: Multiplier when axis is drifting away from
            default (momentum < 0). Must be >= 1.0.
        approaching_damp: Multiplier when axis is approaching default
            (momentum > 0). Must be > 0.0 and <= 1.0.
        max_restoring_fraction: Maximum fraction of displacement that
            the restoring force can correct in a single tick.
            Prevents overshooting. Must be in (0, 0.5].
        min_displacement: Minimum |position - default| to trigger
            restoring force. Below this, the axis is considered
            close enough to homeostasis.
    """

    __slots__ = (
        "_base_weight",
        "_drifting_boost",
        "_approaching_damp",
        "_max_restoring_fraction",
        "_min_displacement",
        "_total_applications",
        "_total_drifting_applications",
        "_total_approaching_applications",
    )

    def __init__(
        self,
        *,
        base_weight: float = 0.02,
        drifting_boost: float = 2.0,
        approaching_damp: float = 0.3,
        max_restoring_fraction: float = 0.1,
        min_displacement: float = 0.05,
    ) -> None:
        """Initialize the momentum-modulated restoring force.

        Args:
            base_weight: Neutral restoring strength per tick.
                Range: (0, 0.1]. Default: 0.02.
            drifting_boost: Boost multiplier when axis is drifting
                away from default. Range: [1.0, 10.0]. Default: 2.0.
            approaching_damp: Dampening multiplier when axis is
                approaching default. Range: (0, 1.0]. Default: 0.3.
            max_restoring_fraction: Max fraction of displacement
                correctable per tick. Range: (0, 0.5]. Default: 0.1.
            min_displacement: Minimum displacement to trigger.
                Range: [0, 1.0). Default: 0.05.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if not (0.0 < base_weight <= 0.1):
            raise ValueError(
                f"base_weight must be in (0, 0.1], got {base_weight}"
            )
        if not (1.0 <= drifting_boost <= 10.0):
            raise ValueError(
                f"drifting_boost must be in [1.0, 10.0], got {drifting_boost}"
            )
        if not (0.0 < approaching_damp <= 1.0):
            raise ValueError(
                f"approaching_damp must be in (0, 1.0], got {approaching_damp}"
            )
        if not (0.0 < max_restoring_fraction <= 0.5):
            raise ValueError(
                f"max_restoring_fraction must be in (0, 0.5], "
                f"got {max_restoring_fraction}"
            )
        if not (0.0 <= min_displacement < 1.0):
            raise ValueError(
                f"min_displacement must be in [0, 1.0), got {min_displacement}"
            )

        self._base_weight = base_weight
        self._drifting_boost = drifting_boost
        self._approaching_damp = approaching_damp
        self._max_restoring_fraction = max_restoring_fraction
        self._min_displacement = min_displacement
        self._total_applications: int = 0
        self._total_drifting_applications: int = 0
        self._total_approaching_applications: int = 0

    # ── Properties ──────────────────────────────────────────────

    @property
    def base_weight(self) -> float:
        """Neutral restoring strength."""
        return self._base_weight

    @property
    def drifting_boost(self) -> float:
        """Multiplier when axis is drifting away from default."""
        return self._drifting_boost

    @property
    def approaching_damp(self) -> float:
        """Multiplier when axis is approaching default."""
        return self._approaching_damp

    @property
    def max_restoring_fraction(self) -> float:
        """Maximum fraction of displacement correctable per tick."""
        return self._max_restoring_fraction

    @property
    def min_displacement(self) -> float:
        """Minimum displacement to trigger restoring force."""
        return self._min_displacement

    @property
    def total_applications(self) -> int:
        """Total number of times restoring force was applied."""
        return self._total_applications

    @property
    def total_drifting_applications(self) -> int:
        """Number of applications where axis was drifting (boosted)."""
        return self._total_drifting_applications

    @property
    def total_approaching_applications(self) -> int:
        """Number of applications where axis was approaching (dampened)."""
        return self._total_approaching_applications

    # ── Core computation ────────────────────────────────────────

    def compute(
        self,
        axis_id: "TensionID",
        position: float,
        default_position: float,
        momentum_score: float = 0.0,
    ) -> float:
        """Compute the restoring force delta for a single axis.

        The restoring delta nudges the position toward its default,
        modulated by the axis's momentum score.

        Args:
            axis_id: The tension axis identifier (for logging).
            position: Current axis position.
            default_position: Homeostasis target.
            momentum_score: From TensionVelocityTracker.
                Positive = approaching default, negative = drifting.

        Returns:
            A small delta to add to the position. Zero if the
            displacement is below min_displacement.
        """
        displacement = position - default_position
        abs_displacement = abs(displacement)

        if abs_displacement < self._min_displacement:
            return 0.0

        # Compute weight based on momentum
        weight = self._base_weight
        is_drifting = False
        is_approaching = False

        if momentum_score < 0:
            # Axis is drifting away from default — boost restoring force
            weight *= self._drifting_boost
            is_drifting = True
        elif momentum_score > 0:
            # Axis is approaching default — dampen restoring force
            weight *= self._approaching_damp
            is_approaching = True
        # else: momentum == 0 → use base_weight (neutral)

        # Restoring delta: nudge toward default
        # Negative of displacement, scaled by weight
        restoring_delta = -displacement * weight

        # Clamp to max_restoring_fraction of displacement
        # to prevent overshooting
        max_correction = abs_displacement * self._max_restoring_fraction
        restoring_delta = max(-max_correction, min(max_correction, restoring_delta))

        # Track statistics
        self._total_applications += 1
        if is_drifting:
            self._total_drifting_applications += 1
        elif is_approaching:
            self._total_approaching_applications += 1

        return restoring_delta

    def compute_no_track(
        self,
        axis_id: "TensionID",
        position: float,
        default_position: float,
        momentum_score: float = 0.0,
    ) -> float:
        """Compute restoring delta without incrementing counters.

        Useful for previews and dry-runs.
        """
        displacement = position - default_position
        abs_displacement = abs(displacement)

        if abs_displacement < self._min_displacement:
            return 0.0

        weight = self._base_weight
        if momentum_score < 0:
            weight *= self._drifting_boost
        elif momentum_score > 0:
            weight *= self._approaching_damp

        restoring_delta = -displacement * weight
        max_correction = abs_displacement * self._max_restoring_fraction
        restoring_delta = max(-max_correction, min(max_correction, restoring_delta))

        return restoring_delta

    # ── Reset / serialize ───────────────────────────────────────

    def reset(self) -> None:
        """Reset all application counters."""
        self._total_applications = 0
        self._total_drifting_applications = 0
        self._total_approaching_applications = 0

    def serialize(self) -> dict:
        """Serialize the controller state for persistence."""
        return {
            "base_weight": self._base_weight,
            "drifting_boost": self._drifting_boost,
            "approaching_damp": self._approaching_damp,
            "max_restoring_fraction": self._max_restoring_fraction,
            "min_displacement": self._min_displacement,
            "total_applications": self._total_applications,
            "total_drifting_applications": self._total_drifting_applications,
            "total_approaching_applications": self._total_approaching_applications,
        }

    @classmethod
    def from_serialized(cls, data: dict) -> "MomentumRestoringForce":
        """Reconstruct from serialized state."""
        obj = cls(
            base_weight=data["base_weight"],
            drifting_boost=data["drifting_boost"],
            approaching_damp=data["approaching_damp"],
            max_restoring_fraction=data["max_restoring_fraction"],
            min_displacement=data["min_displacement"],
        )
        obj._total_applications = data.get("total_applications", 0)
        obj._total_drifting_applications = data.get(
            "total_drifting_applications", 0
        )
        obj._total_approaching_applications = data.get(
            "total_approaching_applications", 0
        )
        return obj

    def __repr__(self) -> str:
        return (
            f"MomentumRestoringForce("
            f"base={self._base_weight}, "
            f"drift_boost={self._drifting_boost}, "
            f"approach_damp={self._approaching_damp}, "
            f"applications={self._total_applications})"
        )
