"""Equilibrium Health Score — composite diagnostic of system health.

Synthesizes multiple equilibrium signals into a single [0, 1] health metric:

  health = w_d * drift_score
         + w_o * oscillation_score
         + w_c * cooldown_score
         + w_v * velocity_score

Where each component score is in [0, 1]:
- **drift_score**: 1.0 at perfect homeostasis, 0.0 at maximum drift.
  Computed as `1.0 - clamp(rms_drift / sqrt(2))` where `rms_drift` is
  the engine's `tension_distance()`. The `sqrt(2)` normalization accounts
  for the worst-case drift of an axis at position ±1.0 from default ±1.0.
- **oscillation_score**: 1.0 when no axes are oscillating, decreasing
  proportionally to the fraction of axes currently oscillating.
- **cooldown_score**: 1.0 when no cooldowns are active, decreasing
  proportionally to the fraction of (pillar, axis) pairs under cooldown.
- **velocity_score**: 1.0 when no axes have imminent oscillation,
  decreasing proportionally to the fraction of axes with imminent oscillation.

Default weights (drift=0.5, oscillation=0.25, cooldown=0.1, velocity=0.15)
reflect that drift is the primary health indicator, oscillation is the
second most important, velocity prediction is a moderate contributor,
and cooldown is a minor signal.

The health score is mapped to a HealthLevel enum for human-readable
classification:

- EXCELLENT (>= 0.9): System is in excellent homeostatic balance
- GOOD (>= 0.7): System is healthy with minor deviations
- FAIR (>= 0.5): System is under moderate stress
- POOR (>= 0.3): System is significantly stressed
- CRITICAL (< 0.3): System is in critical stress, needs intervention

Integration:
- Opt-in via EquilibriumEngine(enable_health_score=True)
- Automatic computation via engine.compute_health()
- PillarEquilibriumView exposes health_score and health_level
- Serialization round-trip preserves scorer configuration
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from isonome.types import TensionID


class HealthLevel(StrEnum):
    """Human-readable classification of equilibrium health.

    Each level has a numeric value for comparison and ordering.
    """

    CRITICAL = "critical"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

    @classmethod
    def from_score(cls, score: float) -> HealthLevel:
        """Classify a health score into a HealthLevel.

        Thresholds:
        - >= 0.9 → EXCELLENT
        - >= 0.7 → GOOD
        - >= 0.5 → FAIR
        - >= 0.3 → POOR
        - < 0.3  → CRITICAL
        """
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.7:
            return cls.GOOD
        elif score >= 0.5:
            return cls.FAIR
        elif score >= 0.3:
            return cls.POOR
        else:
            return cls.CRITICAL

    @property
    def numeric_value(self) -> float:
        """Numeric ordering value for this level."""
        _VALUES = {
            HealthLevel.CRITICAL: 0.0,
            HealthLevel.POOR: 0.3,
            HealthLevel.FAIR: 0.5,
            HealthLevel.GOOD: 0.7,
            HealthLevel.EXCELLENT: 0.9,
        }
        return _VALUES[self]


# Default component weights
_DEFAULT_WEIGHTS: dict[str, float] = {
    "drift": 0.5,
    "oscillation": 0.25,
    "cooldown": 0.1,
    "velocity": 0.15,
}

# Component keys (for validation)
_COMPONENT_KEYS = frozenset({"drift", "oscillation", "cooldown", "velocity"})


class EquilibriumHealthScore:
    """Composite health score for an EquilibriumEngine.

    Computes a weighted combination of drift, oscillation, cooldown,
    and velocity signals into a single [0, 1] score with per-axis
    breakdown and human-readable level classification.

    Usage::

        from isonome.equilibrium.health import EquilibriumHealthScore

        # Standalone
        scorer = EquilibriumHealthScore()
        result = scorer.compute(engine)
        print(result["overall"])       # 0.85
        print(result["drift"])         # 0.90
        print(result["oscillation"])   # 1.00

        # Via engine
        engine = EquilibriumEngine(enable_health_score=True)
        health = engine.compute_health()
        print(health["level"])         # "good"

        # Via pillar view
        view = engine.view_for(Pillar.COGNITION)
        print(view.health_level)       # HealthLevel.GOOD
    """

    __slots__ = ("_weights",)

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
    ):
        """Initialize the health scorer.

        Args:
            weights: Custom component weights. Must have keys
                'drift', 'oscillation', 'cooldown', 'velocity',
                all non-negative, summing to 1.0.
                Defaults to drift=0.5, oscillation=0.25,
                cooldown=0.1, velocity=0.15.

        Raises:
            ValueError: If weights are invalid.
        """
        if weights is None:
            self._weights = dict(_DEFAULT_WEIGHTS)
        else:
            # Validate keys
            w = dict(weights)
            missing = _COMPONENT_KEYS - set(w.keys())
            extra = set(w.keys()) - _COMPONENT_KEYS
            if missing:
                raise ValueError(f"Missing weight keys: {missing}")
            if extra:
                raise ValueError(f"Unknown weight keys: {extra}")

            # Validate values
            for k, v in w.items():
                if v < 0:
                    raise ValueError(
                        f"Weight '{k}' must be non-negative, got {v}"
                    )

            total = sum(w.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {total:.6f}"
                )

            self._weights = w

    @property
    def weights(self) -> dict[str, float]:
        """Component weights (copy)."""
        return dict(self._weights)

    def compute(self, engine: EquilibriumEngine) -> dict[str, float]:  # type: ignore[name-defined]
        """Compute the health score for an engine.

        Returns a dict with keys:
        - 'overall': weighted composite score in [0, 1]
        - 'drift': drift component score in [0, 1]
        - 'oscillation': oscillation component score in [0, 1]
        - 'cooldown': cooldown component score in [0, 1]
        - 'velocity': velocity component score in [0, 1]

        Args:
            engine: The EquilibriumEngine to evaluate.

        Returns:
            Dict of component scores and overall score.
        """
        drift_score = self._compute_drift(engine)
        osc_score = self._compute_oscillation(engine)
        cooldown_score = self._compute_cooldown(engine)
        velocity_score = self._compute_velocity(engine)

        overall = (
            self._weights["drift"] * drift_score
            + self._weights["oscillation"] * osc_score
            + self._weights["cooldown"] * cooldown_score
            + self._weights["velocity"] * velocity_score
        )
        # Clamp to [0, 1] for safety
        overall = max(0.0, min(1.0, overall))

        return {
            "overall": overall,
            "drift": drift_score,
            "oscillation": osc_score,
            "cooldown": cooldown_score,
            "velocity": velocity_score,
        }

    def health_level(self, engine: EquilibriumEngine) -> HealthLevel:  # type: ignore[name-defined]
        """Compute the health level for an engine.

        Args:
            engine: The EquilibriumEngine to evaluate.

        Returns:
            HealthLevel enum member.
        """
        result = self.compute(engine)
        return HealthLevel.from_score(result["overall"])

    def per_axis(self, engine: EquilibriumEngine) -> dict[TensionID, float]:  # type: ignore[name-defined]
        """Compute per-axis health scores.

        Each axis's score is based purely on its drift from default,
        normalized to [0, 1] where 1.0 = at default, 0.0 = maximally
        drifted.

        Args:
            engine: The EquilibriumEngine to evaluate.

        Returns:
            Dict mapping axis ID to per-axis health score.
        """
        result: dict[TensionID, float] = {}
        for axis in engine.axes:
            distance = abs(axis.position - axis.default_position)
            # Max possible drift for a single axis is 2.0
            # (position=1.0, default=-1.0 or vice versa)
            # but typically bounded by [-1, 1] so max is 2.0
            # Normalize: score = 1.0 - distance / max_distance
            # Use sqrt(2) as a reasonable normalization factor
            # so that a drift of ~1.4 gives score 0.0
            score = max(0.0, 1.0 - distance / math.sqrt(2))
            result[axis.id] = score
        return result

    def summary(self, engine: EquilibriumEngine) -> dict[str, Any]:  # type: ignore[name-defined]
        """Compute a full health summary for an engine.

        Returns a dict with:
        - 'overall': float in [0, 1]
        - 'level': HealthLevel string value
        - 'components': dict of component scores
        - 'per_axis': dict of per-axis scores

        Args:
            engine: The EquilibriumEngine to evaluate.

        Returns:
            Complete health diagnostic dict.
        """
        scores = self.compute(engine)
        level = HealthLevel.from_score(scores["overall"])
        return {
            "overall": scores["overall"],
            "level": level.value,
            "components": {
                "drift": scores["drift"],
                "oscillation": scores["oscillation"],
                "cooldown": scores["cooldown"],
                "velocity": scores["velocity"],
            },
            "per_axis": self.per_axis(engine),
        }

    # ── Component computation ──────────────────────────────────

    def _compute_drift(self, engine: EquilibriumEngine) -> float:  # type: ignore[name-defined]
        """Drift component: 1.0 at homeostasis, 0.0 at max drift.

        Uses the engine's tension_distance() (RMS drift across all axes).
        Normalized by sqrt(2) — the worst-case single-axis drift when
        position is ±1 and default is ∓1 (distance = 2, squared = 4,
        RMS contribution = sqrt(4/N), but with sqrt(2) normalization
        we get score = 1 - rms/sqrt(2), which gives:
        - 0 drift → score 1.0
        - rms = sqrt(2) → score 0.0
        """
        rms = engine.tension_distance()
        # Normalize: sqrt(2) ≈ 1.414 is a reasonable max
        return max(0.0, 1.0 - rms / math.sqrt(2))

    def _compute_oscillation(self, engine: EquilibriumEngine) -> float:  # type: ignore[name-defined]
        """Oscillation component: penalizes oscillating axes.

        When adaptive damping is enabled, counts axes with effective
        damping significantly above base damping (indicating recent
        oscillation). Without adaptive damping, falls back to checking
        oscillation events.

        Returns 1.0 when no axes are oscillating, decreasing to 0.0
        as all axes oscillate.
        """
        controller = engine.adaptive_damping
        if controller is None:
            # No adaptive damping → check oscillation events
            # If no events, assume no oscillation
            if engine.total_oscillation_events == 0:
                return 1.0
            # With events but no controller, use a heuristic:
            # each event is mild evidence of oscillation
            n_axes = len(engine.axes)
            if n_axes == 0:
                return 1.0
            # Ratio of oscillation events to feedback received
            # Saturates: more than n_axes events → 0.0
            ratio = min(1.0, engine.total_oscillation_events / max(1, n_axes))
            return max(0.0, 1.0 - ratio)

        # With adaptive damping: count axes whose effective damping
        # is significantly above base damping (boosted by oscillation)
        n_axes = len(engine.axes)
        if n_axes == 0:
            return 1.0

        oscillating = 0
        for axis in engine.axes:
            base_d = axis.damping
            eff_d = controller.effective_damping(axis.id)
            # If effective damping is more than 10% above base,
            # the axis is considered oscillating
            if eff_d > base_d * 1.1:
                oscillating += 1

        if oscillating == 0:
            return 1.0

        # Score decreases proportionally to fraction of oscillating axes
        fraction = oscillating / n_axes
        return max(0.0, 1.0 - fraction)

    def _compute_cooldown(self, engine: EquilibriumEngine) -> float:  # type: ignore[name-defined]
        """Cooldown component: penalizes active cooldowns.

        When cooldown manager is enabled, computes the fraction of
        active cooldown pairs whose current multiplier is below 1.0.
        Returns 1.0 when no cooldowns are active.

        Reads from the cooldown manager's internal _state dict
        (keys are (pillar_value, axis_id) tuples, values are dicts
        with 'current_multiplier' key).
        """
        from isonome.types import Pillar

        cooldown_mgr = engine.feedback_cooldown
        if cooldown_mgr is None:
            return 1.0

        # Access internal state dict directly for aggregate computation
        # The key format is (pillar_value: str, axis_id: str)
        state = cooldown_mgr._state
        if not state:
            return 1.0

        suppressed = 0
        total_multiplier = 0.0
        total = len(state)

        for key, entry in state.items():
            mult = entry.get("current_multiplier", 1.0)
            total_multiplier += mult
            if mult < 1.0:
                suppressed += 1

        if total == 0:
            return 1.0

        # Fraction of suppressed pairs
        fraction_suppressed = suppressed / total
        # Average multiplier across all active pairs
        avg_multiplier = total_multiplier / total

        # Penalty combines fraction suppressed and average suppression depth
        # avg_multiplier ranges from ~0.01 to 1.0
        # depth = 1.0 - avg_multiplier ranges from 0 to ~0.99
        penalty = fraction_suppressed * (1.0 - avg_multiplier)
        return max(0.0, 1.0 - penalty)

    def _compute_velocity(self, engine: EquilibriumEngine) -> float:  # type: ignore[name-defined]
        """Velocity component: penalizes imminent oscillation.

        When velocity tracking is enabled, counts axes where
        is_oscillation_imminent() returns True. Returns 1.0 when
        no axes are imminent.
        """
        tracker = engine.velocity_tracker
        if tracker is None:
            return 1.0

        n_axes = len(engine.axes)
        if n_axes == 0:
            return 1.0

        imminent_count = 0
        for axis in engine.axes:
            if tracker.is_oscillation_imminent(axis.id):
                imminent_count += 1

        if imminent_count == 0:
            return 1.0

        # Score decreases proportionally to fraction of imminent axes
        fraction = imminent_count / n_axes
        return max(0.0, 1.0 - fraction)

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize health scorer configuration."""
        return {
            "weights": dict(self._weights),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EquilibriumHealthScore:
        """Deserialize health scorer from a dict produced by to_dict().

        Args:
            data: A dict produced by to_dict().

        Returns:
            A reconstructed EquilibriumHealthScore.
        """
        weights = data.get("weights")
        if weights is not None:
            return cls(weights=weights)
        return cls()

    def __repr__(self) -> str:
        return (
            f"EquilibriumHealthScore("
            f"drift={self._weights['drift']:.2f}, "
            f"osc={self._weights['oscillation']:.2f}, "
            f"cool={self._weights['cooldown']:.2f}, "
            f"vel={self._weights['velocity']:.2f})"
        )
