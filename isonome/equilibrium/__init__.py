"""The Equilibrium Engine — the heart of isonome.

Continuously balances competing agent tensions (explore/exploit,
speed/thoroughness, etc.) using information-theoretic feedback loops.

Architecture:
    Feedback signals from each pillar → PID-like adjustment →
    damped tension position updates → behavior modulation.

The engine operates as a homeostatic regulator: each tension axis
has a default position (set point), and feedback nudges the current
position. Damping prevents oscillation; learning_rate allows the
set point itself to adapt over time based on outcome signals.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

from isonome.types import (
    AgentState as AgentState,
    EquilibriumError as EquilibriumError,
    Feedback,
    Pillar,
    TensionAxis,
    TensionID,
    TensionOscillationError as TensionOscillationError,
    TensionSnapshot,
    now as now,
)
from isonome.equilibrium.velocity import TensionVelocityTracker
from isonome.equilibrium.cooldown import FeedbackCooldownManager
from isonome.equilibrium.event_log import (
    TensionEventLog as TensionEventLog,
    TensionEvent as TensionEvent,
    TensionEventType as TensionEventType,
)


class PillarEquilibriumView:
    """Structured read-only view of equilibrium state for a single pillar.

    Provides a pillar with direct, organized access to its own tension axes,
    drift metrics, and cross-pillar influence — without needing external
    set_tension_profile() calls.

    This is the pull-side complement to the push-side Feedback mechanism:
    pillars push Feedback to the engine (changing state), and pull
    PillarEquilibriumView from the engine (reading state to modulate behavior).

    Architecture:

     ┌──────────────────┐
     │  EquilibriumEngine │
     │                    │
     │  view_for(pillar) │──→ PillarEquilibriumView
     │                    │     own_axes: {id: pos, ...}
     │                    │     other_axes: {id: pos, ...}
     │                    │     stress_level: float
     │                    │     drift: {id: float, ...}
     │                    │     oscillating: [id, ...]
     └──────────────────┘

    Mathematical foundation:

    Each pillar owns a subset of the 8 tension axes. The view decomposes:

    - **Own axes**: those belonging to this pillar — directly modulate behavior
    - **Cross-pillar influence**: axes from other pillars — indirectly affect
      decision-making (e.g., Cognition reads Praxis autonomy_safety to adjust
      plan risk tolerance)
    - **Stress level**: `σ = √(1/N × Σᵢ (pᵢ - dᵢ)²)` — RMS drift from
      homeostasis across ALL axes. High σ means the agent is in an unfamiliar
      or challenging situation.
    - **Axis drift**: per-axis `|pᵢ - dᵢ|` — how far each axis has moved
      from its learned set point.
    - **Oscillation warning**: axes with high recent variance — the system
      is receiving contradictory feedback and should exercise caution.

    Integration pattern:
    - BasePillar.bind_engine() stores a reference to the engine
    - BasePillar.process_queued() calls engine.view_for(self.pillar) at the
      start of each tick, storing the view as self._equilibrium_view
    - Pillar implementations read self._equilibrium_view instead of requiring
      external update_tension_profile() calls
    """

    __slots__ = (
        "_pillar",
        "_own_axes",
        "_cross_axes",
        "_all_positions",
        "_all_defaults",
        "_stress_level",
        "_drift",
        "_oscillating",
        "_oscillation_threshold",
        "_velocities",
        "_momentum_scores",
        "_oscillation_imminent",
        "_event_log",
        "_own_axis_ids",
        "_feedback_cooldown",
    )

    def __init__(
        self,
        pillar: Pillar,
        axes: dict[TensionID, TensionAxis],
        history: dict[TensionID, deque],
        oscillation_threshold: float,
        velocity_tracker: TensionVelocityTracker | None = None,
        event_log: TensionEventLog | None = None,
        feedback_cooldown: FeedbackCooldownManager | None = None,
    ):
        # Split axes into own and cross-pillar
        self._pillar = pillar
        self._own_axes: dict[TensionID, float] = {}
        self._cross_axes: dict[TensionID, float] = {}
        self._all_positions: dict[TensionID, float] = {}
        self._all_defaults: dict[TensionID, float] = {}
        self._drift: dict[TensionID, float] = {}
        self._oscillating: list[TensionID] = []
        self._oscillation_threshold = oscillation_threshold

        # Velocity data (empty dicts if no tracker)
        self._velocities: dict[TensionID, float] = {}
        self._momentum_scores: dict[TensionID, float] = {}
        self._oscillation_imminent: list[TensionID] = []

        # Event log for audit trail (None if disabled)
        self._event_log = event_log
        self._own_axis_ids: tuple[TensionID, ...] = ()

        # Feedback cooldown manager (None if disabled)
        self._feedback_cooldown = feedback_cooldown

        for axis_id, axis in axes.items():
            pos = axis.position
            default = axis.default_position
            self._all_positions[axis_id] = pos
            self._all_defaults[axis_id] = default
            self._drift[axis_id] = abs(pos - default)

            if axis.pillar == pillar:
                self._own_axes[axis_id] = pos
            else:
                self._cross_axes[axis_id] = pos

            # Check oscillation
            hist = history.get(axis_id)
            if hist is not None and len(hist) >= 4:
                n = len(hist)
                mean = sum(hist) / n
                variance = sum((x - mean) ** 2 for x in hist) / n
                stddev = math.sqrt(variance)
                if stddev > oscillation_threshold:
                    self._oscillating.append(axis_id)

            # Populate velocity data if tracker available
            if velocity_tracker is not None:
                vel = velocity_tracker.get_velocity(axis_id)
                mom = velocity_tracker.get_momentum_score(axis_id)
                self._velocities[axis_id] = vel
                self._momentum_scores[axis_id] = mom
                if velocity_tracker.is_oscillation_imminent(axis_id):
                    self._oscillation_imminent.append(axis_id)

        # Populate own_axis_ids from the own_axes dict keys (for recent_events scoping)
        self._own_axis_ids = tuple(self._own_axes.keys())

        # Compute stress level (RMS drift from homeostasis)
        if self._drift:
            squared = sum(d ** 2 for d in self._drift.values())
            self._stress_level = math.sqrt(squared / len(self._drift))
        else:
            self._stress_level = 0.0

    # ── Properties ──────────────────────────────────────────────

    @property
    def pillar(self) -> Pillar:
        """Which pillar this view belongs to."""
        return self._pillar

    @property
    def own_axes(self) -> dict[TensionID, float]:
        """Positions of axes owned by this pillar: {axis_id: position}.

        These are the tensions the pillar directly controls and should
        use to modulate its own behavior.
        """
        return dict(self._own_axes)

    @property
    def cross_axes(self) -> dict[TensionID, float]:
        """Positions of axes from other pillars: {axis_id: position}.

        These indirectly affect the pillar's decision-making. For example,
        Cognition reads Praxis's autonomy_safety to adjust plan risk tolerance.
        """
        return dict(self._cross_axes)

    @property
    def all_positions(self) -> dict[TensionID, float]:
        """All axis positions regardless of pillar: {axis_id: position}."""
        return dict(self._all_positions)

    @property
    def all_defaults(self) -> dict[TensionID, float]:
        """All axis default positions: {axis_id: default_position}."""
        return dict(self._all_defaults)

    @property
    def stress_level(self) -> float:
        """RMS drift from homeostasis across all axes.

        σ = √(1/N × Σᵢ |pᵢ - dᵢ|²)

        Range: [0.0, ~1.0]. High values indicate the agent is far from
        its learned equilibrium — it may be in an unfamiliar situation
        or receiving contradictory feedback.
        """
        return self._stress_level

    @property
    def drift(self) -> dict[TensionID, float]:
        """Per-axis drift from homeostasis: {axis_id: |position - default|}."""
        return dict(self._drift)

    @property
    def oscillating(self) -> tuple[TensionID, ...]:
        """Axes currently oscillating beyond the threshold.

        An axis is oscillating if its recent position history has
        standard deviation > oscillation_threshold. This means
        the system is receiving contradictory feedback on this axis
        and should exercise caution in modulating behavior based on it.
        """
        return tuple(self._oscillating)

    @property
    def is_stressed(self) -> bool:
        """Whether the stress level exceeds the moderate threshold (0.3)."""
        return self._stress_level > 0.3

    @property
    def is_highly_stressed(self) -> bool:
        """Whether the stress level exceeds the critical threshold (0.5)."""
        return self._stress_level > 0.5

    @property
    def velocities(self) -> dict[TensionID, float]:
        """Per-axis velocity values: {axis_id: velocity}.

        Velocity is the rate of position change per tick. Positive values
        mean the axis is moving right on its pole spectrum, negative means
        moving left. Only populated when velocity tracking is enabled.
        """
        return dict(self._velocities)

    @property
    def momentum_scores(self) -> dict[TensionID, float]:
        """Per-axis momentum scores: {axis_id: momentum_score}.

        Positive = heading toward default (good), negative = drifting
        away (bad). Only populated when velocity tracking is enabled.
        """
        return dict(self._momentum_scores)

    @property
    def oscillation_imminent(self) -> tuple[TensionID, ...]:
        """Axes predicted to oscillate based on velocity reversal patterns.

        Unlike `oscillating` which reports post-hoc (stddev already high),
        this uses velocity reversal rates for early prediction. Only
        populated when velocity tracking is enabled.
        """
        return tuple(self._oscillation_imminent)

    @property
    def feedback_cooldown(self) -> FeedbackCooldownManager | None:
        """The feedback cooldown manager, or None if not enabled."""
        return self._feedback_cooldown

    @property
    def cooldown_axes(self) -> tuple[TensionID, ...]:
        """Axes owned by this pillar that are currently under cooldown.

        An axis is under cooldown if this pillar has sent repeated
        feedback to it within the cooldown window (multiplier < 1.0).
        Only populated when feedback cooldown is enabled.
        """
        if self._feedback_cooldown is None:
            return ()
        cooled = []
        for axis_id in self._own_axes:
            multiplier = self._feedback_cooldown.get_multiplier_for(
                axis_id, self._pillar,
            )
            if multiplier < 1.0:
                cooled.append(axis_id)
        return tuple(cooled)

    # ── Convenience methods ───────────────────────────────────────

    def get(self, axis_id: TensionID, default: float = 0.0) -> float:
        """Get a specific axis position by ID, with default fallback."""
        return self._all_positions.get(axis_id, default)

    def get_drift(self, axis_id: TensionID) -> float:
        """Get drift for a specific axis, 0.0 if unknown."""
        return self._drift.get(axis_id, 0.0)

    def own_axis_ids(self) -> tuple[TensionID, ...]:
        """IDs of axes owned by this pillar."""
        return tuple(self._own_axes.keys())

    def cross_axis_ids(self) -> tuple[TensionID, ...]:
        """IDs of axes owned by other pillars."""
        return tuple(self._cross_axes.keys())

    def is_axis_oscillating(self, axis_id: TensionID) -> bool:
        """Check if a specific axis is oscillating."""
        return axis_id in self._oscillating

    def get_velocity(self, axis_id: TensionID) -> float:
        """Get the current velocity of a specific axis.

        Returns 0.0 if velocity tracking is disabled or axis unknown.
        """
        return self._velocities.get(axis_id, 0.0)

    def get_momentum_score(self, axis_id: TensionID) -> float:
        """Get the momentum score for a specific axis.

        Positive = heading toward default, negative = drifting away.
        Returns 0.0 if velocity tracking is disabled or axis unknown.
        """
        return self._momentum_scores.get(axis_id, 0.0)

    def is_axis_drifting(self, axis_id: TensionID) -> bool:
        """Check if a specific axis is drifting away from its default.

        Returns True if momentum_score < 0 (moving away from homeostasis).
        """
        return self._momentum_scores.get(axis_id, 0.0) < 0

    def recent_events(self, limit: int = 20) -> list[TensionEvent]:
        """Return recent events affecting this pillar's own axes.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of recent TensionEvent objects scoped to this pillar's
            own axes, in chronological order. Empty if event logging
            is disabled.
        """
        if self._event_log is None:
            return []
        all_events = self._event_log.events()
        # Filter to events affecting own axes (or engine-wide events)
        scoped = [
            e for e in all_events
            if e.axis_id in self._own_axis_ids or e.axis_id == ""
        ]
        return scoped[-limit:]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for reporting/logging."""
        result = {
            "pillar": self._pillar.value,
            "own_axes": self._own_axes,
            "cross_axes": self._cross_axes,
            "stress_level": round(self._stress_level, 4),
            "is_stressed": self.is_stressed,
            "is_highly_stressed": self.is_highly_stressed,
            "oscillating": list(self._oscillating),
            "max_drift_axis": max(self._drift, key=self._drift.get) if self._drift else None,
            "max_drift_value": round(max(self._drift.values()), 4) if self._drift else 0.0,
        }
        if self._velocities:
            result["velocities"] = {
                k: round(v, 4) for k, v in self._velocities.items()
            }
            result["momentum_scores"] = {
                k: round(v, 4) for k, v in self._momentum_scores.items()
            }
            result["oscillation_imminent"] = list(self._oscillation_imminent)
        return result

    def __repr__(self) -> str:
        parts = [
            f"pillar={self._pillar.value}",
            f"own={len(self._own_axes)}",
            f"cross={len(self._cross_axes)}",
            f"stress={self._stress_level:.3f}",
            f"oscillating={len(self._oscillating)}",
        ]
        if self._oscillation_imminent:
            parts.append(f"vel_warn={len(self._oscillation_imminent)}")
        return f"PillarEquilibriumView({', '.join(parts)})"


class AdaptiveDampingController:
    """Automatically adjusts per-axis damping based on oscillation signals.

    Static damping creates a fundamental tension: low damping enables
    responsiveness but risks oscillation; high damping prevents oscillation
    but makes the system sluggish. This controller resolves that tension by
    dynamically adapting each axis's effective damping based on its recent
    behavior.

    Adaptation rules:
    - **Oscillation detected**: increase damping (make axis more rigid)
      to stabilize the feedback loop. Scales with oscillation severity.
    - **Oscillation imminent** (velocity-aware): when a TensionVelocityTracker
      is wired in, the controller can boost damping preemptively — before
      position stddev crosses the threshold — using velocity reversal rates
      as an early warning signal. This is a softer, preventive boost.
    - **Stability sustained**: gradually decrease damping (make axis more
      fluid) to restore responsiveness. Only after sustained stability
      (no oscillation for `stability_window` ticks).
    - **Bounds**: effective damping is always clamped to
      [damping_min, damping_max], preventing runaway rigidity or fluidity.

    Mathematical model:
    - On oscillation: d_eff = min(d_max, d_eff + boost_rate * severity)
    - On oscillation imminent (velocity): d_eff = min(d_max, d_eff + preemptive_boost_rate)
    - On stability: d_eff = max(d_min, d_eff - decay_rate)
    - Severity = oscillation_stddev / oscillation_threshold (capped at 2.0)
    - Base damping = axis.damping (the configured static value)

    Integration:
    - The controller is optional -- engines without it use static damping
    - When enabled, engine.apply_feedback() calls the controller after each
      feedback, and uses the controller's effective_damping() instead of
      the axis's static damping value
    - The controller's state is serialized alongside the engine for
      cross-session persistence
    - When both velocity tracking and adaptive damping are enabled, the
      engine wires the tracker to the controller for preemptive detection
    """

    __slots__ = (
        "_effective_damping",
        "_stability_counters",
        "_oscillation_severity",
        "_damping_min",
        "_damping_max",
        "_boost_rate",
        "_decay_rate",
        "_stability_window",
        "_oscillation_threshold",
        "_total_adaptations",
        "_velocity_tracker",
        "_preemptive_boost_rate",
        "_preemptive_oscillation_count",
        "_preemptive_threshold",
    )

    def __init__(
        self,
        *,
        damping_min: float = 0.1,
        damping_max: float = 0.95,
        boost_rate: float = 0.15,
        decay_rate: float = 0.02,
        stability_window: int = 6,
        oscillation_threshold: float = 0.6,
        velocity_tracker: TensionVelocityTracker | None = None,
        preemptive_boost_rate: float | None = None,
        preemptive_threshold: float = 0.4,
    ):
        """Initialize the adaptive damping controller.

        Args:
            damping_min: Minimum effective damping (prevent too-fluid axes).
            damping_max: Maximum effective damping (prevent too-rigid axes).
            boost_rate: How much to increase damping on oscillation.
            decay_rate: How much to decrease damping on sustained stability.
            stability_window: Ticks of stability before damping decays.
            oscillation_threshold: Stddev threshold matching the engine's.
            velocity_tracker: Optional TensionVelocityTracker for preemptive
                oscillation detection. When provided, the controller checks
                is_oscillation_imminent() after each feedback tick and boosts
                damping before position stddev confirms oscillation.
            preemptive_boost_rate: How much to increase damping when
                oscillation is imminent (velocity-based). Defaults to
                boost_rate / 2 (a gentler, preventive boost).
            preemptive_threshold: Reversal rate threshold for velocity-based
                oscillation prediction. Passed to
                TensionVelocityTracker.is_oscillation_imminent().
        """
        if not (0.0 <= damping_min <= 1.0):
            raise ValueError(f"damping_min must be in [0, 1], got {damping_min}")
        if not (0.0 <= damping_max <= 1.0):
            raise ValueError(f"damping_max must be in [0, 1], got {damping_max}")
        if damping_min >= damping_max:
            raise ValueError(
                f"damping_min ({damping_min}) must be < damping_max ({damping_max})"
            )
        if boost_rate <= 0:
            raise ValueError(f"boost_rate must be > 0, got {boost_rate}")
        if decay_rate <= 0:
            raise ValueError(f"decay_rate must be > 0, got {decay_rate}")
        if stability_window < 1:
            raise ValueError(f"stability_window must be >= 1, got {stability_window}")

        self._effective_damping: dict[TensionID, float] = {}
        self._stability_counters: dict[TensionID, int] = {}
        self._oscillation_severity: dict[TensionID, float] = {}
        self._damping_min = damping_min
        self._damping_max = damping_max
        self._boost_rate = boost_rate
        self._decay_rate = decay_rate
        self._stability_window = stability_window
        self._oscillation_threshold = oscillation_threshold
        self._total_adaptations: int = 0
        self._velocity_tracker: TensionVelocityTracker | None = velocity_tracker
        self._preemptive_boost_rate: float = (
            preemptive_boost_rate if preemptive_boost_rate is not None
            else boost_rate / 2.0
        )
        self._preemptive_oscillation_count: int = 0
        self._preemptive_threshold = preemptive_threshold

    # -- Public API --

    def register_axis(self, axis_id: TensionID, base_damping: float) -> None:
        """Register an axis with its static base damping.

        The effective damping starts at the base value. Call this
        for every axis before the first feedback tick.

        Args:
            axis_id: The tension axis identifier.
            base_damping: The axis's configured static damping.
        """
        self._effective_damping[axis_id] = float(base_damping)
        self._stability_counters[axis_id] = 0
        self._oscillation_severity[axis_id] = 0.0

    def unregister_axis(self, axis_id: TensionID) -> None:
        """Remove an axis from adaptive damping tracking."""
        self._effective_damping.pop(axis_id, None)
        self._stability_counters.pop(axis_id, None)
        self._oscillation_severity.pop(axis_id, None)

    def effective_damping(self, axis_id: TensionID) -> float:
        """Get the current effective damping for an axis.

        Returns the base (static) damping if the axis is not registered,
        so unregistered axes fall back gracefully.

        Args:
            axis_id: The tension axis identifier.

        Returns:
            The effective damping value in [damping_min, damping_max].
        """
        return self._effective_damping.get(axis_id, 0.3)

    def on_feedback(
        self,
        axis_id: TensionID,
        position_history: deque[float],
        base_damping: float,
    ) -> float:
        """Adapt damping for an axis after a feedback tick.

        Called by the engine after each apply_feedback(). Analyzes the
        axis's recent position history for oscillation and adjusts the
        effective damping accordingly. When a velocity tracker is bound,
        also checks for imminent oscillation (velocity-based) and applies
        a preemptive boost before position stddev confirms it.

        Priority order:
        1. Position-based oscillation (severity > 1.0): strongest boost
        2. Velocity-based imminent oscillation: gentler preemptive boost
        3. Stability: decay toward base damping

        Args:
            axis_id: The tension axis identifier.
            position_history: Recent position values for this axis.
            base_damping: The axis's static damping (fallback / anchor).

        Returns:
            The updated effective damping for this axis.
        """
        current = self._effective_damping.get(axis_id, base_damping)

        # Auto-register unregistered axis on first contact
        if axis_id not in self._effective_damping:
            self._effective_damping[axis_id] = base_damping
            self._stability_counters[axis_id] = 0
            self._oscillation_severity[axis_id] = 0.0

        # Compute oscillation severity from recent history
        severity = self._compute_severity(position_history)
        self._oscillation_severity[axis_id] = severity

        # Check velocity-based oscillation prediction
        is_preemptive = False
        if (
            severity <= 1.0
            and self._velocity_tracker is not None
            and axis_id in self._velocity_tracker._velocity
        ):
            is_preemptive = self._velocity_tracker.is_oscillation_imminent(
                axis_id, threshold=self._preemptive_threshold
            )

        if severity > 1.0:
            # Position-based oscillation detected -- boost damping
            boost = self._boost_rate * min(severity, 2.0)
            new_damping = min(self._damping_max, current + boost)
            self._effective_damping[axis_id] = new_damping
            self._stability_counters[axis_id] = 0
            self._total_adaptations += 1
        elif is_preemptive:
            # Velocity-based oscillation imminent -- preemptive boost
            new_damping = min(self._damping_max, current + self._preemptive_boost_rate)
            self._effective_damping[axis_id] = new_damping
            self._stability_counters[axis_id] = 0
            self._preemptive_oscillation_count += 1
            self._total_adaptations += 1
        else:
            # Stable -- accumulate stability counter
            counter = self._stability_counters.get(axis_id, 0) + 1
            self._stability_counters[axis_id] = counter

            if counter >= self._stability_window:
                # Sustained stability -- decay damping toward base
                new_damping = max(
                    self._damping_min,
                    current - self._decay_rate,
                )
                # Anchor: never decay below base_damping while above it
                if new_damping > base_damping:
                    new_damping = max(base_damping, new_damping - self._decay_rate)
                else:
                    # Already at or below base -- clamp to base (floor)
                    new_damping = base_damping
                self._effective_damping[axis_id] = new_damping
                self._total_adaptations += 1

        return self._effective_damping[axis_id]

    def reset(self) -> None:
        """Reset all effective damping to base values.

        Call this when the engine resets, to clear learned adaptations.
        Axes must be re-registered via register_axis() after reset.
        The velocity_tracker binding is preserved across resets.
        """
        self._effective_damping.clear()
        self._stability_counters.clear()
        self._oscillation_severity.clear()
        self._total_adaptations = 0
        self._preemptive_oscillation_count = 0

    # -- Properties --

    @property
    def total_adaptations(self) -> int:
        """Total number of damping adjustments made."""
        return self._total_adaptations

    @property
    def damping_min(self) -> float:
        return self._damping_min

    @property
    def damping_max(self) -> float:
        return self._damping_max

    @property
    def boost_rate(self) -> float:
        return self._boost_rate

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def stability_window(self) -> int:
        return self._stability_window

    @property
    def velocity_tracker(self) -> TensionVelocityTracker | None:
        """The velocity tracker bound to this controller, or None."""
        return self._velocity_tracker

    @velocity_tracker.setter
    def velocity_tracker(self, tracker: TensionVelocityTracker | None) -> None:
        """Bind or unbind a velocity tracker for preemptive detection."""
        self._velocity_tracker = tracker

    @property
    def preemptive_boost_rate(self) -> float:
        """The boost rate used when velocity-based oscillation is imminent."""
        return self._preemptive_boost_rate

    @property
    def preemptive_oscillation_count(self) -> int:
        """Total number of preemptive (velocity-based) oscillation boosts."""
        return self._preemptive_oscillation_count

    def get_stability_counter(self, axis_id: TensionID) -> int:
        """How many consecutive stable ticks an axis has had."""
        return self._stability_counters.get(axis_id, 0)

    def get_oscillation_severity(self, axis_id: TensionID) -> float:
        """Latest oscillation severity for an axis (0.0 = stable)."""
        return self._oscillation_severity.get(axis_id, 0.0)

    def all_effective_dampings(self) -> dict[TensionID, float]:
        """Snapshot of all current effective damping values."""
        return dict(self._effective_damping)

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize the controller state for cross-session persistence."""
        return {
            "effective_damping": dict(self._effective_damping),
            "stability_counters": dict(self._stability_counters),
            "oscillation_severity": dict(self._oscillation_severity),
            "damping_min": self._damping_min,
            "damping_max": self._damping_max,
            "boost_rate": self._boost_rate,
            "decay_rate": self._decay_rate,
            "stability_window": self._stability_window,
            "oscillation_threshold": self._oscillation_threshold,
            "total_adaptations": self._total_adaptations,
            "preemptive_oscillation_count": self._preemptive_oscillation_count,
            "preemptive_boost_rate": self._preemptive_boost_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdaptiveDampingController:
        """Deserialize controller state from a dict produced by to_dict()."""
        controller = cls(
            damping_min=data.get("damping_min", 0.1),
            damping_max=data.get("damping_max", 0.95),
            boost_rate=data.get("boost_rate", 0.15),
            decay_rate=data.get("decay_rate", 0.02),
            stability_window=data.get("stability_window", 6),
            oscillation_threshold=data.get("oscillation_threshold", 0.6),
            preemptive_boost_rate=data.get("preemptive_boost_rate"),
        )
        controller._effective_damping = data.get("effective_damping", {})
        controller._stability_counters = {
            k: int(v) for k, v in data.get("stability_counters", {}).items()
        }
        controller._oscillation_severity = data.get("oscillation_severity", {})
        controller._total_adaptations = int(data.get("total_adaptations", 0))
        controller._preemptive_oscillation_count = int(
            data.get("preemptive_oscillation_count", 0)
        )
        return controller

    # -- Internal --

    def _compute_severity(self, history: deque[float]) -> float:
        """Compute oscillation severity from position history.

        Returns:
            Severity ratio = stddev / oscillation_threshold.
            Values > 1.0 indicate oscillation. 0.0 = no variance.
        """
        if len(history) < 4:
            return 0.0
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        stddev = math.sqrt(variance)
        if self._oscillation_threshold <= 0:
            return 0.0
        return stddev / self._oscillation_threshold

    def __repr__(self) -> str:
        n_axes = len(self._effective_damping)
        return (
            f"AdaptiveDampingController("
            f"axes={n_axes}, "
            f"adaptations={self._total_adaptations}, "
            f"preemptive={self._preemptive_oscillation_count}, "
            f"range=[{self._damping_min:.2f}, {self._damping_max:.2f}])"
        )


class EquilibriumEngine:
    """Manages the dynamic balance of all tension axes for an agent.

    The engine receives Feedback signals from pillars, applies
    damped adjustments to tension positions, and provides the
    resulting TensionSnapshot for behavior modulation.

    Key design decisions:
    - Feedback is applied per-axis with per-axis damping
    - Default positions (set points) adapt slowly via outcome signals
    - Oscillation detection prevents runaway feedback loops
    - All operations produce immutable snapshots for audit trails
    - Optional adaptive damping controller adjusts damping dynamically
      based on oscillation signals (see AdaptiveDampingController)
    """

    # Default tension axes that every agent starts with
    DEFAULT_AXES: tuple[TensionAxis, ...] = (
        # --- Cognition tensions ---
        TensionAxis(
            id="explore_exploit",
            pillar=Pillar.COGNITION,
            pole_left="explore",
            pole_right="exploit",
            default_position=0.15,  # Slight exploit bias — safer default
            damping=0.4,
            learning_rate=0.05,
        ),
        TensionAxis(
            id="shallow_deep",
            pillar=Pillar.COGNITION,
            pole_left="shallow",
            pole_right="deep",
            default_position=-0.2,  # Slight shallow bias — start fast
            damping=0.5,
            learning_rate=0.08,
        ),
        TensionAxis(
            id="divergent_convergent",
            pillar=Pillar.COGNITION,
            pole_left="divergent",
            pole_right="convergent",
            default_position=0.3,  # Bias toward convergence
            damping=0.35,
            learning_rate=0.06,
        ),
        # --- Praxis tensions ---
        TensionAxis(
            id="autonomy_safety",
            pillar=Pillar.PRAXIS,
            pole_left="safe",
            pole_right="autonomous",
            default_position=-0.4,  # Safe by default
            damping=0.6,  # Harder to move — safety is sticky
            learning_rate=0.03,  # Slow to adapt — safety priors matter
        ),
        TensionAxis(
            id="sequential_parallel",
            pillar=Pillar.PRAXIS,
            pole_left="sequential",
            pole_right="parallel",
            default_position=0.1,
            damping=0.3,
            learning_rate=0.07,
        ),
        TensionAxis(
            id="verify_execute",
            pillar=Pillar.PRAXIS,
            pole_left="verify_heavy",
            pole_right="execute_fast",
            default_position=0.0,  # Perfect balance
            damping=0.4,
            learning_rate=0.06,
        ),
        # --- Mneme tensions ---
        TensionAxis(
            id="consolidate_prune",
            pillar=Pillar.MNEME,
            pole_left="consolidate",
            pole_right="prune",
            default_position=-0.1,  # Slight consolidate bias
            damping=0.45,
            learning_rate=0.05,
        ),
        TensionAxis(
            id="specific_general",
            pillar=Pillar.MNEME,
            pole_left="specific",
            pole_right="general",
            default_position=0.0,
            damping=0.4,
            learning_rate=0.06,
        ),
    )

    def __init__(
        self,
        axes: Sequence[TensionAxis] | None = None,
        *,
        oscillation_threshold: float = 0.6,
        oscillation_window: int = 8,
        adaptive_damping: AdaptiveDampingController | None = None,
        enable_adaptive_damping: bool = False,
        velocity_tracker: TensionVelocityTracker | None = None,
        enable_velocity_tracking: bool = False,
        event_log: TensionEventLog | None = None,
        enable_event_log: bool = False,
        event_log_max_events: int = 1000,
        feedback_cooldown: FeedbackCooldownManager | None = None,
        enable_feedback_cooldown: bool = False,
    ):
        """Initialize the equilibrium engine.

        Args:
            axes: Custom tension axes. Defaults to DEFAULT_AXES if None.
            oscillation_threshold: Max stddev before oscillation is declared.
            oscillation_window: Number of recent positions to track per axis.
            adaptive_damping: An existing AdaptiveDampingController to bind.
                If None and enable_adaptive_damping is True, a default
                controller is created automatically.
            enable_adaptive_damping: If True and adaptive_damping is None,
                create a default AdaptiveDampingController with the engine's
                oscillation_threshold. If False (default), the engine uses
                static per-axis damping from TensionAxis.damping.
            velocity_tracker: An existing TensionVelocityTracker to bind.
                If None and enable_velocity_tracking is True, a default
                tracker is created automatically.
            enable_velocity_tracking: If True and velocity_tracker is None,
                create a default TensionVelocityTracker. If False (default),
                velocity tracking is disabled for backward compatibility.
            feedback_cooldown: An existing FeedbackCooldownManager to bind.
                If None and enable_feedback_cooldown is True, a default
                manager is created automatically.
            enable_feedback_cooldown: If True and feedback_cooldown is None,
                create a default FeedbackCooldownManager. If False (default),
                feedback cooldown is disabled for backward compatibility.
        """
        axes = tuple(axes) if axes is not None else self.DEFAULT_AXES
        # Initialize each axis with its position set to its default_position,
        # ensuring the agent starts at homeostasis.
        axes = tuple(
            a.model_copy(update={"position": a.default_position}) for a in axes
        )
        self._axes: dict[TensionID, TensionAxis] = {a.id: a for a in axes}
        self._history: dict[TensionID, deque[float]] = {
            a.id: deque(maxlen=oscillation_window) for a in axes
        }
        self._oscillation_threshold = oscillation_threshold
        self._oscillation_window = oscillation_window
        self._feedback_count: int = 0
        self._oscillation_events: int = 0
        self._tick: int = 0

        # -- Adaptive damping integration --
        if adaptive_damping is not None:
            self._adaptive_damping = adaptive_damping
        elif enable_adaptive_damping:
            self._adaptive_damping = AdaptiveDampingController(
                oscillation_threshold=oscillation_threshold,
            )
        else:
            self._adaptive_damping = None

        # Register all axes with the adaptive damping controller
        if self._adaptive_damping is not None:
            for axis in self._axes.values():
                self._adaptive_damping.register_axis(axis.id, axis.damping)

        # -- Velocity tracking integration --
        if velocity_tracker is not None:
            self._velocity_tracker = velocity_tracker
        elif enable_velocity_tracking:
            self._velocity_tracker = TensionVelocityTracker()
        else:
            self._velocity_tracker = None

        # Register all axes with the velocity tracker
        if self._velocity_tracker is not None:
            for axis in self._axes.values():
                self._velocity_tracker.register_axis(axis.id)

        # Wire velocity tracker to adaptive damping controller
        # when both are enabled
        if (
            self._adaptive_damping is not None
            and self._velocity_tracker is not None
            and self._adaptive_damping.velocity_tracker is None
        ):
            self._adaptive_damping.velocity_tracker = self._velocity_tracker

        # -- Event log integration --
        if event_log is not None:
            self._event_log = event_log
        elif enable_event_log:
            self._event_log = TensionEventLog(max_events=event_log_max_events)
        else:
            self._event_log = None

        # -- Feedback cooldown integration --
        if feedback_cooldown is not None:
            self._feedback_cooldown = feedback_cooldown
        elif enable_feedback_cooldown:
            self._feedback_cooldown = FeedbackCooldownManager()
        else:
            self._feedback_cooldown = None

        # ── Public API ───────────────────────────────────────────────

    @property
    def axes(self) -> tuple[TensionAxis, ...]:
        """Current state of all tension axes (immutable)."""
        return tuple(self._axes.values())

    @property
    def total_feedback_received(self) -> int:
        return self._feedback_count

    @property
    def total_oscillation_events(self) -> int:
        return self._oscillation_events

    @property
    def event_log(self) -> TensionEventLog | None:
        """The tension event log, or None if not enabled."""
        return self._event_log

    @property
    def feedback_cooldown(self) -> FeedbackCooldownManager | None:
        """The feedback cooldown manager, or None if not enabled."""
        return self._feedback_cooldown


    def snapshot(self, agent_id=None, trigger=None) -> TensionSnapshot:
        """Capture the current tension state."""
        return TensionSnapshot(
            axes=frozenset(self._axes.values()),
            agent_id=agent_id,
            trigger=trigger,
        )

    def apply_feedback(self, feedback: Feedback) -> TensionAxis:
        """Process a single feedback signal and return the updated axis.

        This is THE central method of the equilibrium engine. Every
        pillar feeds back through this method, and the resulting
        tension shifts modulate future behavior.

        Args:
            feedback: Structured feedback from a pillar.

        Returns:
            The updated TensionAxis after applying the feedback.

        Raises:
            KeyError: If the feedback references an unknown axis.
            TensionOscillationError: If oscillation is detected.
        """
        axis = self._axes.get(feedback.tension_axis_id)
        if axis is None:
            raise KeyError(
                f"Unknown tension axis '{feedback.tension_axis_id}'. "
                f"Known axes: {list(self._axes)}"
            )

        # Weight the signal by confidence — low-confidence feedback
        # moves the needle less
        effective_delta = feedback.signal * feedback.confidence

        # Apply feedback cooldown damping if enabled
        if self._feedback_cooldown is not None:
            cooldown_mult = self._feedback_cooldown.check_and_apply(
                feedback.tension_axis_id, feedback.source, self._feedback_count
            )
            effective_delta *= cooldown_mult

        # Apply adaptive damping if controller is enabled
        if self._adaptive_damping is not None:
            adaptive_d = self._adaptive_damping.effective_damping(
                feedback.tension_axis_id
            )
            axis = axis.model_copy(update={"damping": adaptive_d})

        # Apply the adjustment (damping is internal to TensionAxis.adjust)
        new_axis = axis.adjust(effective_delta)

        # Track history for oscillation detection
        self._history[feedback.tension_axis_id].append(new_axis.position)
        self._check_oscillation(feedback.tension_axis_id)

        # Store and count
        self._axes[feedback.tension_axis_id] = new_axis
        self._feedback_count += 1

        # Feed velocity tracker after position update
        if self._velocity_tracker is not None:
            self._velocity_tracker.on_position_update(
                feedback.tension_axis_id,
                new_axis.position,
                new_axis.default_position,
            )

        # Notify adaptive damping controller after feedback
        if self._adaptive_damping is not None:
            self._adaptive_damping.on_feedback(
                feedback.tension_axis_id,
                self._history[feedback.tension_axis_id],
                self._axes[feedback.tension_axis_id].damping,
            )

        # Record event in the tension event log
        if self._event_log is not None:
            self._tick += 1
            self._event_log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id=feedback.tension_axis_id,
                source_pillar=feedback.source,
                position_before=axis.position,
                position_after=new_axis.position,
                delta=effective_delta,
                confidence=feedback.confidence,
                tick=self._tick,
            )

        return new_axis

    def apply_feedback_batch(self, feedbacks: Sequence[Feedback]) -> TensionSnapshot:
        """Apply multiple feedback signals atomically.

        All adjustments are computed before any axis is updated,
        preventing ordering artifacts.

        When adaptive damping is enabled, each axis uses its
        current effective damping rather than its static value.
        The controller is notified after all updates are applied.
        """
        # Phase 1: compute all new positions
        updates: dict[TensionID, TensionAxis] = {}
        for fb in feedbacks:
            axis = self._axes.get(fb.tension_axis_id)
            if axis is None:
                continue
            effective_delta = fb.signal * fb.confidence
            # Apply feedback cooldown damping if enabled
            if self._feedback_cooldown is not None:
                cooldown_mult = self._feedback_cooldown.check_and_apply(
                    fb.tension_axis_id, fb.source, self._feedback_count
                )
                effective_delta *= cooldown_mult
            # Apply adaptive damping if available
            if self._adaptive_damping is not None:
                adaptive_d = self._adaptive_damping.effective_damping(
                    fb.tension_axis_id
                )
                axis = axis.model_copy(update={"damping": adaptive_d})
            updates[fb.tension_axis_id] = axis.adjust(effective_delta)

        # Phase 2: apply all updates
        for axis_id, new_axis in updates.items():
            self._axes[axis_id] = new_axis
            self._history[axis_id].append(new_axis.position)
            self._feedback_count += 1
            self._check_oscillation(axis_id)

        # Phase 3: feed velocity tracker for each updated axis
        if self._velocity_tracker is not None:
            for axis_id, new_axis in updates.items():
                self._velocity_tracker.on_position_update(
                    axis_id,
                    new_axis.position,
                    new_axis.default_position,
                )

        # Phase 4: notify adaptive damping controller for each updated axis
        if self._adaptive_damping is not None:
            for axis_id in updates:
                self._adaptive_damping.on_feedback(
                    axis_id,
                    self._history[axis_id],
                    self._axes[axis_id].damping,
                )

        # Phase 5: record events in the tension event log
        if self._event_log is not None:
            for fb in feedbacks:
                new_axis = updates.get(fb.tension_axis_id)
                if new_axis is None:
                    continue
                old_axis = self._axes.get(fb.tension_axis_id)
                # We need position_before from before the update.
                # Since updates are already applied, we use the delta to reconstruct.
                pos_after = new_axis.position
                effective_delta = fb.signal * fb.confidence
                pos_before = pos_after - effective_delta * (1 - new_axis.damping)
                self._tick += 1
                self._event_log.record(
                    event_type=TensionEventType.FEEDBACK_APPLIED,
                    axis_id=fb.tension_axis_id,
                    source_pillar=fb.source,
                    position_before=pos_before,
                    position_after=pos_after,
                    delta=effective_delta,
                    confidence=fb.confidence,
                    tick=self._tick,
                )

        return self.snapshot()

    def adjust_default(self, axis_id: TensionID, outcome_signal: float) -> TensionAxis:
        """Slowly adapt the default position (set point) based on outcomes.

        Unlike apply_feedback which moves the current position, this
        moves the homeostasis target — the position the system returns
        to when not stimulated. This is how the agent *learns*.
        """
        axis = self._axes.get(axis_id)
        if axis is None:
            raise KeyError(f"Unknown tension axis '{axis_id}'")

        old_default = axis.default_position
        shift = outcome_signal * axis.learning_rate
        new_default = axis.default_position + shift
        new_default = max(-1.0, min(1.0, new_default))

        new_axis = axis.model_copy(update={"default_position": new_default})
        self._axes[axis_id] = new_axis

        # Record event in the tension event log
        if self._event_log is not None:
            self._tick += 1
            self._event_log.record(
                event_type=TensionEventType.DEFAULT_ADJUSTED,
                axis_id=axis_id,
                source_pillar=axis.pillar,
                position_before=old_default,
                position_after=new_default,
                delta=shift,
                confidence=1.0,
                tick=self._tick,
            )

        return new_axis

    def get_behavior_profile(self) -> dict[TensionID, float]:
        """Extract current positions as a flat dict for behavior modulation.

        This is what pillar implementations consume to decide *how*
        to operate — e.g., a cognition module reads 'shallow_deep' to
        decide how many reasoning steps to take.
        """
        return {axis_id: axis.position for axis_id, axis in self._axes.items()}

    def reset(self, *, keep_defaults: bool = True) -> None:
        """Return all axes to their default positions (or absolute zero)."""
        for axis_id, axis in self._axes.items():
            target = axis.default_position if keep_defaults else 0.0
            self._axes[axis_id] = axis.model_copy(update={"position": target})
            self._history[axis_id].clear()
        # Reset adaptive damping if present
        if self._adaptive_damping is not None:
            self._adaptive_damping.reset()
            for axis in self._axes.values():
                self._adaptive_damping.register_axis(axis.id, axis.damping)
            # Re-wire velocity tracker if both are present
            if self._velocity_tracker is not None:
                self._adaptive_damping.velocity_tracker = self._velocity_tracker

        # Reset velocity tracker if present
        if self._velocity_tracker is not None:
            self._velocity_tracker.reset()
            for axis in self._axes.values():
                self._velocity_tracker.register_axis(axis.id)

        # Record reset event in the tension event log
        if self._event_log is not None:
            self._tick += 1
            self._event_log.record(
                event_type=TensionEventType.RESET,
                axis_id="",
                source_pillar=Pillar.COGNITION,  # Engine-wide event
                position_before=0.0,
                position_after=0.0,
                delta=0.0,
                confidence=0.0,
                tick=self._tick,
            )

    def tension_distance(self) -> float:
        """Aggregate drift from homeostasis — how 'stressed' the agent is.

        Returns the RMS distance of all axes from their defaults.
        High values indicate the agent is far from its learned
        equilibrium and may be in an unfamiliar situation.
        """
        if not self._axes:
            return 0.0
        squared = sum(a.distance_from_default() ** 2 for a in self._axes.values())
        return math.sqrt(squared / len(self._axes))

    def get_axis(self, axis_id: TensionID) -> TensionAxis | None:
        """Look up a single axis by ID."""
        return self._axes.get(axis_id)

    @property
    def adaptive_damping(self) -> AdaptiveDampingController | None:
        """The adaptive damping controller, or None if not enabled."""
        return self._adaptive_damping

    @property
    def velocity_tracker(self) -> TensionVelocityTracker | None:
        """The velocity tracker, or None if not enabled."""
        return self._velocity_tracker

    def view_for(self, pillar: Pillar) -> PillarEquilibriumView:
        """Create a structured read-only view for a specific pillar.

        This is the pull-side complement to push-side Feedback. Each pillar
        calls this at the start of its tick to get a current snapshot of:
        - Its own tension axes (directly modulate behavior)
        - Cross-pillar axes (indirect influence)
        - Aggregate stress level (how far from homeostasis)
        - Per-axis drift from set points
        - Oscillation warnings

        Args:
            pillar: Which pillar is requesting the view.

        Returns:
            A PillarEquilibriumView scoped to the requesting pillar.
        """
        return PillarEquilibriumView(
            pillar=pillar,
            axes=self._axes,
            history=self._history,
            oscillation_threshold=self._oscillation_threshold,
            velocity_tracker=self._velocity_tracker,
            event_log=self._event_log,
            feedback_cooldown=self._feedback_cooldown,
        )

    # ── Internal ─────────────────────────────────────────────────

    def _check_oscillation(self, axis_id: TensionID) -> None:
        """Detect oscillatory behavior on a tension axis.

        Oscillation means the feedback loop is unstable — the axis is
        swinging back and forth too rapidly. This can happen when
        two pillars send contradictory feedback.
        """
        history = self._history[axis_id]
        if len(history) < self._oscillation_window:
            return

        # Compute standard deviation of recent positions.
        # High stddev in a short window = rapid movement = oscillation.
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        stddev = math.sqrt(variance)

        if stddev > self._oscillation_threshold:
            self._oscillation_events += 1

            # Record oscillation detection event in the tension event log
            if self._event_log is not None:
                axis = self._axes.get(axis_id)
                self._tick += 1
                self._event_log.record(
                    event_type=TensionEventType.OSCILLATION_DETECTED,
                    axis_id=axis_id,
                    source_pillar=axis.pillar if axis else Pillar.COGNITION,
                    position_before=mean,
                    position_after=history[-1] if history else 0.0,
                    delta=stddev,
                    confidence=0.0,
                    tick=self._tick,
                )
            # Don't raise — just increment the counter. The caller
            # can decide whether to escalate.

    # ── Serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the equilibrium engine state for cross-session persistence.

        Saves all tension axes with current positions, oscillation history,
        feedback count, and configuration parameters. The agent can be
        resumed with identical equilibrium state — no learning lost.

        Returns:
            A JSON-serializable dict of all engine state.
        """
        return {
            "axes": [
                {
                    "id": a.id,
                    "pillar": a.pillar.value,
                    "pole_left": a.pole_left,
                    "pole_right": a.pole_right,
                    "position": a.position,
                    "default_position": a.default_position,
                    "damping": a.damping,
                    "learning_rate": a.learning_rate,
                }
                for a in self._axes.values()
            ],
            "history": {
                axis_id: list(hist)
                for axis_id, hist in self._history.items()
            },
            "oscillation_threshold": self._oscillation_threshold,
            "oscillation_window": self._oscillation_window,
            "feedback_count": self._feedback_count,
            "oscillation_events": self._oscillation_events,
            "adaptive_damping_state": (
                self._adaptive_damping.to_dict() if self._adaptive_damping is not None else None
            ),
            "velocity_tracker_state": (
                self._velocity_tracker.to_dict() if self._velocity_tracker is not None else None
            ),
            "event_log_state": (
                self._event_log.to_dict() if self._event_log is not None else None
            ),
            "feedback_cooldown_state": (
                self._feedback_cooldown.to_dict() if self._feedback_cooldown is not None else None
            ),
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EquilibriumEngine:
        """Deserialize equilibrium engine state.

        Reconstructs all tension axes with saved positions, restores
        oscillation history, and resets feedback counters from saved
        values.

        Args:
            data: A dict produced by to_dict().

        Returns:
            A reconstructed EquilibriumEngine with full tension state.
        """
        from isonome.types import Pillar, TensionAxis

        # Rebuild axes from saved data
        saved_axes = data.get("axes", [])
        axes = []
        for a_data in saved_axes:
            try:
                pillar = Pillar(a_data.get("pillar", "cognition"))
            except ValueError:
                pillar = Pillar.COGNITION
            axis = TensionAxis(
                id=a_data.get("id", "unknown"),
                pillar=pillar,
                pole_left=a_data.get("pole_left", "left"),
                pole_right=a_data.get("pole_right", "right"),
                position=a_data.get("position", 0.0),
                default_position=a_data.get("default_position", 0.0),
                damping=a_data.get("damping", 0.3),
                learning_rate=a_data.get("learning_rate", 0.1),
            )
            axes.append(axis)

        engine = cls(
            axes=axes,
            oscillation_threshold=data.get("oscillation_threshold", 0.6),
            oscillation_window=data.get("oscillation_window", 8),
        )

        # Restore adaptive damping controller if present
        ad_state = data.get("adaptive_damping_state")
        if ad_state is not None:
            controller = AdaptiveDampingController.from_dict(ad_state)
            engine._adaptive_damping = controller
            # Re-register axes with the restored controller
            for axis in engine._axes.values():
                if axis.id not in controller._effective_damping:
                    controller.register_axis(axis.id, axis.damping)

        # Restore velocity tracker if present
        vt_state = data.get("velocity_tracker_state")
        if vt_state is not None:
            tracker = TensionVelocityTracker.from_dict(vt_state)
            engine._velocity_tracker = tracker
            # Re-register any axes not in the tracker
            for axis in engine._axes.values():
                if axis.id not in tracker._velocity:
                    tracker.register_axis(axis.id)

        # Wire velocity tracker to adaptive damping controller
        # when both are restored
        if engine._adaptive_damping is not None and engine._velocity_tracker is not None:
            engine._adaptive_damping.velocity_tracker = engine._velocity_tracker

        # Restore oscillation history
        saved_history = data.get("history", {})
        for axis_id, hist_list in saved_history.items():
            if axis_id in engine._history:
                engine._history[axis_id].clear()
                for val in hist_list:
                    engine._history[axis_id].append(float(val))

        # Restore counters
        # Restore positions (__init__ resets to default_position, so we override)
        saved_axes = data.get("axes", [])
        for a_data in saved_axes:
            aid = a_data.get("id", "")
            saved_pos = float(a_data.get("position", 0.0))
            if aid in engine._axes:
                engine._axes[aid] = engine._axes[aid].model_copy(
                    update={"position": saved_pos}
                )

        engine._feedback_count = int(data.get("feedback_count", 0))
        engine._oscillation_events = int(data.get("oscillation_events", 0))

        # Restore event log if present
        el_state = data.get("event_log_state")
        if el_state is not None:
            from isonome.equilibrium.event_log import TensionEventLog
            engine._event_log = TensionEventLog.from_dict(el_state)

        # Restore feedback cooldown if present
        fc_state = data.get("feedback_cooldown_state")
        if fc_state is not None:
            engine._feedback_cooldown = FeedbackCooldownManager.from_dict(fc_state)

        return engine
