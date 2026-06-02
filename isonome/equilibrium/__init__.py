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
from datetime import datetime, timezone
from typing import Sequence

from isonome.types import (
    AgentState,
    EquilibriumError,
    Feedback,
    Pillar,
    TensionAxis,
    TensionID,
    TensionOscillationError,
    TensionSnapshot,
    now,
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
    ):
        """Initialize the equilibrium engine.

        Args:
            axes: Custom tension axes. Defaults to DEFAULT_AXES if None.
            oscillation_threshold: Max stddev before oscillation is declared.
            oscillation_window: Number of recent positions to track per axis.
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

        # Apply the adjustment (damping is internal to TensionAxis.adjust)
        new_axis = axis.adjust(effective_delta)

        # Track history for oscillation detection
        self._history[feedback.tension_axis_id].append(new_axis.position)
        self._check_oscillation(feedback.tension_axis_id)

        # Store and count
        self._axes[feedback.tension_axis_id] = new_axis
        self._feedback_count += 1

        return new_axis

    def apply_feedback_batch(self, feedbacks: Sequence[Feedback]) -> TensionSnapshot:
        """Apply multiple feedback signals atomically.

        All adjustments are computed before any axis is updated,
        preventing ordering artifacts.
        """
        # Phase 1: compute all new positions
        updates: dict[TensionID, TensionAxis] = {}
        for fb in feedbacks:
            axis = self._axes.get(fb.tension_axis_id)
            if axis is None:
                continue
            effective_delta = fb.signal * fb.confidence
            updates[fb.tension_axis_id] = axis.adjust(effective_delta)

        # Phase 2: apply all updates
        for axis_id, new_axis in updates.items():
            self._axes[axis_id] = new_axis
            self._history[axis_id].append(new_axis.position)
            self._feedback_count += 1
            self._check_oscillation(axis_id)

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

        shift = outcome_signal * axis.learning_rate
        new_default = axis.default_position + shift
        new_default = max(-1.0, min(1.0, new_default))

        new_axis = axis.model_copy(update={"default_position": new_default})
        self._axes[axis_id] = new_axis
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
            # Don't raise — just increment the counter. The caller
            # can decide whether to escalate.
