"""Equilibrium Convergence Detector — tracks whether the engine converges or diverges.

Detects the overall direction of the equilibrium engine's state:
- **CONVERGING**: Majority of axes are moving toward their default positions
- **DIVERGING**: Majority of axes are moving away from their defaults
- **STABLE**: Most axes are near their defaults with negligible velocity
- **UNKNOWN**: Insufficient data (no velocity tracker, or no velocity data)

The detector uses per-axis velocity data when available (from TensionVelocityTracker),
and falls back to a position-delta heuristic when velocity tracking is disabled.

Mathematical model:
- Per-axis: velocity_sign × sign(default - position) determines direction
  - Positive product → axis is converging (heading home)
  - Negative product → axis is diverging (drifting away)
  - Near-zero velocity → axis is stable
- Overall: majority vote across all axes
- Convergence rate: mean of per-axis velocity projections toward defaults

Integration:
- Opt-in via EquilibriumEngine(enable_convergence_detection=True)
- Automatic detection after each apply_feedback() call
- PillarEquilibriumView exposes convergence_status
- Serialization round-trip preserves detector configuration and history
"""

from __future__ import annotations

import math
from collections import deque
from enum import StrEnum
from typing import Any

from isonome.types import TensionID


class ConvergenceStatus(StrEnum):
    """Direction of equilibrium state movement."""

    CONVERGING = "converging"
    DIVERGING = "diverging"
    STABLE = "stable"
    UNKNOWN = "unknown"


class ConvergenceRecord:
    """A single recorded convergence detection result.

    Attributes:
        tick: The engine tick when this detection occurred.
        status: The overall convergence status.
        convergence_rate: Mean velocity toward defaults.
            Positive = converging, negative = diverging, near-zero = stable.
        n_converging: Number of axes converging.
        n_diverging: Number of axes diverging.
        n_stable: Number of axes stable.
    """

    __slots__ = (
        "tick",
        "status",
        "convergence_rate",
        "n_converging",
        "n_diverging",
        "n_stable",
    )

    def __init__(
        self,
        tick: int,
        status: ConvergenceStatus,
        convergence_rate: float,
        n_converging: int,
        n_diverging: int,
        n_stable: int,
    ):
        self.tick = tick
        self.status = status
        self.convergence_rate = convergence_rate
        self.n_converging = n_converging
        self.n_diverging = n_diverging
        self.n_stable = n_stable

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to a dict."""
        return {
            "tick": self.tick,
            "status": self.status.value,
            "convergence_rate": self.convergence_rate,
            "n_converging": self.n_converging,
            "n_diverging": self.n_diverging,
            "n_stable": self.n_stable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceRecord:
        """Deserialize record from a dict."""
        return cls(
            tick=data["tick"],
            status=ConvergenceStatus(data["status"]),
            convergence_rate=data["convergence_rate"],
            n_converging=data["n_converging"],
            n_diverging=data["n_diverging"],
            n_stable=data["n_stable"],
        )

    def __repr__(self) -> str:
        return (
            f"ConvergenceRecord({self.status.name}, "
            f"tick={self.tick}, "
            f"rate={self.convergence_rate:+.4f}, "
            f"conv={self.n_converging}/div={self.n_diverging}/stab={self.n_stable})"
        )


class ConvergenceDetector:
    """Detects whether the equilibrium engine is converging or diverging.

    Examines per-axis velocity (or position deltas) to determine if the
    system is moving toward homeostasis (converging), drifting away
    (diverging), or sitting near equilibrium (stable).

    The detector maintains a bounded history of convergence records,
    enabling trend analysis and audit trails.

    Usage::

        from isonome.equilibrium.convergence import ConvergenceDetector

        # Standalone
        detector = ConvergenceDetector()
        record = detector.detect(engine)
        print(record.status)  # ConvergenceStatus.CONVERGING

        # Via engine
        engine = EquilibriumEngine(enable_convergence_detection=True)
        result = engine.compute_convergence()
    """

    __slots__ = (
        "_max_history",
        "_convergence_threshold",
        "_history",
        "_total_detections",
        "_current_status",
        "_velocity",
        "_positions",
        "_defaults",
    )

    def __init__(
        self,
        *,
        max_history: int = 100,
        convergence_threshold: float = 0.001,
    ):
        """Initialize the convergence detector.

        Args:
            max_history: Maximum number of ConvergenceRecords to retain.
                When exceeded, oldest records are evicted (FIFO).
                Must be >= 1.
            convergence_threshold: Minimum |velocity| for an axis to be
                classified as converging or diverging. Below this threshold,
                the axis is considered stable. Must be >= 0.

        Raises:
            ValueError: If max_history < 1 or convergence_threshold < 0.
        """
        if max_history < 1:
            raise ValueError(f"max_history must be >= 1, got {max_history}")
        if convergence_threshold < 0:
            raise ValueError(
                f"convergence_threshold must be >= 0, got {convergence_threshold}"
            )

        self._max_history = max_history
        self._convergence_threshold = convergence_threshold
        self._history: deque[ConvergenceRecord] = deque(maxlen=max_history)
        self._total_detections: int = 0
        self._current_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN
        self._velocity: dict[TensionID, float | None] = {}
        self._positions: dict[TensionID, float] = {}
        self._defaults: dict[TensionID, float] = {}

    # ── Properties ─────────────────────────────────────────────

    @property
    def max_history(self) -> int:
        """Maximum number of convergence records retained."""
        return self._max_history

    @property
    def convergence_threshold(self) -> float:
        """Minimum velocity magnitude for non-STABLE classification."""
        return self._convergence_threshold

    @property
    def history_size(self) -> int:
        """Number of convergence records currently stored."""
        return len(self._history)

    @property
    def total_detections(self) -> int:
        """Total detections ever recorded (including evicted)."""
        return self._total_detections

    @property
    def current_status(self) -> ConvergenceStatus:
        """The most recently computed convergence status.

        Updated automatically after each on_position_update() call,
        or after each detect() / compute() call. Returns UNKNOWN
        if no detection has been performed yet.
        """
        return self._current_status

    # ── Per-axis classification ────────────────────────────────

    def classify_axis(
        self,
        axis_id: TensionID,
        position: float,
        default_position: float,
        velocity: float | None,
    ) -> ConvergenceStatus:
        """Classify a single axis's convergence direction.

        The classification is based on the relationship between the axis's
        velocity and the direction toward its default position:
        - If velocity is toward the default → CONVERGING
        - If velocity is away from the default → DIVERGING
        - If |velocity| < convergence_threshold → STABLE
        - If velocity is None → UNKNOWN

        Args:
            axis_id: The tension axis identifier.
            position: Current axis position.
            default_position: Homeostasis target for the axis.
            velocity: Current velocity (position delta per tick).
                None if velocity data is unavailable.

        Returns:
            ConvergenceStatus for this axis.
        """
        if velocity is None:
            # If axis is at or very near default, treat as STABLE
            # even without velocity data — no movement is needed
            if abs(position - default_position) < self._convergence_threshold:
                return ConvergenceStatus.STABLE
            return ConvergenceStatus.UNKNOWN

        abs_velocity = abs(velocity)
        if abs_velocity < self._convergence_threshold:
            return ConvergenceStatus.STABLE

        # Direction toward default from current position
        direction_to_default = default_position - position

        # If at default already, any movement is diverging
        if abs(direction_to_default) < 1e-9:
            # Position is at or very near default — any movement away
            # is diverging, but if velocity is tiny it's already STABLE
            # (handled above). So non-tiny velocity from default = diverging.
            return ConvergenceStatus.DIVERGING

        # Positive product = velocity aligns with direction to default → converging
        # Negative product = velocity opposes direction to default → diverging
        product = velocity * direction_to_default

        if product > 0:
            return ConvergenceStatus.CONVERGING
        else:
            return ConvergenceStatus.DIVERGING

    # ── Engine-level detection ─────────────────────────────────

    def detect(self, engine: 'EquilibriumEngine') -> ConvergenceRecord:  # type: ignore[name-defined]
        """Run convergence detection on an engine.

        Examines all axes, classifies each, and determines the overall
        convergence status by majority vote. Also computes the mean
        convergence rate across all axes.

        Args:
            engine: The EquilibriumEngine to analyze.

        Returns:
            A ConvergenceRecord with the detection results.
        """
        n_converging = 0
        n_diverging = 0
        n_stable = 0
        n_unknown = 0
        rate_sum = 0.0

        for axis in engine.axes:
            # Try to get velocity from tracker
            velocity: float | None = None
            if engine.velocity_tracker is not None:
                velocity = engine.velocity_tracker.get_velocity(axis.id)
            else:
                # Fallback: use position delta from history
                velocity = self._estimate_velocity(engine, axis.id)

            status = self.classify_axis(
                axis_id=axis.id,
                position=axis.position,
                default_position=axis.default_position,
                velocity=velocity,
            )

            if status == ConvergenceStatus.CONVERGING:
                n_converging += 1
                # Rate contribution: velocity projected toward default
                if velocity is not None:
                    direction = axis.default_position - axis.position
                    if abs(direction) > 1e-9:
                        rate_sum += abs(velocity) * (1 if velocity * direction > 0 else -1)
                    else:
                        rate_sum += abs(velocity)
            elif status == ConvergenceStatus.DIVERGING:
                n_diverging += 1
                if velocity is not None:
                    direction = axis.default_position - axis.position
                    if abs(direction) > 1e-9:
                        rate_sum -= abs(velocity) * (1 if velocity * direction < 0 else -1)
                    else:
                        rate_sum -= abs(velocity)
            elif status == ConvergenceStatus.STABLE:
                n_stable += 1
            else:
                n_unknown += 1

        # Compute mean convergence rate
        n_axes = len(engine.axes)
        convergence_rate = rate_sum / n_axes if n_axes > 0 else 0.0

        # Determine overall status by majority
        overall = self._determine_overall(
            n_converging, n_diverging, n_stable, n_unknown
        )

        # Get current tick from engine (approximate)
        tick = getattr(engine, "_tick", self._total_detections)

        record = ConvergenceRecord(
            tick=tick,
            status=overall,
            convergence_rate=convergence_rate,
            n_converging=n_converging,
            n_diverging=n_diverging,
            n_stable=n_stable,
        )

        self._history.append(record)
        self._total_detections += 1

        return record

    def per_axis_status(self, engine: 'EquilibriumEngine') -> dict[TensionID, ConvergenceStatus]:  # type: ignore[name-defined]
        """Return per-axis convergence status for all axes.

        Args:
            engine: The EquilibriumEngine to analyze.

        Returns:
            Dict mapping axis ID to ConvergenceStatus.
        """
        result: dict[TensionID, ConvergenceStatus] = {}
        for axis in engine.axes:
            velocity: float | None = None
            if engine.velocity_tracker is not None:
                velocity = engine.velocity_tracker.get_velocity(axis.id)
            else:
                velocity = self._estimate_velocity(engine, axis.id)

            result[axis.id] = self.classify_axis(
                axis_id=axis.id,
                position=axis.position,
                default_position=axis.default_position,
                velocity=velocity,
            )
        return result

    # ── History access ─────────────────────────────────────────

    def recent_history(self, limit: int = 20) -> list[ConvergenceRecord]:
        """Return the most recent convergence records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of recent ConvergenceRecord objects in chronological order.
        """
        records = list(self._history)
        return records[-limit:] if len(records) > limit else records

    def convergence_trend(self) -> list[float]:
        """Return the convergence rate trend from history.

        Returns:
            List of convergence rates in chronological order.
            Positive = system was converging, negative = diverging.
        """
        return [r.convergence_rate for r in self._history]

    # ── Incremental update (called by EquilibriumEngine) ──────

    def on_position_update(
        self,
        axis_id: TensionID,
        position: float,
        default_position: float,
        velocity: float | None = None,
    ) -> None:
        """Record a position update for an axis and recompute status.

        Called by EquilibriumEngine.apply_feedback() after each
        feedback application. Stores the latest position/default/velocity
        for each axis, then performs a lightweight convergence analysis
        across all known axes.

        Args:
            axis_id: The tension axis that was updated.
            position: New axis position.
            default_position: Axis default (homeostasis target).
            velocity: Current velocity if available. None if unknown.
        """
        self._positions[axis_id] = position
        self._defaults[axis_id] = default_position
        if velocity is not None:
            self._velocity[axis_id] = velocity
        # Don't erase previously known velocity — keep last known value

        # Lightweight recompute across all known axes
        n_converging = 0
        n_diverging = 0
        n_stable = 0
        n_unknown = 0
        rate_sum = 0.0

        for aid in self._positions:
            vel = self._velocity.get(aid)
            pos = self._positions[aid]
            dfl = self._defaults[aid]

            status = self.classify_axis(
                axis_id=aid,
                position=pos,
                default_position=dfl,
                velocity=vel,
            )

            if status == ConvergenceStatus.CONVERGING:
                n_converging += 1
                if vel is not None:
                    direction = dfl - pos
                    if abs(direction) > 1e-9:
                        rate_sum += vel * direction / abs(direction)
                    else:
                        rate_sum += abs(vel)
            elif status == ConvergenceStatus.DIVERGING:
                n_diverging += 1
                if vel is not None:
                    direction = dfl - pos
                    if abs(direction) > 1e-9:
                        rate_sum -= abs(vel)
                    else:
                        rate_sum -= abs(vel)
            elif status == ConvergenceStatus.STABLE:
                n_stable += 1
            else:
                n_unknown += 1

        overall = self._determine_overall(
            n_converging, n_diverging, n_stable, n_unknown
        )
        self._current_status = overall

        # Record a ConvergenceRecord for each auto-update
        # (so history_size grows with each feedback application)
        n_axes = len(self._positions) if self._positions else 1
        convergence_rate = rate_sum / n_axes if n_axes > 0 else 0.0

        record = ConvergenceRecord(
            tick=self._total_detections,
            status=overall,
            convergence_rate=convergence_rate,
            n_converging=n_converging,
            n_diverging=n_diverging,
            n_stable=n_stable,
        )
        self._history.append(record)
        self._total_detections += 1

    # ── Standalone compute (no engine reference needed) ───────

    def compute(
        self,
        positions: dict[TensionID, float],
        defaults: dict[TensionID, float],
    ) -> ConvergenceRecord:
        """Compute convergence from position/default dicts.

        This is the standalone interface that doesn't require an
        EquilibriumEngine reference. It uses stored velocity data
        when available, and estimates velocity from position deltas
        otherwise.

        Args:
            positions: Dict of axis_id → current position.
            defaults: Dict of axis_id → default position.

        Returns:
            A ConvergenceRecord with the detection results.
        """
        n_converging = 0
        n_diverging = 0
        n_stable = 0
        n_unknown = 0
        rate_sum = 0.0

        for aid in positions:
            pos = positions[aid]
            dfl = defaults.get(aid, 0.0)
            vel = self._velocity.get(aid)

            status = self.classify_axis(
                axis_id=aid,
                position=pos,
                default_position=dfl,
                velocity=vel,
            )

            if status == ConvergenceStatus.CONVERGING:
                n_converging += 1
                if vel is not None:
                    direction = dfl - pos
                    if abs(direction) > 1e-9:
                        rate_sum += vel * direction / abs(direction)
                    else:
                        rate_sum += abs(vel)
            elif status == ConvergenceStatus.DIVERGING:
                n_diverging += 1
                if vel is not None:
                    direction = dfl - pos
                    if abs(direction) > 1e-9:
                        rate_sum -= abs(vel)
                    else:
                        rate_sum -= abs(vel)
            elif status == ConvergenceStatus.STABLE:
                n_stable += 1
            else:
                n_unknown += 1

        n_axes = len(positions) if positions else 1
        convergence_rate = rate_sum / n_axes if n_axes > 0 else 0.0

        overall = self._determine_overall(
            n_converging, n_diverging, n_stable, n_unknown
        )

        self._current_status = overall

        record = ConvergenceRecord(
            tick=self._total_detections,
            status=overall,
            convergence_rate=convergence_rate,
            n_converging=n_converging,
            n_diverging=n_diverging,
            n_stable=n_stable,
        )

        self._history.append(record)
        self._total_detections += 1

        return record

    # ── Internal helpers ───────────────────────────────────────

    def _estimate_velocity(
        self, engine: 'EquilibriumEngine', axis_id: TensionID  # type: ignore[name-defined]
    ) -> float | None:
        """Estimate velocity from position history when no tracker is available.

        Uses the last two positions in the engine's history deque.
        Returns None if insufficient history.

        Args:
            engine: The EquilibriumEngine.
            axis_id: The axis to estimate velocity for.

        Returns:
            Estimated velocity, or None if insufficient history.
        """
        history = engine._history.get(axis_id)
        if history is None or len(history) < 2:
            return None
        # Simple finite difference: last position - second-to-last
        positions = list(history)
        return positions[-1] - positions[-2]

    @staticmethod
    def _determine_overall(
        n_converging: int,
        n_diverging: int,
        n_stable: int,
        n_unknown: int,
    ) -> ConvergenceStatus:
        """Determine overall convergence status from per-axis counts.

        Rules:
        - If more axes are UNKNOWN than all others combined → UNKNOWN
        - If n_stable >= 50% of all classified axes → STABLE
        - Otherwise, majority of non-stable, non-unknown axes wins
        - Tie between converging and diverging → STABLE (uncertainty)

        Args:
            n_converging: Number of converging axes.
            n_diverging: Number of diverging axes.
            n_stable: Number of stable axes.
            n_unknown: Number of unknown axes.

        Returns:
            Overall ConvergenceStatus.
        """
        total = n_converging + n_diverging + n_stable + n_unknown

        if total == 0:
            return ConvergenceStatus.UNKNOWN

        # If majority unknown, return unknown
        if n_unknown > total / 2:
            return ConvergenceStatus.UNKNOWN

        # Classified axes (excluding unknown)
        classified = n_converging + n_diverging + n_stable

        if classified == 0:
            return ConvergenceStatus.UNKNOWN

        # If stable is the majority of classified axes
        if n_stable >= classified / 2:
            return ConvergenceStatus.STABLE

        # Compare converging vs diverging among non-stable
        if n_converging > n_diverging:
            return ConvergenceStatus.CONVERGING
        elif n_diverging > n_converging:
            return ConvergenceStatus.DIVERGING
        else:
            # Tie → stable (uncertainty principle)
            return ConvergenceStatus.STABLE

    # ── Reset ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all convergence history and stored state."""
        self._history.clear()
        self._total_detections = 0
        self._current_status = ConvergenceStatus.UNKNOWN
        self._velocity.clear()
        self._positions.clear()
        self._defaults.clear()

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the detector state for cross-session persistence."""
        return {
            "max_history": self._max_history,
            "convergence_threshold": self._convergence_threshold,
            "total_detections": self._total_detections,
            "current_status": self._current_status.value,
            "history": [r.to_dict() for r in self._history],
            "velocity": {k: v for k, v in self._velocity.items() if v is not None},
            "positions": dict(self._positions),
            "defaults": dict(self._defaults),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceDetector:
        """Deserialize detector state from a dict produced by to_dict()."""
        detector = cls(
            max_history=data.get("max_history", 100),
            convergence_threshold=data.get("convergence_threshold", 0.001),
        )
        detector._total_detections = int(data.get("total_detections", 0))
        detector._current_status = ConvergenceStatus(
            data.get("current_status", "unknown")
        )
        for r_data in data.get("history", []):
            detector._history.append(ConvergenceRecord.from_dict(r_data))
        # Restore velocity / positions / defaults
        for k, v in data.get("velocity", {}).items():
            detector._velocity[k] = v
        for k, v in data.get("positions", {}).items():
            detector._positions[k] = v
        for k, v in data.get("defaults", {}).items():
            detector._defaults[k] = v
        return detector

    def __repr__(self) -> str:
        return (
            f"ConvergenceDetector("
            f"history={len(self._history)}/{self._max_history}, "
            f"threshold={self._convergence_threshold:.4f})"
        )
