"""Tension Event Log — bounded, queryable log of equilibrium state changes.

Records every significant event in the equilibrium engine's lifecycle:
- Feedback applied to an axis (from any pillar)
- Default position adjusted (learning signal)
- Oscillation detected on an axis
- Engine reset

The log is bounded (FIFO eviction) to prevent unbounded memory growth in
long-running agents. It supports querying by axis, pillar, event type,
and tick range, enabling audit trail generation and stress pattern analysis.

Integration:
- The event log is optional — engines without it operate exactly as before.
- When enabled, the engine records events after each feedback tick, default
  adjustment, oscillation detection, and reset.
- The PillarEquilibriumView provides scoped access to recent events for
  the requesting pillar's own axes.
"""

from __future__ import annotations

import math
from collections import deque
from enum import StrEnum
from typing import Any

from isonome.types import Pillar, TensionID


class TensionEventType(StrEnum):
    """Types of events recorded in the tension event log."""
    FEEDBACK_APPLIED = "feedback_applied"
    DEFAULT_ADJUSTED = "default_adjusted"
    OSCILLATION_DETECTED = "oscillation_detected"
    COOLDOWN_APPLIED = "cooldown_applied"
    CONVERGENCE_SHIFTED = "convergence_shifted"
    RESET = "reset"


class TensionEvent:
    """A single recorded event in the equilibrium engine's history.

    Attributes:
        event_type: What kind of event occurred.
        axis_id: The tension axis involved (empty string for engine-wide events).
        source_pillar: Which pillar triggered this event.
        position_before: Axis position before the event.
        position_after: Axis position after the event.
        delta: The change applied (signal * confidence for feedback,
               outcome_signal * learning_rate for default adjustments).
        confidence: The confidence associated with the event.
        tick: The engine tick when this event occurred.
    """

    __slots__ = (
        "event_type",
        "axis_id",
        "source_pillar",
        "position_before",
        "position_after",
        "delta",
        "confidence",
        "tick",
    )

    def __init__(
        self,
        event_type: TensionEventType,
        axis_id: TensionID,
        source_pillar: Pillar,
        position_before: float,
        position_after: float,
        delta: float,
        confidence: float,
        tick: int,
    ):
        self.event_type = event_type
        self.axis_id = axis_id
        self.source_pillar = source_pillar
        self.position_before = position_before
        self.position_after = position_after
        self.delta = delta
        self.confidence = confidence
        self.tick = tick

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a dict."""
        return {
            "event_type": self.event_type.value,
            "axis_id": self.axis_id,
            "source_pillar": self.source_pillar.value,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "delta": self.delta,
            "confidence": self.confidence,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensionEvent:
        """Deserialize event from a dict."""
        return cls(
            event_type=TensionEventType(data["event_type"]),
            axis_id=data["axis_id"],
            source_pillar=Pillar(data["source_pillar"]),
            position_before=data["position_before"],
            position_after=data["position_after"],
            delta=data["delta"],
            confidence=data["confidence"],
            tick=data["tick"],
        )

    def __repr__(self) -> str:
        return (
            f"TensionEvent({self.event_type.name}, "
            f"axis={self.axis_id!r}, "
            f"pillar={self.source_pillar.value}, "
            f"tick={self.tick})"
        )


class TensionEventLog:
    """Bounded, queryable log of equilibrium state changes.

    Records every significant event in the equilibrium engine's lifecycle
    and provides efficient querying for audit trails, stress pattern
    analysis, and feedback density tracking.

    The log is bounded (FIFO eviction) to prevent unbounded memory growth
    in long-running agents. When the log exceeds max_events, the oldest
    events are discarded.

    Mathematical foundations:
    - Stress timeline: per-tick RMS drift from homeostasis across all axes
    - Feedback density: feedback events per tick in a sliding window
    """

    __slots__ = (
        "_events",
        "_max_events",
        "_total_events",
        "_positions_at_tick",
    )

    def __init__(self, *, max_events: int = 1000):
        """Initialize the event log.

        Args:
            max_events: Maximum number of events to retain.
                When exceeded, oldest events are evicted (FIFO).
                Must be >= 1.

        Raises:
            ValueError: If max_events < 1.
        """
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events}")

        self._events: deque[TensionEvent] = deque(maxlen=max_events)
        self._max_events = max_events
        self._total_events: int = 0
        # Track per-axis, per-tick positions for stress timeline
        self._positions_at_tick: dict[int, dict[TensionID, float]] = {}

    # ── Recording ────────────────────────────────────────────────

    def record(
        self,
        event_type: TensionEventType,
        axis_id: TensionID,
        source_pillar: Pillar,
        position_before: float,
        position_after: float,
        delta: float,
        confidence: float,
        tick: int,
    ) -> None:
        """Record a new event in the log.

        Args:
            event_type: What kind of event occurred.
            axis_id: The tension axis involved.
            source_pillar: Which pillar triggered this event.
            position_before: Axis position before the event.
            position_after: Axis position after the event.
            delta: The change applied.
            confidence: The confidence associated with the event.
            tick: The engine tick when this event occurred.
        """
        event = TensionEvent(
            event_type=event_type,
            axis_id=axis_id,
            source_pillar=source_pillar,
            position_before=position_before,
            position_after=position_after,
            delta=delta,
            confidence=confidence,
            tick=tick,
        )
        self._events.append(event)
        self._total_events += 1

        # Track position for stress timeline computation
        if tick not in self._positions_at_tick:
            self._positions_at_tick[tick] = {}
        if axis_id:  # Skip empty axis_id (engine-wide events)
            self._positions_at_tick[tick][axis_id] = position_after

    # ── Access ───────────────────────────────────────────────────

    def events(self) -> list[TensionEvent]:
        """Return a snapshot of all current events (copy).

        Returns:
            List of events in chronological order.
        """
        return list(self._events)

    @property
    def max_events(self) -> int:
        """Maximum number of events retained."""
        return self._max_events

    @property
    def total_events(self) -> int:
        """Total events ever recorded (including evicted ones)."""
        return self._total_events

    # ── Querying ─────────────────────────────────────────────────

    def query(
        self,
        *,
        axis_id: TensionID | None = None,
        source_pillar: Pillar | None = None,
        event_type: TensionEventType | None = None,
        tick_range: tuple[int | None, int | None] | None = None,
    ) -> list[TensionEvent]:
        """Query events with optional filters.

        All filters are AND-combined. Omit a filter to skip it.

        Args:
            axis_id: Filter by tension axis ID.
            source_pillar: Filter by source pillar.
            event_type: Filter by event type.
            tick_range: Filter by tick range (inclusive).
                (start, end) where either can be None for open-ended.

        Returns:
            List of matching events in chronological order.
        """
        results = []
        tick_start, tick_end = None, None
        if tick_range is not None:
            tick_start, tick_end = tick_range

        for event in self._events:
            if axis_id is not None and event.axis_id != axis_id:
                continue
            if source_pillar is not None and event.source_pillar != source_pillar:
                continue
            if event_type is not None and event.event_type != event_type:
                continue
            if tick_start is not None and event.tick < tick_start:
                continue
            if tick_end is not None and event.tick > tick_end:
                continue
            results.append(event)
        return results

    # ── Analysis ─────────────────────────────────────────────────

    def stress_timeline(self) -> list[tuple[int, float]]:
        """Compute per-tick stress levels from recorded positions.

        Stress is computed as the RMS drift from homeostasis across
        all tracked axes at each tick. This requires the engine to
        feed position updates through the record() method.

        Returns:
            List of (tick, stress_level) tuples ordered by tick.
            Only includes ticks with recorded positions.
        """
        if not self._positions_at_tick:
            return []

        # We need default positions to compute drift. Since the log
        # doesn't store defaults directly, we compute stress as
        # RMS of absolute position values relative to axis defaults
        # that we infer from the first recorded position for each axis.
        # For engine integration, the stress timeline uses the positions
        # recorded per tick.
        
        # Group positions by tick
        timeline: list[tuple[int, float]] = []
        for tick in sorted(self._positions_at_tick.keys()):
            positions = self._positions_at_tick[tick]
            if not positions:
                continue
            # Compute RMS of absolute position values as a proxy for stress.
            # Higher absolute positions = more stressed (further from neutral).
            # This is a simplified metric; the full engine uses default positions.
            squared = sum(p * p for p in positions.values())
            rms = math.sqrt(squared / len(positions)) if positions else 0.0
            timeline.append((tick, rms))
        return timeline

    def feedback_density(self, window: int = 10) -> float:
        """Compute feedback density (feedback events per tick) in a recent window.

        Args:
            window: Number of recent ticks to consider.

        Returns:
            Average feedback events per tick in the window.
            0.0 if no feedback events in the window.
        """
        if not self._events:
            return 0.0

        # Find the latest tick
        latest_tick = max(e.tick for e in self._events)
        window_start = latest_tick - window + 1

        # Count feedback events and all active ticks in window
        feedback_count = 0
        ticks_in_window: set[int] = set()
        for event in self._events:
            if event.tick >= window_start:
                # Track all ticks with any event activity
                ticks_in_window.add(event.tick)
                if event.event_type == TensionEventType.FEEDBACK_APPLIED:
                    feedback_count += 1

        if not ticks_in_window:
            return 0.0

        # Density = feedback events / all ticks with activity
        num_ticks = len(ticks_in_window)
        return feedback_count / num_ticks

    def count_by_type(self) -> dict[TensionEventType, int]:
        """Count events by type.

        Returns:
            Dict mapping event type to count.
        """
        counts: dict[TensionEventType, int] = {
            et: 0 for et in TensionEventType
        }
        for event in self._events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts

    def count_by_axis(self) -> dict[TensionID, int]:
        """Count events by axis ID.

        Returns:
            Dict mapping axis ID to event count.
        """
        counts: dict[TensionID, int] = {}
        for event in self._events:
            if event.axis_id:
                counts[event.axis_id] = counts.get(event.axis_id, 0) + 1
        return counts

    def count_by_source(self) -> dict[Pillar, int]:
        """Count events by source pillar.

        Returns:
            Dict mapping pillar to event count.
        """
        counts: dict[Pillar, int] = {p: 0 for p in Pillar}
        for event in self._events:
            counts[event.source_pillar] = counts.get(event.source_pillar, 0) + 1
        return counts

    def most_active_axis(self) -> TensionID | None:
        """Return the axis with the most events, or None if empty."""
        counts = self.count_by_axis()
        if not counts:
            return None
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    def cooldown_stats(self) -> dict[str, Any]:
        """Compute summary statistics for COOLDOWN_APPLIED events.

        Returns:
            A dict with:
            - total_cooldown_events: total number of cooldown events
            - affected_axes: list of axis IDs that have been cooled down
            - affected_pillars: list of Pillar values that triggered cooldown
            - avg_multiplier: average cooldown multiplier across all events
            - per_axis: dict mapping axis_id to count of cooldown events
        """
        cooldown_events = self.query(event_type=TensionEventType.COOLDOWN_APPLIED)
        if not cooldown_events:
            return {
                "total_cooldown_events": 0,
                "affected_axes": [],
                "affected_pillars": [],
                "avg_multiplier": 0.0,
                "per_axis": {},
            }

        axes: set[TensionID] = set()
        pillars: set[Pillar] = set()
        total_multiplier = 0.0
        per_axis: dict[TensionID, int] = {}

        for event in cooldown_events:
            axes.add(event.axis_id)
            pillars.add(event.source_pillar)
            total_multiplier += event.delta
            per_axis[event.axis_id] = per_axis.get(event.axis_id, 0) + 1

        avg = total_multiplier / len(cooldown_events)

        return {
            "total_cooldown_events": len(cooldown_events),
            "affected_axes": sorted(axes),
            "affected_pillars": sorted(pillars, key=lambda p: p.value),
            "avg_multiplier": avg,
            "per_axis": per_axis,
        }

    # ── Advanced Analysis ──────────────────────────────────────

    def pillar_stress_scores(self) -> dict[Pillar, float]:
        """Compute per-pillar stress scores from feedback events.

        Each pillar's stress score is the sum of |delta| * confidence
        across all FEEDBACK_APPLIED events from that pillar. This
        measures how much force each pillar is exerting on the system.

        Returns:
            Dict mapping Pillar to its cumulative stress score.
            Empty dict if no feedback events exist.
        """
        scores: dict[Pillar, float] = {}
        for event in self._events:
            if event.event_type != TensionEventType.FEEDBACK_APPLIED:
                continue
            weight = abs(event.delta) * event.confidence
            scores[event.source_pillar] = (
                scores.get(event.source_pillar, 0.0) + weight
            )
        return scores

    def axis_volatility(self) -> dict[TensionID, float]:
        """Compute per-axis volatility (standard deviation of positions).

        Volatility is the standard deviation of the position_after
        values for each axis across all events. A high volatility
        indicates the axis is moving erratically; low volatility
        indicates stability.

        Returns:
            Dict mapping axis_id to its position standard deviation.
            Axes with a single position have volatility 0.0.
            Engine-wide events (empty axis_id) are excluded.
        """
        # Collect per-axis positions
        axis_positions: dict[TensionID, list[float]] = {}
        for event in self._events:
            if not event.axis_id:
                continue  # Skip engine-wide events
            if event.axis_id not in axis_positions:
                axis_positions[event.axis_id] = []
            axis_positions[event.axis_id].append(event.position_after)

        # Compute stddev for each axis
        result: dict[TensionID, float] = {}
        for axis_id, positions in axis_positions.items():
            n = len(positions)
            if n <= 1:
                result[axis_id] = 0.0
                continue
            mean = sum(positions) / n
            variance = sum((p - mean) ** 2 for p in positions) / n
            result[axis_id] = math.sqrt(variance)
        return result

    def detect_feedback_bursts(
        self,
        *,
        window: int = 5,
        threshold: int = 3,
    ) -> list[dict[str, Any]]:
        """Detect feedback bursts -- rapid consecutive feedback on one axis.

        A burst is a contiguous window of ticks where the number of
        FEEDBACK_APPLIED events on a single axis exceeds the threshold.

        Args:
            window: Number of consecutive ticks to check.
            threshold: Minimum number of feedback events in the window
                to qualify as a burst.

        Returns:
            List of burst dicts, each with:
            - axis_id: The axis experiencing the burst
            - tick_start: First tick of the burst window
            - tick_end: Last tick of the burst window
            - event_count: Number of feedback events in the window
        """
        if not self._events:
            return []

        # Group FEEDBACK_APPLIED events by axis, preserving tick order
        axis_events: dict[TensionID, list[TensionEvent]] = {}
        for event in self._events:
            if event.event_type != TensionEventType.FEEDBACK_APPLIED:
                continue
            if event.axis_id not in axis_events:
                axis_events[event.axis_id] = []
            axis_events[event.axis_id].append(event)

        bursts: list[dict[str, Any]] = []
        for axis_id, events in axis_events.items():
            if len(events) < threshold:
                continue
            # Sliding window over events
            for i in range(len(events)):
                window_events = [events[i]]
                for j in range(i + 1, len(events)):
                    if events[j].tick - events[i].tick <= window:
                        window_events.append(events[j])
                    else:
                        break
                if len(window_events) >= threshold:
                    bursts.append({
                        'axis_id': axis_id,
                        'tick_start': window_events[0].tick,
                        'tick_end': window_events[-1].tick,
                        'event_count': len(window_events),
                    })
                    break  # Only report first burst per axis

        return bursts

    def dominant_feedback_source(self) -> dict[TensionID, dict[str, Any]]:
        """Identify the dominant feedback source for each axis.

        For each axis that has received feedback, determines which pillar
        has the highest cumulative |delta| * confidence weight, and reports
        that pillar along with its total weight and event count.

        Returns:
            Dict mapping axis_id to a dict with:
            - pillar: The Pillar with the most influence on this axis
            - total_weight: Sum of |delta| * confidence for that pillar
            - event_count: Number of feedback events from that pillar
        """
        # Collect per-axis, per-pillar weights
        axis_pillar_data: dict[
            TensionID, dict[Pillar, list[float]]
        ] = {}
        for event in self._events:
            if event.event_type != TensionEventType.FEEDBACK_APPLIED:
                continue
            if event.axis_id not in axis_pillar_data:
                axis_pillar_data[event.axis_id] = {}
            if event.source_pillar not in axis_pillar_data[event.axis_id]:
                axis_pillar_data[event.axis_id][event.source_pillar] = []
            axis_pillar_data[event.axis_id][event.source_pillar].append(
                abs(event.delta) * event.confidence
            )

        result: dict[TensionID, dict[str, Any]] = {}
        for axis_id, pillar_weights in axis_pillar_data.items():
            best_pillar: Pillar | None = None
            best_total = -1.0
            best_count = 0
            for pillar, weights in pillar_weights.items():
                total = sum(weights)
                if total > best_total:
                    best_total = total
                    best_pillar = pillar
                    best_count = len(weights)
            if best_pillar is not None:
                result[axis_id] = {
                    'pillar': best_pillar,
                    'total_weight': best_total,
                    'event_count': best_count,
                }
        return result

    def detect_convergence_from_events(self) -> dict[str, Any]:
        """Detect convergence/divergence trends from feedback event deltas.

        Analyzes the trend of |delta| values over FEEDBACK_APPLIED
        events. Decreasing deltas suggest convergence (the system is
        settling), increasing deltas suggest divergence.

        Also considers OSCILLATION_DETECTED events as a divergence
        signal.

        Returns:
            Dict with:
            - direction: 'converging', 'diverging', 'stable', or 'unknown'
            - confidence: 0.0 to 1.0 confidence in the direction
            - trend_slope: Linear regression slope of |delta| over time
        """
        # Collect (tick, |delta|) pairs for feedback events
        feedback_deltas: list[tuple[int, float]] = []
        has_oscillation = False
        for event in self._events:
            if event.event_type == TensionEventType.FEEDBACK_APPLIED:
                feedback_deltas.append((event.tick, abs(event.delta)))
            elif event.event_type == TensionEventType.OSCILLATION_DETECTED:
                has_oscillation = True

        if not feedback_deltas:
            return {
                'direction': 'unknown',
                'confidence': 0.0,
                'trend_slope': 0.0,
            }

        # Simple linear regression of |delta| on tick order
        n = len(feedback_deltas)
        if n == 1:
            direction = 'unknown'
            if has_oscillation:
                direction = 'diverging'
            return {
                'direction': direction,
                'confidence': 0.0,
                'trend_slope': 0.0,
            }

        # x = order index (0, 1, 2, ...), y = |delta|
        x_mean = (n - 1) / 2.0
        y_mean = sum(d for _, d in feedback_deltas) / n

        numerator = 0.0
        denom_x = 0.0
        denom_y = 0.0
        for i, (_, delta) in enumerate(feedback_deltas):
            dx = i - x_mean
            dy = delta - y_mean
            numerator += dx * dy
            denom_x += dx * dx
            denom_y += dy * dy

        slope = numerator / denom_x if denom_x != 0 else 0.0

        # R-squared for confidence
        if denom_y != 0:
            r_squared = (numerator ** 2) / (denom_x * denom_y)
        else:
            r_squared = 0.0

        # Determine direction from slope
        # Use a small epsilon to avoid noise
        eps = 0.005
        if slope < -eps:
            direction = 'converging'
        elif slope > eps:
            direction = 'diverging'
        else:
            direction = 'stable'

        # Oscillation events bias toward divergence
        if has_oscillation and direction != 'diverging':
            direction = 'diverging' if slope > 0 else 'stable'
            r_squared = min(r_squared + 0.2, 1.0)

        confidence = min(r_squared, 1.0)

        return {
            'direction': direction,
            'confidence': confidence,
            'trend_slope': slope,
        }

    def detect_cross_pillar_conflicts(self) -> list[dict[str, Any]]:
        """Detect cross-pillar conflicts -- same axis, opposing directions.

        A conflict occurs when two different pillars send feedback
        to the same axis in opposing directions (one positive delta,
        one negative delta). The conflict intensity measures how
        strongly the pillars disagree.

        Returns:
            List of conflict dicts, each with:
            - axis_id: The contested axis
            - pillars: Set of Pillars in conflict
            - opposing_deltas: Dict of pillar -> signed total delta
            - conflict_intensity: 0.0 to 1.0 (higher = stronger conflict)
        """
        # Collect per-axis, per-pillar signed delta totals
        axis_deltas: dict[TensionID, dict[Pillar, list[float]]] = {}
        for event in self._events:
            if event.event_type != TensionEventType.FEEDBACK_APPLIED:
                continue
            if event.axis_id not in axis_deltas:
                axis_deltas[event.axis_id] = {}
            if event.source_pillar not in axis_deltas[event.axis_id]:
                axis_deltas[event.axis_id][event.source_pillar] = []
            axis_deltas[event.axis_id][event.source_pillar].append(
                event.delta
            )

        conflicts: list[dict[str, Any]] = []
        for axis_id, pillar_deltas in axis_deltas.items():
            if len(pillar_deltas) < 2:
                continue  # Need at least 2 different pillars

            # Sum deltas per pillar
            pillar_totals: dict[Pillar, float] = {}
            for pillar, deltas in pillar_deltas.items():
                pillar_totals[pillar] = sum(deltas)

            # Check for opposing signs
            positive_pillars = [
                (p, t) for p, t in pillar_totals.items() if t > 0
            ]
            negative_pillars = [
                (p, t) for p, t in pillar_totals.items() if t < 0
            ]

            if not positive_pillars or not negative_pillars:
                continue  # Same direction = no conflict

            # Compute conflict intensity: normalized sum of |totals|
            total_force = sum(abs(t) for t in pillar_totals.values())
            if total_force == 0:
                continue

            conflict_intensity = min(total_force / 2.0, 1.0)

            # Build the opposing deltas dict
            opposing_deltas = {
                p: t for p, t in pillar_totals.items() if t != 0
            }

            conflicts.append({
                'axis_id': axis_id,
                'pillars': set(pillar_totals.keys()),
                'opposing_deltas': opposing_deltas,
                'conflict_intensity': conflict_intensity,
            })

        return conflicts

    # ── Lifecycle ────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all events and tracking state."""
        self._events.clear()
        self._total_events = 0
        self._positions_at_tick.clear()

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event log state for cross-session persistence."""
        return {
            "max_events": self._max_events,
            "total_events": self._total_events,
            "events": [e.to_dict() for e in self._events],
            "positions_at_tick": {
                str(k): v for k, v in self._positions_at_tick.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensionEventLog:
        """Deserialize event log state from a dict produced by to_dict()."""
        log = cls(max_events=data.get("max_events", 1000))
        log._total_events = int(data.get("total_events", 0))
        for e_data in data.get("events", []):
            log._events.append(TensionEvent.from_dict(e_data))
        # Restore positions_at_tick
        for tick_str, positions in data.get("positions_at_tick", {}).items():
            log._positions_at_tick[int(tick_str)] = dict(positions)
        return log

    def __repr__(self) -> str:
        return (
            f"TensionEventLog("
            f"events={len(self._events)}, "
            f"max={self._max_events}, "
            f"total={self._total_events})"
        )