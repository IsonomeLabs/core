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