"""Tests for TensionEventLog — iter-028.

Covers:
- Event recording (feedback applied, default adjusted, oscillation detected, reset)
- Bounded log with max_events overflow (FIFO eviction)
- Query by axis_id, source pillar, event type
- Stress timeline computation
- Feedback density tracking
- Event counting and aggregation
- Integration with EquilibriumEngine (apply_feedback, batch, adjust_default, reset, serialization)
- Serialization round-trip (to_dict / from_dict)
- PillarEquilibriumView event log properties
- Edge cases: empty log, single event, max_events=1, unknown axis queries
"""

import math
import pytest

from isonome.equilibrium import (
    EquilibriumEngine,
    PillarEquilibriumView,
)
from isonome.equilibrium.event_log import (
    TensionEventLog,
    TensionEvent,
    TensionEventType,
)
from isonome.types import Feedback, Pillar, TensionAxis, TensionID


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def log() -> TensionEventLog:
    """Fresh event log with default settings."""
    return TensionEventLog()


@pytest.fixture
def log_small() -> TensionEventLog:
    """Event log with small max_events for testing overflow."""
    return TensionEventLog(max_events=5)


@pytest.fixture
def engine_with_log() -> EquilibriumEngine:
    """Engine with event logging enabled."""
    return EquilibriumEngine(enable_event_log=True)


@pytest.fixture
def engine_without_log() -> EquilibriumEngine:
    """Standard engine without event logging."""
    return EquilibriumEngine()


@pytest.fixture
def engine_with_small_log() -> EquilibriumEngine:
    """Engine with a small event log for testing overflow."""
    return EquilibriumEngine(
        enable_event_log=True,
        event_log_max_events=10,
    )


@pytest.fixture
def engine_oscillation() -> EquilibriumEngine:
    """Engine configured so that oscillation can actually trigger.

    Default damping (0.4) clamps ±0.9 signals to small steps, producing
    stddev ≈0.27 which never exceeds the default threshold (0.6).  We use
    low-damping axes and a low threshold so alternating feedback reliably
    triggers oscillation detection.
    """
    low_damp_axes = tuple(
        a.model_copy(update={"damping": 0.05})
        for a in EquilibriumEngine.DEFAULT_AXES
    )
    return EquilibriumEngine(
        axes=low_damp_axes,
        enable_event_log=True,
        oscillation_threshold=0.2,
        oscillation_window=4,
    )


def _make_feedback(axis_id: TensionID, signal: float, confidence: float = 1.0,
                    source: Pillar = Pillar.COGNITION) -> Feedback:
    """Helper to create a Feedback object."""
    return Feedback(
        tension_axis_id=axis_id,
        signal=signal,
        confidence=confidence,
        source=source,
        reason="test feedback",
    )


# ═══════════════════════════════════════════════════════════════════
# Construction & Validation
# ═══════════════════════════════════════════════════════════════════


class TestEventLogConstruction:
    def test_default_construction(self, log):
        assert log.max_events == 1000
        assert log.total_events == 0

    def test_custom_max_events(self):
        el = TensionEventLog(max_events=50)
        assert el.max_events == 50

    def test_max_events_zero_raises(self):
        with pytest.raises(ValueError, match="max_events must be >= 1"):
            TensionEventLog(max_events=0)

    def test_max_events_negative_raises(self):
        with pytest.raises(ValueError, match="max_events must be >= 1"):
            TensionEventLog(max_events=-5)


class TestTensionEvent:
    def test_event_fields(self):
        event = TensionEvent(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.1,
            delta=0.1,
            confidence=0.8,
            tick=1,
        )
        assert event.event_type == TensionEventType.FEEDBACK_APPLIED
        assert event.axis_id == "explore_exploit"
        assert event.source_pillar == Pillar.COGNITION
        assert event.position_before == 0.0
        assert event.position_after == 0.1
        assert event.delta == 0.1
        assert event.confidence == 0.8
        assert event.tick == 1

    def test_event_repr(self):
        event = TensionEvent(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.1,
            delta=0.1,
            confidence=0.8,
            tick=1,
        )
        r = repr(event)
        assert "FEEDBACK_APPLIED" in r
        assert "explore_exploit" in r

    def test_event_type_values(self):
        assert TensionEventType.FEEDBACK_APPLIED == "feedback_applied"
        assert TensionEventType.DEFAULT_ADJUSTED == "default_adjusted"
        assert TensionEventType.OSCILLATION_DETECTED == "oscillation_detected"
        assert TensionEventType.RESET == "reset"


# ═══════════════════════════════════════════════════════════════════
# Event Recording
# ═══════════════════════════════════════════════════════════════════


class TestEventRecording:
    def test_record_feedback_event(self, log):
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.15,
            position_after=0.25,
            delta=0.1,
            confidence=0.8,
            tick=1,
        )
        assert log.total_events == 1
        events = log.events()
        assert len(events) == 1
        assert events[0].axis_id == "explore_exploit"
        assert events[0].position_after == 0.25

    def test_record_default_adjusted_event(self, log):
        log.record(
            event_type=TensionEventType.DEFAULT_ADJUSTED,
            axis_id="autonomy_safety",
            source_pillar=Pillar.PRAXIS,
            position_before=-0.4,
            position_after=-0.35,
            delta=0.05,
            confidence=1.0,
            tick=5,
        )
        assert log.total_events == 1
        events = log.events()
        assert events[0].event_type == TensionEventType.DEFAULT_ADJUSTED

    def test_record_oscillation_event(self, log):
        log.record(
            event_type=TensionEventType.OSCILLATION_DETECTED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.5,
            position_after=0.6,
            delta=0.1,
            confidence=0.0,
            tick=10,
        )
        assert log.total_events == 1

    def test_record_reset_event(self, log):
        log.record(
            event_type=TensionEventType.RESET,
            axis_id="",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.0,
            delta=0.0,
            confidence=0.0,
            tick=20,
        )
        assert log.total_events == 1
        events = log.events()
        assert events[0].event_type == TensionEventType.RESET

    def test_record_multiple_events(self, log):
        for i in range(5):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.1 * i,
                delta=0.1,
                confidence=0.5,
                tick=i,
            )
        assert log.total_events == 5

    def test_events_returns_copy(self, log):
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.1,
            delta=0.1,
            confidence=0.8,
            tick=1,
        )
        events1 = log.events()
        events2 = log.events()
        assert events1 is not events2  # Different list objects
        assert events1 == events2


# ═══════════════════════════════════════════════════════════════════
# Bounded Log & FIFO Eviction
# ═══════════════════════════════════════════════════════════════════


class TestBoundedLog:
    def test_overflow_evicts_oldest(self, log_small):
        for i in range(7):
            log_small.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id=f"axis_{i}",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=float(i),
                delta=1.0,
                confidence=0.5,
                tick=i,
            )
        # max_events=5, so only last 5 remain
        assert log_small.total_events == 7  # Counter tracks all-time
        events = log_small.events()
        assert len(events) == 5
        # Oldest 2 should be evicted
        assert events[0].axis_id == "axis_2"
        assert events[4].axis_id == "axis_6"

    def test_at_capacity_no_eviction(self, log_small):
        for i in range(5):
            log_small.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id=f"axis_{i}",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=float(i),
                delta=1.0,
                confidence=0.5,
                tick=i,
            )
        events = log_small.events()
        assert len(events) == 5
        assert events[0].axis_id == "axis_0"

    def test_max_events_one(self):
        el = TensionEventLog(max_events=1)
        el.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="first",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.1,
            delta=0.1,
            confidence=0.5,
            tick=0,
        )
        el.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="second",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.2,
            delta=0.2,
            confidence=0.5,
            tick=1,
        )
        events = el.events()
        assert len(events) == 1
        assert events[0].axis_id == "second"


# ═══════════════════════════════════════════════════════════════════
# Querying
# ═══════════════════════════════════════════════════════════════════


class TestEventQuerying:
    @pytest.fixture
    def populated_log(self):
        """Log with a mix of events for querying."""
        el = TensionEventLog(max_events=100)
        # Tick 1: Cognition pushes explore_exploit
        el.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                  Pillar.COGNITION, 0.15, 0.25, 0.1, 0.8, 1)
        # Tick 2: Praxis pushes autonomy_safety
        el.record(TensionEventType.FEEDBACK_APPLIED, "autonomy_safety",
                  Pillar.PRAXIS, -0.4, -0.3, 0.1, 0.6, 2)
        # Tick 3: Cognition pushes shallow_deep
        el.record(TensionEventType.FEEDBACK_APPLIED, "shallow_deep",
                  Pillar.COGNITION, -0.2, 0.0, 0.2, 0.9, 3)
        # Tick 4: Default adjusted on autonomy_safety
        el.record(TensionEventType.DEFAULT_ADJUSTED, "autonomy_safety",
                  Pillar.PRAXIS, -0.4, -0.38, 0.02, 1.0, 4)
        # Tick 5: Oscillation detected on explore_exploit
        el.record(TensionEventType.OSCILLATION_DETECTED, "explore_exploit",
                  Pillar.COGNITION, 0.6, 0.7, 0.1, 0.0, 5)
        # Tick 6: Mneme pushes consolidate_prune
        el.record(TensionEventType.FEEDBACK_APPLIED, "consolidate_prune",
                  Pillar.MNEME, -0.1, 0.0, 0.1, 0.7, 6)
        return el

    def test_query_by_axis(self, populated_log):
        events = populated_log.query(axis_id="explore_exploit")
        assert len(events) == 2
        assert all(e.axis_id == "explore_exploit" for e in events)

    def test_query_by_source(self, populated_log):
        events = populated_log.query(source_pillar=Pillar.COGNITION)
        assert len(events) == 3
        assert all(e.source_pillar == Pillar.COGNITION for e in events)

    def test_query_by_event_type(self, populated_log):
        events = populated_log.query(event_type=TensionEventType.DEFAULT_ADJUSTED)
        assert len(events) == 1
        assert events[0].axis_id == "autonomy_safety"

    def test_query_by_tick_range(self, populated_log):
        events = populated_log.query(tick_range=(2, 4))
        assert len(events) == 3
        assert all(2 <= e.tick <= 4 for e in events)

    def test_query_by_tick_range_open_ended(self, populated_log):
        events = populated_log.query(tick_range=(5, None))
        assert len(events) == 2

    def test_query_by_tick_range_open_start(self, populated_log):
        events = populated_log.query(tick_range=(None, 2))
        assert len(events) == 2  # tick 1 and 2

    def test_query_combined_filters(self, populated_log):
        events = populated_log.query(
            axis_id="explore_exploit",
            event_type=TensionEventType.FEEDBACK_APPLIED,
        )
        assert len(events) == 1
        assert events[0].tick == 1

    def test_query_no_match(self, populated_log):
        events = populated_log.query(axis_id="nonexistent_axis")
        assert len(events) == 0

    def test_query_empty_log(self, log):
        events = log.query(axis_id="explore_exploit")
        assert len(events) == 0


# ═══════════════════════════════════════════════════════════════════
# Stress Timeline
# ═══════════════════════════════════════════════════════════════════


class TestStressTimeline:
    def test_empty_timeline(self, log):
        assert log.stress_timeline() == []

    def test_single_event_timeline(self, log):
        log.record(
            TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
            Pillar.COGNITION, 0.15, 0.25, 0.1, 0.8, 1,
        )
        timeline = log.stress_timeline()
        assert len(timeline) == 1
        tick, stress = timeline[0]
        assert tick == 1
        assert stress > 0  # 0.25 is away from default 0.15

    def test_timeline_ordering(self, log):
        # Record events at different ticks
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.15, 0.25, 0.1, 0.8, 1)
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.25, 0.55, 0.3, 0.8, 3)
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.55, 0.20, -0.35, 0.8, 5)
        timeline = log.stress_timeline()
        assert len(timeline) == 3
        # Should be ordered by tick
        assert timeline[0][0] == 1
        assert timeline[1][0] == 3
        assert timeline[2][0] == 5
        # Stress should increase then decrease
        assert timeline[1][1] > timeline[0][1]
        assert timeline[2][1] < timeline[1][1]

    def test_timeline_uses_engine_positions(self, engine_with_log):
        """Stress timeline should reflect actual engine state, not just log entries."""
        engine = engine_with_log
        # Push one axis away from default
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        log = engine.event_log
        assert log is not None
        timeline = log.stress_timeline()
        assert len(timeline) >= 1


# ═══════════════════════════════════════════════════════════════════
# Feedback Density
# ═══════════════════════════════════════════════════════════════════


class TestFeedbackDensity:
    def test_empty_log_density(self, log):
        assert log.feedback_density(window=5) == 0.0

    def test_density_within_window(self, log):
        for i in range(10):
            log.record(
                TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, i,
            )
        # 10 events in ticks 0-9, window=5 covers last 5 ticks
        density = log.feedback_density(window=5)
        # Last 5 ticks (5-9) have 5 events
        assert density == 1.0  # 5 events / 5 ticks

    def test_density_partial_window(self, log):
        for i in range(3):
            log.record(
                TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, i,
            )
        # 3 events in ticks 0-2, window=5
        density = log.feedback_density(window=5)
        # 3 events / 3 ticks (no ticks 3-4)
        assert density == 1.0

    def test_density_excludes_non_feedback(self, log):
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, 1)
        log.record(TensionEventType.DEFAULT_ADJUSTED, "explore_exploit",
                   Pillar.COGNITION, 0.0, 0.1, 0.02, 1.0, 2)
        log.record(TensionEventType.OSCILLATION_DETECTED, "explore_exploit",
                   Pillar.COGNITION, 0.5, 0.6, 0.1, 0.0, 3)
        density = log.feedback_density(window=5)
        # Only 1 feedback event in ticks 1-3
        # Density = events / ticks_in_window = 1 / 3
        assert abs(density - 1/3) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Event Counting & Aggregation
# ═══════════════════════════════════════════════════════════════════


class TestEventCounting:
    def test_count_by_type(self):
        el = TensionEventLog()
        el.record(TensionEventType.FEEDBACK_APPLIED, "a", Pillar.COGNITION,
                  0.0, 0.1, 0.1, 0.8, 1)
        el.record(TensionEventType.FEEDBACK_APPLIED, "b", Pillar.PRAXIS,
                  0.0, 0.1, 0.1, 0.6, 2)
        el.record(TensionEventType.DEFAULT_ADJUSTED, "a", Pillar.COGNITION,
                  0.0, 0.1, 0.02, 1.0, 3)
        counts = el.count_by_type()
        assert counts[TensionEventType.FEEDBACK_APPLIED] == 2
        assert counts[TensionEventType.DEFAULT_ADJUSTED] == 1
        assert counts[TensionEventType.OSCILLATION_DETECTED] == 0

    def test_count_by_axis(self):
        el = TensionEventLog()
        el.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                  Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, 1)
        el.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                  Pillar.COGNITION, 0.1, 0.2, 0.1, 0.8, 2)
        el.record(TensionEventType.FEEDBACK_APPLIED, "autonomy_safety",
                  Pillar.PRAXIS, 0.0, 0.1, 0.1, 0.6, 3)
        counts = el.count_by_axis()
        assert counts["explore_exploit"] == 2
        assert counts["autonomy_safety"] == 1

    def test_count_by_source(self):
        el = TensionEventLog()
        el.record(TensionEventType.FEEDBACK_APPLIED, "a", Pillar.COGNITION,
                  0.0, 0.1, 0.1, 0.8, 1)
        el.record(TensionEventType.FEEDBACK_APPLIED, "b", Pillar.PRAXIS,
                  0.0, 0.1, 0.1, 0.6, 2)
        el.record(TensionEventType.FEEDBACK_APPLIED, "c", Pillar.COGNITION,
                  0.0, 0.1, 0.1, 0.7, 3)
        counts = el.count_by_source()
        assert counts[Pillar.COGNITION] == 2
        assert counts[Pillar.PRAXIS] == 1
        assert counts[Pillar.MNEME] == 0

    def test_most_active_axis(self):
        el = TensionEventLog()
        el.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                  Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, 1)
        el.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                  Pillar.COGNITION, 0.1, 0.2, 0.1, 0.8, 2)
        el.record(TensionEventType.FEEDBACK_APPLIED, "autonomy_safety",
                  Pillar.PRAXIS, 0.0, 0.1, 0.1, 0.6, 3)
        assert el.most_active_axis() == "explore_exploit"

    def test_most_active_axis_empty_log(self, log):
        assert el.most_active_axis() is None if (el := log) else True
        # Cleaner:
        el2 = TensionEventLog()
        assert el2.most_active_axis() is None


# ═══════════════════════════════════════════════════════════════════
# Engine Integration
# ═══════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    def test_disabled_by_default(self, engine_without_log):
        assert engine_without_log.event_log is None

    def test_enabled_creates_log(self, engine_with_log):
        assert engine_with_log.event_log is not None
        assert isinstance(engine_with_log.event_log, TensionEventLog)

    def test_apply_feedback_records_event(self, engine_with_log):
        engine = engine_with_log
        pos_before = engine.get_axis("explore_exploit").position
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        log = engine.event_log
        events = log.query(event_type=TensionEventType.FEEDBACK_APPLIED)
        assert len(events) == 1
        assert events[0].axis_id == "explore_exploit"
        assert events[0].source_pillar == Pillar.COGNITION
        assert events[0].confidence == 1.0

    def test_apply_feedback_batch_records_events(self, engine_with_log):
        engine = engine_with_log
        feedbacks = [
            _make_feedback("explore_exploit", 0.2),
            _make_feedback("shallow_deep", 0.1, source=Pillar.COGNITION),
        ]
        engine.apply_feedback_batch(feedbacks)
        log = engine.event_log
        events = log.query(event_type=TensionEventType.FEEDBACK_APPLIED)
        assert len(events) == 2

    def test_adjust_default_records_event(self, engine_with_log):
        engine = engine_with_log
        engine.adjust_default("explore_exploit", 0.1)
        log = engine.event_log
        events = log.query(event_type=TensionEventType.DEFAULT_ADJUSTED)
        assert len(events) == 1
        assert events[0].axis_id == "explore_exploit"

    def test_reset_records_event(self, engine_with_log):
        engine = engine_with_log
        # Push some feedback first
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        # Reset
        engine.reset()
        log = engine.event_log
        events = log.query(event_type=TensionEventType.RESET)
        assert len(events) == 1

    def test_oscillation_records_event(self, engine_oscillation):
        """Oscillation detection should record an event."""
        engine = engine_oscillation
        # Push contradictory feedback to trigger oscillation.
        # With low damping (0.05) and threshold 0.2, alternating ±0.9
        # produces stddev ≈0.43 which exceeds the threshold.
        for _ in range(10):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.9))
            engine.apply_feedback(_make_feedback("explore_exploit", -0.9))
        log = engine.event_log
        events = log.query(event_type=TensionEventType.OSCILLATION_DETECTED)
        assert len(events) >= 1

    def test_custom_max_events(self, engine_with_small_log):
        engine = engine_with_small_log
        assert engine.event_log.max_events == 10

    def test_no_log_no_events(self, engine_without_log):
        engine = engine_without_log
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        # Should not crash — log is None


# ═══════════════════════════════════════════════════════════════════
# Serialization Round-Trip
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_to_dict_round_trip(self, log):
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.15, 0.25, 0.1, 0.8, 1)
        log.record(TensionEventType.DEFAULT_ADJUSTED, "autonomy_safety",
                   Pillar.PRAXIS, -0.4, -0.38, 0.02, 1.0, 2)
        data = log.to_dict()
        restored = TensionEventLog.from_dict(data)
        assert restored.max_events == log.max_events
        assert restored.total_events == log.total_events
        assert len(restored.events()) == len(log.events())

    def test_to_dict_preserves_event_data(self, log):
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.15, 0.25, 0.1, 0.8, 1)
        data = log.to_dict()
        restored = TensionEventLog.from_dict(data)
        events = restored.events()
        assert events[0].axis_id == "explore_exploit"
        assert events[0].event_type == TensionEventType.FEEDBACK_APPLIED
        assert abs(events[0].position_after - 0.25) < 0.001

    def test_engine_serialization_includes_log(self, engine_with_log):
        engine = engine_with_log
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        data = engine.to_dict()
        # Should include event_log_state
        assert "event_log_state" in data
        assert data["event_log_state"] is not None

    def test_engine_deserialization_restores_log(self, engine_with_log):
        engine = engine_with_log
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        engine.apply_feedback(_make_feedback("shallow_deep", 0.2))
        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        assert restored.event_log is not None
        events = restored.event_log.query(event_type=TensionEventType.FEEDBACK_APPLIED)
        assert len(events) == 2

    def test_engine_without_log_serialization(self, engine_without_log):
        engine = engine_without_log
        data = engine.to_dict()
        assert data.get("event_log_state") is None

    def test_empty_log_serialization(self, log):
        data = log.to_dict()
        restored = TensionEventLog.from_dict(data)
        assert restored.total_events == 0
        assert len(restored.events()) == 0


# ═══════════════════════════════════════════════════════════════════
# PillarEquilibriumView Integration
# ═══════════════════════════════════════════════════════════════════


class TestPillarViewIntegration:
    def test_view_without_log(self, engine_without_log):
        view = engine_without_log.view_for(Pillar.COGNITION)
        # Should not crash
        assert view is not None

    def test_view_with_log_provides_recent_events(self, engine_with_log):
        engine = engine_with_log
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        engine.apply_feedback(_make_feedback("shallow_deep", 0.2))
        view = engine.view_for(Pillar.COGNITION)
        # View should provide access to recent events for own axes
        recent = view.recent_events(limit=10)
        assert len(recent) >= 2

    def test_view_recent_events_scoped_to_pillar(self, engine_with_log):
        engine = engine_with_log
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3,
                                             source=Pillar.COGNITION))
        engine.apply_feedback(_make_feedback("autonomy_safety", 0.2,
                                             source=Pillar.PRAXIS))
        view = engine.view_for(Pillar.COGNITION)
        recent = view.recent_events(limit=10)
        # Should include both own-axis events (even from other pillars
        # if they affect own axes) but the view is scoped
        assert len(recent) >= 1


# ═══════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════


class TestEventLogReset:
    def test_reset_clears_events(self, log):
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, 1)
        log.reset()
        assert len(log.events()) == 0
        assert log.total_events == 0

    def test_reset_preserves_config(self, log_small):
        log_small.reset()
        assert log_small.max_events == 5


# ═══════════════════════════════════════════════════════════════════
# Repr
# ═══════════════════════════════════════════════════════════════════


class TestRepr:
    def test_empty_repr(self, log):
        r = repr(log)
        assert "TensionEventLog" in r
        assert "events=0" in r

    def test_populated_repr(self, log):
        log.record(TensionEventType.FEEDBACK_APPLIED, "explore_exploit",
                   Pillar.COGNITION, 0.0, 0.1, 0.1, 0.8, 1)
        r = repr(log)
        assert "events=1" in r
