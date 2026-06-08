"""Tests for iter-031: Cooldown-Aware Event Logging.

When both feedback cooldown and event logging are enabled, the engine
should record COOLDOWN_APPLIED events whenever the cooldown system
dampens a feedback signal. This bridges iter-030 (cooldown) with the
event log, enabling audit trails and pattern analysis for cooldown
suppression events.

Core invariants:
1. COOLDOWN_APPLIED events are only emitted when both cooldown AND event log are enabled
2. Each COOLDOWN_APPLIED event records the multiplier applied and the original delta
3. COOLDOWN_APPLIED events are recorded BEFORE the FEEDBACK_APPLIED event for the same tick
4. When cooldown multiplier is 1.0 (no dampening), no COOLDOWN_APPLIED event is recorded
5. The cooldown_stats() convenience method on TensionEventLog summarizes cooldown events
6. Cooldown events are serialized/deserialized correctly
7. Backward compatibility: engines without cooldown or event log are unaffected
"""

from __future__ import annotations

import pytest

from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.equilibrium.cooldown import FeedbackCooldownManager
from isonome.equilibrium.event_log import TensionEventLog, TensionEventType
from isonome.types import Feedback, Pillar


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def engine_both() -> EquilibriumEngine:
    """Engine with both cooldown and event log enabled."""
    return EquilibriumEngine(
        enable_feedback_cooldown=True,
        enable_event_log=True,
        enable_adaptive_damping=False,
    )


@pytest.fixture
def engine_cooldown_only() -> EquilibriumEngine:
    """Engine with cooldown but no event log."""
    return EquilibriumEngine(
        enable_feedback_cooldown=True,
        enable_adaptive_damping=False,
    )


@pytest.fixture
def engine_eventlog_only() -> EquilibriumEngine:
    """Engine with event log but no cooldown."""
    return EquilibriumEngine(
        enable_event_log=True,
        enable_adaptive_damping=False,
    )


@pytest.fixture
def engine_neither() -> EquilibriumEngine:
    """Engine with neither cooldown nor event log."""
    return EquilibriumEngine(enable_adaptive_damping=False)


# ======================================================================
# TestTensionEventTypeExtension
# ======================================================================


class TestTensionEventTypeExtension:
    """COOLDOWN_APPLIED event type exists in TensionEventType."""

    def test_cooldown_applied_exists(self):
        """COOLDOWN_APPLIED is a valid TensionEventType member."""
        assert hasattr(TensionEventType, "COOLDOWN_APPLIED")

    def test_cooldown_applied_value(self):
        """COOLDOWN_APPLIED has the correct string value."""
        assert TensionEventType.COOLDOWN_APPLIED.value == "cooldown_applied"

    def test_all_event_types_includes_cooldown(self):
        """COOLDOWN_APPLIED appears in the full set of event types."""
        all_types = set(TensionEventType)
        assert TensionEventType.COOLDOWN_APPLIED in all_types

    def test_count_by_type_includes_cooldown(self):
        """count_by_type() includes COOLDOWN_APPLIED in its keys."""
        log = TensionEventLog()
        counts = log.count_by_type()
        assert TensionEventType.COOLDOWN_APPLIED in counts


# ======================================================================
# TestCooldownEventRecording
# ======================================================================


class TestCooldownEventRecording:
    """Engine records COOLDOWN_APPLIED events when both systems are enabled."""

    def test_no_cooldown_event_on_first_feedback(self, engine_both: EquilibriumEngine):
        """First feedback (no cooldown applied) does not generate COOLDOWN_APPLIED."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) == 0

    def test_cooldown_event_on_second_feedback(self, engine_both: EquilibriumEngine):
        """Second feedback from same (pillar, axis) generates COOLDOWN_APPLIED."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) == 1

    def test_cooldown_event_records_multiplier(self, engine_both: EquilibriumEngine):
        """COOLDOWN_APPLIED event's delta field contains the multiplier."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        event = cooldown_events[0]
        # The delta field should hold the cooldown multiplier (0.5 for default decay)
        assert event.delta == pytest.approx(0.5, abs=1e-6)

    def test_cooldown_event_records_correct_axis(self, engine_both: EquilibriumEngine):
        """COOLDOWN_APPLIED event references the correct axis."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert cooldown_events[0].axis_id == "explore_exploit"

    def test_cooldown_event_records_correct_pillar(self, engine_both: EquilibriumEngine):
        """COOLDOWN_APPLIED event references the correct source pillar."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert cooldown_events[0].source_pillar == Pillar.COGNITION

    def test_cooldown_before_feedback_event(self, engine_both: EquilibriumEngine):
        """COOLDOWN_APPLIED is recorded before FEEDBACK_APPLIED for the same tick."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        all_events = engine_both.event_log.events()
        # Find the second feedback's events
        second_tick_events = [e for e in all_events if e.tick > all_events[0].tick]
        if len(second_tick_events) >= 2:
            # COOLDOWN_APPLIED should come before FEEDBACK_APPLIED
            types_in_order = [e.event_type for e in second_tick_events]
            cooldown_idx = types_in_order.index(TensionEventType.COOLDOWN_APPLIED)
            feedback_idx = types_in_order.index(TensionEventType.FEEDBACK_APPLIED)
            assert cooldown_idx < feedback_idx

    def test_multiple_cooldown_events_accumulate(self, engine_both: EquilibriumEngine):
        """Multiple cooldown events accumulate across repeated feedbacks."""
        for i in range(5):
            engine_both.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.3,
                confidence=1.0,
                reason=f"test: feedback {i}",
            ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        # First feedback has no cooldown, remaining 4 do
        assert len(cooldown_events) == 4

    def test_cooldown_event_positions_unchanged(self, engine_both: EquilibriumEngine):
        """COOLDOWN_APPLIED event records position_before == position_after.

        Cooldown doesn't change position — it modifies the delta of the
        subsequent FEEDBACK_APPLIED event.
        """
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        event = cooldown_events[0]
        assert event.position_before == event.position_after

    def test_different_pillar_no_cooldown_event(self, engine_both: EquilibriumEngine):
        """Different pillar on same axis doesn't generate COOLDOWN_APPLIED."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: cognition first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.PRAXIS,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: praxis first",
        ))
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) == 0


# ======================================================================
# TestCooldownEventNoEffectWhenDisabled
# ======================================================================


class TestCooldownEventNoEffectWhenDisabled:
    """No COOLDOWN_APPLIED events when either system is disabled."""

    def test_no_cooldown_events_cooldown_only(
        self, engine_cooldown_only: EquilibriumEngine,
    ):
        """Engine with cooldown but no event log produces no events at all."""
        engine_cooldown_only.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_cooldown_only.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        assert engine_cooldown_only.event_log is None

    def test_no_cooldown_events_eventlog_only(
        self, engine_eventlog_only: EquilibriumEngine,
    ):
        """Engine with event log but no cooldown has no COOLDOWN_APPLIED events."""
        engine_eventlog_only.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: first",
        ))
        engine_eventlog_only.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: second",
        ))
        cooldown_events = engine_eventlog_only.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) == 0

    def test_neither_system_unaffected(self, engine_neither: EquilibriumEngine):
        """Engine with neither system works as before."""
        engine_neither.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test",
        ))
        assert engine_neither.event_log is None
        assert engine_neither.feedback_cooldown is None


# ======================================================================
# TestCooldownEventBatch
# ======================================================================


class TestCooldownEventBatch:
    """Cooldown events are recorded in apply_feedback_batch as well."""

    def test_batch_records_cooldown_events(self, engine_both: EquilibriumEngine):
        """apply_feedback_batch records COOLDOWN_APPLIED events."""
        engine_both.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.3, confidence=1.0, reason="test: batch 1"),
        ])
        engine_both.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.3, confidence=1.0, reason="test: batch 2"),
        ])
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) >= 1

    def test_batch_multi_source_cooldowns(self, engine_both: EquilibriumEngine):
        """Batch with multiple sources tracks cooldowns independently."""
        # First batch
        engine_both.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: cog batch 1"),
            Feedback(source=Pillar.PRAXIS, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: prax batch 1"),
        ])
        # Second batch — Cognition gets cooldown, Praxis gets cooldown too
        engine_both.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: cog batch 2"),
            Feedback(source=Pillar.PRAXIS, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: prax batch 2"),
        ])
        cooldown_events = engine_both.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        # Both Cognition and Praxis should have cooldown events
        assert len(cooldown_events) >= 2


# ======================================================================
# TestCooldownStatsMethod
# ======================================================================


class TestCooldownStatsMethod:
    """TensionEventLog.cooldown_stats() convenience method."""

    def test_cooldown_stats_empty_log(self):
        """cooldown_stats on empty log returns zeros."""
        log = TensionEventLog()
        stats = log.cooldown_stats()
        assert stats["total_cooldown_events"] == 0
        assert stats["affected_axes"] == []
        assert stats["affected_pillars"] == []

    def test_cooldown_stats_with_events(self, engine_both: EquilibriumEngine):
        """cooldown_stats returns correct summary after cooldown events."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))
        stats = engine_both.event_log.cooldown_stats()
        assert stats["total_cooldown_events"] >= 1
        assert "explore_exploit" in stats["affected_axes"]
        assert Pillar.COGNITION in stats["affected_pillars"]

    def test_cooldown_stats_average_multiplier(self, engine_both: EquilibriumEngine):
        """cooldown_stats includes average multiplier across cooldown events."""
        for i in range(4):
            engine_both.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.3, confidence=1.0, reason=f"test: fb {i}",
            ))
        stats = engine_both.event_log.cooldown_stats()
        assert "avg_multiplier" in stats
        # Average multiplier should be between MIN_MULTIPLIER and 1.0
        assert 0.0 < stats["avg_multiplier"] <= 1.0

    def test_cooldown_stats_per_axis_counts(self, engine_both: EquilibriumEngine):
        """cooldown_stats includes per-axis cooldown event counts."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first ee",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second ee",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="shallow_deep",
            signal=0.3, confidence=1.0, reason="test: first sd",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="shallow_deep",
            signal=0.3, confidence=1.0, reason="test: second sd",
        ))
        stats = engine_both.event_log.cooldown_stats()
        assert "per_axis" in stats
        assert stats["per_axis"].get("explore_exploit", 0) >= 1
        assert stats["per_axis"].get("shallow_deep", 0) >= 1


# ======================================================================
# TestCooldownEventSerialization
# ======================================================================


class TestCooldownEventSerialization:
    """COOLDOWN_APPLIED events survive serialization round-trip."""

    def test_cooldown_event_round_trip(self):
        """A COOLDOWN_APPLIED event serializes and deserializes correctly."""
        log = TensionEventLog()
        log.record(
            event_type=TensionEventType.COOLDOWN_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.3,
            position_after=0.3,
            delta=0.5,
            confidence=1.0,
            tick=2,
        )
        data = log.to_dict()
        restored = TensionEventLog.from_dict(data)
        events = restored.query(event_type=TensionEventType.COOLDOWN_APPLIED)
        assert len(events) == 1
        assert events[0].event_type == TensionEventType.COOLDOWN_APPLIED
        assert events[0].delta == 0.5
        assert events[0].axis_id == "explore_exploit"

    def test_engine_round_trip_preserves_cooldown_events(
        self, engine_both: EquilibriumEngine,
    ):
        """Engine serialization preserves COOLDOWN_APPLIED events in the log."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))
        data = engine_both.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        assert restored.event_log is not None
        cooldown_events = restored.event_log.query(
            event_type=TensionEventType.COOLDOWN_APPLIED,
        )
        assert len(cooldown_events) >= 1


# ======================================================================
# TestCooldownEventWithPillarView
# ======================================================================


class TestCooldownEventWithPillarView:
    """PillarEquilibriumView provides access to cooldown events."""

    def test_view_recent_events_includes_cooldown(self, engine_both: EquilibriumEngine):
        """PillarEquilibriumView.recent_events() includes COOLDOWN_APPLIED events."""
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine_both.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))
        view = engine_both.view_for(Pillar.COGNITION)
        recent = view.recent_events(limit=20)
        cooldown_events = [e for e in recent if e.event_type == TensionEventType.COOLDOWN_APPLIED]
        assert len(cooldown_events) >= 1
