"""Tests for TensionEventLog analysis methods — iter-032.

Covers:
- Pillar stress scores (per-pillar stress timeline)
- Axis volatility (stability metric from position history)
- Feedback burst detection (consecutive rapid feedback)
- Dominant feedback source (which pillar most affects an axis)
- Convergence/divergence detection from events
- Cross-pillar conflict detection (same axis, opposing signals)
- PillarEquilibriumView integration for analysis access
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
def engine_all_features() -> EquilibriumEngine:
    """Engine with all features enabled for integration tests."""
    return EquilibriumEngine(
        enable_event_log=True,
        enable_velocity_tracking=True,
        enable_adaptive_damping=True,
        enable_feedback_cooldown=True,
        enable_health_score=True,
        enable_convergence_detection=True,
    )


@pytest.fixture
def engine_with_log() -> EquilibriumEngine:
    """Engine with event logging enabled."""
    return EquilibriumEngine(enable_event_log=True)


@pytest.fixture
def populated_log() -> TensionEventLog:
    """Log with events from multiple pillars and axes."""
    log = TensionEventLog(max_events=100)
    # Cognition pushing explore_exploit right
    log.record(
        event_type=TensionEventType.FEEDBACK_APPLIED,
        axis_id="explore_exploit",
        source_pillar=Pillar.COGNITION,
        position_before=0.0,
        position_after=0.2,
        delta=0.2,
        confidence=0.8,
        tick=1,
    )
    # Praxis pushing autonomy_safety left
    log.record(
        event_type=TensionEventType.FEEDBACK_APPLIED,
        axis_id="autonomy_safety",
        source_pillar=Pillar.PRAXIS,
        position_before=0.0,
        position_after=-0.3,
        delta=-0.3,
        confidence=0.9,
        tick=1,
    )
    # Cognition pushing shallow_deep right
    log.record(
        event_type=TensionEventType.FEEDBACK_APPLIED,
        axis_id="shallow_deep",
        source_pillar=Pillar.COGNITION,
        position_before=0.0,
        position_after=0.15,
        delta=0.15,
        confidence=0.7,
        tick=2,
    )
    # Oscillation detected on explore_exploit
    log.record(
        event_type=TensionEventType.OSCILLATION_DETECTED,
        axis_id="explore_exploit",
        source_pillar=Pillar.COGNITION,
        position_before=0.2,
        position_after=0.5,
        delta=0.3,
        confidence=0.0,
        tick=3,
    )
    # Praxis pushing explore_exploit left (conflict)
    log.record(
        event_type=TensionEventType.FEEDBACK_APPLIED,
        axis_id="explore_exploit",
        source_pillar=Pillar.PRAXIS,
        position_before=0.2,
        position_after=-0.1,
        delta=-0.3,
        confidence=0.6,
        tick=3,
    )
    return log


# ── pillar_stress_scores ──────────────────────────────────────────


class TestPillarStressScores:
    """Tests for TensionEventLog.pillar_stress_scores()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns empty dict."""
        assert log.pillar_stress_scores() == {}

    def test_single_pillar_single_event(self, log: TensionEventLog):
        """One pillar, one feedback event produces a stress score."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        scores = log.pillar_stress_scores()
        assert Pillar.COGNITION in scores
        # Single event, stress = |delta| * confidence = 0.24
        assert abs(scores[Pillar.COGNITION] - 0.24) < 1e-9

    def test_multiple_pillars(self, populated_log: TensionEventLog):
        """Multiple pillars tracked separately."""
        scores = populated_log.pillar_stress_scores()
        assert Pillar.COGNITION in scores
        assert Pillar.PRAXIS in scores

    def test_non_feedback_events_ignored(self, log: TensionEventLog):
        """Only FEEDBACK_APPLIED events contribute to stress."""
        log.record(
            event_type=TensionEventType.OSCILLATION_DETECTED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.5,
            delta=0.5,
            confidence=0.0,
            tick=1,
        )
        assert log.pillar_stress_scores() == {}

    def test_zero_delta_feedback(self, log: TensionEventLog):
        """Zero delta feedback contributes zero stress."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.0,
            delta=0.0,
            confidence=1.0,
            tick=1,
        )
        scores = log.pillar_stress_scores()
        assert scores[Pillar.COGNITION] == 0.0


# ── axis_volatility ──────────────────────────────────────────────


class TestAxisVolatility:
    """Tests for TensionEventLog.axis_volatility()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns empty dict."""
        assert log.axis_volatility() == {}

    def test_single_position_no_volatility(self, log: TensionEventLog):
        """Single position entry means zero volatility."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        vol = log.axis_volatility()
        assert "explore_exploit" in vol
        assert vol["explore_exploit"] == 0.0

    def test_stable_axis_low_volatility(self, log: TensionEventLog):
        """Axis with consistent positions has low volatility."""
        for i in range(5):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.1 * i,
                position_after=0.1 * (i + 1),
                delta=0.1,
                confidence=0.8,
                tick=i + 1,
            )
        vol = log.axis_volatility()
        assert "explore_exploit" in vol
        # Small consistent changes → low volatility
        assert vol["explore_exploit"] < 0.2

    def test_oscillating_axis_high_volatility(self, log: TensionEventLog):
        """Oscillating axis has high volatility."""
        positions = [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
        for i, pos in enumerate(positions):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=positions[i - 1] if i > 0 else 0.0,
                position_after=pos,
                delta=pos - (positions[i - 1] if i > 0 else 0.0),
                confidence=0.8,
                tick=i + 1,
            )
        vol = log.axis_volatility()
        assert "explore_exploit" in vol
        # Large swings → high volatility
        assert vol["explore_exploit"] > 0.3

    def test_multiple_axes_tracked(self, populated_log: TensionEventLog):
        """Multiple axes have independent volatility scores."""
        vol = populated_log.axis_volatility()
        assert "explore_exploit" in vol
        assert "autonomy_safety" in vol
        assert "shallow_deep" in vol

    def test_engine_wide_events_skipped(self, log: TensionEventLog):
        """Engine-wide events (empty axis_id) are not included."""
        log.record(
            event_type=TensionEventType.RESET,
            axis_id="",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.0,
            delta=0.0,
            confidence=0.0,
            tick=1,
        )
        assert log.axis_volatility() == {}


# ── detect_feedback_bursts ───────────────────────────────────────


class TestDetectFeedbackBursts:
    """Tests for TensionEventLog.detect_feedback_bursts()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns empty list."""
        assert log.detect_feedback_bursts() == []

    def test_no_bursts_spaced_out(self, log: TensionEventLog):
        """Widely spaced events produce no bursts."""
        for tick in [1, 10, 20, 30]:
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.1 * tick,
                delta=0.1,
                confidence=0.8,
                tick=tick,
            )
        bursts = log.detect_feedback_bursts(window=3, threshold=3)
        assert len(bursts) == 0

    def test_burst_detected(self, log: TensionEventLog):
        """Rapid consecutive feedback triggers a burst."""
        for tick in [1, 1, 1, 2, 2]:
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.1,
                delta=0.1,
                confidence=0.8,
                tick=tick,
            )
        bursts = log.detect_feedback_bursts(window=3, threshold=3)
        assert len(bursts) >= 1
        # The burst should identify the axis
        burst = bursts[0]
        assert "axis_id" in burst
        assert burst["axis_id"] == "explore_exploit"

    def test_burst_structure(self, log: TensionEventLog):
        """Burst dict has required keys."""
        for tick in [1, 1, 1, 1]:
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.1,
                delta=0.1,
                confidence=0.8,
                tick=tick,
            )
        bursts = log.detect_feedback_bursts(window=5, threshold=3)
        assert len(bursts) >= 1
        burst = bursts[0]
        assert "axis_id" in burst
        assert "tick_start" in burst
        assert "tick_end" in burst
        assert "event_count" in burst
        assert burst["event_count"] >= 3

    def test_non_feedback_events_excluded(self, log: TensionEventLog):
        """Non-feedback events don't contribute to burst detection."""
        for tick in [1, 1, 1]:
            log.record(
                event_type=TensionEventType.OSCILLATION_DETECTED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.5,
                delta=0.5,
                confidence=0.0,
                tick=tick,
            )
        bursts = log.detect_feedback_bursts(window=5, threshold=3)
        assert len(bursts) == 0

    def test_custom_threshold(self, log: TensionEventLog):
        """Higher threshold requires more events to trigger a burst."""
        for tick in [1, 1, 1]:
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=0.1,
                delta=0.1,
                confidence=0.8,
                tick=tick,
            )
        # threshold=3 should detect
        assert len(log.detect_feedback_bursts(window=5, threshold=3)) >= 1
        # threshold=4 should not
        assert len(log.detect_feedback_bursts(window=5, threshold=4)) == 0


# ── dominant_feedback_source ─────────────────────────────────────


class TestDominantFeedbackSource:
    """Tests for TensionEventLog.dominant_feedback_source()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns empty dict."""
        assert log.dominant_feedback_source() == {}

    def test_single_source(self, log: TensionEventLog):
        """One pillar dominating one axis."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        dominant = log.dominant_feedback_source()
        assert "explore_exploit" in dominant
        assert dominant["explore_exploit"]["pillar"] == Pillar.COGNITION
        assert dominant["explore_exploit"]["total_weight"] == pytest.approx(0.24)

    def test_contested_axis(self, populated_log: TensionEventLog):
        """Axis with feedback from multiple pillars shows the dominant one."""
        dominant = populated_log.dominant_feedback_source()
        # explore_exploit gets feedback from Cognition (0.2*0.8=0.16)
        # and Praxis (-0.3*0.6=0.18, but we use |delta|*confidence)
        assert "explore_exploit" in dominant
        # The dominant pillar is the one with the highest total |delta|*confidence
        entry = dominant["explore_exploit"]
        assert "pillar" in entry
        assert entry["pillar"] in (Pillar.COGNITION, Pillar.PRAXIS)

    def test_non_feedback_ignored(self, log: TensionEventLog):
        """Only FEEDBACK_APPLIED events are considered."""
        log.record(
            event_type=TensionEventType.OSCILLATION_DETECTED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.5,
            delta=0.5,
            confidence=0.0,
            tick=1,
        )
        assert log.dominant_feedback_source() == {}

    def test_entry_structure(self, log: TensionEventLog):
        """Each entry has pillar, total_weight, and event_count."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        entry = log.dominant_feedback_source()["explore_exploit"]
        assert "pillar" in entry
        assert "total_weight" in entry
        assert "event_count" in entry
        assert entry["event_count"] == 1


# ── detect_convergence_from_events ────────────────────────────────


class TestDetectConvergenceFromEvents:
    """Tests for TensionEventLog.detect_convergence_from_events()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns 'unknown'."""
        result = log.detect_convergence_from_events()
        assert result["direction"] == "unknown"
        assert result["confidence"] == 0.0

    def test_converging_events(self, log: TensionEventLog):
        """Decreasing deltas indicate convergence."""
        deltas = [0.5, 0.3, 0.2, 0.1, 0.05]
        for i, d in enumerate(deltas):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=d,
                delta=d,
                confidence=0.8,
                tick=i + 1,
            )
        result = log.detect_convergence_from_events()
        assert result["direction"] in ("converging", "stable")

    def test_diverging_events(self, log: TensionEventLog):
        """Increasing deltas indicate divergence."""
        deltas = [0.05, 0.1, 0.2, 0.4, 0.6]
        for i, d in enumerate(deltas):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=d,
                delta=d,
                confidence=0.8,
                tick=i + 1,
            )
        result = log.detect_convergence_from_events()
        assert result["direction"] == "diverging"

    def test_stable_events(self, log: TensionEventLog):
        """Consistent small deltas indicate stability."""
        deltas = [0.01, 0.02, 0.01, 0.02, 0.01]
        for i, d in enumerate(deltas):
            log.record(
                event_type=TensionEventType.FEEDBACK_APPLIED,
                axis_id="explore_exploit",
                source_pillar=Pillar.COGNITION,
                position_before=0.0,
                position_after=d,
                delta=d,
                confidence=0.8,
                tick=i + 1,
            )
        result = log.detect_convergence_from_events()
        assert result["direction"] in ("converging", "stable", "unknown")

    def test_result_structure(self, log: TensionEventLog):
        """Result has direction, confidence, and trend_slope."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        result = log.detect_convergence_from_events()
        assert "direction" in result
        assert "confidence" in result
        assert "trend_slope" in result

    def test_oscillation_events_suggest_divergence(self, log: TensionEventLog):
        """Presence of oscillation events biases toward divergence."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.OSCILLATION_DETECTED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.3,
            position_after=0.5,
            delta=0.3,
            confidence=0.0,
            tick=2,
        )
        result = log.detect_convergence_from_events()
        # With oscillation, direction should not be "converging"
        assert result["direction"] != "converging"


# ── detect_cross_pillar_conflicts ─────────────────────────────────


class TestDetectCrossPillarConflicts:
    """Tests for TensionEventLog.detect_cross_pillar_conflicts()."""

    def test_empty_log(self, log: TensionEventLog):
        """Empty log returns empty list."""
        assert log.detect_cross_pillar_conflicts() == []

    def test_no_conflict_same_direction(self, log: TensionEventLog):
        """Same direction from different pillars = no conflict."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.PRAXIS,
            position_before=0.3,
            position_after=0.5,
            delta=0.2,
            confidence=0.7,
            tick=2,
        )
        conflicts = log.detect_cross_pillar_conflicts()
        # Both positive = same direction = no conflict
        assert len(conflicts) == 0

    def test_conflict_opposing_directions(self, log: TensionEventLog):
        """Opposing feedback from different pillars = conflict."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.PRAXIS,
            position_before=0.3,
            position_after=-0.1,
            delta=-0.4,
            confidence=0.7,
            tick=2,
        )
        conflicts = log.detect_cross_pillar_conflicts()
        assert len(conflicts) >= 1
        conflict = conflicts[0]
        assert conflict["axis_id"] == "explore_exploit"
        assert conflict["pillars"] == {Pillar.COGNITION, Pillar.PRAXIS}

    def test_conflict_structure(self, log: TensionEventLog):
        """Conflict dict has required keys."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.PRAXIS,
            position_before=0.3,
            position_after=-0.1,
            delta=-0.4,
            confidence=0.7,
            tick=2,
        )
        conflicts = log.detect_cross_pillar_conflicts()
        conflict = conflicts[0]
        assert "axis_id" in conflict
        assert "pillars" in conflict
        assert "opposing_deltas" in conflict
        assert "conflict_intensity" in conflict

    def test_single_pillar_no_conflict(self, log: TensionEventLog):
        """Feedback from only one pillar cannot create a cross-pillar conflict."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.3,
            delta=0.3,
            confidence=0.8,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.3,
            position_after=-0.1,
            delta=-0.4,
            confidence=0.7,
            tick=2,
        )
        # Same pillar pushing both ways is NOT a cross-pillar conflict
        conflicts = log.detect_cross_pillar_conflicts()
        assert len(conflicts) == 0

    def test_conflict_intensity_range(self, log: TensionEventLog):
        """Conflict intensity is in [0, 1]."""
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            position_before=0.0,
            position_after=0.5,
            delta=0.5,
            confidence=1.0,
            tick=1,
        )
        log.record(
            event_type=TensionEventType.FEEDBACK_APPLIED,
            axis_id="explore_exploit",
            source_pillar=Pillar.PRAXIS,
            position_before=0.5,
            position_after=-0.5,
            delta=-1.0,
            confidence=1.0,
            tick=2,
        )
        conflicts = log.detect_cross_pillar_conflicts()
        assert len(conflicts) >= 1
        for conflict in conflicts:
            assert 0.0 <= conflict["conflict_intensity"] <= 1.0


# ── Integration tests ─────────────────────────────────────────────


class TestEngineIntegration:
    """Tests for EquilibriumEngine integration with analysis methods."""

    def test_engine_log_has_analysis_methods(self, engine_with_log: EquilibriumEngine):
        """Engine's event log supports the new analysis methods."""
        log = engine_with_log.event_log
        assert log is not None
        assert hasattr(log, "pillar_stress_scores")
        assert hasattr(log, "axis_volatility")
        assert hasattr(log, "detect_feedback_bursts")
        assert hasattr(log, "dominant_feedback_source")
        assert hasattr(log, "detect_convergence_from_events")
        assert hasattr(log, "detect_cross_pillar_conflicts")

    def test_engine_feedback_populates_analysis(self, engine_with_log: EquilibriumEngine):
        """Applying feedback populates the analysis-accessible data."""
        fb = Feedback(
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=0.8,
            source=Pillar.COGNITION,
            reason="test feedback",
        )
        engine_with_log.apply_feedback(fb)
        log = engine_with_log.event_log
        assert log is not None
        # Analysis should work on engine-generated data
        scores = log.pillar_stress_scores()
        assert Pillar.COGNITION in scores

    def test_pillar_view_exposes_analysis(self, engine_with_log: EquilibriumEngine):
        """PillarEquilibriumView exposes analysis through the event log."""
        fb = Feedback(
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=0.8,
            source=Pillar.COGNITION,
            reason="test feedback",
        )
        engine_with_log.apply_feedback(fb)
        view = engine_with_log.view_for(Pillar.COGNITION)
        # View provides access to the event log
        assert view.event_log is not None

    def test_analysis_after_serialization_roundtrip(self, engine_with_log: EquilibriumEngine):
        """Analysis works on a deserialized event log."""
        fb = Feedback(
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=0.8,
            source=Pillar.COGNITION,
            reason="test feedback",
        )
        engine_with_log.apply_feedback(fb)

        # Serialize and deserialize
        data = engine_with_log.event_log.to_dict()
        restored = TensionEventLog.from_dict(data)

        # Analysis should work on restored data
        scores = restored.pillar_stress_scores()
        assert Pillar.COGNITION in scores
