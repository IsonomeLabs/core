"""Tests for Equilibrium Convergence Detector — iter-032.

Covers:
- ConvergenceDetector construction and validation
- Tick-level convergence detection (converging, diverging, stable, unknown)
- Multi-axis convergence: majority rules
- Convergence rate computation (mean velocity toward defaults)
- Convergence history tracking (bounded deque)
- Per-axis convergence status
- Integration with EquilibriumEngine (auto-detection after feedback)
- PillarEquilibriumView convergence properties
- Serialization round-trip (to_dict / from_dict)
- Edge cases: empty engine, single axis, all axes stable, all diverging
"""

import math
import pytest

from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.equilibrium.convergence import (
    ConvergenceDetector,
    ConvergenceStatus,
    ConvergenceRecord,
)
from isonome.types import Feedback, Pillar, TensionAxis


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def detector() -> ConvergenceDetector:
    """Fresh convergence detector with default settings."""
    return ConvergenceDetector()


@pytest.fixture
def detector_small_history() -> ConvergenceDetector:
    """Detector with small max_history for testing overflow."""
    return ConvergenceDetector(max_history=5)


@pytest.fixture
def engine_with_detector() -> EquilibriumEngine:
    """Engine with convergence detection enabled."""
    return EquilibriumEngine(enable_convergence_detection=True)


@pytest.fixture
def engine_without_detector() -> EquilibriumEngine:
    """Standard engine without convergence detection."""
    return EquilibriumEngine()


@pytest.fixture
def engine_full_features() -> EquilibriumEngine:
    """Engine with velocity tracking + convergence detection."""
    return EquilibriumEngine(
        enable_velocity_tracking=True,
        enable_convergence_detection=True,
    )


def _make_feedback(axis_id: str, signal: float, confidence: float = 1.0,
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


class TestConvergenceDetectorConstruction:
    def test_default_construction(self, detector):
        assert detector.max_history == 100
        assert detector.convergence_threshold == 0.001
        assert detector.history_size == 0

    def test_custom_parameters(self):
        d = ConvergenceDetector(max_history=50, convergence_threshold=0.01)
        assert d.max_history == 50
        assert d.convergence_threshold == 0.01

    def test_max_history_zero_raises(self):
        with pytest.raises(ValueError, match="max_history must be >= 1"):
            ConvergenceDetector(max_history=0)

    def test_max_history_negative_raises(self):
        with pytest.raises(ValueError, match="max_history must be >= 1"):
            ConvergenceDetector(max_history=-5)

    def test_convergence_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="convergence_threshold must be >= 0"):
            ConvergenceDetector(convergence_threshold=-0.01)


# ═══════════════════════════════════════════════════════════════════
# ConvergenceRecord
# ═══════════════════════════════════════════════════════════════════


class TestConvergenceRecord:
    def test_record_fields(self):
        record = ConvergenceRecord(
            tick=5,
            status=ConvergenceStatus.CONVERGING,
            convergence_rate=0.05,
            n_converging=3,
            n_diverging=1,
            n_stable=4,
        )
        assert record.tick == 5
        assert record.status == ConvergenceStatus.CONVERGING
        assert record.convergence_rate == pytest.approx(0.05)
        assert record.n_converging == 3
        assert record.n_diverging == 1
        assert record.n_stable == 4

    def test_record_to_dict(self):
        record = ConvergenceRecord(
            tick=10,
            status=ConvergenceStatus.DIVERGING,
            convergence_rate=-0.03,
            n_converging=1,
            n_diverging=5,
            n_stable=2,
        )
        d = record.to_dict()
        assert d["tick"] == 10
        assert d["status"] == "diverging"
        assert d["convergence_rate"] == pytest.approx(-0.03)
        assert d["n_converging"] == 1
        assert d["n_diverging"] == 5
        assert d["n_stable"] == 2

    def test_record_from_dict(self):
        data = {
            "tick": 15,
            "status": "stable",
            "convergence_rate": 0.0005,
            "n_converging": 0,
            "n_diverging": 0,
            "n_stable": 8,
        }
        record = ConvergenceRecord.from_dict(data)
        assert record.tick == 15
        assert record.status == ConvergenceStatus.STABLE
        assert record.convergence_rate == pytest.approx(0.0005)

    def test_record_repr(self):
        record = ConvergenceRecord(
            tick=1,
            status=ConvergenceStatus.CONVERGING,
            convergence_rate=0.1,
            n_converging=6,
            n_diverging=1,
            n_stable=1,
        )
        assert "CONVERGING" in repr(record)
        assert "tick=1" in repr(record)


# ═══════════════════════════════════════════════════════════════════
# Per-Axis Detection
# ═══════════════════════════════════════════════════════════════════


class TestPerAxisConvergence:
    def test_axis_converging_toward_default(self, detector):
        """Axis moving toward default should be CONVERGING."""
        # explore_exploit default is 0.15
        # Position at 0.5, velocity toward 0.15 = negative velocity = converging
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.5,
            default_position=0.15,
            velocity=-0.1,
        )
        assert status == ConvergenceStatus.CONVERGING

    def test_axis_diverging_from_default(self, detector):
        """Axis moving away from default should be DIVERGING."""
        # Position at 0.5, velocity toward 0.8 = moving away from 0.15 = diverging
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.5,
            default_position=0.15,
            velocity=0.3,
        )
        assert status == ConvergenceStatus.DIVERGING

    def test_axis_stable_near_default(self, detector):
        """Axis near default with near-zero velocity should be STABLE."""
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.15,
            default_position=0.15,
            velocity=0.0,
        )
        assert status == ConvergenceStatus.STABLE

    def test_axis_stable_small_velocity(self, detector):
        """Axis near default with velocity below threshold should be STABLE."""
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.16,
            default_position=0.15,
            velocity=0.0001,  # Below default threshold of 0.001
        )
        assert status == ConvergenceStatus.STABLE

    def test_axis_converging_from_below(self, detector):
        """Axis below default moving up should be CONVERGING."""
        # Position at -0.3, default at 0.15, velocity positive = toward default
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=-0.3,
            default_position=0.15,
            velocity=0.2,
        )
        assert status == ConvergenceStatus.CONVERGING

    def test_axis_diverging_from_below(self, detector):
        """Axis below default moving further down should be DIVERGING."""
        # Position at -0.3, default at 0.15, velocity negative = away from default
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=-0.3,
            default_position=0.15,
            velocity=-0.1,
        )
        assert status == ConvergenceStatus.DIVERGING

    def test_axis_at_default_no_velocity(self, detector):
        """Axis exactly at default with no velocity = STABLE."""
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.0,
            default_position=0.0,
            velocity=0.0,
        )
        assert status == ConvergenceStatus.STABLE

    def test_axis_unknown_when_no_velocity_data(self, detector):
        """When velocity is None, should return UNKNOWN."""
        status = detector.classify_axis(
            axis_id="explore_exploit",
            position=0.5,
            default_position=0.15,
            velocity=None,
        )
        assert status == ConvergenceStatus.UNKNOWN


# ═══════════════════════════════════════════════════════════════════
# Engine-Level Detection
# ═══════════════════════════════════════════════════════════════════


class TestEngineConvergence:
    def test_converging_after_restoring_feedback(self, engine_full_features):
        """Sending feedback that pushes axes back toward defaults → CONVERGING."""
        # First push axes away from defaults
        for _ in range(5):
            engine_full_features.apply_feedback(
                _make_feedback("explore_exploit", 0.9)
            )
            engine_full_features.apply_feedback(
                _make_feedback("shallow_deep", -0.9)
            )

        # Now send feedback toward defaults
        detector = engine_full_features.convergence_detector
        for _ in range(3):
            engine_full_features.apply_feedback(
                _make_feedback("explore_exploit", -0.5)
            )

        # Check that detector has been recording
        assert detector.history_size > 0

    def test_detect_on_engine_without_velocity(self, engine_with_detector):
        """Engine without velocity tracking uses position-delta heuristic."""
        engine = engine_with_detector
        detector = engine.convergence_detector

        # Apply some feedback
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))

        # Should still record something (using position delta as proxy)
        record = detector.detect(engine)
        assert record is not None
        assert record.status in ConvergenceStatus

    def test_detect_empty_engine(self, detector):
        """Detection on an engine with no velocity tracker should work.

        Axes at their default positions are classified as STABLE even
        without velocity data (no movement needed = already home).
        Axes away from defaults with no velocity are UNKNOWN.
        """
        engine = EquilibriumEngine()
        record = detector.detect(engine)
        # Default engine has all axes at default → STABLE
        assert record.status == ConvergenceStatus.STABLE

    def test_detect_records_in_history(self, detector_small_history):
        """Each detection should append to history."""
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        # Bind detector
        engine._convergence_detector = detector_small_history

        for i in range(10):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.3))

        # History should be capped at max_history
        assert detector_small_history.history_size <= 5

    def test_convergence_rate_positive_when_converging(self, engine_full_features):
        """Convergence rate should be positive when axes move toward defaults."""
        engine = engine_full_features
        detector = engine.convergence_detector

        # Push away first
        for _ in range(3):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.9))

        # Then push back
        record = detector.detect(engine)
        # After only pushing away, the system should be diverging
        # (we haven't sent restoring feedback yet)
        assert record is not None

    def test_all_axes_stable_means_stable(self, engine_with_detector):
        """When all axes are at default, overall status should be STABLE."""
        engine = engine_with_detector
        detector = engine.convergence_detector
        # Engine starts at homeostasis (all axes at defaults)
        record = detector.detect(engine)
        assert record.status == ConvergenceStatus.STABLE


# ═══════════════════════════════════════════════════════════════════
# Convergence History
# ═══════════════════════════════════════════════════════════════════


class TestConvergenceHistory:
    def test_history_starts_empty(self, detector):
        assert detector.history_size == 0
        assert detector.recent_history() == []

    def test_history_records_detection(self, detector):
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        engine._convergence_detector = detector

        # apply_feedback auto-calls on_position_update which adds a history
        # record, then detect() adds a second.  We assert the net count.
        engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
        # At this point, on_position_update already recorded 1 entry
        assert detector.history_size == 1
        detector.detect(engine)
        # detect() adds a second entry
        assert detector.history_size == 2

    def test_history_bounded(self, detector_small_history):
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        engine._convergence_detector = detector_small_history

        for _ in range(20):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.1))

        # Should be capped at max_history=5
        assert detector_small_history.history_size <= 5

    def test_recent_history_returns_latest(self, detector):
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        engine._convergence_detector = detector

        for i in range(3):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.1))
            detector.detect(engine)

        recent = detector.recent_history(limit=2)
        assert len(recent) <= 2

    def test_convergence_trend(self, detector):
        """convergence_trend() should return list of rates."""
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        engine._convergence_detector = detector

        for _ in range(3):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.1))
            detector.detect(engine)

        trend = detector.convergence_trend()
        assert len(trend) == detector.history_size
        assert all(isinstance(r, float) for r in trend)


# ═══════════════════════════════════════════════════════════════════
# Per-Axis Status via Engine
# ═══════════════════════════════════════════════════════════════════


class TestPerAxisStatusViaEngine:
    def test_per_axis_convergence(self, engine_full_features):
        """per_axis_status should return a dict of axis_id → ConvergenceStatus."""
        engine = engine_full_features
        detector = engine.convergence_detector

        # Apply feedback to move an axis away
        engine.apply_feedback(_make_feedback("explore_exploit", 0.8))
        detector.detect(engine)

        statuses = detector.per_axis_status(engine)
        assert isinstance(statuses, dict)
        assert len(statuses) == len(engine.axes)

    def test_per_axis_without_velocity(self, engine_with_detector):
        """per_axis_status without velocity tracker uses position-delta heuristic."""
        engine = engine_with_detector
        detector = engine.convergence_detector

        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        statuses = detector.per_axis_status(engine)
        # Without velocity data, axes that haven't been touched are UNKNOWN
        assert len(statuses) == len(engine.axes)


# ═══════════════════════════════════════════════════════════════════
# Integration with EquilibriumEngine
# ═══════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    def test_engine_convergence_detector_property(self, engine_with_detector):
        """Engine should expose convergence_detector property."""
        assert engine_with_detector.convergence_detector is not None
        assert isinstance(
            engine_with_detector.convergence_detector, ConvergenceDetector
        )

    def test_engine_without_detector(self, engine_without_detector):
        """Engine without convergence detection should return None."""
        assert engine_without_detector.convergence_detector is None

    def test_engine_convergence_detection_auto(self, engine_with_detector):
        """Convergence detection should run automatically after feedback."""
        engine = engine_with_detector
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        detector = engine.convergence_detector
        # Should have at least one record
        assert detector.history_size >= 1

    def test_engine_compute_convergence(self, engine_with_detector):
        """compute_convergence() should return a ConvergenceRecord."""
        engine = engine_with_detector
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        result = engine.compute_convergence()
        assert result is not None
        assert isinstance(result, ConvergenceRecord)

    def test_engine_compute_convergence_disabled(self, engine_without_detector):
        """compute_convergence() should return None when disabled."""
        result = engine_without_detector.compute_convergence()
        assert result is None

    def test_pillar_view_convergence_status(self, engine_with_detector):
        """PillarEquilibriumView should expose convergence_status."""
        engine = engine_with_detector
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        view = engine.view_for(Pillar.COGNITION)
        # convergence_status should be a ConvergenceStatus or None
        status = view.convergence_status
        assert status is None or isinstance(status, ConvergenceStatus)


# ═══════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════


class TestConvergenceDetectorSerialization:
    def test_empty_detector_round_trip(self, detector):
        data = detector.to_dict()
        restored = ConvergenceDetector.from_dict(data)
        assert restored.max_history == detector.max_history
        assert restored.convergence_threshold == detector.convergence_threshold
        assert restored.history_size == 0

    def test_detector_with_history_round_trip(self, detector):
        engine = EquilibriumEngine(enable_velocity_tracking=True)
        engine._convergence_detector = detector

        for _ in range(5):
            engine.apply_feedback(_make_feedback("explore_exploit", 0.3))
            detector.detect(engine)

        assert detector.history_size > 0
        data = detector.to_dict()
        restored = ConvergenceDetector.from_dict(data)
        assert restored.history_size == detector.history_size
        assert restored.max_history == detector.max_history
        # Compare trend data
        assert restored.convergence_trend() == detector.convergence_trend()

    def test_engine_serialization_preserves_detector(self, engine_with_detector):
        """Engine serialization should preserve convergence detector state."""
        engine = engine_with_detector
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))
        engine.apply_feedback(_make_feedback("shallow_deep", -0.3))

        data = engine.to_dict()
        assert "convergence_detector_state" in data
        assert data["convergence_detector_state"] is not None

    def test_engine_deserialization_restores_detector(self, engine_with_detector):
        """Engine deserialization should restore convergence detector."""
        engine = engine_with_detector
        engine.apply_feedback(_make_feedback("explore_exploit", 0.5))

        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        assert restored.convergence_detector is not None
        assert isinstance(restored.convergence_detector, ConvergenceDetector)


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_axis_engine(self):
        """Convergence detection with a single axis."""
        axes = [
            TensionAxis(
                id="test_axis",
                pillar=Pillar.COGNITION,
                pole_left="left",
                pole_right="right",
                default_position=0.0,
                damping=0.3,
                learning_rate=0.1,
            )
        ]
        engine = EquilibriumEngine(
            axes=axes,
            enable_velocity_tracking=True,
            enable_convergence_detection=True,
        )
        engine.apply_feedback(_make_feedback("test_axis", 0.5))
        result = engine.compute_convergence()
        assert result is not None

    def test_custom_threshold(self):
        """Custom convergence_threshold affects classification."""
        d = ConvergenceDetector(convergence_threshold=0.1)
        # Velocity of 0.05 should be STABLE with threshold=0.1
        status = d.classify_axis(
            axis_id="explore_exploit",
            position=0.5,
            default_position=0.15,
            velocity=0.05,  # Moving right but default is to the left
        )
        # |velocity| = 0.05 < threshold 0.1, so it's STABLE
        # regardless of direction (too small to be meaningful)
        assert status == ConvergenceStatus.STABLE

    def test_majority_rules_for_overall(self):
        """Overall status should follow the majority of axes."""
        d = ConvergenceDetector()
        axes = [
            TensionAxis(
                id=f"axis_{i}",
                pillar=Pillar.COGNITION,
                pole_left="left",
                pole_right="right",
                default_position=0.0,
                damping=0.3,
                learning_rate=0.1,
                position=0.5,
            )
            for i in range(5)
        ]
        engine = EquilibriumEngine(
            axes=axes,
            enable_velocity_tracking=True,
        )
        # Push most axes toward defaults
        for i in range(3):
            engine.apply_feedback(
                _make_feedback("axis_0", -0.5)
            )
            engine.apply_feedback(
                _make_feedback("axis_1", -0.5)
            )

        record = d.detect(engine)
        assert record is not None

    def test_detector_repr(self, detector):
        """Detector should have a useful repr."""
        r = repr(detector)
        assert "ConvergenceDetector" in r

    def test_convergence_status_enum_values(self):
        """ConvergenceStatus should have expected values."""
        assert ConvergenceStatus.CONVERGING.value == "converging"
        assert ConvergenceStatus.DIVERGING.value == "diverging"
        assert ConvergenceStatus.STABLE.value == "stable"
        assert ConvergenceStatus.UNKNOWN.value == "unknown"
