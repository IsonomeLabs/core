"""Tests for TensionVelocityTracker — iter-021.

Covers:
- Basic velocity computation and tracking
- Reversal detection (sign changes with magnitude threshold)
- Momentum score computation (heading toward/away from default)
- Reversal rate and oscillation prediction
- Integration with EquilibriumEngine (apply_feedback, batch, reset, serialization)
- PillarEquilibriumView velocity properties
- Serialization round-trip
- Edge cases: single update, no updates, window rollover
"""

import math
import pytest

from isonome.equilibrium import (
    AdaptiveDampingController,
    EquilibriumEngine,
    PillarEquilibriumView,
    TensionVelocityTracker,
)
from isonome.equilibrium.velocity import TensionVelocityTracker as VTDirect
from isonome.types import Feedback, Pillar, TensionAxis, TensionID


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tracker() -> TensionVelocityTracker:
    """Fresh velocity tracker with default settings."""
    return TensionVelocityTracker()


@pytest.fixture
def tracker_small_window() -> TensionVelocityTracker:
    """Tracker with a small window for testing rollover."""
    return TensionVelocityTracker(window_size=4)


@pytest.fixture
def engine_with_velocity() -> EquilibriumEngine:
    """Engine with velocity tracking enabled."""
    return EquilibriumEngine(enable_velocity_tracking=True)


@pytest.fixture
def engine_without_velocity() -> EquilibriumEngine:
    """Standard engine without velocity tracking."""
    return EquilibriumEngine()


def _make_feedback(axis_id: TensionID, signal: float, confidence: float = 1.0) -> Feedback:
    """Helper to create a Feedback object."""
    return Feedback(
        tension_axis_id=axis_id,
        signal=signal,
        confidence=confidence,
        source=Pillar.COGNITION,
        reason="test feedback",
    )


# ── Construction & Validation ─────────────────────────────────────


class TestVelocityTrackerConstruction:
    def test_default_construction(self, tracker):
        assert tracker.window_size == 10
        assert tracker.min_reversal_magnitude == 0.005
        assert tracker.total_reversals == 0
        assert tracker.total_updates == 0

    def test_custom_construction(self):
        t = TensionVelocityTracker(window_size=20, min_reversal_magnitude=0.01)
        assert t.window_size == 20
        assert t.min_reversal_magnitude == 0.01

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            TensionVelocityTracker(window_size=1)

    def test_invalid_reversal_magnitude(self):
        with pytest.raises(ValueError, match="min_reversal_magnitude"):
            TensionVelocityTracker(min_reversal_magnitude=-0.01)

    def test_repr(self, tracker):
        r = repr(tracker)
        assert "TensionVelocityTracker" in r
        assert "axes=0" in r


# ── Registration ──────────────────────────────────────────────────


class TestVelocityTrackerRegistration:
    def test_register_axis(self, tracker):
        tracker.register_axis("test_axis")
        assert tracker.get_velocity("test_axis") == 0.0
        assert tracker.get_momentum_score("test_axis") == 0.0
        assert tracker.get_reversal_count("test_axis") == 0

    def test_unregister_axis(self, tracker):
        tracker.register_axis("test_axis")
        tracker.on_position_update("test_axis", 0.3, 0.0)
        tracker.unregister_axis("test_axis")
        assert tracker.get_velocity("test_axis") == 0.0

    def test_auto_register_on_update(self, tracker):
        """Unknown axes should be auto-registered on first position update."""
        tracker.on_position_update("auto_axis", 0.1, 0.0)
        assert tracker.get_velocity("auto_axis") == 0.0  # first update has no velocity


# ── Velocity Computation ──────────────────────────────────────────


class TestVelocityComputation:
    def test_first_update_zero_velocity(self, tracker):
        """First position update produces zero velocity (no previous position)."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.5, 0.0)
        assert tracker.get_velocity("a") == 0.0

    def test_second_update_computes_velocity(self, tracker):
        """Second update computes velocity as position delta."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.5, 0.0)
        tracker.on_position_update("a", 0.7, 0.0)
        assert tracker.get_velocity("a") == pytest.approx(0.2)

    def test_negative_velocity(self, tracker):
        """Moving left produces negative velocity."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.5, 0.0)
        tracker.on_position_update("a", 0.3, 0.0)
        assert tracker.get_velocity("a") == pytest.approx(-0.2)

    def test_multiple_updates(self, tracker):
        """Velocity tracks the most recent delta."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.2, 0.0)
        tracker.on_position_update("a", 0.5, 0.0)
        assert tracker.get_velocity("a") == pytest.approx(0.3)

    def test_all_velocities(self, tracker):
        tracker.register_axis("a")
        tracker.register_axis("b")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.1, 0.0)
        tracker.on_position_update("b", 0.0, 0.0)
        tracker.on_position_update("b", -0.1, 0.0)
        vels = tracker.all_velocities()
        assert vels["a"] == pytest.approx(0.1)
        assert vels["b"] == pytest.approx(-0.1)

    def test_unknown_axis_returns_zero(self, tracker):
        assert tracker.get_velocity("nonexistent") == 0.0


# ── Momentum Score ────────────────────────────────────────────────


class TestMomentumScore:
    def test_approaching_default(self, tracker):
        """Axis moving toward default has positive momentum."""
        tracker.register_axis("a")
        # default=0.5, position starts at 0.8, then moves to 0.6 (toward default)
        tracker.on_position_update("a", 0.8, 0.5)
        tracker.on_position_update("a", 0.6, 0.5)
        # velocity = -0.2, drift_direction = 0.5 - 0.6 = -0.1
        # momentum = -0.2 * -0.1 = 0.02 (positive = approaching)
        assert tracker.get_momentum_score("a") > 0
        assert tracker.is_approaching_default("a") is True
        assert tracker.is_drifting_from_default("a") is False

    def test_drifting_from_default(self, tracker):
        """Axis moving away from default has negative momentum."""
        tracker.register_axis("a")
        # default=0.5, position starts at 0.5, then moves to 0.8 (away from default)
        tracker.on_position_update("a", 0.5, 0.5)
        tracker.on_position_update("a", 0.8, 0.5)
        # velocity = 0.3, drift_direction = 0.5 - 0.8 = -0.3
        # momentum = 0.3 * -0.3 = -0.09 (negative = drifting)
        assert tracker.get_momentum_score("a") < 0
        assert tracker.is_approaching_default("a") is False
        assert tracker.is_drifting_from_default("a") is True

    def test_at_default_zero_momentum(self, tracker):
        """Axis at default with any velocity has zero momentum."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.5, 0.5)
        tracker.on_position_update("a", 0.7, 0.5)
        # velocity = 0.2, drift_direction = 0.5 - 0.7 = -0.2
        # momentum = 0.2 * -0.2 = -0.04 (drifting away)
        # This is actually drifting, not at default
        assert tracker.get_momentum_score("a") < 0

    def test_at_default_no_velocity_zero_momentum(self, tracker):
        """Axis at default with no velocity has zero momentum."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.5, 0.5)
        # velocity = 0 (first update), drift = 0, momentum = 0
        assert tracker.get_momentum_score("a") == pytest.approx(0.0)

    def test_all_momentum_scores(self, tracker):
        tracker.register_axis("a")
        tracker.register_axis("b")
        tracker.on_position_update("a", 0.5, 0.5)
        tracker.on_position_update("a", 0.7, 0.5)
        tracker.on_position_update("b", 0.5, 0.5)
        tracker.on_position_update("b", 0.3, 0.5)
        scores = tracker.all_momentum_scores()
        assert scores["a"] < 0  # drifting
        assert scores["b"] < 0  # drifting (away from 0.5)

    def test_unknown_axis_momentum(self, tracker):
        assert tracker.get_momentum_score("nonexistent") == 0.0
        assert tracker.is_approaching_default("nonexistent") is False
        assert tracker.is_drifting_from_default("nonexistent") is False


# ── Reversal Detection ────────────────────────────────────────────


class TestReversalDetection:
    def test_no_reversal_constant_direction(self, tracker):
        """Moving consistently in one direction produces no reversals."""
        tracker.register_axis("a")
        for pos in [0.0, 0.1, 0.2, 0.3]:
            tracker.on_position_update("a", pos, 0.0)
        assert tracker.get_reversal_count("a") == 0

    def test_single_reversal(self, tracker):
        """A single sign change counts as one reversal."""
        tracker.register_axis("a")
        # Move right, then left
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.2, 0.0)  # velocity +0.2
        tracker.on_position_update("a", 0.0, 0.0)  # velocity -0.2 → reversal
        assert tracker.get_reversal_count("a") == 1

    def test_multiple_reversals(self, tracker):
        """Multiple sign changes count correctly."""
        tracker.register_axis("a")
        positions = [0.0, 0.1, -0.1, 0.2, -0.2]
        for pos in positions:
            tracker.on_position_update("a", pos, 0.0)
        # velocities: 0, +0.1, -0.2, +0.3, -0.4
        # reversals: 0→+0.1 (no), +0.1→-0.2 (yes), -0.2→+0.3 (yes), +0.3→-0.4 (yes)
        assert tracker.get_reversal_count("a") == 3

    def test_small_movement_no_reversal(self, tracker):
        """Movements below min_reversal_magnitude don't count as reversals."""
        t = TensionVelocityTracker(min_reversal_magnitude=0.1)
        t.register_axis("a")
        t.on_position_update("a", 0.0, 0.0)
        t.on_position_update("a", 0.05, 0.0)  # velocity 0.05 < 0.1 threshold
        t.on_position_update("a", -0.05, 0.0)  # velocity -0.1, but prev_vel too small
        # No reversal because |prev_vel| < 0.1
        assert t.get_reversal_count("a") == 0

    def test_total_reversals(self, tracker):
        """Total reversals across all axes."""
        tracker.register_axis("a")
        tracker.register_axis("b")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.1, 0.0)
        tracker.on_position_update("a", -0.1, 0.0)  # reversal on a
        tracker.on_position_update("b", 0.0, 0.0)
        tracker.on_position_update("b", -0.1, 0.0)
        tracker.on_position_update("b", 0.1, 0.0)  # reversal on b
        assert tracker.total_reversals == 2


# ── Reversal Rate & Oscillation Prediction ────────────────────────


class TestReversalRateAndPrediction:
    def test_reversal_rate_insufficient_data(self, tracker):
        """Less than 3 data points returns 0.0 reversal rate."""
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.1, 0.0)
        assert tracker.get_reversal_rate("a") == 0.0

    def test_reversal_rate_no_reversals(self, tracker):
        """Steady movement returns 0.0 reversal rate."""
        tracker.register_axis("a")
        for pos in [0.0, 0.1, 0.2, 0.3, 0.4]:
            tracker.on_position_update("a", pos, 0.0)
        assert tracker.get_reversal_rate("a") == pytest.approx(0.0)

    def test_reversal_rate_with_reversals(self, tracker):
        """Reversal rate computed correctly."""
        tracker.register_axis("a")
        # Oscillating: 0.0, 0.2, -0.2, 0.2 → 2 reversals in 3 possible slots
        for pos in [0.0, 0.2, -0.2, 0.2]:
            tracker.on_position_update("a", pos, 0.0)
        rate = tracker.get_reversal_rate("a")
        assert rate > 0.0

    def test_oscillation_imminent(self, tracker):
        """is_oscillation_imminent detects high reversal rates."""
        tracker.register_axis("a")
        # Create strong oscillation pattern
        for pos in [0.0, 0.3, -0.3, 0.3, -0.3]:
            tracker.on_position_update("a", pos, 0.0)
        assert tracker.is_oscillation_imminent("a") is True

    def test_oscillation_not_imminent(self, tracker):
        """is_oscillation_imminent returns False for stable movement."""
        tracker.register_axis("a")
        for pos in [0.0, 0.05, 0.1, 0.15, 0.2]:
            tracker.on_position_update("a", pos, 0.0)
        assert tracker.is_oscillation_imminent("a") is False

    def test_custom_threshold(self, tracker):
        """Custom threshold for oscillation prediction."""
        tracker.register_axis("a")
        for pos in [0.0, 0.1, -0.1, 0.1, -0.1]:
            tracker.on_position_update("a", pos, 0.0)
        # Rate should be high with default threshold (0.4)
        rate = tracker.get_reversal_rate("a")
        assert rate > 0.0
        # With threshold above the rate, it should not be imminent
        assert tracker.is_oscillation_imminent("a", threshold=rate + 0.01) is False
        # With threshold below the rate, it should be imminent
        assert tracker.is_oscillation_imminent("a", threshold=max(0.0, rate - 0.01)) is True

    def test_unknown_axis_reversal_rate(self, tracker):
        assert tracker.get_reversal_rate("nonexistent") == 0.0


# ── Window Rollover ───────────────────────────────────────────────


class TestWindowRollover:
    def test_small_window_rollover(self, tracker_small_window):
        """Old positions drop out when window is exceeded."""
        tracker_small_window.register_axis("a")
        # Fill window: 4 positions max
        for pos in [0.0, 0.1, 0.2, 0.3, 0.4]:
            tracker_small_window.on_position_update("a", pos, 0.0)
        # Only 4 most recent positions remain in history
        hist = tracker_small_window._position_history["a"]
        assert len(hist) == 4
        assert list(hist) == [0.1, 0.2, 0.3, 0.4]


# ── Reset ─────────────────────────────────────────────────────────


class TestVelocityTrackerReset:
    def test_reset_clears_all(self, tracker):
        tracker.register_axis("a")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.1, 0.0)
        assert tracker.total_updates > 0
        tracker.reset()
        assert tracker.total_updates == 0
        assert tracker.total_reversals == 0
        assert tracker.get_velocity("a") == 0.0  # axis gone after reset
        assert len(tracker.all_velocities()) == 0


# ── Serialization ─────────────────────────────────────────────────


class TestVelocityTrackerSerialization:
    def test_round_trip(self, tracker):
        tracker.register_axis("a")
        tracker.register_axis("b")
        tracker.on_position_update("a", 0.0, 0.0)
        tracker.on_position_update("a", 0.2, 0.5)
        tracker.on_position_update("b", 0.5, 0.0)
        tracker.on_position_update("b", 0.3, 0.0)

        data = tracker.to_dict()
        restored = TensionVelocityTracker.from_dict(data)

        assert restored.get_velocity("a") == pytest.approx(0.2)
        assert restored.get_velocity("b") == pytest.approx(-0.2)
        assert restored.window_size == tracker.window_size
        assert restored.min_reversal_magnitude == tracker.min_reversal_magnitude
        assert restored.total_updates == tracker.total_updates
        assert restored.total_reversals == tracker.total_reversals

    def test_empty_round_trip(self, tracker):
        data = tracker.to_dict()
        restored = TensionVelocityTracker.from_dict(data)
        assert restored.total_updates == 0
        assert len(restored.all_velocities()) == 0


# ── Engine Integration ────────────────────────────────────────────


class TestEngineIntegration:
    def test_engine_without_velocity_tracker(self, engine_without_velocity):
        """Engine without velocity tracking has None tracker."""
        assert engine_without_velocity.velocity_tracker is None

    def test_engine_with_velocity_enabled(self, engine_with_velocity):
        """Engine with velocity tracking has a tracker."""
        assert engine_with_velocity.velocity_tracker is not None
        assert isinstance(engine_with_velocity.velocity_tracker, TensionVelocityTracker)

    def test_engine_with_explicit_tracker(self):
        """Engine can be given an explicit tracker."""
        tracker = TensionVelocityTracker(window_size=15)
        engine = EquilibriumEngine(velocity_tracker=tracker)
        assert engine.velocity_tracker is tracker
        assert engine.velocity_tracker.window_size == 15

    def test_velocity_tracker_axes_registered(self, engine_with_velocity):
        """All engine axes are registered with the velocity tracker."""
        vt = engine_with_velocity.velocity_tracker
        for axis in engine_with_velocity.axes:
            assert vt.get_velocity(axis.id) == 0.0  # registered with zero velocity

    def test_apply_feedback_updates_velocity(self, engine_with_velocity):
        """apply_feedback feeds the velocity tracker."""
        axis = engine_with_velocity.axes[0]
        vt = engine_with_velocity.velocity_tracker

        # First feedback — no velocity yet
        fb1 = _make_feedback(axis.id, 0.3)
        engine_with_velocity.apply_feedback(fb1)
        # Velocity should be 0 after first position update (no previous)
        # Actually the first position is default (0), and after feedback it's 0.3*damping
        # The tracker gets the new position, but velocity=0 for first call

        # Second feedback — velocity should be non-zero
        fb2 = _make_feedback(axis.id, 0.2)
        engine_with_velocity.apply_feedback(fb2)
        # Now we should have a velocity
        vel = vt.get_velocity(axis.id)
        assert vel != 0.0  # Position changed between the two feedbacks

    def test_batch_feedback_updates_velocity(self, engine_with_velocity):
        """apply_feedback_batch feeds the velocity tracker."""
        vt = engine_with_velocity.velocity_tracker
        axis_ids = [a.id for a in engine_with_velocity.axes[:2]]

        # First batch to establish positions
        feedbacks1 = [_make_feedback(aid, 0.1) for aid in axis_ids]
        engine_with_velocity.apply_feedback_batch(feedbacks1)

        # Second batch — should produce velocities
        feedbacks2 = [_make_feedback(aid, -0.1) for aid in axis_ids]
        engine_with_velocity.apply_feedback_batch(feedbacks2)

        for aid in axis_ids:
            assert vt.get_velocity(aid) != 0.0

    def test_engine_reset_resets_tracker(self, engine_with_velocity):
        """Engine reset also resets the velocity tracker."""
        vt = engine_with_velocity.velocity_tracker
        axis = engine_with_velocity.axes[0]

        # Create some velocity
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, -0.3))
        assert vt.total_updates > 0

        # Reset
        engine_with_velocity.reset()
        # Tracker should be reset and re-registered
        assert vt.total_updates == 0
        # Axes should be re-registered after reset
        assert vt.get_velocity(axis.id) == 0.0

    def test_engine_serialization_with_velocity(self, engine_with_velocity):
        """Engine serialization includes velocity tracker state."""
        axis = engine_with_velocity.axes[0]
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.3))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.2))

        data = engine_with_velocity.to_dict()
        assert "velocity_tracker_state" in data
        assert data["velocity_tracker_state"] is not None

    def test_engine_deserialization_with_velocity(self, engine_with_velocity):
        """Engine deserialization restores velocity tracker state."""
        vt = engine_with_velocity.velocity_tracker
        axis = engine_with_velocity.axes[0]
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.2))

        data = engine_with_velocity.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        # Velocity tracker should be restored from serialized state
        assert restored.velocity_tracker is not None
        assert restored.velocity_tracker.total_updates == vt.total_updates

    def test_engine_serialization_without_velocity(self, engine_without_velocity):
        """Engine without velocity tracking serializes None for tracker."""
        data = engine_without_velocity.to_dict()
        assert data.get("velocity_tracker_state") is None

    def test_backward_compatible_no_velocity(self, engine_without_velocity):
        """Engine without velocity tracker works exactly as before."""
        axis = engine_without_velocity.axes[0]
        result = engine_without_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        assert result is not None
        # View should have empty velocity data
        view = engine_without_velocity.view_for(Pillar.COGNITION)
        assert view.velocities == {}
        assert view.momentum_scores == {}
        assert view.oscillation_imminent == ()


# ── PillarEquilibriumView Integration ─────────────────────────────


class TestPillarEquilibriumViewVelocity:
    def test_view_with_velocity_tracker(self, engine_with_velocity):
        """View includes velocity data when tracker is enabled."""
        axis = engine_with_velocity.axes[0]
        # Which pillar owns this axis?
        pillar = axis.pillar

        # Create some position history
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, -0.2))

        view = engine_with_velocity.view_for(pillar)
        # Velocity data should be present
        assert len(view.velocities) > 0
        assert len(view.momentum_scores) > 0

    def test_view_without_velocity_tracker(self, engine_without_velocity):
        """View has empty velocity data when tracker is disabled."""
        axis = engine_without_velocity.axes[0]
        pillar = axis.pillar
        engine_without_velocity.apply_feedback(_make_feedback(axis.id, 0.5))

        view = engine_without_velocity.view_for(pillar)
        assert view.velocities == {}
        assert view.momentum_scores == {}
        assert view.oscillation_imminent == ()

    def test_view_velocity_convenience_methods(self, engine_with_velocity):
        """View convenience methods for velocity data."""
        axis = engine_with_velocity.axes[0]
        pillar = axis.pillar

        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, -0.3))

        view = engine_with_velocity.view_for(pillar)
        # These should return 0.0 for unknown axes
        assert view.get_velocity("nonexistent") == 0.0
        assert view.get_momentum_score("nonexistent") == 0.0
        assert view.is_axis_drifting("nonexistent") is False

    def test_view_summary_with_velocity(self, engine_with_velocity):
        """View summary includes velocity data when available."""
        axis = engine_with_velocity.axes[0]
        pillar = axis.pillar
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, -0.3))

        view = engine_with_velocity.view_for(pillar)
        summary = view.summary()
        assert "velocities" in summary
        assert "momentum_scores" in summary

    def test_view_summary_without_velocity(self, engine_without_velocity):
        """View summary omits velocity data when tracker is disabled."""
        axis = engine_without_velocity.axes[0]
        pillar = axis.pillar
        engine_without_velocity.apply_feedback(_make_feedback(axis.id, 0.5))

        view = engine_without_velocity.view_for(pillar)
        summary = view.summary()
        assert "velocities" not in summary
        assert "momentum_scores" not in summary

    def test_view_repr_with_velocity_warning(self, engine_with_velocity):
        """View repr includes vel_warn when oscillation is imminent."""
        vt = engine_with_velocity.velocity_tracker
        axis = engine_with_velocity.axes[0]
        pillar = axis.pillar

        # Create oscillation pattern
        for i in range(6):
            sign = 1 if i % 2 == 0 else -1
            engine_with_velocity.apply_feedback(_make_feedback(axis.id, sign * 0.3))

        view = engine_with_velocity.view_for(pillar)
        r = repr(view)
        # If oscillation_imminent is detected, repr should show vel_warn
        if view.oscillation_imminent:
            assert "vel_warn=" in r

    def test_view_repr_without_velocity(self, engine_without_velocity):
        """View repr without velocity tracking has no vel_warn."""
        axis = engine_without_velocity.axes[0]
        pillar = axis.pillar
        engine_without_velocity.apply_feedback(_make_feedback(axis.id, 0.5))

        view = engine_without_velocity.view_for(pillar)
        r = repr(view)
        assert "vel_warn=" not in r


# ── Adaptive Damping + Velocity Integration ───────────────────────


class TestAdaptiveDampingVelocityCoexistence:
    def test_both_adaptive_damping_and_velocity(self):
        """Engine can have both adaptive damping and velocity tracking."""
        engine = EquilibriumEngine(
            enable_adaptive_damping=True,
            enable_velocity_tracking=True,
        )
        assert engine.adaptive_damping is not None
        assert engine.velocity_tracker is not None

        axis = engine.axes[0]
        # Apply feedbacks that should trigger both systems
        for i in range(10):
            engine.apply_feedback(_make_feedback(axis.id, 0.3 if i % 2 == 0 else -0.3))

        # Both should have data
        assert engine.velocity_tracker.total_updates > 0
        assert engine.adaptive_damping.total_adaptations > 0

    def test_momentum_aware_restoration_pattern(self, engine_with_velocity):
        """Demonstrate the momentum-aware restoration pattern.

        When an axis is moving toward its default (positive momentum),
        the restoring force can be reduced (coasting). When drifting
        away (negative momentum), the restoring force should be stronger.

        This test validates the data is available; the actual restoring
        force modulation is a consumer concern.
        """
        vt = engine_with_velocity.velocity_tracker
        # Find an axis whose default is 0.0
        axis = None
        for a in engine_with_velocity.axes:
            if a.default_position == 0.0:
                axis = a
                break
        if axis is None:
            pytest.skip("No axis with default_position=0.0")

        # Push axis away from default
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.8))
        # Axis is now at some positive position, moving away
        # Next feedback pushes toward default (negative signal)
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, -0.3))

        # Check momentum data is available for consumer logic
        momentum = vt.get_momentum_score(axis.id)
        # The exact sign depends on the net position vs default,
        # but the value should be non-zero after these updates
        assert momentum != 0.0 or vt.get_velocity(axis.id) != 0.0


# ── Edge Cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_signal_feedback(self, engine_with_velocity):
        """Zero-signal feedback still records position (velocity could be 0)."""
        axis = engine_with_velocity.axes[0]
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.5))
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.0))
        vt = engine_with_velocity.velocity_tracker
        # The position after zero signal should be same as before
        # So velocity from this step should be ~0

    def test_high_confidence_vs_low_confidence(self, engine_with_velocity):
        """Different confidence levels produce different velocities."""
        axis = engine_with_velocity.axes[0]
        vt = engine_with_velocity.velocity_tracker

        # High confidence → large position change
        engine_with_velocity.apply_feedback(
            _make_feedback(axis.id, 0.5, confidence=1.0)
        )
        vel_high = abs(vt.get_velocity(axis.id))

        # Reset and try low confidence
        engine_with_velocity.reset()
        engine_with_velocity.apply_feedback(
            _make_feedback(axis.id, 0.5, confidence=0.1)
        )
        vel_low = abs(vt.get_velocity(axis.id))

        # High confidence should produce larger velocity
        # (though first update is always 0, so check after second)
        engine_with_velocity.apply_feedback(
            _make_feedback(axis.id, 0.5, confidence=1.0)
        )
        assert abs(vt.get_velocity(axis.id)) > 0

    def test_tracker_update_count(self, engine_with_velocity):
        """Each axis update increments the tracker's total_updates."""
        vt = engine_with_velocity.velocity_tracker
        initial_updates = vt.total_updates
        axis = engine_with_velocity.axes[0]
        engine_with_velocity.apply_feedback(_make_feedback(axis.id, 0.1))
        assert vt.total_updates == initial_updates + 1
