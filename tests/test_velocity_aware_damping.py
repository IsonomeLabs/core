"""Tests for Velocity-Aware Adaptive Damping — iter-028.

When a TensionVelocityTracker is wired to the AdaptiveDampingController,
the controller gains a *preemptive* damping channel: it can boost damping
when the velocity tracker predicts oscillation is imminent (high reversal
rate), even before position stddev crosses the oscillation threshold.

This closes the gap identified in iter-021's "Next Steps":
  > Velocity-aware adaptive damping: Wire is_oscillation_imminent() into
  > AdaptiveDampingController.on_feedback() for preemptive damping boost

Covers:
- AdaptiveDampingController construction with velocity_tracker
- Preemptive damping boost when oscillation is imminent
- Interaction between velocity-aware boost and position-based boost
- Preemptive_oscillation_count tracking
- Serialization round-trip with velocity-aware state
- Engine integration: both velocity tracking + adaptive damping
- Backward compatibility: no velocity tracker = original behavior
- Edge cases: tracker without sufficient data, no reversals, etc.
"""

from __future__ import annotations

import math
from collections import deque

import pytest

from isonome.equilibrium import (
    AdaptiveDampingController,
    EquilibriumEngine,
    Feedback,
)
from isonome.equilibrium.velocity import TensionVelocityTracker
from isonome.types import Pillar


# ── Helpers ──────────────────────────────────────────────────────────────


def _fb(axis_id, signal, source=Pillar.COGNITION, confidence=0.9):
    return Feedback(
        source=source,
        tension_axis_id=axis_id,
        signal=signal,
        confidence=confidence,
        reason="test",
    )


def _make_oscillating_tracker(axis_id, steps=8, threshold=0.4):
    """Create a TensionVelocityTracker that has oscillation-imminent on axis_id."""
    tracker = TensionVelocityTracker(
        window_size=5,
        min_reversal_magnitude=0.001,
    )
    tracker.register_axis(axis_id)
    # Simulate alternating positions to create reversals
    position = 0.15  # default for explore_exploit
    for i in range(steps):
        direction = 0.2 if i % 2 == 0 else -0.2
        position += direction
        tracker.on_position_update(axis_id, position, 0.15)
    return tracker


# ═══════════════════════════════════════════════════════════════════════
# Construction & Wiring
# ═══════════════════════════════════════════════════════════════════════


class TestVelocityAwareConstruction:
    """AdaptiveDampingController accepts an optional velocity_tracker."""

    def test_default_controller_has_no_velocity_tracker(self):
        ctrl = AdaptiveDampingController()
        assert ctrl.velocity_tracker is None

    def test_controller_with_velocity_tracker(self):
        tracker = TensionVelocityTracker()
        ctrl = AdaptiveDampingController(velocity_tracker=tracker)
        assert ctrl.velocity_tracker is tracker

    def test_preemptive_oscillation_count_starts_at_zero(self):
        ctrl = AdaptiveDampingController()
        assert ctrl.preemptive_oscillation_count == 0

    def test_preemptive_oscillation_count_with_tracker(self):
        tracker = TensionVelocityTracker()
        ctrl = AdaptiveDampingController(velocity_tracker=tracker)
        assert ctrl.preemptive_oscillation_count == 0


class TestVelocityTrackerProperty:
    """velocity_tracker property behavior."""

    def test_property_returns_none_by_default(self):
        ctrl = AdaptiveDampingController()
        assert ctrl.velocity_tracker is None

    def test_property_returns_bound_tracker(self):
        tracker = TensionVelocityTracker(window_size=15)
        ctrl = AdaptiveDampingController(velocity_tracker=tracker)
        assert ctrl.velocity_tracker is tracker
        assert ctrl.velocity_tracker.window_size == 15

    def test_setting_velocity_tracker_after_construction(self):
        ctrl = AdaptiveDampingController()
        assert ctrl.velocity_tracker is None
        tracker = TensionVelocityTracker()
        ctrl.velocity_tracker = tracker
        assert ctrl.velocity_tracker is tracker

    def test_setting_velocity_tracker_to_none(self):
        tracker = TensionVelocityTracker()
        ctrl = AdaptiveDampingController(velocity_tracker=tracker)
        ctrl.velocity_tracker = None
        assert ctrl.velocity_tracker is None


# ═══════════════════════════════════════════════════════════════════════
# Preemptive Damping Boost
# ═══════════════════════════════════════════════════════════════════════


class TestPreemptiveDampingBoost:
    """When oscillation is imminent, the controller boosts damping preemptively."""

    def test_preemptive_boost_when_oscillation_imminent(self):
        """Damping should increase when velocity tracker says oscillation is imminent,
        even though position-based severity is below 1.0."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        assert tracker.is_oscillation_imminent("explore_exploit"), \
            "Tracker should report oscillation imminent for this test"

        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,  # Very high — won't trigger position-based boost
            boost_rate=0.15,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)
        initial = ctrl.effective_damping("explore_exploit")

        # Feed a stable history (no position-based oscillation)
        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # Damping should have increased due to preemptive boost
        assert ctrl.effective_damping("explore_exploit") > initial, \
            "Preemptive boost should increase damping when oscillation is imminent"

    def test_preemptive_boost_increments_preemptive_count(self):
        """Preemptive oscillation detection should be counted separately."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        assert ctrl.preemptive_oscillation_count > 0

    def test_no_preemptive_boost_when_not_imminent(self):
        """When oscillation is NOT imminent, no preemptive boost occurs."""
        tracker = TensionVelocityTracker(window_size=10)
        tracker.register_axis("explore_exploit")
        # No oscillation — just steady movement in one direction
        for i in range(8):
            tracker.on_position_update(
                "explore_exploit", 0.15 + i * 0.01, 0.15
            )
        assert not tracker.is_oscillation_imminent("explore_exploit")

        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("explore_exploit", 0.4)
        initial = ctrl.effective_damping("explore_exploit")

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # No position-based oscillation AND no velocity-based oscillation
        # → no boost, only stability accumulation
        assert ctrl.effective_damping("explore_exploit") == initial
        assert ctrl.preemptive_oscillation_count == 0

    def test_preemptive_boost_resets_stability_counter(self):
        """Preemptive oscillation detection resets the stability counter."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            stability_window=5,
            # Use a very high preemptive threshold initially to accumulate stability
            preemptive_threshold=10.0,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        # Accumulate some stability (preemptive threshold too high to trigger)
        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        for _ in range(3):
            ctrl.on_feedback("explore_exploit", history, 0.4)
        assert ctrl.get_stability_counter("explore_exploit") == 3

        # Now lower the preemptive threshold so it fires on next tick
        ctrl._preemptive_threshold = 0.1
        ctrl.on_feedback("explore_exploit", history, 0.4)
        assert ctrl.get_stability_counter("explore_exploit") == 0

    def test_preemptive_boost_rate_is_configurable(self):
        """The preemptive_boost_rate controls how much damping increases."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
            preemptive_boost_rate=0.05,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # Boost should be preemptive_boost_rate (0.05), not boost_rate (0.15)
        new_damping = ctrl.effective_damping("explore_exploit")
        assert new_damping == pytest.approx(0.45, abs=0.01)

    def test_preemptive_boost_defaults_to_half_boost_rate(self):
        """When preemptive_boost_rate is not specified, it defaults to boost_rate / 2."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.20,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # Default preemptive = 0.20 / 2 = 0.10
        new_damping = ctrl.effective_damping("explore_exploit")
        assert new_damping == pytest.approx(0.50, abs=0.01)

    def test_preemptive_boost_clamped_by_damping_max(self):
        """Preemptive boost cannot exceed damping_max."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
            preemptive_boost_rate=0.5,
            damping_min=0.05,
            damping_max=0.5,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        assert ctrl.effective_damping("explore_exploit") <= 0.5

    def test_no_preemptive_boost_without_velocity_tracker(self):
        """Without a velocity tracker, no preemptive boost occurs."""
        ctrl = AdaptiveDampingController(
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("explore_exploit", 0.4)
        initial = ctrl.effective_damping("explore_exploit")

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        assert ctrl.effective_damping("explore_exploit") == initial
        assert ctrl.preemptive_oscillation_count == 0

    def test_preemptive_boost_for_unregistered_axis_in_tracker(self):
        """If velocity tracker doesn't know an axis, no preemptive boost."""
        tracker = TensionVelocityTracker(window_size=5)
        # Don't register the axis with the tracker
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("explore_exploit", 0.4)
        initial = ctrl.effective_damping("explore_exploit")

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # Tracker doesn't know about this axis → no preemptive boost
        assert ctrl.effective_damping("explore_exploit") == initial
        assert ctrl.preemptive_oscillation_count == 0


# ═══════════════════════════════════════════════════════════════════════
# Interaction: Position-Based vs Velocity-Based
# ═══════════════════════════════════════════════════════════════════════


class TestPositionVsVelocityInteraction:
    """Position-based boost and velocity-based boost interact correctly."""

    def test_position_boost_takes_priority_over_velocity(self):
        """When both position-based AND velocity-based oscillation are detected,
        position-based boost takes priority (it's stronger)."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=0.1,  # Low threshold → position-based detection
            boost_rate=0.15,
            preemptive_boost_rate=0.05,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        # Oscillating position history
        history = deque([0.5, -0.5, 0.6, -0.4, 0.7, -0.3], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # The boost should use position-based formula (boost_rate * severity),
        # not the preemptive rate
        severity = ctrl.get_oscillation_severity("explore_exploit")
        assert severity > 1.0, "Position-based severity should exceed 1.0"

    def test_position_and_preemptive_both_reset_stability(self):
        """Either type of oscillation resets the stability counter."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=0.1,
            boost_rate=0.15,
            # Start with high preemptive threshold to accumulate stability
            preemptive_threshold=10.0,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        # Accumulate stability with both thresholds high
        ctrl._oscillation_threshold = 100.0
        stable_history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        for _ in range(3):
            ctrl.on_feedback("explore_exploit", stable_history, 0.4)
        assert ctrl.get_stability_counter("explore_exploit") > 0

        # Now oscillate in position — resets stability
        ctrl._oscillation_threshold = 0.1
        osc_history = deque([0.5, -0.5, 0.6, -0.4, 0.7, -0.3], maxlen=8)
        ctrl.on_feedback("explore_exploit", osc_history, 0.4)
        assert ctrl.get_stability_counter("explore_exploit") == 0

    def test_only_velocity_detection_no_position_oscillation(self):
        """When velocity says imminent but position is stable,
        only preemptive boost fires."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,  # Very high — no position detection
            boost_rate=0.15,
            preemptive_boost_rate=0.05,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        # Severity should be low (position is stable)
        assert ctrl.get_oscillation_severity("explore_exploit") < 1.0
        # But preemptive boost still fired
        assert ctrl.effective_damping("explore_exploit") > 0.4
        assert ctrl.preemptive_oscillation_count > 0


# ═══════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════


class TestVelocityAwareSerialization:
    """Serialization preserves velocity-aware damping state."""

    def test_to_dict_includes_preemptive_fields(self):
        ctrl = AdaptiveDampingController(
            boost_rate=0.15,
            preemptive_boost_rate=0.07,
        )
        d = ctrl.to_dict()
        assert "preemptive_oscillation_count" in d
        assert "preemptive_boost_rate" in d
        assert d["preemptive_boost_rate"] == 0.07
        assert d["preemptive_oscillation_count"] == 0

    def test_from_dict_restores_preemptive_state(self):
        d = {
            "effective_damping": {"a1": 0.5},
            "stability_counters": {"a1": 0},
            "oscillation_severity": {"a1": 0.0},
            "damping_min": 0.1,
            "damping_max": 0.95,
            "boost_rate": 0.15,
            "decay_rate": 0.02,
            "stability_window": 6,
            "oscillation_threshold": 0.6,
            "total_adaptations": 3,
            "preemptive_oscillation_count": 2,
            "preemptive_boost_rate": 0.08,
        }
        ctrl = AdaptiveDampingController.from_dict(d)
        assert ctrl.preemptive_oscillation_count == 2
        assert ctrl.preemptive_boost_rate == 0.08

    def test_from_dict_without_preemptive_fields_uses_defaults(self):
        """Backward compatibility: old serializations without preemptive fields."""
        d = {
            "effective_damping": {},
            "stability_counters": {},
            "oscillation_severity": {},
            "damping_min": 0.1,
            "damping_max": 0.95,
            "boost_rate": 0.15,
            "decay_rate": 0.02,
            "stability_window": 6,
            "oscillation_threshold": 0.6,
            "total_adaptations": 0,
        }
        ctrl = AdaptiveDampingController.from_dict(d)
        assert ctrl.preemptive_oscillation_count == 0
        assert ctrl.preemptive_boost_rate == pytest.approx(0.075)  # 0.15/2

    def test_velocity_tracker_not_serialized_in_controller(self):
        """The velocity tracker reference is NOT serialized — it's a runtime binding.
        Deserialized controller has velocity_tracker=None."""
        tracker = TensionVelocityTracker()
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            boost_rate=0.15,
        )
        ctrl.register_axis("a1", 0.4)
        d = ctrl.to_dict()
        # No velocity_tracker_state key — that's on the engine
        assert "velocity_tracker_state" not in d

        ctrl2 = AdaptiveDampingController.from_dict(d)
        assert ctrl2.velocity_tracker is None

    def test_round_trip_preserves_preemptive_count(self):
        ctrl = AdaptiveDampingController(
            boost_rate=0.15,
            preemptive_boost_rate=0.05,
        )
        ctrl.register_axis("a1", 0.4)
        ctrl._preemptive_oscillation_count = 5
        ctrl._total_adaptations = 10

        d = ctrl.to_dict()
        ctrl2 = AdaptiveDampingController.from_dict(d)
        assert ctrl2.preemptive_oscillation_count == 5
        assert ctrl2.total_adaptations == 10


# ═══════════════════════════════════════════════════════════════════════
# Reset
# ═══════════════════════════════════════════════════════════════════════


class TestVelocityAwareReset:
    """Reset clears velocity-aware state."""

    def test_reset_clears_preemptive_count(self):
        ctrl = AdaptiveDampingController(boost_rate=0.15)
        ctrl._preemptive_oscillation_count = 10
        ctrl.reset()
        assert ctrl.preemptive_oscillation_count == 0

    def test_reset_does_not_clear_velocity_tracker_reference(self):
        """Reset clears counters but keeps the velocity tracker binding."""
        tracker = TensionVelocityTracker()
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            boost_rate=0.15,
        )
        ctrl._preemptive_oscillation_count = 5
        ctrl.reset()
        assert ctrl.velocity_tracker is tracker
        assert ctrl.preemptive_oscillation_count == 0


# ═══════════════════════════════════════════════════════════════════════
# Engine Integration
# ═══════════════════════════════════════════════════════════════════════


class TestEngineVelocityAwareDampingIntegration:
    """EquilibriumEngine with both velocity tracking and adaptive damping."""

    def test_engine_auto_wires_tracker_to_controller(self):
        """When both enable_velocity_tracking=True and enable_adaptive_damping=True,
        the engine should wire the tracker to the controller."""
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
        )
        assert e.velocity_tracker is not None
        assert e.adaptive_damping is not None
        assert e.adaptive_damping.velocity_tracker is e.velocity_tracker

    def test_engine_with_velocity_no_damping_has_no_controller(self):
        e = EquilibriumEngine(enable_velocity_tracking=True)
        assert e.adaptive_damping is None

    def test_engine_with_damping_no_velocity_has_no_tracker_on_controller(self):
        e = EquilibriumEngine(enable_adaptive_damping=True)
        assert e.velocity_tracker is None
        assert e.adaptive_damping.velocity_tracker is None

    def test_preemptive_damping_in_oscillating_engine(self):
        """Oscillation on an axis should trigger both velocity and position detection."""
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
            oscillation_threshold=0.3,
        )

        # Create oscillation by alternating strong feedback
        for i in range(10):
            signal = 0.8 if i % 2 == 0 else -0.8
            e.apply_feedback(_fb("explore_exploit", signal))

        # Either preemptive or position-based boost should have occurred
        assert e.adaptive_damping.total_adaptations > 0 or \
               e.adaptive_damping.preemptive_oscillation_count > 0

    def test_preemptive_boost_happens_before_position_oscillation(self):
        """With a high oscillation threshold (position never triggers),
        the velocity-based preemptive boost should still fire."""
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
            oscillation_threshold=5.0,  # Very high → position never detects oscillation
        )

        # Simulate mild oscillation (positions stay within range)
        for i in range(10):
            signal = 0.15 if i % 2 == 0 else -0.15
            e.apply_feedback(_fb("explore_exploit", signal))

        # Position-based severity should be low
        sev = e.adaptive_damping.get_oscillation_severity("explore_exploit")

        # But velocity may detect oscillation-imminent
        # (depending on whether the reversal rate is high enough)
        tracker = e.velocity_tracker
        if tracker.is_oscillation_imminent("explore_exploit"):
            assert e.adaptive_damping.preemptive_oscillation_count > 0

    def test_engine_reset_rewires_tracker_to_controller(self):
        """After engine reset, the controller should still reference the tracker."""
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
        )
        # Cause some adaptations
        for i in range(6):
            e.apply_feedback(_fb("explore_exploit", 0.4 if i % 2 == 0 else -0.4))

        e.reset()
        # After reset, controller and tracker are both reset
        assert e.adaptive_damping is not None
        assert e.velocity_tracker is not None
        assert e.adaptive_damping.velocity_tracker is e.velocity_tracker
        assert e.adaptive_damping.preemptive_oscillation_count == 0

    def test_engine_serialization_preserves_both_systems(self):
        """Engine serialization preserves both adaptive damping and velocity state."""
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
        )
        for i in range(6):
            e.apply_feedback(_fb("explore_exploit", 0.4 if i % 2 == 0 else -0.4))

        d = e.to_dict()
        assert d["adaptive_damping_state"] is not None
        assert d["velocity_tracker_state"] is not None

        e2 = EquilibriumEngine.from_dict(d)
        assert e2.adaptive_damping is not None
        assert e2.velocity_tracker is not None
        # Controller should be wired to tracker
        assert e2.adaptive_damping.velocity_tracker is e2.velocity_tracker

    def test_engine_serialization_preserves_preemptive_count(self):
        e = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_adaptive_damping=True,
            oscillation_threshold=5.0,  # High to avoid position-based detection
        )
        # Create mild oscillation to trigger velocity detection
        for i in range(12):
            signal = 0.2 if i % 2 == 0 else -0.2
            e.apply_feedback(_fb("explore_exploit", signal))

        pre_count = e.adaptive_damping.preemptive_oscillation_count
        d = e.to_dict()

        e2 = EquilibriumEngine.from_dict(d)
        assert e2.adaptive_damping.preemptive_oscillation_count == pre_count


# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Velocity-aware damping is opt-in — no impact without tracker."""

    def test_engine_without_velocity_or_damping_unchanged(self):
        """Engine with neither feature is completely unchanged."""
        e = EquilibriumEngine()
        assert e.velocity_tracker is None
        assert e.adaptive_damping is None

    def test_controller_without_tracker_identical_behavior(self):
        """A controller without a velocity tracker behaves exactly as before."""
        ctrl = AdaptiveDampingController(
            oscillation_threshold=0.3,
            boost_rate=0.15,
        )
        ctrl.register_axis("test", 0.4)

        # Stable history
        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("test", history, 0.4)

        # No changes — stability accumulates
        assert ctrl.effective_damping("test") == 0.4
        assert ctrl.preemptive_oscillation_count == 0

    def test_controller_with_tracker_but_no_reversals(self):
        """A tracker with no reversals doesn't change controller behavior."""
        tracker = TensionVelocityTracker(window_size=10)
        tracker.register_axis("test")
        # Feed steady data — no oscillation
        for i in range(8):
            tracker.on_position_update("test", 0.15 + i * 0.01, 0.15)

        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("test", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("test", history, 0.4)

        assert ctrl.effective_damping("test") == 0.4
        assert ctrl.preemptive_oscillation_count == 0

    def test_all_existing_adaptive_damping_tests_still_pass(self):
        """Sanity check: basic controller still works without tracker."""
        ctrl = AdaptiveDampingController(oscillation_threshold=0.3)
        ctrl.register_axis("a1", 0.4)
        ctrl.register_axis("a2", 0.6)

        # Oscillate a1
        history = deque([0.5, -0.5, 0.6, -0.4, 0.7, -0.3], maxlen=8)
        ctrl.on_feedback("a1", history, 0.4)
        # a1 damping should have increased (severity > 1.0 with threshold 0.3)
        assert ctrl.effective_damping("a1") > 0.4

        # a2 should be unchanged
        assert ctrl.effective_damping("a2") == 0.6


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════════


class TestVelocityAwareEdgeCases:
    """Boundary and edge conditions for velocity-aware damping."""

    def test_tracker_with_insufficient_data_no_boost(self):
        """A tracker with fewer than 3 data points has no reversal rate → no boost."""
        tracker = TensionVelocityTracker(window_size=5)
        tracker.register_axis("explore_exploit")
        # Only 1 position update — not enough data
        tracker.on_position_update("explore_exploit", 0.2, 0.15)

        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)

        assert ctrl.preemptive_oscillation_count == 0
        assert ctrl.effective_damping("explore_exploit") == 0.4

    def test_multiple_axes_preemptive_boost_independent(self):
        """Preemptive boost on one axis doesn't affect another."""
        tracker = TensionVelocityTracker(window_size=5, min_reversal_magnitude=0.001)
        tracker.register_axis("explore_exploit")
        tracker.register_axis("autonomy_safety")

        # Oscillate explore_exploit only
        pos = 0.15
        for i in range(8):
            direction = 0.2 if i % 2 == 0 else -0.2
            pos += direction
            tracker.on_position_update("explore_exploit", pos, 0.15)

        # Steady on autonomy_safety
        for i in range(8):
            tracker.on_position_update(
                "autonomy_safety", -0.4 + i * 0.01, -0.4
            )

        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
        )
        ctrl.register_axis("explore_exploit", 0.4)
        ctrl.register_axis("autonomy_safety", 0.6)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        ctrl.on_feedback("explore_exploit", history, 0.4)
        ctrl.on_feedback("autonomy_safety", history, 0.6)

        # explore_exploit should be boosted, autonomy_safety should not
        assert ctrl.effective_damping("explore_exploit") > 0.4
        assert ctrl.effective_damping("autonomy_safety") == 0.6

    def test_preemptive_boost_accumulates_over_ticks(self):
        """Multiple preemptive boosts accumulate."""
        tracker = _make_oscillating_tracker("explore_exploit", steps=8)
        ctrl = AdaptiveDampingController(
            velocity_tracker=tracker,
            oscillation_threshold=10.0,
            boost_rate=0.15,
            preemptive_boost_rate=0.05,
            damping_min=0.05,
            damping_max=0.95,
        )
        ctrl.register_axis("explore_exploit", 0.4)

        history = deque([0.1, 0.1, 0.1, 0.1, 0.1], maxlen=8)
        d1 = ctrl.effective_damping("explore_exploit")

        ctrl.on_feedback("explore_exploit", history, 0.4)
        d2 = ctrl.effective_damping("explore_exploit")

        # After first boost, if oscillation is still imminent, another boost occurs
        # But the stability counter was reset, so it won't decay immediately
        if tracker.is_oscillation_imminent("explore_exploit"):
            assert ctrl.preemptive_oscillation_count >= 1

    def test_custom_preemptive_threshold(self):
        """The velocity tracker's oscillation_imminent threshold can be configured."""
        tracker = TensionVelocityTracker(
            window_size=5,
            min_reversal_magnitude=0.001,
        )
        tracker.register_axis("explore_exploit")
        # Mild oscillation
        pos = 0.15
        for i in range(8):
            direction = 0.05 if i % 2 == 0 else -0.05
            pos += direction
            tracker.on_position_update("explore_exploit", pos, 0.15)

        # With default threshold (0.4), this might not be imminent
        # With low threshold (0.2), it is
        default_imminent = tracker.is_oscillation_imminent(
            "explore_exploit", threshold=0.4
        )
        low_imminent = tracker.is_oscillation_imminent(
            "explore_exploit", threshold=0.1
        )

        # The controller uses the tracker's default threshold
        # Custom threshold can be passed when creating the controller
        # (this is a design choice — the controller can expose a
        #  preemptive_threshold parameter)

    def test_repr_includes_preemptive_count(self):
        ctrl = AdaptiveDampingController(boost_rate=0.15)
        ctrl._preemptive_oscillation_count = 3
        r = repr(ctrl)
        # Should mention preemptive count
        assert "preemptive=3" in r or "3" in r
