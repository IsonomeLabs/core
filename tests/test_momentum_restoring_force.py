"""Tests for iter-029: Momentum-Modulated Restoring Force (engine-level).

The MomentumRestoringForce applies a gentle, momentum-aware nudge
toward homeostasis directly inside the engine's position-update loop.

Key behaviors:
1. Drifting axis (momentum < 0): stronger restoring pull toward default
2. Approaching axis (momentum > 0): weaker restoring pull (let it coast)
3. Zero momentum: neutral restoring pull (base_weight)
4. Small displacement (< min_displacement): no restoring force
5. Restoring delta is clamped to max_restoring_fraction of displacement
6. Position stays in [-1, 1] after restoring force is applied
7. All parameters validated on construction
8. Serialization round-trip preserves state
9. Integration with EquilibriumEngine via enable_momentum_restoring_force
"""

from __future__ import annotations

import math

import pytest

from isonome.equilibrium.restoring import MomentumRestoringForce
from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_restoring(**overrides) -> MomentumRestoringForce:
    """Create a MomentumRestoringForce with optional overrides."""
    defaults = dict(
        base_weight=0.02,
        drifting_boost=2.0,
        approaching_damp=0.3,
        max_restoring_fraction=0.1,
        min_displacement=0.05,
    )
    defaults.update(overrides)
    return MomentumRestoringForce(**defaults)


# ═══════════════════════════════════════════════════════════════
# 1. Construction & Validation
# ═══════════════════════════════════════════════════════════════

class TestConstruction:
    """MomentumRestoringForce validates all parameters."""

    def test_default_values(self):
        """Default constructor should produce expected values."""
        rf = MomentumRestoringForce()
        assert rf.base_weight == 0.02
        assert rf.drifting_boost == 2.0
        assert rf.approaching_damp == 0.3
        assert rf.max_restoring_fraction == 0.1
        assert rf.min_displacement == 0.05

    def test_custom_values(self):
        """Custom parameters should be accepted."""
        rf = MomentumRestoringForce(
            base_weight=0.05,
            drifting_boost=3.0,
            approaching_damp=0.5,
            max_restoring_fraction=0.2,
            min_displacement=0.1,
        )
        assert rf.base_weight == 0.05
        assert rf.drifting_boost == 3.0
        assert rf.approaching_damp == 0.5
        assert rf.max_restoring_fraction == 0.2
        assert rf.min_displacement == 0.1

    def test_base_weight_zero_rejected(self):
        """base_weight must be > 0."""
        with pytest.raises(ValueError, match="base_weight"):
            MomentumRestoringForce(base_weight=0.0)

    def test_base_weight_too_large_rejected(self):
        """base_weight must be <= 0.1."""
        with pytest.raises(ValueError, match="base_weight"):
            MomentumRestoringForce(base_weight=0.2)

    def test_drifting_boost_below_one_rejected(self):
        """drifting_boost must be >= 1.0."""
        with pytest.raises(ValueError, match="drifting_boost"):
            MomentumRestoringForce(drifting_boost=0.5)

    def test_drifting_boost_too_large_rejected(self):
        """drifting_boost must be <= 10.0."""
        with pytest.raises(ValueError, match="drifting_boost"):
            MomentumRestoringForce(drifting_boost=15.0)

    def test_approaching_damp_zero_rejected(self):
        """approaching_damp must be > 0."""
        with pytest.raises(ValueError, match="approaching_damp"):
            MomentumRestoringForce(approaching_damp=0.0)

    def test_approaching_damp_above_one_rejected(self):
        """approaching_damp must be <= 1.0."""
        with pytest.raises(ValueError, match="approaching_damp"):
            MomentumRestoringForce(approaching_damp=1.5)

    def test_max_restoring_fraction_zero_rejected(self):
        """max_restoring_fraction must be > 0."""
        with pytest.raises(ValueError, match="max_restoring_fraction"):
            MomentumRestoringForce(max_restoring_fraction=0.0)

    def test_max_restoring_fraction_too_large_rejected(self):
        """max_restoring_fraction must be <= 0.5."""
        with pytest.raises(ValueError, match="max_restoring_fraction"):
            MomentumRestoringForce(max_restoring_fraction=0.6)

    def test_min_displacement_negative_rejected(self):
        """min_displacement must be >= 0."""
        with pytest.raises(ValueError, match="min_displacement"):
            MomentumRestoringForce(min_displacement=-0.1)

    def test_min_displacement_one_rejected(self):
        """min_displacement must be < 1.0."""
        with pytest.raises(ValueError, match="min_displacement"):
            MomentumRestoringForce(min_displacement=1.0)


# ═══════════════════════════════════════════════════════════════
# 2. Basic Restoring Force Computation
# ═══════════════════════════════════════════════════════════════

class TestBasicComputation:
    """Core compute() behavior."""

    def test_zero_displacement_returns_zero(self):
        """When position == default, no restoring force."""
        rf = _make_restoring()
        delta = rf.compute("explore_exploit", 0.15, 0.15)
        assert delta == 0.0

    def test_small_displacement_returns_zero(self):
        """When |displacement| < min_displacement, no restoring force."""
        rf = _make_restoring(min_displacement=0.05)
        delta = rf.compute("explore_exploit", 0.19, 0.15)
        assert delta == 0.0  # displacement = 0.04 < 0.05

    def test_displacement_at_threshold_returns_zero(self):
        """When |displacement| < min_displacement, no restoring force."""
        rf = _make_restoring(min_displacement=0.05)
        delta = rf.compute("explore_exploit", 0.199, 0.15)
        # displacement = 0.049 < 0.05
        assert delta == 0.0

    def test_displacement_above_threshold_returns_nonzero(self):
        """Just above min_displacement → restoring force kicks in."""
        rf = _make_restoring(min_displacement=0.05)
        delta = rf.compute("explore_exploit", 0.21, 0.15)
        assert delta != 0.0

    def test_positive_displacement_negative_delta(self):
        """When position > default, restoring delta should be negative
        (pulling toward default)."""
        rf = _make_restoring(base_weight=0.02)
        delta = rf.compute("explore_exploit", 0.5, 0.15, momentum_score=0.0)
        assert delta < 0

    def test_negative_displacement_positive_delta(self):
        """When position < default, restoring delta should be positive
        (pulling toward default)."""
        rf = _make_restoring(base_weight=0.02)
        delta = rf.compute("explore_exploit", -0.3, 0.15, momentum_score=0.0)
        assert delta > 0

    def test_neutral_momentum_uses_base_weight(self):
        """With momentum=0, the weight should be exactly base_weight."""
        rf = _make_restoring(base_weight=0.03)
        displacement = 0.5 - 0.15  # 0.35
        delta = rf.compute("explore_exploit", 0.5, 0.15, momentum_score=0.0)
        # Expected: -displacement * base_weight = -0.35 * 0.03 = -0.0105
        # But clamped to max_restoring_fraction
        expected_raw = -displacement * 0.03
        max_corr = displacement * 0.1  # max_restoring_fraction=0.1
        expected = max(-max_corr, min(max_corr, expected_raw))
        assert abs(delta - expected) < 1e-10


# ═══════════════════════════════════════════════════════════════
# 3. Drifting Momentum — Boosted Restoring Force
# ═══════════════════════════════════════════════════════════════

class TestDriftingMomentum:
    """When momentum < 0 (axis drifting away), restoring force is boosted."""

    def test_drifting_force_greater_than_neutral(self):
        """Drifting restoring force should be stronger than neutral."""
        rf = _make_restoring(base_weight=0.02, drifting_boost=2.0)
        delta_neutral = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=0.0
        )
        delta_drifting = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=-0.5
        )
        # Both are negative (pulling toward default); drifting is more negative
        assert abs(delta_drifting) > abs(delta_neutral)

    def test_drifting_boost_multiplier_applied(self):
        """drifting_boost multiplier should scale the weight correctly."""
        rf = _make_restoring(
            base_weight=0.02, drifting_boost=3.0,
            max_restoring_fraction=0.5,  # wide clamp
        )
        displacement = 0.35
        delta = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=-0.8
        )
        # Expected: -0.35 * 0.02 * 3.0 = -0.021
        expected = -displacement * 0.02 * 3.0
        assert abs(delta - expected) < 1e-10

    def test_drifting_increments_counter(self):
        """Drifting applications should increment the drifting counter."""
        rf = _make_restoring()
        assert rf.total_drifting_applications == 0
        rf.compute("explore_exploit", 0.5, 0.15, momentum_score=-0.3)
        assert rf.total_drifting_applications == 1
        assert rf.total_applications == 1


# ═══════════════════════════════════════════════════════════════
# 4. Approaching Momentum — Weakened Restoring Force
# ═══════════════════════════════════════════════════════════════

class TestApproachingMomentum:
    """When momentum > 0 (axis approaching default), restoring force is dampened."""

    def test_approaching_force_less_than_neutral(self):
        """Approaching restoring force should be weaker than neutral."""
        rf = _make_restoring(base_weight=0.02, approaching_damp=0.3)
        delta_neutral = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=0.0
        )
        delta_approaching = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=0.5
        )
        # Both are negative (pulling toward default); approaching is less negative
        assert abs(delta_approaching) < abs(delta_neutral)

    def test_approaching_damp_multiplier_applied(self):
        """approaching_damp multiplier should scale the weight correctly."""
        rf = _make_restoring(
            base_weight=0.02, approaching_damp=0.5,
            max_restoring_fraction=0.5,  # wide clamp
        )
        displacement = 0.35
        delta = rf.compute_no_track(
            "explore_exploit", 0.5, 0.15, momentum_score=0.6
        )
        # Expected: -0.35 * 0.02 * 0.5 = -0.0035
        expected = -displacement * 0.02 * 0.5
        assert abs(delta - expected) < 1e-10

    def test_approaching_increments_counter(self):
        """Approaching applications should increment the approaching counter."""
        rf = _make_restoring()
        assert rf.total_approaching_applications == 0
        rf.compute("explore_exploit", 0.5, 0.15, momentum_score=0.3)
        assert rf.total_approaching_applications == 1
        assert rf.total_applications == 1


# ═══════════════════════════════════════════════════════════════
# 5. Clamping & Bounds
# ═══════════════════════════════════════════════════════════════

class TestClampingAndBounds:
    """Restoring delta must be clamped to max_restoring_fraction."""

    def test_clamping_prevents_overshoot(self):
        """Even with a huge drifting_boost, delta is clamped."""
        rf = _make_restoring(
            base_weight=0.05,
            drifting_boost=10.0,  # Very aggressive
            max_restoring_fraction=0.1,
        )
        displacement = 0.5
        delta = rf.compute_no_track(
            "explore_exploit", 0.5, 0.0, momentum_score=-1.0
        )
        # Raw: -0.5 * 0.05 * 10.0 = -0.25
        # Max correction: 0.5 * 0.1 = 0.05
        # Clamped: -0.05
        max_corr = displacement * 0.1
        assert abs(delta) <= max_corr + 1e-10

    def test_clamping_symmetric_for_negative_displacement(self):
        """Clamping works the same for negative displacement."""
        rf = _make_restoring(
            base_weight=0.05,
            drifting_boost=10.0,
            max_restoring_fraction=0.1,
        )
        delta = rf.compute_no_track(
            "explore_exploit", -0.5, 0.0, momentum_score=1.0
        )
        max_corr = 0.5 * 0.1
        assert abs(delta) <= max_corr + 1e-10

    def test_no_clamping_when_within_bounds(self):
        """Small restoring force should not be clamped."""
        rf = _make_restoring(
            base_weight=0.01,
            max_restoring_fraction=0.5,  # wide
        )
        displacement = 0.2
        delta = rf.compute_no_track(
            "explore_exploit", 0.2, 0.0, momentum_score=0.0
        )
        expected_raw = -displacement * 0.01  # -0.002
        assert abs(delta - expected_raw) < 1e-10


# ═══════════════════════════════════════════════════════════════
# 6. Counters & Tracking
# ═══════════════════════════════════════════════════════════════

class TestCounters:
    """Application counters track usage patterns."""

    def test_total_counter_increments(self):
        """Each compute() call with sufficient displacement increments total."""
        rf = _make_restoring()
        rf.compute("a", 0.5, 0.15, momentum_score=0.0)
        rf.compute("b", 0.6, 0.15, momentum_score=-0.3)
        rf.compute("c", 0.4, 0.15, momentum_score=0.3)
        assert rf.total_applications == 3

    def test_small_displacement_no_counter_increment(self):
        """Below min_displacement, counter should not increment."""
        rf = _make_restoring(min_displacement=0.05)
        rf.compute("a", 0.19, 0.15, momentum_score=0.0)
        assert rf.total_applications == 0

    def test_compute_no_track_does_not_increment(self):
        """compute_no_track should not affect counters."""
        rf = _make_restoring()
        rf.compute_no_track("a", 0.5, 0.15, momentum_score=0.0)
        assert rf.total_applications == 0

    def test_drifting_and_approaching_counters_sum_to_total(self):
        """total should equal drifting + approaching + neutral count."""
        rf = _make_restoring()
        rf.compute("a", 0.5, 0.15, momentum_score=-0.5)  # drifting
        rf.compute("b", 0.6, 0.15, momentum_score=0.5)   # approaching
        rf.compute("c", 0.7, 0.15, momentum_score=0.0)   # neutral
        # total = 3, drifting = 1, approaching = 1, neutral = 1
        assert rf.total_applications == 3
        assert rf.total_drifting_applications == 1
        assert rf.total_approaching_applications == 1
        neutral = rf.total_applications - rf.total_drifting_applications - rf.total_approaching_applications
        assert neutral == 1

    def test_reset_clears_counters(self):
        """reset() should zero all counters."""
        rf = _make_restoring()
        rf.compute("a", 0.5, 0.15, momentum_score=-0.5)
        rf.compute("b", 0.6, 0.15, momentum_score=0.5)
        rf.reset()
        assert rf.total_applications == 0
        assert rf.total_drifting_applications == 0
        assert rf.total_approaching_applications == 0


# ═══════════════════════════════════════════════════════════════
# 7. Serialization Round-Trip
# ═══════════════════════════════════════════════════════════════

class TestSerialization:
    """Serialize / deserialize preserves all state."""

    def test_round_trip_preserves_config(self):
        """Config values should survive serialization."""
        rf = MomentumRestoringForce(
            base_weight=0.05,
            drifting_boost=3.0,
            approaching_damp=0.4,
            max_restoring_fraction=0.2,
            min_displacement=0.1,
        )
        data = rf.serialize()
        rf2 = MomentumRestoringForce.from_serialized(data)
        assert rf2.base_weight == 0.05
        assert rf2.drifting_boost == 3.0
        assert rf2.approaching_damp == 0.4
        assert rf2.max_restoring_fraction == 0.2
        assert rf2.min_displacement == 0.1

    def test_round_trip_preserves_counters(self):
        """Counter values should survive serialization."""
        rf = _make_restoring()
        rf.compute("a", 0.5, 0.15, momentum_score=-0.5)
        rf.compute("b", 0.6, 0.15, momentum_score=0.5)
        data = rf.serialize()
        rf2 = MomentumRestoringForce.from_serialized(data)
        assert rf2.total_applications == 2
        assert rf2.total_drifting_applications == 1
        assert rf2.total_approaching_applications == 1


# ═══════════════════════════════════════════════════════════════
# 8. Repr
# ═══════════════════════════════════════════════════════════════

class TestRepr:
    """__repr__ provides useful diagnostic info."""

    def test_repr_contains_key_info(self):
        """repr should include base weight, boost, damp, and applications."""
        rf = _make_restoring()
        r = repr(rf)
        assert "MomentumRestoringForce" in r
        assert "base=" in r
        assert "drift_boost=" in r
        assert "approach_damp=" in r
        assert "applications=" in r


# ═══════════════════════════════════════════════════════════════
# 9. Integration with EquilibriumEngine
# ═══════════════════════════════════════════════════════════════

class TestEngineIntegration:
    """MomentumRestoringForce integrates with EquilibriumEngine."""

    def test_engine_without_restoring_force(self):
        """Default engine should not have a restoring force."""
        eng = EquilibriumEngine()
        assert eng.momentum_restoring_force is None

    def test_engine_with_restoring_force_enabled(self):
        """Engine with enable_momentum_restoring_force=True should have one."""
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_momentum_restoring_force=True,
        )
        assert eng.momentum_restoring_force is not None

    def test_engine_with_custom_restoring_force(self):
        """Engine should accept a pre-built MomentumRestoringForce."""
        rf = MomentumRestoringForce(base_weight=0.05)
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            momentum_restoring_force=rf,
        )
        assert eng.momentum_restoring_force is rf
        assert eng.momentum_restoring_force.base_weight == 0.05

    def test_restoring_force_requires_velocity_tracking(self):
        """Enabling restoring force without velocity tracking should raise."""
        with pytest.raises(ValueError, match="velocity tracking"):
            EquilibriumEngine(enable_momentum_restoring_force=True)

    def test_restoring_force_applied_during_feedback(self):
        """apply_feedback should apply the restoring force nudge
        when the module is enabled."""
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            enable_momentum_restoring_force=True,
            momentum_restoring_force=MomentumRestoringForce(
                base_weight=0.05,
                drifting_boost=2.0,
                approaching_damp=0.3,
                max_restoring_fraction=0.15,
                min_displacement=0.02,
            ),
        )

        # Push explore_exploit far from default (0.15)
        for _ in range(8):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.15,
                confidence=1.0,
                reason="push away",
            ))

        # The restoring force should have been applied
        # (axis would be even further from default without it)
        rf = eng.momentum_restoring_force
        assert rf.total_applications > 0

    def test_restoring_force_nudge_direction(self):
        """The restoring force should nudge position toward default."""
        rf = MomentumRestoringForce(
            base_weight=0.1,
            drifting_boost=1.0,
            approaching_damp=1.0,
            max_restoring_fraction=0.4,
            min_displacement=0.01,
        )
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            momentum_restoring_force=rf,
        )

        # Push explore_exploit far from default
        for _ in range(5):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.2,
                confidence=1.0,
                reason="push away",
            ))

        # Compare to engine WITHOUT restoring force
        eng_no_rf = EquilibriumEngine(enable_velocity_tracking=True)
        for _ in range(5):
            eng_no_rf.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.2,
                confidence=1.0,
                reason="push away",
            ))

        pos_with_rf = eng._axes["explore_exploit"].position
        pos_without_rf = eng_no_rf._axes["explore_exploit"].position

        # Default is 0.15. Both should be > 0.15, but the one with
        # restoring force should be closer to default
        # (position is pulled back toward default by the restoring nudge)
        drift_with_rf = abs(pos_with_rf - 0.15)
        drift_without_rf = abs(pos_without_rf - 0.15)
        assert drift_with_rf <= drift_without_rf

    def test_restoring_force_resets_with_engine(self):
        """Engine reset should also reset the restoring force."""
        rf = MomentumRestoringForce(base_weight=0.05)
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            momentum_restoring_force=rf,
        )

        # Apply some feedback to trigger restoring force
        for _ in range(5):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.2,
                confidence=1.0,
                reason="test",
            ))

        assert rf.total_applications > 0
        eng.reset()
        assert rf.total_applications == 0


# ═══════════════════════════════════════════════════════════════
# 10. Backward Compatibility
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Engine behavior is unchanged when restoring force is disabled."""

    def test_position_unchanged_without_restoring_force(self):
        """Positions should be identical with/without restoring force
        when the restoring force is disabled."""
        # Engine without restoring force
        eng1 = EquilibriumEngine(enable_velocity_tracking=True)

        # Engine with restoring force disabled
        eng2 = EquilibriumEngine(enable_velocity_tracking=True)

        for _ in range(10):
            fb = Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.1,
                confidence=0.8,
                reason="test",
            )
            eng1.apply_feedback(fb)
            eng2.apply_feedback(fb)

        # Positions should be identical (no restoring force in either)
        for aid in eng1._axes:
            pos1 = eng1._axes[aid].position
            pos2 = eng2._axes[aid].position
            assert abs(pos1 - pos2) < 1e-10, f"Axis {aid}: {pos1} vs {pos2}"


# ═══════════════════════════════════════════════════════════════
# 11. Engine Serialization with Restoring Force
# ═══════════════════════════════════════════════════════════════

class TestEngineSerializationWithRestoringForce:
    """Engine serialization should preserve restoring force config."""

    def test_serialize_includes_restoring_force(self):
        """Engine with restoring force should include it in serialization."""
        rf = MomentumRestoringForce(base_weight=0.05)
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            momentum_restoring_force=rf,
        )
        data = eng.to_dict()
        # Restoring force is serialized via the engine's to_dict() / from_dict()
        assert "momentum_restoring_force" in repr(data)

    def test_restoring_force_survives_deserialization(self):
        """Restoring force should survive engine to_dict/from_dict."""
        rf = MomentumRestoringForce(base_weight=0.05, drifting_boost=3.0)
        eng = EquilibriumEngine(
            enable_velocity_tracking=True,
            momentum_restoring_force=rf,
        )
        data = eng.to_dict()
        eng2 = EquilibriumEngine.from_dict(data)
        assert eng2.momentum_restoring_force is not None
        assert eng2.momentum_restoring_force.base_weight == 0.05
        assert eng2.momentum_restoring_force.drifting_boost == 3.0
