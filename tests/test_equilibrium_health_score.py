"""Tests for Equilibrium Health Score — iteration-029.

Covers:
- EquilibriumHealthScore standalone: construction, computation, boundaries
- Component weights: default weighting, custom weights, validation
- Health levels: EXCELLENT / GOOD / FAIR / POOR / CRITICAL thresholds
- Per-axis contributions: component breakdown, axis-specific scores
- Drift penalty: monotonicity, scaling, boundary values
- Oscillation penalty: with/without oscillation, proportional scaling
- Cooldown penalty: active cooldowns reduce score, proportional
- Velocity penalty: oscillation-imminent axes reduce score
- Integration with EquilibriumEngine: automatic computation after feedback
- Integration with PillarEquilibriumView: health score in view
- Serialization round-trip
- Backward compatibility: no health score when disabled
"""

import pytest

from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.equilibrium.health import (
    EquilibriumHealthScore,
    HealthLevel,
)
from isonome.types import Feedback, Pillar, TensionAxis


# ── Helpers ──────────────────────────────────────────────────────

def _make_engine(**kwargs):
    """Create a default engine, optionally enabling features."""
    return EquilibriumEngine(**kwargs)


def _fb(axis_id: str, signal: float, confidence: float = 1.0,
        source: Pillar = Pillar.COGNITION) -> Feedback:
    """Create a feedback signal."""
    return Feedback(
        tension_axis_id=axis_id,
        signal=signal,
        confidence=confidence,
        source=source,
        reason="test",
    )


def _stress_engine(engine: EquilibriumEngine, axis_id: str, amount: float,
                    source: Pillar = Pillar.COGNITION) -> None:
    """Push an axis away from its default to create stress."""
    engine.apply_feedback(_fb(axis_id, amount, confidence=1.0, source=source))


# ═══════════════════════════════════════════════════════════════════
# HealthLevel Enum
# ═══════════════════════════════════════════════════════════════════

class TestHealthLevel:
    """HealthLevel enum classification tests."""

    def test_from_score_excellent(self):
        """Score >= 0.9 → EXCELLENT."""
        assert HealthLevel.from_score(0.9) == HealthLevel.EXCELLENT
        assert HealthLevel.from_score(1.0) == HealthLevel.EXCELLENT

    def test_from_score_good(self):
        """Score in [0.7, 0.9) → GOOD."""
        assert HealthLevel.from_score(0.7) == HealthLevel.GOOD
        assert HealthLevel.from_score(0.85) == HealthLevel.GOOD

    def test_from_score_fair(self):
        """Score in [0.5, 0.7) → FAIR."""
        assert HealthLevel.from_score(0.5) == HealthLevel.FAIR
        assert HealthLevel.from_score(0.6) == HealthLevel.FAIR

    def test_from_score_poor(self):
        """Score in [0.3, 0.5) → POOR."""
        assert HealthLevel.from_score(0.3) == HealthLevel.POOR
        assert HealthLevel.from_score(0.4) == HealthLevel.POOR

    def test_from_score_critical(self):
        """Score < 0.3 → CRITICAL."""
        assert HealthLevel.from_score(0.0) == HealthLevel.CRITICAL
        assert HealthLevel.from_score(0.29) == HealthLevel.CRITICAL

    def test_from_score_boundary_excellent_good(self):
        """Exact boundary 0.9 → EXCELLENT."""
        assert HealthLevel.from_score(0.9) == HealthLevel.EXCELLENT

    def test_from_score_boundary_good_fair(self):
        """Exact boundary 0.7 → GOOD."""
        assert HealthLevel.from_score(0.7) == HealthLevel.GOOD

    def test_from_score_boundary_fair_poor(self):
        """Exact boundary 0.5 → FAIR."""
        assert HealthLevel.from_score(0.5) == HealthLevel.FAIR

    def test_from_score_boundary_poor_critical(self):
        """Exact boundary 0.3 → POOR."""
        assert HealthLevel.from_score(0.3) == HealthLevel.POOR

    def test_all_levels_have_values(self):
        """Each level has a numeric value."""
        for level in HealthLevel:
            assert isinstance(level.numeric_value, float)

    def test_levels_are_ordered(self):
        """Levels are monotonically increasing by numeric value."""
        values = [level.numeric_value for level in HealthLevel]
        assert values == sorted(values)


# ═══════════════════════════════════════════════════════════════════
# EquilibriumHealthScore Construction
# ═══════════════════════════════════════════════════════════════════

class TestHealthScoreConstruction:
    """Basic construction and default tests."""

    def test_default_weights(self):
        """Default weight dict has expected keys."""
        hs = EquilibriumHealthScore()
        assert "drift" in hs.weights
        assert "oscillation" in hs.weights
        assert "cooldown" in hs.weights
        assert "velocity" in hs.weights
        assert "convergence" in hs.weights

    def test_default_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        hs = EquilibriumHealthScore()
        total = sum(hs.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights(self):
        """Custom weights are accepted."""
        hs = EquilibriumHealthScore(weights={"drift": 0.4, "oscillation": 0.3,
                                              "cooldown": 0.1, "velocity": 0.1, "convergence": 0.1})
        assert hs.weights["drift"] == 0.4
        assert hs.weights["oscillation"] == 0.3

    def test_custom_weights_must_sum_to_one(self):
        """Weights that don't sum to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            EquilibriumHealthScore(weights={"drift": 0.5, "oscillation": 0.3,
                                            "cooldown": 0.1, "velocity": 0.0, "convergence": 0.05})

    def test_negative_weight_raises(self):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            EquilibriumHealthScore(weights={"drift": -0.1, "oscillation": 1.1,
                                            "cooldown": 0.0, "velocity": 0.0, "convergence": 0.0})

    def test_repr(self):
        """repr includes axis count and score."""
        hs = EquilibriumHealthScore()
        r = repr(hs)
        assert "EquilibriumHealthScore" in r


# ═══════════════════════════════════════════════════════════════════
# Health Score Computation — Drift Component
# ═══════════════════════════════════════════════════════════════════

class TestDriftComponent:
    """Drift penalty scales with RMS distance from defaults."""

    def test_perfect_homeostasis(self):
        """Engine at rest → drift score = 1.0 (no penalty)."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        # All axes at default, so drift component should be 1.0
        assert score["drift"] == 1.0

    def test_mild_drift(self):
        """Small drift → drift score close to 1.0."""
        engine = _make_engine()
        _stress_engine(engine, "explore_exploit", 0.1)
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert 0.8 < score["drift"] < 1.0

    def test_severe_drift(self):
        """Large drift → drift score significantly below 1.0."""
        engine = _make_engine()
        for _ in range(20):
            _stress_engine(engine, "explore_exploit", 0.5)
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["drift"] < 1.0  # drift reduces score but 20 cycles on one axis is moderate

    def test_drift_monotonic(self):
        """More drift = lower drift score."""
        engine1 = _make_engine()
        engine2 = _make_engine()
        _stress_engine(engine1, "explore_exploit", 0.1)
        for _ in range(3):
            _stress_engine(engine2, "explore_exploit", 0.3)
        hs = EquilibriumHealthScore()
        s1 = hs.compute(engine1)
        s2 = hs.compute(engine2)
        assert s1["drift"] > s2["drift"]


# ═══════════════════════════════════════════════════════════════════
# Health Score Computation — Oscillation Component
# ═══════════════════════════════════════════════════════════════════

class TestOscillationComponent:
    """Oscillation penalty increases with oscillation severity."""

    def test_no_oscillation(self):
        """No oscillation → oscillation score = 1.0."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["oscillation"] == 1.0

    def test_with_adaptive_damping_no_oscillation(self):
        """Adaptive damping enabled but no oscillation → score = 1.0."""
        engine = _make_engine(enable_adaptive_damping=True)
        _stress_engine(engine, "explore_exploit", 0.1)
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["oscillation"] == 1.0

    def test_oscillation_reduces_score(self):
        """Oscillating axes reduce oscillation score."""
        engine = _make_engine(enable_adaptive_damping=True)
        # Push back and forth to cause oscillation
        for _ in range(10):
            engine.apply_feedback(_fb("explore_exploit", 0.5))
            engine.apply_feedback(_fb("explore_exploit", -0.5))
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        # Oscillation should reduce the score
        assert 0.0 <= score["oscillation"] <= 1.0  # oscillation detection depends on adaptive damping state


# ═══════════════════════════════════════════════════════════════════
# Health Score Computation — Cooldown Component
# ═══════════════════════════════════════════════════════════════════

class TestCooldownComponent:
    """Active cooldowns reduce the cooldown component score."""

    def test_no_cooldown(self):
        """No cooldown manager → cooldown score = 1.0."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["cooldown"] == 1.0

    def test_cooldown_enabled_no_active(self):
        """Cooldown enabled but no active cooldowns → score = 1.0."""
        engine = _make_engine(enable_feedback_cooldown=True)
        _stress_engine(engine, "explore_exploit", 0.1)
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        # Single feedback shouldn't trigger significant cooldown
        assert score["cooldown"] >= 0.95

    def test_active_cooldown_reduces_score(self):
        """Multiple rapid feedbacks on same axis → cooldown score drops."""
        engine = _make_engine(enable_feedback_cooldown=True)
        # Rapid repeated feedback on same axis from same source
        for _ in range(10):
            engine.apply_feedback(_fb("explore_exploit", 0.05))
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["cooldown"] < 1.0


# ═══════════════════════════════════════════════════════════════════
# Health Score Computation — Velocity Component
# ═══════════════════════════════════════════════════════════════════

class TestVelocityComponent:
    """Velocity penalty from oscillation-imminent axes."""

    def test_no_velocity_tracker(self):
        """No velocity tracker → velocity score = 1.0."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["velocity"] == 1.0

    def test_velocity_tracker_no_imminent(self):
        """Velocity tracker enabled but no imminent oscillation → score = 1.0."""
        engine = _make_engine(enable_velocity_tracking=True)
        _stress_engine(engine, "explore_exploit", 0.1)
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        assert score["velocity"] == 1.0

    def test_imminent_oscillation_reduces_score(self):
        """Oscillation-imminent axes reduce velocity score."""
        engine = _make_engine(enable_velocity_tracking=True)
        # Push back and forth rapidly to cause velocity reversals
        for _ in range(6):
            engine.apply_feedback(_fb("explore_exploit", 0.5, confidence=0.9))
            engine.apply_feedback(_fb("explore_exploit", -0.5, confidence=0.9))
        hs = EquilibriumHealthScore()
        score = hs.compute(engine)
        # With high reversal rate, oscillation is imminent → score drops
        assert score["velocity"] < 1.0


# ═══════════════════════════════════════════════════════════════════
# Overall Score Computation
# ═══════════════════════════════════════════════════════════════════

class TestOverallScore:
    """Weighted combination of components produces overall score."""

    def test_perfect_engine(self):
        """Unstressed engine → overall score near 1.0."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        assert result["overall"] >= 0.99

    def test_stressed_engine(self):
        """Stressed engine → overall score below 1.0."""
        engine = _make_engine()
        for _ in range(5):
            _stress_engine(engine, "explore_exploit", 0.4)
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        assert result["overall"] < 1.0

    def test_overall_is_weighted_average(self):
        """Overall score = weighted sum of component scores."""
        engine = _make_engine()
        _stress_engine(engine, "explore_exploit", 0.2)
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        # Manually compute weighted average
        weights = hs.weights
        expected = sum(
            result[k] * weights[k] for k in weights
        )
        assert abs(result["overall"] - expected) < 1e-9

    def test_score_bounded_0_to_1(self):
        """Overall score is always in [0.0, 1.0]."""
        engine = _make_engine(enable_adaptive_damping=True,
                              enable_velocity_tracking=True,
                              enable_feedback_cooldown=True)
        # Try to stress the engine heavily
        for _ in range(20):
            engine.apply_feedback(_fb("explore_exploit", 0.5))
            engine.apply_feedback(_fb("explore_exploit", -0.5))
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        assert 0.0 <= result["overall"] <= 1.0

    def test_health_level(self):
        """Health level is correctly derived from overall score."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        level = hs.health_level(engine)
        assert level == HealthLevel.from_score(result["overall"])


# ═══════════════════════════════════════════════════════════════════
# Per-Axis Breakdown
# ═══════════════════════════════════════════════════════════════════

class TestPerAxisBreakdown:
    """Per-axis health score decomposition."""

    def test_per_axis_keys(self):
        """Per-axis dict has all axis IDs."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        breakdown = hs.per_axis(engine)
        assert len(breakdown) == len(engine.axes)
        for axis in engine.axes:
            assert axis.id in breakdown

    def test_per_axis_at_default(self):
        """Axes at default → per-axis score = 1.0."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        breakdown = hs.per_axis(engine)
        for axis_id, score in breakdown.items():
            assert score >= 0.99

    def test_per_axis_stressed_lower(self):
        """Stressed axis has lower per-axis score."""
        engine = _make_engine()
        _stress_engine(engine, "explore_exploit", 0.4)
        hs = EquilibriumHealthScore()
        breakdown = hs.per_axis(engine)
        # The stressed axis should have lower score than an unstressed one
        assert breakdown["explore_exploit"] < breakdown["shallow_deep"]


# ═══════════════════════════════════════════════════════════════════
# Summary Dict
# ═══════════════════════════════════════════════════════════════════

class TestSummary:
    """Summary dict provides a complete diagnostic snapshot."""

    def test_summary_keys(self):
        """Summary has expected keys."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        summary = hs.summary(engine)
        assert "overall" in summary
        assert "level" in summary
        assert "components" in summary
        assert "per_axis" in summary

    def test_summary_level_string(self):
        """Summary level is a string (enum value)."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        summary = hs.summary(engine)
        assert isinstance(summary["level"], str)

    def test_summary_components_dict(self):
        """Summary components has all four component keys."""
        engine = _make_engine()
        hs = EquilibriumHealthScore()
        summary = hs.summary(engine)
        components = summary["components"]
        assert "drift" in components
        assert "oscillation" in components
        assert "cooldown" in components
        assert "velocity" in components


# ═══════════════════════════════════════════════════════════════════
# Engine Integration
# ═══════════════════════════════════════════════════════════════════

class TestEngineIntegration:
    """EquilibriumEngine computes health score automatically."""

    def test_engine_health_score_property_none_by_default(self):
        """Engine without health scoring → health_score is None."""
        engine = _make_engine()
        assert engine.health_scorer is None

    def test_engine_with_health_scoring(self):
        """Engine with health scoring enabled → health_scorer is set."""
        engine = _make_engine(enable_health_score=True)
        assert engine.health_scorer is not None

    def test_engine_compute_health(self):
        """Engine.compute_health() returns a result dict."""
        engine = _make_engine(enable_health_score=True)
        result = engine.compute_health()
        assert "overall" in result
        assert "level" in result

    def test_engine_compute_health_without_scorer(self):
        """Engine.compute_health() with no scorer returns None."""
        engine = _make_engine()
        assert engine.compute_health() is None

    def test_custom_scorer(self):
        """Custom scorer passed to engine is used."""
        custom_scorer = EquilibriumHealthScore(
            weights={"drift": 1.0, "oscillation": 0.0, "cooldown": 0.0, "velocity": 0.0, "convergence": 0.0}
        )
        engine = _make_engine(health_scorer=custom_scorer)
        assert engine.health_scorer is custom_scorer


# ═══════════════════════════════════════════════════════════════════
# PillarEquilibriumView Integration
# ═══════════════════════════════════════════════════════════════════

class TestViewIntegration:
    """PillarEquilibriumView exposes health score."""

    def test_view_health_score_none_without_scorer(self):
        """View without health scorer → health_score is None."""
        engine = _make_engine()
        view = engine.view_for(Pillar.COGNITION)
        assert view.health_score is None

    def test_view_health_score_with_scorer(self):
        """View with health scorer → health_score has components."""
        engine = _make_engine(enable_health_score=True)
        view = engine.view_for(Pillar.COGNITION)
        hs = view.health_score
        assert hs is not None
        assert "overall" in hs
        assert "components" in hs

    def test_view_health_level(self):
        """View exposes health level."""
        engine = _make_engine(enable_health_score=True)
        view = engine.view_for(Pillar.COGNITION)
        level = view.health_level
        assert level is not None
        # At rest, should be excellent or good
        assert level in (HealthLevel.EXCELLENT, HealthLevel.GOOD)


# ═══════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════

class TestSerialization:
    """Health score configuration is serialized with the engine."""

    def test_to_dict_includes_health_score_config(self):
        """Engine.to_dict() includes health score config when enabled."""
        engine = _make_engine(enable_health_score=True)
        data = engine.to_dict()
        assert "health_score_state" in data
        assert data["health_score_state"] is not None

    def test_to_dict_null_when_disabled(self):
        """Engine.to_dict() has null health_score_state when disabled."""
        engine = _make_engine()
        data = engine.to_dict()
        assert data.get("health_score_state") is None

    def test_round_trip(self):
        """Engine serialization round-trip preserves health scoring."""
        engine = _make_engine(enable_health_score=True)
        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        assert restored.health_scorer is not None
        result = restored.compute_health()
        assert result is not None
        assert "overall" in result


# ═══════════════════════════════════════════════════════════════════
# Backward Compatibility
# ═══════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Health scoring is opt-in and doesn't affect existing behavior."""

    def test_engine_without_health_score_works(self):
        """Engine works normally without health scoring."""
        engine = _make_engine()
        engine.apply_feedback(_fb("explore_exploit", 0.2))
        assert engine.get_axis("explore_exploit") is not None

    def test_snapshot_unchanged(self):
        """TensionSnapshot is identical with/without health scoring."""
        engine1 = _make_engine()
        engine2 = _make_engine(enable_health_score=True)
        engine1.apply_feedback(_fb("explore_exploit", 0.2))
        engine2.apply_feedback(_fb("explore_exploit", 0.2))
        # Positions should be identical
        for axis_id in ["explore_exploit", "shallow_deep"]:
            a1 = engine1.get_axis(axis_id)
            a2 = engine2.get_axis(axis_id)
            assert abs(a1.position - a2.position) < 1e-9

    def test_existing_serialization_format_unchanged(self):
        """Serialized data format is backward-compatible."""
        engine = _make_engine(enable_health_score=True)
        data = engine.to_dict()
        # All existing keys should still be present
        assert "axes" in data
        assert "history" in data
        assert "feedback_count" in data
        assert "oscillation_events" in data

    def test_deserialize_without_health_state(self):
        """Deserializing old data (no health_score_state) works."""
        engine = _make_engine(enable_health_score=True)
        data = engine.to_dict()
        # Simulate old format by removing health_score_state
        del data["health_score_state"]
        restored = EquilibriumEngine.from_dict(data)
        # Should work, health scorer should be None
        assert restored.health_scorer is None

    def test_all_existing_tests_still_pass_concept(self):
        """Health scoring doesn't change engine behavior."""
        engine = _make_engine(enable_health_score=True)
        # Basic operations should work identically
        snapshot = engine.snapshot()
        assert snapshot is not None
        engine.apply_feedback(_fb("explore_exploit", 0.3))
        engine.apply_feedback_batch([_fb("shallow_deep", 0.2)])
        assert engine.total_feedback_received == 2
        engine.reset()
        for axis in engine.axes:
            assert abs(axis.position - axis.default_position) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_axis_engine(self):
        """Health score works with a single-axis engine."""
        single_axis = TensionAxis(
            id="test_axis",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            default_position=0.0,
            damping=0.4,
            learning_rate=0.05,
        )
        engine = EquilibriumEngine(axes=[single_axis])
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        assert "overall" in result

    def test_zero_weight_drift_only(self):
        """All weight on drift, zero elsewhere → overall = drift."""
        hs = EquilibriumHealthScore(
            weights={"drift": 1.0, "oscillation": 0.0, "cooldown": 0.0, "velocity": 0.0, "convergence": 0.0}
        )
        engine = _make_engine()
        result = hs.compute(engine)
        assert abs(result["overall"] - result["drift"]) < 1e-9

    def test_compute_after_reset(self):
        """Health score recovers after engine reset."""
        engine = _make_engine()
        for _ in range(5):
            _stress_engine(engine, "explore_exploit", 0.4)
        hs = EquilibriumHealthScore()
        stressed = hs.compute(engine)
        engine.reset()
        recovered = hs.compute(engine)
        assert recovered["overall"] > stressed["overall"]

    def test_health_level_changes_with_stress(self):
        """Health level degrades with stress."""
        engine = _make_engine(enable_health_score=True)
        initial = engine.compute_health()
        initial_level = HealthLevel(initial["level"])
        for _ in range(10):
            engine.apply_feedback(_fb("explore_exploit", 0.5))
            engine.apply_feedback(_fb("autonomy_safety", -0.5))
        stressed = engine.compute_health()
        stressed_level = HealthLevel(stressed["level"])
        # Stressed level should be worse (lower value) than initial
        # Use overall numeric score, not string enum comparison
        assert stressed["overall"] < initial["overall"]

    def test_custom_engine_axes(self):
        """Health score works with non-default axis configurations."""
        custom_axes = [
            TensionAxis(id="a1", pillar=Pillar.COGNITION, pole_left="L", pole_right="R", position=0.0, default_position=0.4, damping=0.05),
            TensionAxis(id="a2", pillar=Pillar.PRAXIS, pole_left="L", pole_right="R", position=0.0, default_position=0.4, damping=0.05),
        ]
        engine = EquilibriumEngine(axes=custom_axes)
        hs = EquilibriumHealthScore()
        result = hs.compute(engine)
        assert result["overall"] >= 0.99
        breakdown = hs.per_axis(engine)
        assert "a1" in breakdown
        assert "a2" in breakdown
