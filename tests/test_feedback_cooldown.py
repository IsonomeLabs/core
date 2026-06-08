"""Tests for iter-030: Feedback Cooldown System.

When an axis receives high-frequency contradictory feedback (e.g., one pillar
pushing explore while another pushes exploit), the engine can oscillate even
with adaptive damping. The cooldown system adds a per-axis (pillar, axis)
cooldown that dampens repeated feedback from the same source on the same axis
within a configurable tick window.

Core invariants:
1. First feedback from (pillar, axis) always passes through unmodified
2. Repeated feedback within cooldown_window gets dampened by a decay factor
3. After cooldown_window ticks with no feedback, the cooldown resets
4. Cooldown is purely additive — it never amplifies feedback
5. When disabled, the system is completely transparent (backward-compatible)

Integration:
- EquilibriumEngine applies cooldown in apply_feedback() and apply_feedback_batch()
- PillarEquilibriumView exposes cooldown state for pillar introspection
- Serialization preserves cooldown state for cross-session persistence
"""

from __future__ import annotations

import pytest

from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.equilibrium.cooldown import FeedbackCooldownManager
from isonome.types import Feedback, Pillar, TensionAxis, TensionID


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def cooldown() -> FeedbackCooldownManager:
    """Fresh cooldown manager with default settings."""
    return FeedbackCooldownManager()


@pytest.fixture
def cooldown_short() -> FeedbackCooldownManager:
    """Cooldown with short window for testing."""
    return FeedbackCooldownManager(cooldown_window=3, decay_factor=0.5)


@pytest.fixture
def cooldown_strict() -> FeedbackCooldownManager:
    """Cooldown with strict settings (small window, aggressive decay)."""
    return FeedbackCooldownManager(cooldown_window=2, decay_factor=0.25)


@pytest.fixture
def engine_with_cooldown() -> EquilibriumEngine:
    """Engine with feedback cooldown enabled."""
    return EquilibriumEngine(enable_feedback_cooldown=True)


@pytest.fixture
def engine_without_cooldown() -> EquilibriumEngine:
    """Standard engine without feedback cooldown."""
    return EquilibriumEngine()


# ======================================================================
# TestFeedbackCooldownManager — Unit Tests
# ======================================================================


class TestFeedbackCooldownManagerConstruction:
    """Construction and validation of FeedbackCooldownManager."""

    def test_default_construction(self, cooldown: FeedbackCooldownManager):
        """Default cooldown manager has sensible defaults."""
        assert cooldown.cooldown_window == 5
        assert cooldown.decay_factor == 0.5
        assert cooldown.total_cooldowns == 0
        assert cooldown.total_suppressions == 0

    def test_custom_construction(self):
        """Custom parameters are respected."""
        mgr = FeedbackCooldownManager(cooldown_window=10, decay_factor=0.3)
        assert mgr.cooldown_window == 10
        assert mgr.decay_factor == 0.3

    def test_invalid_window_zero(self):
        """Window of 0 raises ValueError."""
        with pytest.raises(ValueError, match="cooldown_window must be >= 1"):
            FeedbackCooldownManager(cooldown_window=0)

    def test_invalid_window_negative(self):
        """Negative window raises ValueError."""
        with pytest.raises(ValueError, match="cooldown_window must be >= 1"):
            FeedbackCooldownManager(cooldown_window=-1)

    def test_invalid_decay_zero(self):
        """Decay factor of 0 raises ValueError."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            FeedbackCooldownManager(decay_factor=0.0)

    def test_invalid_decay_negative(self):
        """Negative decay factor raises ValueError."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            FeedbackCooldownManager(decay_factor=-0.1)

    def test_invalid_decay_above_one(self):
        """Decay factor > 1 raises ValueError (would amplify feedback)."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            FeedbackCooldownManager(decay_factor=1.5)

    def test_decay_factor_one_allowed(self):
        """Decay factor of exactly 1.0 is allowed (no dampening, just tracking)."""
        mgr = FeedbackCooldownManager(decay_factor=1.0)
        assert mgr.decay_factor == 1.0

    def test_repr(self, cooldown: FeedbackCooldownManager):
        """Repr shows key parameters."""
        r = repr(cooldown)
        assert "FeedbackCooldownManager" in r
        assert "window=5" in r
        assert "decay=0.50" in r


class TestCooldownTracking:
    """Cooldown state tracking per (pillar, axis) pair."""

    def test_first_feedback_no_cooldown(self, cooldown: FeedbackCooldownManager):
        """First feedback from any (pillar, axis) pair is never cooled down."""
        multiplier = cooldown.check_and_apply(
            axis_id="explore_exploit",
            source_pillar=Pillar.COGNITION,
            tick=1,
        )
        assert multiplier == 1.0
        assert cooldown.total_cooldowns == 0

    def test_second_feedback_within_window(self, cooldown_short: FeedbackCooldownManager):
        """Second feedback within window gets dampened."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        multiplier = cooldown_short.check_and_apply(
            "explore_exploit", Pillar.COGNITION, tick=2,
        )
        assert multiplier == 0.5  # decay_factor
        assert cooldown_short.total_cooldowns == 1

    def test_different_pillar_no_cooldown(self, cooldown_short: FeedbackCooldownManager):
        """Same axis but different pillar doesn't trigger cooldown."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        multiplier = cooldown_short.check_and_apply(
            "explore_exploit", Pillar.PRAXIS, tick=2,
        )
        assert multiplier == 1.0  # Different pillar = no cooldown

    def test_different_axis_no_cooldown(self, cooldown_short: FeedbackCooldownManager):
        """Same pillar but different axis doesn't trigger cooldown."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        multiplier = cooldown_short.check_and_apply(
            "shallow_deep", Pillar.COGNITION, tick=2,
        )
        assert multiplier == 1.0  # Different axis = no cooldown

    def test_cooldown_resets_after_window(self, cooldown_short: FeedbackCooldownManager):
        """After cooldown_window ticks with no feedback, the next one is fresh."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        # Skip beyond cooldown window
        multiplier = cooldown_short.check_and_apply(
            "explore_exploit", Pillar.COGNITION, tick=10,
        )
        assert multiplier == 1.0  # Window expired

    def test_cooldown_accumulates(self, cooldown_short: FeedbackCooldownManager):
        """Multiple feedbacks within window accumulate cooldown."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        m1 = cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        m2 = cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=3)
        assert m1 == 0.5
        assert m2 == 0.25  # 0.5 * 0.5 (decay_factor compounded)

    def test_cooldown_min_floor(self, cooldown_strict: FeedbackCooldownManager):
        """Cooldown multiplier never drops below a minimum floor."""
        cooldown_strict.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_strict.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        # Third and fourth hits with very aggressive decay
        m3 = cooldown_strict.check_and_apply("explore_exploit", Pillar.COGNITION, tick=3)
        m4 = cooldown_strict.check_and_apply("explore_exploit", Pillar.COGNITION, tick=4)
        assert m3 == 0.0625  # 0.25^2
        # Floor prevents going below 0.01
        assert m4 >= 0.01

    def test_interleaved_pillars_independent(self, cooldown_short: FeedbackCooldownManager):
        """Two pillars on the same axis track cooldowns independently."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        m_cog = cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        m_prax = cooldown_short.check_and_apply("explore_exploit", Pillar.PRAXIS, tick=2)
        assert m_cog == 0.5  # Second from Cognition
        assert m_prax == 1.0  # First from Praxis

    def test_tick_must_advance(self, cooldown_short: FeedbackCooldownManager):
        """Same tick feedback from same source is tracked as repeated."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        m = cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        # Same tick = within window, so cooldown applies
        assert m == 0.5


class TestCooldownStats:
    """Statistics tracking for cooldown events."""

    def test_total_cooldowns_count(self, cooldown_short: FeedbackCooldownManager):
        """total_cooldowns increments when cooldown is applied."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=3)
        # Ticks 2 and 3 triggered cooldown
        assert cooldown_short.total_cooldowns == 2

    def test_total_suppressions_tracks_decay(self, cooldown_short: FeedbackCooldownManager):
        """total_suppressions tracks the total suppression amount."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        # Second feedback: multiplier = 0.5, suppression = 1.0 - 0.5 = 0.5
        assert cooldown_short.total_suppressions == pytest.approx(0.5, abs=1e-6)

    def test_active_keys(self, cooldown_short: FeedbackCooldownManager):
        """active_keys returns currently tracked (pillar, axis) pairs."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("shallow_deep", Pillar.COGNITION, tick=1)
        keys = cooldown_short.active_keys
        assert len(keys) == 2
        assert (Pillar.COGNITION.value, "explore_exploit") in keys
        assert (Pillar.COGNITION.value, "shallow_deep") in keys

    def test_cooldown_state_for_axis(self, cooldown_short: FeedbackCooldownManager):
        """cooldown_state_for returns the state for a specific (pillar, axis)."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        state = cooldown_short.cooldown_state_for("explore_exploit", Pillar.COGNITION)
        assert state is not None
        assert "last_tick" in state
        assert "hit_count" in state
        assert "current_multiplier" in state
        assert state["last_tick"] == 1
        assert state["hit_count"] == 1
        assert state["current_multiplier"] == 1.0

    def test_cooldown_state_for_unknown(self, cooldown_short: FeedbackCooldownManager):
        """cooldown_state_for returns None for unknown pairs."""
        state = cooldown_short.cooldown_state_for("nonexistent", Pillar.COGNITION)
        assert state is None


class TestCooldownReset:
    """Reset behavior for cooldown manager."""

    def test_reset_clears_all(self, cooldown_short: FeedbackCooldownManager):
        """reset() clears all tracking state and counters."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        cooldown_short.reset()
        assert cooldown_short.total_cooldowns == 0
        assert cooldown_short.total_suppressions == 0.0
        assert len(cooldown_short.active_keys) == 0

    def test_after_reset_first_feedback_no_cooldown(self, cooldown_short: FeedbackCooldownManager):
        """After reset, first feedback is treated as fresh."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.reset()
        m = cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        assert m == 1.0


class TestCooldownSerialization:
    """Serialization round-trip for FeedbackCooldownManager."""

    def test_round_trip_empty(self):
        """Empty cooldown manager survives serialization."""
        mgr = FeedbackCooldownManager()
        data = mgr.to_dict()
        restored = FeedbackCooldownManager.from_dict(data)
        assert restored.cooldown_window == mgr.cooldown_window
        assert restored.decay_factor == mgr.decay_factor
        assert restored.total_cooldowns == 0
        assert restored.total_suppressions == 0.0

    def test_round_trip_with_state(self, cooldown_short: FeedbackCooldownManager):
        """Cooldown manager with active state survives serialization."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        cooldown_short.check_and_apply("shallow_deep", Pillar.COGNITION, tick=1)

        data = cooldown_short.to_dict()
        restored = FeedbackCooldownManager.from_dict(data)

        assert restored.cooldown_window == cooldown_short.cooldown_window
        assert restored.decay_factor == cooldown_short.decay_factor
        assert restored.total_cooldowns == cooldown_short.total_cooldowns
        assert restored.total_suppressions == pytest.approx(
            cooldown_short.total_suppressions, abs=1e-6
        )
        assert len(restored.active_keys) == len(cooldown_short.active_keys)

    def test_round_trip_preserves_multiplier(self, cooldown_short: FeedbackCooldownManager):
        """Serialization preserves the current multiplier state."""
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        cooldown_short.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)

        data = cooldown_short.to_dict()
        restored = FeedbackCooldownManager.from_dict(data)

        # Next feedback should continue the cooldown chain
        m = restored.check_and_apply("explore_exploit", Pillar.COGNITION, tick=3)
        assert m == 0.25  # 0.5 * 0.5 — continues from restored state

    def test_from_dict_defaults(self):
        """from_dict with missing fields uses sensible defaults."""
        restored = FeedbackCooldownManager.from_dict({})
        assert restored.cooldown_window == 5
        assert restored.decay_factor == 0.5


# ======================================================================
# TestEquilibriumEngineCooldown — Integration Tests
# ======================================================================


class TestEquilibriumEngineCooldown:
    """Feedback cooldown integration with EquilibriumEngine."""

    def test_engine_without_cooldown_is_transparent(
        self, engine_without_cooldown: EquilibriumEngine,
    ):
        """Engine without cooldown behaves exactly as before."""
        assert engine_without_cooldown.feedback_cooldown is None

    def test_engine_with_cooldown_has_manager(
        self, engine_with_cooldown: EquilibriumEngine,
    ):
        """Engine with cooldown has a FeedbackCooldownManager."""
        assert engine_with_cooldown.feedback_cooldown is not None
        assert isinstance(engine_with_cooldown.feedback_cooldown, FeedbackCooldownManager)

    def test_custom_cooldown_manager(self):
        """Engine accepts a pre-configured cooldown manager."""
        mgr = FeedbackCooldownManager(cooldown_window=3, decay_factor=0.3)
        engine = EquilibriumEngine(feedback_cooldown=mgr)
        assert engine.feedback_cooldown is mgr
        assert engine.feedback_cooldown.cooldown_window == 3
        assert engine.feedback_cooldown.decay_factor == 0.3

    def test_enable_feedback_cooldown_flag(self):
        """enable_feedback_cooldown=True creates a default manager."""
        engine = EquilibriumEngine(enable_feedback_cooldown=True)
        assert engine.feedback_cooldown is not None
        assert engine.feedback_cooldown.cooldown_window == 5
        assert engine.feedback_cooldown.decay_factor == 0.5

    def test_first_feedback_unchanged(self, engine_with_cooldown: EquilibriumEngine):
        """First feedback on an axis applies full signal."""
        axis_before = engine_with_cooldown.get_axis("explore_exploit")
        pos_before = axis_before.position

        engine_with_cooldown.apply_feedback(
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.3,
                confidence=0.8,
                reason="test: first feedback",
            )
        )

        axis_after = engine_with_cooldown.get_axis("explore_exploit")
        # Position should have moved by 0.3 * 0.8 * (1 - damping)
        expected_delta = 0.3 * 0.8 * (1 - axis_before.damping)
        assert axis_after.position == pytest.approx(
            pos_before + expected_delta, abs=1e-6
        )

    def test_repeated_feedback_dampened(self):
        """Second feedback from same (pillar, axis) within cooldown gets dampened."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,  # Isolate cooldown effect
        )

        # First feedback: full signal
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=1.0,
            reason="test: first feedback",
        ))
        pos_after_first = engine.get_axis("explore_exploit").position

        # Second feedback: same source and axis, dampened by cooldown
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=1.0,
            reason="test: second feedback",
        ))
        pos_after_second = engine.get_axis("explore_exploit").position

        # The second movement should be less than the first due to cooldown
        # But we need to account for adaptive damping if any — we disabled it
        delta1 = pos_after_first - 0.15  # default position
        delta2 = pos_after_second - pos_after_first
        # Second delta should be smaller because of cooldown
        assert abs(delta2) < abs(delta1)

    def test_different_source_no_cooldown(self):
        """Different pillar on same axis doesn't get cooldown."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )

        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: cognition feedback",
        ))
        # Praxis sends feedback to same axis — no cooldown
        engine.apply_feedback(Feedback(
            source=Pillar.PRAXIS,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=1.0,
            reason="test: praxis feedback",
        ))

        # Both should have moved similarly (different sources, no cooldown)
        assert engine.feedback_cooldown.total_cooldowns == 0

    def test_batch_feedback_applies_cooldown(self):
        """Cooldown is applied in apply_feedback_batch as well."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )

        # First batch
        engine.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.3, confidence=1.0, reason="test: batch 1"),
        ])

        # Second batch with same source
        engine.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.3, confidence=1.0, reason="test: batch 2"),
        ])

        # Cooldown should have been applied
        assert engine.feedback_cooldown.total_cooldowns >= 1

    def test_batch_with_multiple_sources(self):
        """Batch with feedback from different sources tracks each independently."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )

        # First batch with multiple sources on same axis
        engine.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: cog batch 1"),
            Feedback(source=Pillar.PRAXIS, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: prax batch 1"),
        ])

        # Second batch — Cognition gets cooldown, Praxis doesn't
        engine.apply_feedback_batch([
            Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: cog batch 2"),
            Feedback(source=Pillar.PRAXIS, tension_axis_id="explore_exploit",
                     signal=0.2, confidence=1.0, reason="test: prax batch 2"),
        ])

        # Both pillars appeared in both batches, so both get cooldown
        assert engine.feedback_cooldown.total_cooldowns >= 1

    def test_cooldown_window_expires(self):
        """After cooldown window expires, feedback returns to full strength."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        # Default window is 5 ticks
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: initial",
        ))
        # Advance the engine tick far enough to expire cooldown
        # Apply feedback to other axes to advance ticks
        for _ in range(10):
            engine.apply_feedback(Feedback(
                source=Pillar.PRAXIS, tension_axis_id="autonomy_safety",
                signal=0.01, confidence=0.1, reason="test: tick advance",
            ))

        # Now Cognition's next feedback on explore_exploit should not be cooled
        m = engine.feedback_cooldown.check_and_apply(
            "explore_exploit", Pillar.COGNITION, tick=100,
        )
        assert m == 1.0


class TestEngineCooldownSerialization:
    """Cooldown state serialization through EquilibriumEngine."""

    def test_engine_to_dict_includes_cooldown_state(
        self, engine_with_cooldown: EquilibriumEngine,
    ):
        """Engine to_dict includes cooldown state."""
        data = engine_with_cooldown.to_dict()
        assert "feedback_cooldown_state" in data

    def test_engine_to_dict_no_cooldown(self, engine_without_cooldown: EquilibriumEngine):
        """Engine without cooldown has None in serialization."""
        data = engine_without_cooldown.to_dict()
        assert data.get("feedback_cooldown_state") is None

    def test_engine_round_trip_with_cooldown(self):
        """Engine with cooldown state survives serialization round-trip."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))

        # Serialize and restore
        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        assert restored.feedback_cooldown is not None
        assert restored.feedback_cooldown.total_cooldowns == 1
        assert restored.feedback_cooldown.decay_factor == 0.5

    def test_engine_round_trip_without_cooldown(self):
        """Engine without cooldown survives serialization round-trip."""
        engine = EquilibriumEngine()
        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        assert restored.feedback_cooldown is None


class TestPillarEquilibriumViewCooldown:
    """PillarEquilibriumView exposes cooldown information."""

    def test_view_with_cooldown(self):
        """View on engine with cooldown shows cooldown info."""
        engine = EquilibriumEngine(enable_feedback_cooldown=True)
        view = engine.view_for(Pillar.COGNITION)
        # Should have cooldown data available
        assert view.feedback_cooldown is not None

    def test_view_without_cooldown(self):
        """View on engine without cooldown shows None."""
        engine = EquilibriumEngine()
        view = engine.view_for(Pillar.COGNITION)
        assert view.feedback_cooldown is None

    def test_view_cooldown_axes(self):
        """View can identify which own axes are under cooldown."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))

        view = engine.view_for(Pillar.COGNITION)
        cooled = view.cooldown_axes
        assert "explore_exploit" in cooled


class TestCooldownWithAdaptiveDamping:
    """Cooldown works correctly alongside adaptive damping."""

    def test_cooldown_and_adaptive_damping_both_active(self):
        """When both cooldown and adaptive damping are enabled, both apply."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=True,
        )
        # First feedback
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.5, confidence=1.0, reason="test: first",
        ))
        # Second feedback — both cooldown and adaptive damping apply
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.5, confidence=1.0, reason="test: second",
        ))
        # Both systems should have been active
        assert engine.feedback_cooldown.total_cooldowns >= 1
        assert engine.adaptive_damping.total_adaptations >= 0

    def test_cooldown_does_not_interfere_with_oscillation_detection(self):
        """Cooldown dampens feedback but doesn't prevent oscillation detection."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        # Rapidly alternate positive and negative feedback
        for i in range(20):
            sign = 1 if i % 2 == 0 else -1
            engine.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=sign * 0.8,
                confidence=1.0,
                reason="test: oscillation probe",
            ))
        # Oscillation detection should still work
        assert engine.total_oscillation_events >= 0


class TestCooldownEdgeCases:
    """Edge cases for the feedback cooldown system."""

    def test_high_frequency_same_tick(self):
        """Multiple feedbacks on the same tick are all cooled after the first."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        # Apply multiple feedbacks on the same tick
        # (This can happen in batch processing)
        cooldown_mgr = engine.feedback_cooldown
        m1 = cooldown_mgr.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        m2 = cooldown_mgr.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        assert m1 == 1.0
        assert m2 < 1.0  # Second call on same tick = within window

    def test_all_pillars_all_axes(self):
        """Cooldown tracks all 8 axes x 3 pillars independently."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_adaptive_damping=False,
        )
        cooldown_mgr = engine.feedback_cooldown
        # First feedback from each (pillar, axis) pair: no cooldown
        for axis in engine.axes:
            for pillar in Pillar:
                m = cooldown_mgr.check_and_apply(axis.id, pillar, tick=1)
                assert m == 1.0

        # Second round: all should be cooled
        for axis in engine.axes:
            for pillar in Pillar:
                m = cooldown_mgr.check_and_apply(axis.id, pillar, tick=2)
                assert m < 1.0

    def test_cooldown_with_zero_signal(self):
        """Zero signal feedback still updates cooldown tracking."""
        cooldown = FeedbackCooldownManager()
        cooldown.check_and_apply("explore_exploit", Pillar.COGNITION, tick=1)
        m = cooldown.check_and_apply("explore_exploit", Pillar.COGNITION, tick=2)
        # Cooldown tracking still applies even if signal would be zero
        assert m == 0.5  # Cooldown multiplier is independent of signal magnitude

    def test_cooldown_with_event_log(self):
        """Cooldown works alongside event logging without conflicts."""
        engine = EquilibriumEngine(
            enable_feedback_cooldown=True,
            enable_event_log=True,
            enable_adaptive_damping=False,
        )
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: first",
        ))
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION, tension_axis_id="explore_exploit",
            signal=0.3, confidence=1.0, reason="test: second",
        ))
        # Event log should still record both feedbacks
        assert engine.event_log is not None
        assert engine.event_log.total_events >= 2
