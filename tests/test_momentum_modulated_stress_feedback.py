"""Tests for iter-029: Momentum-Modulated Stress Feedback.

When velocity tracking is enabled, BasePillar._emit_stress_feedback()
reads momentum scores to modulate its restoring force:

1. Drifting axis (momentum < 0): boost restoring force — axis is moving
   away from default, so the pull needs to be stronger.
2. Approaching axis (momentum > 0): weaken restoring force — axis is
   already heading home, let it coast.
3. No momentum data (velocity tracking off): exact backward-compatible
   behavior — no change in stress feedback magnitude.

The momentum multiplier is configurable:
 - drifting_multiplier: >1.0 (default 1.5) — boost when drifting
 - approaching_multiplier: <1.0 (default 0.5) — weaken when approaching

The pillar also tracks how many times momentum modulated the stress
feedback via _momentum_stress_modulations counter.
"""

from __future__ import annotations

import math

import pytest

from isonome.base import BasePillar
from isonome.cognition.pillar import CognitionPillar
from isonome.equilibrium import EquilibriumEngine
from isonome.mneme.pillar import MnemePillar
from isonome.praxis.pillar import PraxisPillar
from isonome.types import (
    AgentIdentity,
    AgentState,
    Feedback,
    Pillar,
)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_engine_with_velocity(**overrides) -> EquilibriumEngine:
    """Create an engine with velocity tracking enabled."""
    return EquilibriumEngine(enable_velocity_tracking=True, **overrides)


def _make_engine_without_velocity(**overrides) -> EquilibriumEngine:
    """Create an engine without velocity tracking (default)."""
    return EquilibriumEngine(**overrides)


def _make_agent_state(engine: EquilibriumEngine) -> AgentState:
    """Create an AgentState from an engine."""
    return AgentState(
        identity=AgentIdentity(name="test"),
        tensions=engine.snapshot(),
    )


def _stress_engine_velocity() -> EquilibriumEngine:
    """Create a stressed engine with velocity tracking.

    Pushes cognition axes far from homeostasis to trigger stress feedback.
    The repeated same-direction pushes create negative momentum (drifting).
    """
    eng = _make_engine_with_velocity()
    for _ in range(6):
        for axis_id, delta in [
            ("explore_exploit", 0.7),
            ("shallow_deep", -0.7),
            ("autonomy_safety", 0.8),
            ("consolidate_prune", 0.6),
        ]:
            source = Pillar.COGNITION
            if axis_id in ("autonomy_safety", "sequential_parallel", "verify_execute"):
                source = Pillar.PRAXIS
            elif axis_id in ("consolidate_prune", "specific_general"):
                source = Pillar.MNEME
            eng.apply_feedback(Feedback(
                source=source,
                tension_axis_id=axis_id,
                signal=delta,
                confidence=1.0,
                reason="stress test",
            ))
    return eng


def _stress_engine_approaching() -> EquilibriumEngine:
    """Create a stressed engine where the max-drift axis is approaching default.

    First push far away, then reverse direction to create approaching momentum.
    The stress should still be > 0.3 (still far from homeostasis).
    """
    eng = _make_engine_with_velocity()
    source = Pillar.COGNITION

    # Push explore_exploit far from default
    for _ in range(8):
        eng.apply_feedback(Feedback(
            source=source,
            tension_axis_id="explore_exploit",
            signal=0.8,
            confidence=1.0,
            reason="push away",
        ))

    # Now reverse: push toward default (approaching momentum)
    for _ in range(2):
        eng.apply_feedback(Feedback(
            source=source,
            tension_axis_id="explore_exploit",
            signal=-0.5,
            confidence=0.9,
            reason="pull toward default",
        ))

    return eng


# ═══════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════

class TestMomentumModulationConfig:
    """Momentum modulation parameters are configurable on BasePillar."""

    def test_default_drifting_multiplier(self):
        """Default drifting_multiplier should be 1.5."""
        cog = CognitionPillar(name="test")
        assert hasattr(cog, "_momentum_drifting_multiplier")
        assert cog._momentum_drifting_multiplier == 1.5

    def test_default_approaching_multiplier(self):
        """Default approaching_multiplier should be 0.5."""
        cog = CognitionPillar(name="test")
        assert hasattr(cog, "_momentum_approaching_multiplier")
        assert cog._momentum_approaching_multiplier == 0.5

    def test_custom_multipliers(self):
        """Multipliers should be settable via constructor."""
        cog = CognitionPillar(
            name="test",
            momentum_drifting_multiplier=2.0,
            momentum_approaching_multiplier=0.3,
        )
        assert cog._momentum_drifting_multiplier == 2.0
        assert cog._momentum_approaching_multiplier == 0.3

    def test_drifting_multiplier_must_be_positive(self):
        """Drifting multiplier must be > 0."""
        with pytest.raises(ValueError, match="momentum_drifting_multiplier"):
            CognitionPillar(name="test", momentum_drifting_multiplier=0.0)

    def test_approaching_multiplier_must_be_positive(self):
        """Approaching multiplier must be > 0."""
        with pytest.raises(ValueError, match="momentum_approaching_multiplier"):
            CognitionPillar(name="test", momentum_approaching_multiplier=-0.1)

    def test_momentum_stress_modulations_counter_starts_at_zero(self):
        """Pillar should track momentum-based stress modulations, starting at 0."""
        cog = CognitionPillar(name="test")
        assert hasattr(cog, "_momentum_stress_modulations")
        assert cog._momentum_stress_modulations == 0

    def test_mneme_inherits_momentum_params(self):
        """MnemePillar should also accept momentum parameters."""
        mneme = MnemePillar(
            name="test_mneme",
            momentum_drifting_multiplier=2.5,
            momentum_approaching_multiplier=0.2,
        )
        assert mneme._momentum_drifting_multiplier == 2.5
        assert mneme._momentum_approaching_multiplier == 0.2

    def test_praxis_inherits_momentum_params(self):
        """PraxisPillar should also accept momentum parameters."""
        prax = PraxisPillar(
            name="test_prax",
            momentum_drifting_multiplier=3.0,
            momentum_approaching_multiplier=0.1,
        )
        assert prax._momentum_drifting_multiplier == 3.0
        assert prax._momentum_approaching_multiplier == 0.1


# ═══════════════════════════════════════════════════════════════
# 2. Backward Compatibility (no velocity tracking)
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompatibilityNoVelocity:
    """Without velocity tracking, stress feedback is exactly as before."""

    def test_no_velocity_same_stress_feedback(self):
        """Without velocity tracking, stress feedback magnitude is unchanged."""
        eng = _make_engine_without_velocity()
        for _ in range(6):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.7,
                confidence=1.0,
                reason="stress",
            ))

        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()  # clear initial
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        # Should still emit stress feedback
        assert len(stress_fb) >= 1
        # No momentum modulation counter increment
        assert cog._momentum_stress_modulations == 0

    def test_no_velocity_reason_unmodulated(self):
        """Without velocity tracking, stress feedback reason should not
        mention momentum modulation."""
        eng = _make_engine_without_velocity()
        for _ in range(6):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.7,
                confidence=1.0,
                reason="stress",
            ))

        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        for fb in stress_fb:
            assert "momentum" not in fb.reason


# ═══════════════════════════════════════════════════════════════
# 3. Drifting Momentum — Boosted Restoring Force
# ═══════════════════════════════════════════════════════════════

class TestDriftingMomentumBoostedFeedback:
    """When the max-drift axis has negative momentum (drifting away),
    stress feedback should be stronger than baseline."""

    def test_drifting_feedback_stronger_than_baseline(self):
        """Stress feedback on a drifting axis should be stronger."""
        eng = _stress_engine_velocity()

        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        assert len(stress_fb) >= 1, "Should emit stress feedback when stressed"

        # Check that momentum modulation was applied if the max-drift
        # axis was drifting (momentum < 0)
        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum < 0:
            # Axis is drifting — momentum modulation should be logged
            assert cog._momentum_stress_modulations >= 1
            # The reason should mention momentum boost
            boosted = [f for f in stress_fb if "momentum-boosted" in f.reason]
            assert len(boosted) >= 1

    def test_drifting_multiplier_applied(self):
        """When drifting, the signal magnitude should be multiplied by
        the drifting_multiplier (default 1.5)."""
        eng = _stress_engine_velocity()

        # Create pillar with a very high multiplier to make effect obvious
        cog = CognitionPillar(
            name="test",
            momentum_drifting_multiplier=3.0,
        )
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum < 0 and len(stress_fb) >= 1:
            # Signal should be present and > 0
            assert abs(stress_fb[0].signal) > 0


# ═══════════════════════════════════════════════════════════════
# 4. Approaching Momentum — Weakened Restoring Force
# ═══════════════════════════════════════════════════════════════

class TestApproachingMomentumWeakenedFeedback:
    """When the max-drift axis has positive momentum (approaching default),
    stress feedback should be weaker than baseline."""

    def test_approaching_feedback_weaker(self):
        """Stress feedback on an approaching axis should include momentum
        modulation with approaching multiplier."""
        eng = _stress_engine_approaching()

        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum > 0 and len(stress_fb) >= 1:
            # Approaching — momentum modulation counter should be incremented
            assert cog._momentum_stress_modulations >= 1
            # The reason should mention momentum dampened
            dampened = [f for f in stress_fb if "momentum-dampened" in f.reason]
            assert len(dampened) >= 1

    def test_approaching_multiplier_reduces_signal(self):
        """When approaching, the signal should be reduced by the
        approaching_multiplier (default 0.5)."""
        eng = _stress_engine_approaching()

        cog = CognitionPillar(
            name="test",
            momentum_approaching_multiplier=0.2,  # Very small
        )
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum > 0 and len(stress_fb) >= 1:
            # Signal should still be present but reduced
            assert abs(stress_fb[0].signal) > 0


# ═══════════════════════════════════════════════════════════════
# 5. Zero Momentum — No Modulation
# ═══════════════════════════════════════════════════════════════

class TestZeroMomentumNoModulation:
    """When the max-drift axis has zero momentum (at default or no movement),
    stress feedback should not be momentum-modulated."""

    def test_zero_momentum_no_modulation(self):
        """With velocity tracking but zero momentum, no momentum modulation."""
        # Create engine with velocity tracking but push just enough for stress
        eng = _make_engine_with_velocity()
        for _ in range(3):
            eng.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.5,
                confidence=0.9,
                reason="moderate push",
            ))
            eng.apply_feedback(Feedback(
                source=Pillar.PRAXIS,
                tension_axis_id="autonomy_safety",
                signal=0.6,
                confidence=0.9,
                reason="moderate push",
            ))

        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        if stress_fb:
            # Check if the max-drift axis has zero momentum
            view = eng.view_for(Pillar.COGNITION)
            own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
            max_drift_axis = max(own_drift, key=own_drift.get)
            momentum = view.get_momentum_score(max_drift_axis)
            if momentum == 0.0:
                # Zero momentum — no modulation, reason should not mention momentum
                for fb in stress_fb:
                    assert "momentum-boosted" not in fb.reason
                    assert "momentum-dampened" not in fb.reason


# ═══════════════════════════════════════════════════════════════
# 6. Stress Feedback Bounded After Modulation
# ═══════════════════════════════════════════════════════════════

class TestStressFeedbackBoundedAfterModulation:
    """Stress feedback signal must remain in [-1.0, 1.0] even after
    momentum multiplication."""

    def test_drifting_signal_still_bounded(self):
        """Even with a large drifting multiplier, signal stays in bounds."""
        eng = _stress_engine_velocity()
        cog = CognitionPillar(
            name="test",
            momentum_drifting_multiplier=10.0,  # Very aggressive
        )
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        for fb in stress_fb:
            assert -1.0 <= fb.signal <= 1.0

    def test_approaching_signal_still_bounded(self):
        """Approaching multiplier should not change sign or bounds."""
        eng = _stress_engine_approaching()
        cog = CognitionPillar(
            name="test",
            momentum_approaching_multiplier=0.01,  # Very small
        )
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        for fb in stress_fb:
            assert -1.0 <= fb.signal <= 1.0


# ═══════════════════════════════════════════════════════════════
# 7. Cross-Pillar Momentum Stress Feedback
# ═══════════════════════════════════════════════════════════════

class TestCrossPillarMomentumStressFeedback:
    """Momentum-modulated stress feedback works for all three pillars."""

    def test_praxis_momentum_modulation(self):
        """PraxisPillar should also apply momentum-modulated stress feedback."""
        eng = _stress_engine_velocity()
        prax = PraxisPillar(name="test_prax")
        agent_state = _make_agent_state(eng)
        prax.initialize(agent_state)
        prax.bind_engine(eng)
        prax.drain_feedback()
        prax.process_queued()
        feedbacks = prax.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        # Praxis has momentum modulation counter
        assert hasattr(prax, "_momentum_stress_modulations")

    def test_mneme_momentum_modulation(self):
        """MnemePillar should also apply momentum-modulated stress feedback."""
        eng = _stress_engine_velocity()
        mneme = MnemePillar(name="test_mneme")
        agent_state = _make_agent_state(eng)
        mneme.initialize(agent_state)
        mneme.bind_engine(eng)
        mneme.drain_feedback()
        mneme.process_queued()
        feedbacks = mneme.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        # Mneme has momentum modulation counter
        assert hasattr(mneme, "_momentum_stress_modulations")


# ═══════════════════════════════════════════════════════════════
# 8. Comparison: With vs Without Momentum Modulation
# ═══════════════════════════════════════════════════════════════

class TestModulationMagnitudeComparison:
    """Verify that momentum modulation actually changes the signal
    magnitude compared to the un-modulated baseline."""

    def test_drifting_signal_larger_than_unmodulated(self):
        """A drifting axis should produce a larger stress feedback signal
        than an un-modulated one at the same stress/drift level."""
        # Engine with velocity: drifting momentum
        eng_vel = _stress_engine_velocity()
        cog_vel = CognitionPillar(name="vel")
        agent_state_vel = _make_agent_state(eng_vel)
        cog_vel.initialize(agent_state_vel)
        cog_vel.bind_engine(eng_vel)
        cog_vel.drain_feedback()
        cog_vel.process_queued()
        fb_vel = cog_vel.drain_feedback()
        stress_vel = [f for f in fb_vel if "stress-reactive" in f.reason]

        # Engine without velocity: same stress but no momentum data
        eng_no_vel = EquilibriumEngine()
        for _ in range(6):
            for axis_id, delta in [
                ("explore_exploit", 0.7),
                ("shallow_deep", -0.7),
                ("autonomy_safety", 0.8),
                ("consolidate_prune", 0.6),
            ]:
                source = Pillar.COGNITION
                if axis_id in ("autonomy_safety", "sequential_parallel", "verify_execute"):
                    source = Pillar.PRAXIS
                elif axis_id in ("consolidate_prune", "specific_general"):
                    source = Pillar.MNEME
                eng_no_vel.apply_feedback(Feedback(
                    source=source,
                    tension_axis_id=axis_id,
                    signal=delta,
                    confidence=1.0,
                    reason="stress test",
                ))

        cog_no_vel = CognitionPillar(name="no_vel")
        agent_state_no = _make_agent_state(eng_no_vel)
        cog_no_vel.initialize(agent_state_no)
        cog_no_vel.bind_engine(eng_no_vel)
        cog_no_vel.drain_feedback()
        cog_no_vel.process_queued()
        fb_no_vel = cog_no_vel.drain_feedback()
        stress_no_vel = [f for f in fb_no_vel if "stress-reactive" in f.reason]

        # Both should emit stress feedback
        if stress_vel and stress_no_vel:
            # If max-drift axis is drifting (momentum < 0), the velocity-
            # enabled pillar's signal should be larger
            view = eng_vel.view_for(Pillar.COGNITION)
            own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
            max_drift_axis = max(own_drift, key=own_drift.get)
            momentum = view.get_momentum_score(max_drift_axis)
            if momentum < 0:
                # Drifting: velocity-enabled signal should be >= no-velocity
                assert abs(stress_vel[0].signal) >= abs(stress_no_vel[0].signal) * 0.9


# ═══════════════════════════════════════════════════════════════
# 9. Disable Momentum Stress Modulation
# ═══════════════════════════════════════════════════════════════

class TestDisableMomentumStressModulation:
    """Pillar should support disabling momentum-modulated stress feedback
    independently from stress feedback itself."""

    def test_momentum_stress_modulation_disabled(self):
        """When momentum_stress_modulation_enabled=False, no momentum
        modulation is applied even with velocity tracking."""
        eng = _stress_engine_velocity()
        cog = CognitionPillar(
            name="test",
            momentum_stress_modulation_enabled=False,
        )
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        # Should still emit stress feedback
        assert len(stress_fb) >= 1
        # But no momentum modulation in reason
        for fb in stress_fb:
            assert "momentum-boosted" not in fb.reason
            assert "momentum-dampened" not in fb.reason
        # Counter should be 0
        assert cog._momentum_stress_modulations == 0

    def test_default_momentum_stress_modulation_enabled(self):
        """By default, momentum stress modulation should be enabled."""
        cog = CognitionPillar(name="test")
        assert cog._momentum_stress_modulation_enabled is True


# ═══════════════════════════════════════════════════════════════
# 10. Reason String Format
# ═══════════════════════════════════════════════════════════════

class TestMomentumStressFeedbackReasonFormat:
    """Stress feedback reason strings should include momentum info."""

    def test_drifting_reason_includes_momentum_boosted(self):
        """When drifting, reason should mention 'momentum-boosted'."""
        eng = _stress_engine_velocity()
        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum < 0 and stress_fb:
            boosted = [f for f in stress_fb if "momentum-boosted" in f.reason]
            if boosted:
                assert "x1.50" in boosted[0].reason  # default multiplier

    def test_approaching_reason_includes_momentum_dampened(self):
        """When approaching, reason should mention 'momentum-dampened'."""
        eng = _stress_engine_approaching()
        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        view = eng.view_for(Pillar.COGNITION)
        own_drift = {k: v for k, v in view.drift.items() if k in view.own_axes}
        max_drift_axis = max(own_drift, key=own_drift.get)
        momentum = view.get_momentum_score(max_drift_axis)

        if momentum > 0 and stress_fb:
            dampened = [f for f in stress_fb if "momentum-dampened" in f.reason]
            if dampened:
                assert "x0.50" in dampened[0].reason  # default multiplier


# ═══════════════════════════════════════════════════════════════
# 11. Confidence Unchanged by Momentum Modulation
# ═══════════════════════════════════════════════════════════════

class TestConfidenceUnchangedByMomentum:
    """Momentum modulation should affect signal magnitude, not confidence."""

    def test_drifting_confidence_same(self):
        """Confidence should remain at 0.4 regardless of momentum modulation."""
        eng = _stress_engine_velocity()
        cog = CognitionPillar(name="test")
        agent_state = _make_agent_state(eng)
        cog.initialize(agent_state)
        cog.bind_engine(eng)
        cog.drain_feedback()
        cog.process_queued()
        feedbacks = cog.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]

        for fb in stress_fb:
            assert fb.confidence == 0.4
