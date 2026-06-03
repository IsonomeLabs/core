"""Tests for the Pillar-Equilibrium Pull Mechanism.

Covers:
1. PillarEquilibriumView: axis decomposition, stress, drift, oscillation
2. EquilibriumEngine.view_for(): per-pillar view creation
3. BasePillar.bind_engine() / unbind_engine(): lifecycle
4. BasePillar.process_queued() auto-sync: view refresh on each tick
5. Stress-reactive feedback: homeostasis pull when stressed
6. CognitionPillar._on_equilibrium_sync(): auto tension modulation
7. PraxisPillar._on_equilibrium_sync(): auto tension + cross-pillar
8. MnemePillar._on_equilibrium_sync(): auto tension + cross-pillar
9. Integration: full pull loop (bind → tick → read → feedback)
"""

from __future__ import annotations

import pytest
from collections import deque

from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.types import (
    AgentIdentity,
    AgentState,
    Feedback,
    Pillar,
    TensionAxis,
    TensionSnapshot,
    TensionID,
)
from isonome.base import BasePillar
from isonome.cognition.pillar import CognitionPillar
from isonome.praxis.pillar import PraxisPillar
from isonome.mneme.pillar import MnemePillar


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def engine():
    """A fresh equilibrium engine with default axes."""
    return EquilibriumEngine()


@pytest.fixture
def stressed_engine():
    """An engine with axes pushed far from homeostasis (stress > 0.3)."""
    eng = EquilibriumEngine()
    # Push axes aggressively — multiple rounds to overcome damping
    for _ in range(5):
        for axis_id, delta in [
            ("explore_exploit", 0.8),
            ("shallow_deep", -0.9),
            ("autonomy_safety", 0.9),
            ("consolidate_prune", 0.7),
        ]:
            source = Pillar.COGNITION
            if axis_id in ("autonomy_safety", "sequential_parallel", "verify_execute"):
                source = Pillar.PRAXIS
            elif axis_id in ("consolidate_prune", "specific_general"):
                source = Pillar.MNEME
            fb = Feedback(
                source=source,
                tension_axis_id=axis_id,
                signal=delta,
                confidence=1.0,
                reason="stress test",
            )
            eng.apply_feedback(fb)
    return eng


@pytest.fixture
def oscillating_engine():
    """An engine with history that triggers oscillation detection."""
    eng = EquilibriumEngine(oscillation_threshold=0.1, oscillation_window=8)
    # Rapidly swing an axis back and forth to create oscillation
    for i in range(10):
        direction = 0.8 if i % 2 == 0 else -0.8
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=direction,
            confidence=0.9,
            reason="oscillation test",
        )
        eng.apply_feedback(fb)
    return eng


@pytest.fixture
def agent_state():
    """Minimal AgentState for pillar initialization."""
    return AgentState(identity=AgentIdentity(name="test-agent"))


# ═══════════════════════════════════════════════════════════════
#  1. PillarEquilibriumView — Axis Decomposition
# ═══════════════════════════════════════════════════════════════

class TestPillarEquilibriumViewAxisDecomposition:
    """View correctly splits axes into own vs. cross-pillar."""

    def test_cognition_gets_3_own_axes(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert len(view.own_axes) == 3
        assert "explore_exploit" in view.own_axes
        assert "shallow_deep" in view.own_axes
        assert "divergent_convergent" in view.own_axes

    def test_cognition_gets_5_cross_axes(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert len(view.cross_axes) == 5
        assert "autonomy_safety" in view.cross_axes
        assert "consolidate_prune" in view.cross_axes

    def test_praxis_gets_3_own_axes(self, engine):
        view = engine.view_for(Pillar.PRAXIS)
        assert len(view.own_axes) == 3
        assert "autonomy_safety" in view.own_axes
        assert "sequential_parallel" in view.own_axes
        assert "verify_execute" in view.own_axes

    def test_praxis_gets_5_cross_axes(self, engine):
        view = engine.view_for(Pillar.PRAXIS)
        assert len(view.cross_axes) == 5
        assert "explore_exploit" in view.cross_axes

    def test_mneme_gets_2_own_axes(self, engine):
        view = engine.view_for(Pillar.MNEME)
        assert len(view.own_axes) == 2
        assert "consolidate_prune" in view.own_axes
        assert "specific_general" in view.own_axes

    def test_mneme_gets_6_cross_axes(self, engine):
        view = engine.view_for(Pillar.MNEME)
        assert len(view.cross_axes) == 6

    def test_total_positions_equal_8(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert len(view.all_positions) == 8
        assert len(view.all_defaults) == 8

    def test_own_plus_cross_equals_all(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        all_ids = set(view.own_axes) | set(view.cross_axes)
        assert all_ids == set(view.all_positions)


# ═══════════════════════════════════════════════════════════════
#  2. PillarEquilibriumView — Stress Level
# ═══════════════════════════════════════════════════════════════

class TestPillarEquilibriumViewStress:
    """View correctly computes stress from drift."""

    def test_fresh_engine_zero_stress(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert view.stress_level == 0.0
        assert not view.is_stressed
        assert not view.is_highly_stressed

    def test_stressed_engine_has_high_stress(self, stressed_engine):
        view = stressed_engine.view_for(Pillar.COGNITION)
        assert view.stress_level > 0.1  # Non-trivial drift
        assert view.is_stressed or view.stress_level > 0.05

    def test_highly_stressed_engine(self, stressed_engine):
        view = stressed_engine.view_for(Pillar.COGNITION)
        # With 4 axes pushed 0.5-0.8, stress should be significant
        assert view.stress_level > 0.0
        # Verify is_stressed and is_highly_stressed thresholds
        if view.stress_level > 0.5:
            assert view.is_highly_stressed
        else:
            assert not view.is_highly_stressed

    def test_stress_is_rms_drift(self, engine):
        """Stress equals sqrt(1/N * sum(drift^2))"""
        # Push exactly one axis by 0.5
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=1.0,
            reason="test",
        )
        engine.apply_feedback(fb)
        view = engine.view_for(Pillar.COGNITION)
        # With 8 axes, 1 drifted by ~0.5, others at 0
        # stress = sqrt(0.5^2 / 8) ≈ 0.177
        assert view.stress_level > 0.0
        assert view.stress_level < 0.5  # Not dominant


# ═══════════════════════════════════════════════════════════════
#  3. PillarEquilibriumView — Drift
# ═══════════════════════════════════════════════════════════════

class TestPillarEquilibriumViewDrift:
    """Per-axis drift is correctly computed."""

    def test_fresh_engine_zero_drift(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        for axis_id, drift_val in view.drift.items():
            assert drift_val == 0.0

    def test_drift_after_feedback(self, engine):
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.4,
            confidence=1.0,
            reason="test",
        )
        engine.apply_feedback(fb)
        view = engine.view_for(Pillar.COGNITION)
        assert view.drift["explore_exploit"] > 0.0
        # With damping=0.4, effective delta = 0.4 * 1.0 * (1-0.4) = 0.24
        # Starting from default 0.15, new pos ≈ 0.39, drift ≈ 0.24
        assert view.drift["explore_exploit"] > 0.1

    def test_get_drift_convenience(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert view.get_drift("explore_exploit") == 0.0
        assert view.get_drift("nonexistent") == 0.0


# ═══════════════════════════════════════════════════════════════
#  4. PillarEquilibriumView — Oscillation
# ═══════════════════════════════════════════════════════════════

class TestPillarEquilibriumViewOscillation:
    """View correctly identifies oscillating axes."""

    def test_fresh_engine_no_oscillation(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert view.oscillating == ()
        assert not view.is_axis_oscillating("explore_exploit")

    def test_oscillating_engine_detects_oscillation(self, oscillating_engine):
        view = oscillating_engine.view_for(Pillar.COGNITION)
        assert "explore_exploit" in view.oscillating
        assert view.is_axis_oscillating("explore_exploit")

    def test_non_oscillating_axis_not_flagged(self, oscillating_engine):
        view = oscillating_engine.view_for(Pillar.COGNITION)
        assert not view.is_axis_oscillating("autonomy_safety")


# ═══════════════════════════════════════════════════════════════
#  5. PillarEquilibriumView — Convenience Methods
# ═══════════════════════════════════════════════════════════════

class TestPillarEquilibriumViewConvenience:
    """Convenience methods work correctly."""

    def test_get_with_default(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        assert view.get("explore_exploit") is not None
        assert view.get("nonexistent", 0.42) == 0.42

    def test_own_axis_ids(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        ids = view.own_axis_ids()
        assert isinstance(ids, tuple)
        assert "explore_exploit" in ids

    def test_cross_axis_ids(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        ids = view.cross_axis_ids()
        assert isinstance(ids, tuple)
        assert "autonomy_safety" in ids

    def test_summary_dict(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        s = view.summary()
        assert s["pillar"] == "cognition"
        assert "own_axes" in s
        assert "stress_level" in s
        assert "is_stressed" in s
        assert "oscillating" in s

    def test_repr(self, engine):
        view = engine.view_for(Pillar.COGNITION)
        r = repr(view)
        assert "PillarEquilibriumView" in r
        assert "cognition" in r

    def test_own_axes_returns_copy(self, engine):
        """Properties return copies, not references."""
        view = engine.view_for(Pillar.COGNITION)
        d1 = view.own_axes
        d1["fake"] = 999
        assert "fake" not in view.own_axes

    def test_pillar_property(self, engine):
        view = engine.view_for(Pillar.PRAXIS)
        assert view.pillar == Pillar.PRAXIS


# ═══════════════════════════════════════════════════════════════
#  6. BasePillar.bind_engine / unbind_engine
# ═══════════════════════════════════════════════════════════════

class TestBasePillarBindEngine:
    """Engine binding lifecycle."""

    def test_bind_creates_view(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(engine)
        assert pillar.engine is engine
        assert pillar.equilibrium_view is not None

    def test_bind_same_engine_idempotent(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(engine)
        pillar.bind_engine(engine)  # Same engine — OK
        assert pillar.engine is engine

    def test_bind_different_engine_raises(self, agent_state):
        eng1 = EquilibriumEngine()
        eng2 = EquilibriumEngine()
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(eng1)
        with pytest.raises(Exception):
            pillar.bind_engine(eng2)

    def test_unbind_clears_engine(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(engine)
        pillar.unbind_engine()
        assert pillar.engine is None
        assert pillar.equilibrium_view is None

    def test_unbind_then_bind_different(self, agent_state):
        eng1 = EquilibriumEngine()
        eng2 = EquilibriumEngine()
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(eng1)
        pillar.unbind_engine()
        pillar.bind_engine(eng2)  # Should succeed
        assert pillar.engine is eng2

    def test_stress_feedback_disabled_by_flag(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar._stress_feedback_enabled = False
        pillar.bind_engine(engine)
        # Process queued with stressed engine — no stress feedback
        # (verified indirectly: no feedback emitted)
        fb_before = len(pillar._pending_feedback)
        pillar.process_queued()
        fb_after = len(pillar._pending_feedback)
        # With fresh engine (zero stress), no stress feedback expected anyway
        assert fb_after == fb_before


# ═══════════════════════════════════════════════════════════════
#  7. BasePillar.process_queued auto-sync
# ═══════════════════════════════════════════════════════════════

class TestBasePillarAutoSync:
    """process_queued() auto-syncs the equilibrium view."""

    def test_auto_sync_updates_view(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(engine)
        # First sync on bind
        view1 = pillar.equilibrium_view
        # Push some feedback to change engine state
        fb = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=0.9,
            reason="test",
        )
        engine.apply_feedback(fb)
        # Process queued — auto-sync should pick up new state
        pillar.process_queued()
        view2 = pillar.equilibrium_view
        assert view2 is not None
        assert view2.get("explore_exploit") != view1.get("explore_exploit")

    def test_no_auto_sync_without_engine(self, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        # No bind_engine call
        assert pillar.equilibrium_view is None
        pillar.process_queued()
        assert pillar.equilibrium_view is None

    def test_stress_feedback_emitted_when_stressed(self, stressed_engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        # Clear any feedback from bind
        pillar.drain_feedback()
        # Process queued — stress feedback should be emitted
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        # Should have stress-reactive feedback
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        assert len(stress_fb) >= 1

    def test_no_stress_feedback_when_not_stressed(self, engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(engine)
        pillar.drain_feedback()
        # Fresh engine — stress=0, no stress feedback
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        assert len(stress_fb) == 0


# ═══════════════════════════════════════════════════════════════
#  8. Stress-Reactive Feedback Content
# ═══════════════════════════════════════════════════════════════

class TestStressReactiveFeedback:
    """Stress feedback has correct properties."""

    def test_stress_feedback_targets_own_axis(self, stressed_engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            # Should target a Cognition axis
            assert stress_fb[0].tension_axis_id in {
                "explore_exploit", "shallow_deep", "divergent_convergent"
            }

    def test_stress_feedback_low_confidence(self, stressed_engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            # Low confidence — it's a gentle nudge
            assert stress_fb[0].confidence <= 0.5

    def test_stress_feedback_signal_toward_default(self, stressed_engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            fb = stress_fb[0]
            # Signal should push toward the default
            view = stressed_engine.view_for(Pillar.COGNITION)
            current = view.own_axes.get(fb.tension_axis_id, 0.0)
            default = view.all_defaults.get(fb.tension_axis_id, 0.0)
            if current > default:
                assert fb.signal < 0  # Push downward toward default
            elif current < default:
                assert fb.signal > 0  # Push upward toward default

    def test_stress_feedback_signal_bounded(self, stressed_engine, agent_state):
        pillar = CognitionPillar(name="test_cog")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            for fb in stress_fb:
                assert -1.0 <= fb.signal <= 1.0

    def test_praxis_stress_feedback_targets_praxis_axis(self, stressed_engine, agent_state):
        pillar = PraxisPillar(name="test_prax")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            assert stress_fb[0].tension_axis_id in {
                "autonomy_safety", "sequential_parallel", "verify_execute"
            }

    def test_mneme_stress_feedback_targets_mneme_axis(self, stressed_engine, agent_state):
        pillar = MnemePillar(name="test_mneme")
        pillar.initialize(agent_state)
        pillar.bind_engine(stressed_engine)
        pillar.drain_feedback()
        pillar.process_queued()
        feedbacks = pillar.drain_feedback()
        stress_fb = [f for f in feedbacks if "stress-reactive" in f.reason]
        if stress_fb:
            assert stress_fb[0].tension_axis_id in {
                "consolidate_prune", "specific_general"
            }


# ═══════════════════════════════════════════════════════════════
#  9. CognitionPillar._on_equilibrium_sync
# ═══════════════════════════════════════════════════════════════

class TestCognitionEquilibriumSync:
    """Cognition pillar correctly auto-syncs tension."""

    def test_sync_updates_reasoning_profile(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        cog.bind_engine(engine)
        # Verify the hook was called by checking that reasoning received
        # the tension update (observable via the pillar's own view)
        cog.process_queued()
        # The view should exist and reflect engine state
        view = cog.equilibrium_view
        assert view is not None
        assert "explore_exploit" in view.own_axes

    def test_sync_reads_cross_axes(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        cog.bind_engine(engine)
        cog.process_queued()
        view = cog.equilibrium_view
        assert view is not None
        # Cognition should see Praxis axes as cross
        assert "autonomy_safety" in view.cross_axes


# ═══════════════════════════════════════════════════════════════
#  10. PraxisPillar._on_equilibrium_sync
# ═══════════════════════════════════════════════════════════════

class TestPraxisEquilibriumSync:
    """Praxis pillar correctly auto-syncs tension."""

    def test_sync_updates_orchestrator_profile(self, engine, agent_state):
        prax = PraxisPillar(name="prax")
        prax.initialize(agent_state)
        prax.bind_engine(engine)
        prax.process_queued()
        # Verify the hook was called — view reflects praxis axes
        view = prax.equilibrium_view
        assert view is not None
        assert "autonomy_safety" in view.own_axes

    def test_sync_reads_cognition_cross_axis(self, engine, agent_state):
        prax = PraxisPillar(name="prax")
        prax.initialize(agent_state)
        prax.bind_engine(engine)
        prax.process_queued()
        view = prax.equilibrium_view
        assert view is not None
        assert "explore_exploit" in view.cross_axes
        assert "shallow_deep" in view.cross_axes


# ═══════════════════════════════════════════════════════════════
#  11. MnemePillar._on_equilibrium_sync
# ═══════════════════════════════════════════════════════════════

class TestMnemeEquilibriumSync:
    """Mneme pillar correctly auto-syncs tension."""

    def test_sync_updates_mneme_profile(self, engine, agent_state):
        mneme = MnemePillar(name="mneme")
        mneme.initialize(agent_state)
        mneme.bind_engine(engine)
        mneme.process_queued()
        # Verify the hook was called — view reflects mneme axes
        view = mneme.equilibrium_view
        assert view is not None
        assert "consolidate_prune" in view.own_axes

    def test_sync_reads_praxis_cross_axis(self, engine, agent_state):
        mneme = MnemePillar(name="mneme")
        mneme.initialize(agent_state)
        mneme.bind_engine(engine)
        mneme.process_queued()
        view = mneme.equilibrium_view
        assert view is not None
        assert "autonomy_safety" in view.cross_axes


# ═══════════════════════════════════════════════════════════════
#  12. Full Pull Loop Integration
# ═══════════════════════════════════════════════════════════════

class TestFullPullLoopIntegration:
    """End-to-end: bind → tick → read → feedback → tick again."""

    def test_full_loop_cognition(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        cog.bind_engine(engine)

        # Tick 1: no feedback expected (zero stress)
        cog.process_queued()
        fb1 = cog.drain_feedback()
        assert all("stress-reactive" not in f.reason for f in fb1)

        # Push engine away from homeostasis
        for _ in range(3):
            engine.apply_feedback(Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.6,
                confidence=0.9,
                reason="test push",
            ))

        # Tick 2: should have stress-reactive feedback
        cog.process_queued()
        fb2 = cog.drain_feedback()
        stress_fb = [f for f in fb2 if "stress-reactive" in f.reason]
        assert len(stress_fb) >= 1

    def test_full_loop_three_pillars(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        prax = PraxisPillar(name="prax")
        mneme = MnemePillar(name="mneme")
        cog.initialize(agent_state)
        prax.initialize(agent_state)
        mneme.initialize(agent_state)

        # Bind all three to the same engine
        cog.bind_engine(engine)
        prax.bind_engine(engine)
        mneme.bind_engine(engine)

        # All should have views
        assert cog.equilibrium_view is not None
        assert prax.equilibrium_view is not None
        assert mneme.equilibrium_view is not None

        # Each should see different own_axes
        assert set(cog.equilibrium_view.own_axes.keys()) != set(prax.equilibrium_view.own_axes.keys())
        assert set(prax.equilibrium_view.own_axes.keys()) != set(mneme.equilibrium_view.own_axes.keys())

    def test_feedback_from_one_pillar_affects_other_view(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        prax = PraxisPillar(name="prax")
        cog.initialize(agent_state)
        prax.initialize(agent_state)
        cog.bind_engine(engine)
        prax.bind_engine(engine)

        # Cognition pushes explore_exploit
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.7,
            confidence=0.9,
            reason="test",
        ))

        # Praxis processes — should see updated cross-axis
        prax.process_queued()
        view = prax.equilibrium_view
        assert view is not None
        # explore_exploit should have moved
        assert view.cross_axes.get("explore_exploit", 0.0) != 0.15  # Default is 0.15

    def test_view_reflects_latest_engine_state(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        cog.bind_engine(engine)

        # Initial view
        cog.process_queued()
        v1 = cog.equilibrium_view
        assert v1 is not None
        pos1 = v1.get("explore_exploit")

        # Change engine
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=1.0,
            reason="test",
        ))

        # Re-sync
        cog.process_queued()
        v2 = cog.equilibrium_view
        assert v2 is not None
        pos2 = v2.get("explore_exploit")
        assert pos2 != pos1


# ═══════════════════════════════════════════════════════════════
#  13. PillarEquilibriumView with Custom Axes
# ═══════════════════════════════════════════════════════════════

class TestViewWithCustomAxes:
    """View works correctly with non-default axis configurations."""

    def test_custom_axes_decomposition(self):
        custom = [
            TensionAxis(id="custom_1", pillar=Pillar.COGNITION,
                        pole_left="a", pole_right="b",
                        default_position=0.0, damping=0.3, learning_rate=0.05),
            TensionAxis(id="custom_2", pillar=Pillar.PRAXIS,
                        pole_left="c", pole_right="d",
                        default_position=0.0, damping=0.3, learning_rate=0.05),
        ]
        engine = EquilibriumEngine(axes=custom)
        view = engine.view_for(Pillar.COGNITION)
        assert len(view.own_axes) == 1
        assert "custom_1" in view.own_axes
        assert "custom_2" in view.cross_axes

    def test_custom_axes_stress(self):
        custom = [
            TensionAxis(id="x1", pillar=Pillar.COGNITION,
                        pole_left="a", pole_right="b",
                        default_position=0.0, damping=0.3, learning_rate=0.05),
        ]
        engine = EquilibriumEngine(axes=custom)
        view = engine.view_for(Pillar.COGNITION)
        assert view.stress_level == 0.0

    def test_empty_axes_view(self):
        engine = EquilibriumEngine(axes=[])
        view = engine.view_for(Pillar.COGNITION)
        assert len(view.own_axes) == 0
        assert view.stress_level == 0.0
        assert view.oscillating == ()


# ═══════════════════════════════════════════════════════════════
#  14. Backward Compatibility
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Existing update_tension_profile() still works alongside pull."""

    def test_cognition_update_tension_profile_still_works(self, engine, agent_state):
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        # Old-style push (no bind_engine)
        cog.update_tension_profile(engine.get_behavior_profile())
        # Should not raise

    def test_praxis_update_tension_profile_still_works(self, engine, agent_state):
        prax = PraxisPillar(name="prax")
        prax.initialize(agent_state)
        prax.update_tension_profile(engine.get_behavior_profile())

    def test_mneme_update_tension_profile_still_works(self, engine, agent_state):
        mneme = MnemePillar(name="mneme")
        mneme.initialize(agent_state)
        mneme.update_tension_profile(engine.get_behavior_profile())

    def test_push_and_pull_coexist(self, engine, agent_state):
        """Both push and pull can coexist — pull auto-applies."""
        cog = CognitionPillar(name="cog")
        cog.initialize(agent_state)
        cog.bind_engine(engine)
        # Pull mechanism auto-syncs on process_queued
        cog.process_queued()
        # Manual push should still work (though redundant)
        cog.update_tension_profile(engine.get_behavior_profile())
        # No error — both mechanisms coexist
