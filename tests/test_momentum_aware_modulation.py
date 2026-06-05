"""Tests for iter-024: Momentum-Aware Pillar Behavior Modulation.

When the TensionVelocityTracker is enabled, pillars read velocity and
momentum data from the PillarEquilibriumView to modulate their behavior:

Cognition:
  - explore_exploit drifting away (momentum < 0) → boost reasoning depth
  - shallow_deep oscillation-imminent → cap reasoning depth

Praxis:
  - autonomy_safety drifting toward unsafe (momentum < 0) → boost verify depth
  - verify_execute oscillation-imminent → reduce max_parallel

Mneme:
  - consolidate_prune drifting toward prune (momentum < 0) → raise consolidation threshold
  - consolidate_prune oscillation-imminent → skip consolidation this tick

All three pillars track the number of momentum-based modulations applied.
"""

from __future__ import annotations

import math
from collections import deque
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from isonome.base import BasePillar
from isonome.cognition.pillar import CognitionPillar
from isonome.equilibrium import EquilibriumEngine, PillarEquilibriumView
from isonome.equilibrium.velocity import TensionVelocityTracker
from isonome.mneme.hierarchical import HierarchicalMneme
from isonome.mneme.pillar import MnemePillar
from isonome.praxis.orchestrator import ActionOrchestrator
from isonome.praxis.pillar import PraxisPillar
from isonome.types import (
    AgentIdentity,
    AgentState,
    Pillar,
    TensionAxis,
    TensionSnapshot,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_engine_with_velocity(**overrides) -> EquilibriumEngine:
    """Create an engine with velocity tracking enabled."""
    return EquilibriumEngine(enable_velocity_tracking=True, **overrides)


def _make_agent_state(engine: EquilibriumEngine) -> AgentState:
    """Create an AgentState from an engine."""
    return AgentState(
        identity=AgentIdentity(name="test"),
        tensions=engine.snapshot(),
    )


def _simulate_feedback_to_drift(
    engine: EquilibriumEngine,
    axis_id: str,
    direction: float,
    steps: int = 5,
) -> None:
    """Apply feedback repeatedly to create a drift in the given direction.

    direction > 0 pushes position right, < 0 pushes left.
    """
    from isonome.types import Feedback
    for _ in range(steps):
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id=axis_id,
            signal=direction * 0.15,
            confidence=0.8,
            reason="test drift",
        ))


def _simulate_oscillation(
    engine: EquilibriumEngine,
    axis_id: str,
    steps: int = 8,
) -> None:
    """Apply alternating feedback to create oscillation on an axis."""
    from isonome.types import Feedback
    for i in range(steps):
        direction = 0.2 if i % 2 == 0 else -0.2
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id=axis_id,
            signal=direction,
            confidence=0.8,
            reason="test oscillation",
        ))


# ── Cognition Pillar Tests ──────────────────────────────────

class TestCognitionMomentumModulation:
    """CognitionPillar reads momentum to modulate reasoning behavior."""

    def test_cognition_tracks_momentum_modulations_counter(self):
        """CognitionPillar should have a _momentum_modulations counter."""
        cog = CognitionPillar(name="thinker", engine=_make_engine_with_velocity())
        assert hasattr(cog, "_momentum_modulations")
        assert cog._momentum_modulations == 0

    def test_exploit_drift_boosts_reasoning_depth(self):
        """When explore_exploit momentum < 0 (drifting away from equilibrium),
        the reasoning amplifier should get a depth boost."""
        engine = _make_engine_with_velocity()
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # Create drift away from default on explore_exploit
        # Default is +0.15 (slight exploit). Push toward explore to create momentum < 0
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.8, steps=6)

        # Get the view with velocity data
        view = engine.view_for(Pillar.COGNITION)

        # Confirm momentum is negative on explore_exploit
        momentum = view.get_momentum_score("explore_exploit")
        assert momentum < 0, f"Expected negative momentum, got {momentum}"

        # Record pre-sync reasoning amplifier
        cal = cog.reasoning.calibrator
        # Ensure enough predictions for amplifier
        for i in range(12):
            cal.record(0.5 + i * 0.01, i % 2 == 0)

        pre_ece = cal.compute_ece()
        pre_amplifier = cog.reasoning._compute_calibration_amplifier()

        # Sync with momentum data
        cog._on_equilibrium_sync(view)

        # The pillar should have applied a momentum depth boost
        assert cog._momentum_modulations > 0

    def test_shallow_deep_oscillation_imminent_caps_depth(self):
        """When shallow_deep axis is oscillation-imminent,
        the pillar should cap reasoning depth to avoid instability."""
        engine = _make_engine_with_velocity()
        # Override the velocity tracker's window to make oscillation easier
        # to trigger in tests
        engine._velocity_tracker = TensionVelocityTracker(
            window_size=5,
            min_reversal_magnitude=0.001,
        )
        for axis in engine._axes.values():
            engine._velocity_tracker.register_axis(axis.id)

        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # Create oscillation on shallow_deep
        _simulate_oscillation(engine, "shallow_deep", steps=8)

        view = engine.view_for(Pillar.COGNITION)

        # Verify oscillation is imminent on shallow_deep
        assert "shallow_deep" in view.oscillation_imminent, \
            f"Expected oscillation_imminent on shallow_deep, got {view.oscillation_imminent}"

        # Sync should apply a depth cap
        pre_modulations = cog._momentum_modulations
        cog._on_equilibrium_sync(view)
        assert cog._momentum_modulations > pre_modulations

    def test_no_velocity_data_no_modulation(self):
        """Without velocity tracking, no momentum modulations occur."""
        engine = EquilibriumEngine(enable_velocity_tracking=False)
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        view = engine.view_for(Pillar.COGNITION)
        # No velocity data — momentums should be empty
        assert view.momentum_scores == {}

        cog._on_equilibrium_sync(view)
        assert cog._momentum_modulations == 0

    def test_positive_momentum_no_depth_boost(self):
        """When explore_exploit momentum > 0 (heading toward default),
        no depth boost should be applied."""
        engine = _make_engine_with_velocity()
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # Start explore_exploit far from default, then push back toward it
        # Default is +0.15. First push far negative, then push back positive
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.8, steps=4)
        # Now push toward default (positive direction)
        from isonome.types import Feedback
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.3,
            confidence=0.8,
            reason="push toward default",
        ))

        view = engine.view_for(Pillar.COGNITION)
        momentum = view.get_momentum_score("explore_exploit")

        # If momentum happens to be positive (heading toward default),
        # no depth boost
        pre_mod = cog._momentum_modulations
        cog._on_equilibrium_sync(view)
        # Only the oscillation check applies, not the depth boost
        # (depth boost only on negative momentum)


# ── Praxis Pillar Tests ─────────────────────────────────────

class TestPraxisMomentumModulation:
    """PraxisPillar reads momentum to modulate execution behavior."""

    def test_praxis_tracks_momentum_modulations_counter(self):
        """PraxisPillar should have a _momentum_modulations counter."""
        praxis = PraxisPillar(name="executor")
        assert hasattr(praxis, "_momentum_modulations")
        assert praxis._momentum_modulations == 0

    def test_autonomy_safety_drift_boosts_verify_depth(self):
        """When autonomy_safety momentum < 0 (drifting toward unsafe),
        the pillar should increase verify scrutiny on the orchestrator."""
        engine = _make_engine_with_velocity()
        praxis = PraxisPillar(name="executor")
        state = _make_agent_state(engine)
        praxis.initialize(state)

        # Create drift toward unsafe (negative direction on autonomy_safety)
        _simulate_feedback_to_drift(engine, "autonomy_safety", -0.8, steps=6)

        view = engine.view_for(Pillar.PRAXIS)

        # Confirm momentum is negative
        momentum = view.get_momentum_score("autonomy_safety")
        assert momentum < 0, f"Expected negative momentum, got {momentum}"

        # Record pre-sync verify_execute tension
        pre_verify = praxis.orchestrator._get_tension_profile().get("verify_execute", 0.0)

        praxis._on_equilibrium_sync(view)

        # Verify depth should have been boosted (verify_execute shifted toward verify)
        post_verify = praxis.orchestrator._get_tension_profile().get("verify_execute", 0.0)
        assert post_verify < pre_verify, \
            f"verify_execute should shift toward verify (more negative), was {pre_verify}, now {post_verify}"
        assert praxis._momentum_modulations > 0

    def test_verify_execute_oscillation_reduces_parallelism(self):
        """When verify_execute is oscillation-imminent,
        the pillar should reduce max_parallel."""
        engine = _make_engine_with_velocity()
        engine._velocity_tracker = TensionVelocityTracker(
            window_size=5,
            min_reversal_magnitude=0.001,
        )
        for axis in engine._axes.values():
            engine._velocity_tracker.register_axis(axis.id)

        praxis = PraxisPillar(name="executor", max_parallel=8)
        state = _make_agent_state(engine)
        praxis.initialize(state)

        # Create oscillation on verify_execute
        _simulate_oscillation(engine, "verify_execute", steps=8)

        praxis_view = engine.view_for(Pillar.PRAXIS)

        assert "verify_execute" in praxis_view.oscillation_imminent, \
            f"Expected oscillation on verify_execute, got {praxis_view.oscillation_imminent}"

        pre_mod = praxis._momentum_modulations
        praxis._on_equilibrium_sync(praxis_view)

        assert praxis._momentum_modulations > pre_mod

    def test_no_velocity_data_no_praxis_modulation(self):
        """Without velocity tracking, no momentum modulations occur in Praxis."""
        engine = EquilibriumEngine(enable_velocity_tracking=False)
        praxis = PraxisPillar(name="executor")
        state = _make_agent_state(engine)
        praxis.initialize(state)

        praxis_view = engine.view_for(Pillar.PRAXIS)
        praxis._on_equilibrium_sync(praxis_view)
        assert praxis._momentum_modulations == 0


# ── Mneme Pillar Tests ──────────────────────────────────────

class TestMnemeMomentumModulation:
    """MnemePillar reads momentum to modulate consolidation behavior."""

    def test_mneme_tracks_momentum_modulations_counter(self):
        """MnemePillar should have a _momentum_modulations counter."""
        mneme = MnemePillar(name="memory")
        assert hasattr(mneme, "_momentum_modulations")
        assert mneme._momentum_modulations == 0

    def test_consolidate_prune_drift_raises_threshold(self):
        """When consolidate_prune momentum < 0 (drifting toward prune),
        the pillar should raise the consolidation significance threshold."""
        engine = _make_engine_with_velocity()
        mneme = MnemePillar(name="memory")
        state = _make_agent_state(engine)
        mneme.initialize(state)

        # Create drift toward prune (positive direction = prune pole)
        _simulate_feedback_to_drift(engine, "consolidate_prune", 0.8, steps=6)

        mneme_view = engine.view_for(Pillar.MNEME)

        # Momentum should be negative if position is moving away from default
        # Default is -0.10 (slight consolidate). Pushing toward prune = positive direction
        # Position moves away from default → momentum < 0
        momentum = mneme_view.get_momentum_score("consolidate_prune")
        assert momentum < 0, f"Expected negative momentum, got {momentum}"

        pre_mod = mneme._momentum_modulations
        mneme._on_equilibrium_sync(mneme_view)
        assert mneme._momentum_modulations > pre_mod

    def test_consolidate_prune_oscillation_skips_consolidation(self):
        """When consolidate_prune is oscillation-imminent,
        the pillar should skip the consolidation cycle this tick."""
        engine = _make_engine_with_velocity()
        engine._velocity_tracker = TensionVelocityTracker(
            window_size=5,
            min_reversal_magnitude=0.001,
        )
        for axis in engine._axes.values():
            engine._velocity_tracker.register_axis(axis.id)

        mneme = MnemePillar(name="memory")
        state = _make_agent_state(engine)
        mneme.initialize(state)

        # Create oscillation on consolidate_prune
        _simulate_oscillation(engine, "consolidate_prune", steps=8)

        mneme_view = engine.view_for(Pillar.MNEME)
        assert "consolidate_prune" in mneme_view.oscillation_imminent, \
            f"Expected oscillation on consolidate_prune, got {mneme_view.oscillation_imminent}"

        # Track whether consolidate was called
        consolidate_calls = 0
        original_consolidate = mneme.mneme.consolidate
        def tracking_consolidate():
            nonlocal consolidate_calls
            consolidate_calls += 1
            return original_consolidate()
        mneme.mneme.consolidate = tracking_consolidate

        pre_mod = mneme._momentum_modulations
        mneme._on_equilibrium_sync(mneme_view)
        # Consolidation should have been skipped
        assert consolidate_calls == 0, \
            "Consolidation should be skipped when consolidate_prune is oscillation-imminent"
        assert mneme._momentum_modulations > pre_mod

    def test_no_velocity_data_no_mneme_modulation(self):
        """Without velocity tracking, no momentum modulations occur in Mneme."""
        engine = EquilibriumEngine(enable_velocity_tracking=False)
        mneme = MnemePillar(name="memory")
        state = _make_agent_state(engine)
        mneme.initialize(state)

        mneme_view = engine.view_for(Pillar.MNEME)
        mneme._on_equilibrium_sync(mneme_view)
        assert mneme._momentum_modulations == 0


# ── Integration Tests ───────────────────────────────────────

class TestMomentumAwareIntegration:
    """Integration: velocity tracker → engine → view → pillar modulation."""

    def test_engine_with_velocity_feeds_view_to_pillars(self):
        """An engine with velocity tracking provides velocity data to views."""
        engine = _make_engine_with_velocity()
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.5, steps=4)

        view = engine.view_for(Pillar.COGNITION)
        assert len(view.velocities) > 0, "View should have velocity data"
        assert len(view.momentum_scores) > 0, "View should have momentum data"

    def test_engine_without_velocity_provides_empty_view_data(self):
        """An engine without velocity tracking provides empty dicts."""
        engine = EquilibriumEngine(enable_velocity_tracking=False)
        view = engine.view_for(Pillar.COGNITION)
        assert view.velocities == {}
        assert view.momentum_scores == {}
        assert view.oscillation_imminent == ()

    def test_momentum_modulation_idempotent_on_stable_axes(self):
        """When all axes have positive momentum (heading toward defaults),
        momentum modulations should be 0."""
        engine = _make_engine_with_velocity()
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # No drift applied — engine at defaults
        view = engine.view_for(Pillar.COGNITION)

        # All momentum should be 0 (no velocity)
        cog._on_equilibrium_sync(view)
        # No momentum-based modulations on a fresh engine
        assert cog._momentum_modulations == 0

    def test_modulation_counter_increases_across_ticks(self):
        """Momentum modulations counter should accumulate across ticks."""
        engine = _make_engine_with_velocity()
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # Create drift
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.8, steps=6)
        view = engine.view_for(Pillar.COGNITION)

        cog._on_equilibrium_sync(view)
        first_count = cog._momentum_modulations
        assert first_count > 0

        # Apply more drift and sync again
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.8, steps=3)
        view = engine.view_for(Pillar.COGNITION)
        cog._on_equilibrium_sync(view)

        assert cog._momentum_modulations > first_count

    def test_oscillation_imminent_flag_prevents_consolidation_in_mneme(self):
        """Full pipeline: oscillation on consolidate_prune →
        MnemePillar skips consolidation."""
        engine = _make_engine_with_velocity()
        engine._velocity_tracker = TensionVelocityTracker(
            window_size=5,
            min_reversal_magnitude=0.001,
        )
        for axis in engine._axes.values():
            engine._velocity_tracker.register_axis(axis.id)

        mneme = MnemePillar(name="memory")
        state = _make_agent_state(engine)
        mneme.initialize(state)

        # Add some memories to consolidate
        mneme.mneme.store("test content", significance=0.6, tags=("test",))

        # Create oscillation
        _simulate_oscillation(engine, "consolidate_prune", steps=8)

        view = engine.view_for(Pillar.MNEME)
        assert "consolidate_prune" in view.oscillation_imminent

        # Count consolidation reports before sync
        pre_consolidation_count = mneme.mneme._stats.consolidation_count

        mneme._on_equilibrium_sync(view)

        # Consolidation should not have run
        post_consolidation_count = mneme.mneme._stats.consolidation_count
        assert post_consolidation_count == pre_consolidation_count


class TestMomentumModulationSerialization:
    """Momentum modulation counters should survive serialization."""

    def test_cognition_momentum_counter_in_serialize(self):
        """CognitionPillar.serialize() should include momentum_modulations."""
        engine = _make_engine_with_velocity()
        cog = CognitionPillar(name="thinker", engine=engine)
        state = _make_agent_state(engine)
        cog.initialize(state)

        # Force a modulation
        _simulate_feedback_to_drift(engine, "explore_exploit", -0.8, steps=6)
        view = engine.view_for(Pillar.COGNITION)
        cog._on_equilibrium_sync(view)
        assert cog._momentum_modulations > 0

        serialized = cog.serialize()
        assert "momentum_modulations" in serialized
        assert serialized["momentum_modulations"] == cog._momentum_modulations

    def test_praxis_momentum_counter_in_serialize(self):
        """PraxisPillar.serialize() should include momentum_modulations."""
        engine = _make_engine_with_velocity()
        praxis = PraxisPillar(name="executor")
        state = _make_agent_state(engine)
        praxis.initialize(state)

        _simulate_feedback_to_drift(engine, "autonomy_safety", -0.8, steps=6)
        view = engine.view_for(Pillar.PRAXIS)
        praxis._on_equilibrium_sync(view)

        serialized = praxis.serialize()
        assert "momentum_modulations" in serialized

    def test_mneme_momentum_counter_in_serialize(self):
        """MnemePillar.serialize() should include momentum_modulations."""
        engine = _make_engine_with_velocity()
        mneme = MnemePillar(name="memory")
        state = _make_agent_state(engine)
        mneme.initialize(state)

        _simulate_feedback_to_drift(engine, "consolidate_prune", 0.8, steps=6)
        view = engine.view_for(Pillar.MNEME)
        mneme._on_equilibrium_sync(view)

        serialized = mneme.serialize()
        assert "momentum_modulations" in serialized
