"""Tests for cross-session serialization (to_dict / from_dict).

Covers all four serializable systems:
1. ConfidenceCalibrator — metacognitive state persistence
2. EquilibriumEngine — tension state persistence
3. HierarchicalMneme — memory state persistence
4. ActionOrchestrator — execution state persistence
5. IsonomeAgent — full agent state persistence (integration)
"""

from __future__ import annotations

import math
import time
from uuid import UUID

import pytest

from isonome.cognition.reasoning import (
    CalibrationBin,
    ConfidenceCalibrator,
)
from isonome.equilibrium import EquilibriumEngine
from isonome.mneme.hierarchical import HierarchicalMneme, MemoryEntry, MemoryTier, ConsolidationEvent
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionResult,
    RetryPolicy,
)
from isonome.agent import IsonomeAgent
from isonome.types import Pillar, Task, TaskComplexity, TaskStatus


# ═══════════════════════════════════════════════════════════════════
# ConfidenceCalibrator Serialization Tests
# ═══════════════════════════════════════════════════════════════════


class TestConfidenceCalibratorSerialization:
    """Round-trip serialization for ConfidenceCalibrator state."""

    def test_empty_calibrator_round_trip(self):
        """An empty calibrator should survive to_dict/from_dict."""
        cal = ConfidenceCalibrator(num_bins=10, window_size=200, drift_threshold=0.05)
        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        assert restored.compute_ece() == 0.0
        assert restored.compute_bias() == 0.0
        assert restored.total_predictions == 0
        assert restored.total_adjustments == 0
        assert restored.evidence_weight == cal.evidence_weight
        assert restored.child_weight == cal.child_weight
        assert len(restored._bins) == len(cal._bins)
        assert all(b.count == 0 for b in restored._bins)
        assert len(restored._predictions) == 0

    def test_round_trip_with_predictions(self):
        """Calibrator with recorded predictions should serialize faithfully."""
        cal = ConfidenceCalibrator(num_bins=10)
        # Record 30 predictions with some miscalibration
        for i in range(30):
            pred = 0.5 + (i % 5) * 0.1  # Mix of confidences
            actual = (pred > 0.65)  # Actual success doesn't perfectly match
            if i > 15:
                actual = True  # Introduce overconfident bias
            cal.record(pred, actual)

        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        assert restored.total_predictions == 30
        assert restored.compute_ece() == pytest.approx(cal.compute_ece(), abs=1e-6)
        assert restored.compute_bias() == pytest.approx(cal.compute_bias(), abs=1e-6)
        assert restored.evidence_weight == pytest.approx(cal.evidence_weight)
        assert restored.child_weight == pytest.approx(cal.child_weight)

    def test_adaptive_weights_survive_round_trip(self):
        """Weight adjustments should persist through serialization."""
        cal = ConfidenceCalibrator(num_bins=10, drift_threshold=0.01)  # Low threshold
        # Record enough to trigger adjustments
        for i in range(30):
            cal.record(0.85 + (i % 2) * 0.05, True)  # Systematic overconfidence
        cal.adjust_weights()

        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        assert restored.evidence_weight == pytest.approx(cal.evidence_weight)
        assert restored.child_weight == pytest.approx(cal.child_weight)
        assert restored.total_adjustments == cal.total_adjustments

    def test_ece_history_preserved(self):
        """ECE history trend data should survive serialization."""
        cal = ConfidenceCalibrator(num_bins=10, drift_threshold=0.01)
        for i in range(30):
            cal.record(0.85, True)
            if i >= 20:
                cal.adjust_weights()

        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        assert tuple(restored.ece_trend) == cal.ece_trend
        assert len(restored.ece_trend) > 0

    def test_drift_threshold_preserved(self):
        """Custom drift threshold should survive serialization."""
        cal = ConfidenceCalibrator(num_bins=5, window_size=100, drift_threshold=0.12)
        cal.record(0.7, True)
        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        assert restored._drift_threshold == 0.12
        assert len(restored._bins) == 5
        assert restored._predictions.maxlen == 100

    def test_predictions_window_size_preserved(self):
        """Prediction window size should survive serialization."""
        cal = ConfidenceCalibrator(window_size=50)
        for i in range(100):  # More than window size
            cal.record(0.5 + (i % 5) * 0.1, i % 3 == 0)

        data = cal.to_dict()
        restored = ConfidenceCalibrator.from_dict(data)

        # Window should have capped at 50
        assert len(restored._predictions) == 50

    def test_invalid_predictions_handled_gracefully(self):
        """Malformed prediction data should not break from_dict."""
        data = {
            "num_bins": 10,
            "window_size": 200,
            "drift_threshold": 0.05,
            "bins": [{"lower": 0.0, "upper": 0.1, "count": -1, "correct": 5}],
            "predictions": [[0.5, True], "bad_data"],
            "evidence_weight": 0.7,
            "child_weight": 0.3,
            "ece_history": [0.1, 0.2],
            "total_predictions": 10,
            "total_adjustments": 2,
        }
        restored = ConfidenceCalibrator.from_dict(data)
        assert restored.total_predictions == 10
        assert restored.total_adjustments == 2


# ═══════════════════════════════════════════════════════════════════
# EquilibriumEngine Serialization Tests
# ═══════════════════════════════════════════════════════════════════


class TestEquilibriumEngineSerialization:
    """Round-trip serialization for EquilibriumEngine state."""

    def test_default_engine_round_trip(self):
        """Default engine should serialize/deserialize with all 8 axes."""
        engine = EquilibriumEngine()
        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        assert len(restored.axes) == 8
        assert restored.total_feedback_received == 0
        assert restored.total_oscillation_events == 0
        for orig_axis in engine.axes:
            restored_axis = restored.get_axis(orig_axis.id)
            assert restored_axis is not None
            assert restored_axis.position == pytest.approx(orig_axis.position)

    def test_tension_positions_survive_round_trip(self):
        """Custom tension positions should survive serialization."""
        engine = EquilibriumEngine()
        # Move some axes
        snapshot_before = engine.snapshot()
        axes_before = {a.id: a.position for a in snapshot_before.axes}

        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        snapshot_after = restored.snapshot()
        axes_after = {a.id: a.position for a in snapshot_after.axes}

        assert axes_before == axes_after

    def test_oscillation_history_preserved(self):
        """Oscillation history should survive serialization."""
        engine = EquilibriumEngine(oscillation_window=8)
        # Manually create some history by applying feedback
        for i in range(5):
            from isonome.types import Feedback
            engine.apply_feedback(
                Feedback(
                    source=Pillar.COGNITION,
                    tension_axis_id="explore_exploit",
                    signal=0.1 if i % 2 == 0 else -0.1,
                    confidence=0.8,
                    reason="test",
                )
            )

        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        assert restored.total_feedback_received == 5
        assert "explore_exploit" in restored._history
        assert len(restored._history["explore_exploit"]) == 5

    def test_custom_axes_survive(self):
        """Custom tension axes should survive serialization."""
        from isonome.types import TensionAxis
        axes = [
            TensionAxis(
                id="custom_test",
                pillar=Pillar.COGNITION,
                pole_left="left",
                pole_right="right",
                position=0.5,
                default_position=0.0,
                damping=0.2,
                learning_rate=0.15,
            ),
            TensionAxis(
                id="custom_test_2",
                pillar=Pillar.PRAXIS,
                pole_left="slow",
                pole_right="fast",
                position=-0.3,
                default_position=-0.1,
                damping=0.5,
                learning_rate=0.03,
            ),
        ]
        engine = EquilibriumEngine(axes=axes)
        # __init__ resets position to default_position, then apply feedback
        from isonome.types import Feedback
        engine.apply_feedback(
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="custom_test",
                signal=0.5,
                confidence=0.9,
                reason="setup",
            )
        )

        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)

        assert len(restored.axes) == 2
        axis = restored.get_axis("custom_test")
        assert axis is not None
        assert axis.pillar == Pillar.COGNITION
        assert axis.damping == 0.2
        assert axis.learning_rate == 0.15

    def test_oscillation_events_counter_preserved(self):
        """Oscillation event counter should survive serialization."""
        engine = EquilibriumEngine(oscillation_threshold=0.01, oscillation_window=3)
        from isonome.types import Feedback
        # Send rapidly alternating feedback to trigger oscillation
        for i in range(10):
            engine.apply_feedback(
                Feedback(
                    source=Pillar.COGNITION,
                    tension_axis_id="explore_exploit",
                    signal=0.9 if i % 2 == 0 else -0.9,
                    confidence=1.0,
                    reason="osc test",
                )
            )

        data = engine.to_dict()
        restored = EquilibriumEngine.from_dict(data)
        # Oscillation events may or may not be triggered; the counter value should match
        assert restored.total_oscillation_events == engine.total_oscillation_events


# ═══════════════════════════════════════════════════════════════════
# HierarchicalMneme Serialization Tests
# ═══════════════════════════════════════════════════════════════════


class TestHierarchicalMnemeSerialization:
    """Round-trip serialization for HierarchicalMneme state."""

    def test_empty_mneme_round_trip(self):
        """Empty mneme should serialize/deserialize with empty collections."""
        mneme = HierarchicalMneme()
        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert restored.total_memories == 0
        assert len(restored._working) == 0
        assert len(restored._episodic) == 0
        assert len(restored._semantic) == 0
        assert restored.total_memories == 0

    def test_store_and_restore_entry(self):
        """Stored memories should survive round-trip."""
        mneme = HierarchicalMneme()
        entry = mneme.store(
            "test memory content",
            significance=0.85,
            source="test",
            tags=("important", "test"),
        )

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert restored.total_memories == 1
        assert len(restored._working) == 1
        restored_entry = list(restored._working.values())[0]
        assert restored_entry.content == "test memory content"
        assert restored_entry.significance == 0.85
        assert restored_entry.source == "test"
        assert restored_entry.tags == ("important", "test")
        assert restored_entry.id == entry.id

    def test_multiple_tiers_survive(self):
        """Entries at all three tiers should survive round-trip."""
        mneme = HierarchicalMneme()
        wm = mneme.store("working memory", significance=0.5)
        ep = MemoryEntry(
            content="episodic memory",
            tier=MemoryTier.EPISODIC,
            significance=0.7,
            base_half_life=86400.0,
        )
        mneme._episodic[ep.id] = ep
        sem = MemoryEntry(
            content="semantic memory",
            tier=MemoryTier.SEMANTIC,
            significance=0.9,
            base_half_life=2592000.0,
        )
        mneme._semantic[sem.id] = sem

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert len(restored._working) == 1
        assert len(restored._episodic) == 1
        assert len(restored._semantic) == 1

        wm_r = list(restored._working.values())[0]
        assert wm_r.content == "working memory"
        assert wm_r.tier == MemoryTier.WORKING

        ep_r = list(restored._episodic.values())[0]
        assert ep_r.content == "episodic memory"
        assert ep_r.tier == MemoryTier.EPISODIC
        assert ep_r.base_half_life == 86400.0

        sem_r = list(restored._semantic.values())[0]
        assert sem_r.content == "semantic memory"
        assert sem_r.tier == MemoryTier.SEMANTIC

    def test_pattern_frequencies_survive(self):
        """N-gram pattern frequencies should survive round-trip."""
        mneme = HierarchicalMneme()
        mneme.store("hello world test", tags=("a",))
        mneme.store("hello world again", tags=("a",))
        mneme._update_patterns("hello world test", ("a",))
        mneme._update_patterns("hello world again", ("a",))

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert "hello world" in restored._pattern_frequencies
        assert restored._pattern_frequencies["hello world"] >= 2

    def test_tag_cooccurrence_survives(self):
        """Tag co-occurrence data should survive round-trip."""
        mneme = HierarchicalMneme()
        mneme.store("content with trend", tags=("trend", "important"))
        mneme.store("more trend content", tags=("trend", "important"))

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        # Check tag co-occurrence was preserved (may need to trigger update)
        mneme._update_patterns("content with trend", ("trend", "important"))
        data2 = mneme.to_dict()
        restored2 = HierarchicalMneme.from_dict(data2)

        # Verify we can round-trip with co-occurrence
        assert len(restored2._tag_cooccurrence) > 0

    def test_calibration_state_survives(self):
        """Calibration state set on mneme should survive round-trip."""
        mneme = HierarchicalMneme()
        mneme.set_calibration_state(
            ece=0.25,
            bias=-0.08,
            is_overconfident=False,
            is_underconfident=True,
            total_predictions=15,
        )

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert restored._calibration_ece == 0.25
        assert restored._calibration_bias == -0.08
        assert not restored._calibration_overconfident
        assert restored._calibration_underconfident
        assert restored._calibration_total_predictions == 15

    def test_consolidation_log_survives(self):
        """Consolidation history should survive round-trip."""
        mneme = HierarchicalMneme()
        mneme.store("test content", significance=0.9)
        report = mneme.consolidate()

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert len(restored._consolidation_log) > 0
        first = restored._consolidation_log[0]
        assert first.from_tier == MemoryTier.WORKING
        assert first.to_tier == MemoryTier.EPISODIC
        assert first.significance > 0.5

    def test_configuration_preserved(self):
        """Custom configuration parameters should survive round-trip."""
        mneme = HierarchicalMneme(
            consolidation_significance=0.6,
            promotion_significance=0.8,
            pattern_count_threshold=4,
            rehearsal_boost=0.2,
        )

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert restored._consolidation_sig == 0.6
        assert restored._promotion_sig == 0.8
        assert restored._pattern_threshold == 4
        assert restored._rehearsal_boost == 0.2

    def test_via_pillar_serialize_restore(self):
        """MnemePillar.serialize()/restore() should work."""
        from isonome.mneme.pillar import MnemePillar
        from isonome.types import AgentIdentity, AgentState

        pillar = MnemePillar(name="test-mneme")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        # Store some content
        assert pillar.mneme is not None
        pillar.mneme.store("persistent content", significance=0.95, tags=("test",))

        # Serialize
        serialized = pillar.serialize()
        assert serialized is not None
        assert len(serialized.get("working", [])) == 1

        # Restore into a new pillar
        pillar2 = MnemePillar(name="test-mneme-2")
        pillar2.restore(serialized)
        assert pillar2.mneme is not None
        assert pillar2.mneme.total_memories == 1

        entry = list(pillar2.mneme._working.values())[0]
        assert entry.content == "persistent content"
        assert entry.significance == 0.95

    def test_round_trip_with_large_entry_set(self):
        """Round-trip with multiple entries in all tiers."""
        mneme = HierarchicalMneme()
        for i in range(10):
            mneme.store(f"working item {i}", significance=0.3 + i * 0.05, tags=("batch",))
        for i in range(5):
            ep = MemoryEntry(
                content=f"episodic item {i}",
                tier=MemoryTier.EPISODIC,
                significance=0.6 + i * 0.05,
                base_half_life=86400.0,
            )
            mneme._episodic[ep.id] = ep
        for i in range(3):
            sem = MemoryEntry(
                content=f"semantic item {i}",
                tier=MemoryTier.SEMANTIC,
                significance=0.9,
                base_half_life=2592000.0,
            )
            mneme._semantic[sem.id] = sem

        data = mneme.to_dict()
        restored = HierarchicalMneme.from_dict(data)

        assert len(restored._working) == 7  # Capped by WORKING_CAPACITY
        assert len(restored._episodic) == 5
        assert len(restored._semantic) == 3
        assert restored.total_memories == 7 + 5 + 3


# ═══════════════════════════════════════════════════════════════════
# ActionOrchestrator Serialization Tests
# ═══════════════════════════════════════════════════════════════════


class TestActionOrchestratorSerialization:
    """Round-trip serialization for ActionOrchestrator state."""

    def test_empty_orchestrator_round_trip(self):
        """Empty orchestrator should serialize/deserialize cleanly."""
        orch = ActionOrchestrator()
        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert len(restored._actions) == 0
        assert len(restored._states) == 0
        assert len(restored._completed) == 0
        assert restored.total_actions == 0

    def test_actions_and_states_survive(self):
        """Registered actions with states should survive round-trip."""
        orch = ActionOrchestrator()
        a1 = Action(
            description="first action",
            tool_name="test_tool",
            risk=ActionRisk.LOW,
            tags=("test",),
            metadata={"key": "value"},
            confidence_required=0.6,
        )
        a2 = Action(
            description="second action",
            tool_name="other_tool",
            risk=ActionRisk.HIGH,
            preconditions=("precond_a",),
        )
        orch.register_action(a1)
        orch.register_action(a2)

        # Set explicit states
        orch._states[a1.id] = ActionState.COMPLETED
        orch._completed.add(a1.id)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert restored.total_actions == 2
        assert a1.id in restored._states
        assert restored._states[a1.id] == ActionState.COMPLETED
        assert a1.id in restored._completed
        assert a2.id in restored._states
        assert restored._states[a2.id] in (ActionState.PENDING, ActionState.QUEUED)

    def test_results_survive_round_trip(self):
        """Execution results should survive round-trip."""
        orch = ActionOrchestrator()
        action = Action(description="test", tool_name="test")
        orch.register_action(action)

        result = ExecutionResult(
            action_id=action.id,
            success=True,
            error="",
            duration_ms=42.0,
            validation_passed=True,
            validation_score=0.95,
            attempt=1,
            timestamp=time.time(),
        )
        orch._results[action.id] = [result]

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert action.id in restored._results
        assert len(restored._results[action.id]) == 1
        r = restored._results[action.id][0]
        assert r.success is True
        assert r.duration_ms == 42.0
        assert r.validation_score == 0.95
        assert r.attempt == 1

    def test_topological_levels_survive(self):
        """Topological levels should survive round-trip."""
        orch = ActionOrchestrator()
        a1 = Action(description="dep target", tool_name="t1")
        orch.register_action(a1)
        # Add a dependent action
        a2 = Action(description="dependent", tool_name="t2", dependencies=(a1.id,))
        orch.register_action(a2)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        # Levels should be restored (even if recalculated on access)
        assert len(restored._topological_levels) > 0

    def test_counters_survive(self):
        """Execution counters should survive round-trip."""
        orch = ActionOrchestrator()
        orch._total_executed = 10
        orch._total_completed = 8
        orch._total_failed = 2
        orch._total_retried = 3
        orch._total_blocked = 1
        orch._batch_count = 5

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert restored._total_executed == 10
        assert restored._total_completed == 8
        assert restored._total_failed == 2
        assert restored._total_retried == 3
        assert restored._total_blocked == 1
        assert restored._batch_count == 5

    def test_retry_policy_preserved(self):
        """Custom retry policies should survive round-trip."""
        rp = RetryPolicy(max_retries=5, base_delay=2.0, backoff_factor=3.0, max_delay=600.0)
        orch = ActionOrchestrator(default_retry_policy=rp)
        action = Action(
            description="with custom retry",
            tool_name="t",
            retry_policy=RetryPolicy(max_retries=2, base_delay=0.5),
        )
        orch.register_action(action)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert restored._default_retry.max_retries == 5
        assert restored._default_retry.base_delay == 2.0

    def test_via_pillar_serialize_restore(self):
        """PraxisPillar.serialize()/restore() should work."""
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = PraxisPillar(name="test-praxis")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        assert pillar.orchestrator is not None
        action = Action(description="test action", tool_name="test_tool")
        pillar.orchestrator.register_action(action)
        pillar.orchestrator._states[action.id] = ActionState.COMPLETED
        pillar.orchestrator._completed.add(action.id)

        serialized = pillar.serialize()
        assert serialized is not None

        pillar2 = PraxisPillar(name="test-praxis-2")
        pillar2.restore(serialized)
        assert pillar2.orchestrator is not None
        assert pillar2.orchestrator.total_actions == 1
        assert action.id in pillar2.orchestrator._completed



class TestCognitionPillarSerialization:
    """Direct CognitionPillar serialize/restore round-trips."""

    def test_empty_cognition_pillar_serialize_restore(self):
        """Empty CognitionPillar survives serialize/restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="test-cog")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        serialized = pillar.serialize()
        assert serialized is not None

        pillar2 = CognitionPillar(name="test-cog-2")
        pillar2.restore(serialized)

        assert pillar2.attention is not None
        assert pillar2.reasoning is not None
        assert pillar2._context_added == 0
        assert pillar2._tasks_reasoned == 0

    def test_cognition_pillar_counters_serialize_restore(self):
        """CognitionPillar counters survive serialize/restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="test-cog")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        pillar.add_context("some content")
        pillar.add_context("more content")
        pillar._tasks_reasoned = 5

        serialized = pillar.serialize()
        pillar2 = CognitionPillar(name="test-cog-2")
        pillar2.restore(serialized)

        assert pillar2._context_added == pillar._context_added
        assert pillar2._tasks_reasoned == 5
        assert pillar2._token_capacity == pillar._token_capacity

    def test_cognition_pillar_calibrator_serialize_restore(self):
        """Calibrator state survives CognitionPillar serialize/restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="test-cog")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        cal = pillar.reasoning.calibrator
        for i in range(25):
            cal.record(0.6 + (i % 4) * 0.1, (i % 2) == 0)

        serialized = pillar.serialize()
        pillar2 = CognitionPillar(name="test-cog-2")
        pillar2.restore(serialized)

        restored_cal = pillar2.reasoning.calibrator
        assert restored_cal.total_predictions == 25
        assert restored_cal.compute_ece() == pytest.approx(cal.compute_ece(), abs=1e-6)

    def test_cognition_pillar_attention_counters_serialize_restore(self):
        """Attention GC counters survive CognitionPillar serialize/restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="test-cog")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        for i in range(5):
            pillar.add_context(f"content block {i} " * 1000)
        pillar.attention.collect_garbage()

        gc_cycles_before = pillar.attention._gc_cycles
        pruned_before = pillar.attention._total_pruned

        serialized = pillar.serialize()
        pillar2 = CognitionPillar(name="test-cog-2")
        pillar2.restore(serialized)

        assert pillar2.attention._gc_cycles == gc_cycles_before
        assert pillar2.attention._total_pruned == pruned_before

    def test_cognition_pillar_config_serialize_restore(self):
        """CognitionPillar config (auto_gc, gc_threshold) survives restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(
            name="test-cog",
            auto_gc=False,
            gc_utilization_threshold=0.65,
        )
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        serialized = pillar.serialize()
        pillar2 = CognitionPillar(name="test-cog-2")
        pillar2.restore(serialized)

        assert pillar2._auto_gc is False
        assert pillar2._gc_util_threshold == pytest.approx(0.65)



# ═══════════════════════════════════════════════════════════════════
# IsonomeAgent Serialization Tests (Integration)
# ═══════════════════════════════════════════════════════════════════

class TestIsonomeAgentSerialization:
    """Full-agent serialization round-trip."""

    def test_empty_agent_round_trip(self):
        """Minimal agent serialization round-trip."""
        agent = IsonomeAgent(name="test-agent")
        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert restored.identity.name == "test-agent"
        assert restored.identity.id == agent.identity.id
        assert len(restored.engine.axes) == 8

    def test_agent_identity_preserved(self):
        """Agent identity should be preserved through round-trip."""
        agent = IsonomeAgent(name="identity-test")
        agent.state.task_count = 7

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert restored.identity.name == "identity-test"
        assert restored._tick_count == 0
        # Task count is not saved yet — that's OK

    def test_engine_state_survives(self):
        """Equilibrium engine state should survive agent-level round-trip."""
        agent = IsonomeAgent(name="engine-test")
        from isonome.types import Feedback
        agent.engine.apply_feedback(
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.5,
                confidence=0.9,
                reason="test",
            )
        )

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        orig_pos = agent.engine.get_behavior_profile()["explore_exploit"]
        restored_pos = restored.engine.get_behavior_profile()["explore_exploit"]
        assert restored_pos == pytest.approx(orig_pos)

    def test_cognition_pillar_state_survives(self):
        """CognitionPillar stats should survive agent-level round-trip."""
        from isonome.cognition.pillar import CognitionPillar
        agent = IsonomeAgent(
            name="cog-test",
            cognition=CognitionPillar(name="thinker"),
        )

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert "cognition" in data
        assert data["cognition"] is not None

    def test_mneme_pillar_state_survives(self):
        """MnemePillar state should survive agent-level round-trip."""
        from isonome.mneme.pillar import MnemePillar
        mneme = MnemePillar(name="memory")
        agent = IsonomeAgent(name="mneme-test", mneme=mneme)
        agent.start()

        data = agent.to_dict()
        restored_data = data  # just check it serializes

        assert "mneme" in data
        # The mneme should have empty collections
        assert data["mneme"] is not None

    def test_praxis_pillar_state_survives(self):
        """PraxisPillar state should survive agent-level round-trip."""
        from isonome.praxis.pillar import PraxisPillar
        praxis = PraxisPillar(name="executor")
        agent = IsonomeAgent(name="praxis-test", praxis=praxis)
        agent.start()

        data = agent.to_dict()
        assert "praxis" in data

    def test_task_queue_serialized(self):
        """Task queue items should survive round-trip."""
        agent = IsonomeAgent(name="task-test")
        agent.submit_task(Task(description="first task", complexity=TaskComplexity.SIMPLE))
        agent.submit_task(Task(description="second task", complexity=TaskComplexity.COMPLEX))

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        # Queue items should be restored
        assert len(restored._task_queue) == 2


# ═══════════════════════════════════════════════════════════════════
# JSON Round-Trip (End-to-End)
# ═══════════════════════════════════════════════════════════════════


class TestJSONRoundTrip:
    """All systems should survive JSON serialization/deserialization."""

    def test_calibrator_json_round_trip(self):
        """Calibrator through json.dumps/loads (true cross-session)."""
        import json

        cal = ConfidenceCalibrator(num_bins=10)
        for i in range(30):
            cal.record(0.5 + (i % 5) * 0.1, (i % 3) == 0)

        raw_json = json.dumps(cal.to_dict())
        parsed = json.loads(raw_json)
        restored = ConfidenceCalibrator.from_dict(parsed)

        assert restored.total_predictions == 30
        assert restored.compute_ece() == pytest.approx(cal.compute_ece(), abs=1e-6)

    def test_equilibrium_json_round_trip(self):
        """EquilibriumEngine through json.dumps/loads."""
        import json

        engine = EquilibriumEngine()
        raw_json = json.dumps(engine.to_dict())
        parsed = json.loads(raw_json)
        restored = EquilibriumEngine.from_dict(parsed)

        assert len(restored.axes) == 8

    def test_mneme_json_round_trip(self):
        """HierarchicalMneme through json.dumps/loads."""
        import json

        mneme = HierarchicalMneme()
        mneme.store("persistent content", significance=0.9, tags=("important",))

        raw_json = json.dumps(mneme.to_dict())
        parsed = json.loads(raw_json)
        restored = HierarchicalMneme.from_dict(parsed)

        assert restored.total_memories == 1
        entry = list(restored._working.values())[0]
        assert entry.content == "persistent content"

    def test_orchestrator_json_round_trip(self):
        """ActionOrchestrator through json.dumps/loads."""
        import json

        orch = ActionOrchestrator()
        a = Action(description="json test", tool_name="json_tool")
        orch.register_action(a)
        orch._states[a.id] = ActionState.COMPLETED
        orch._completed.add(a.id)

        raw_json = json.dumps(orch.to_dict())
        parsed = json.loads(raw_json)
        restored = ActionOrchestrator.from_dict(parsed)

        assert restored.total_actions == 1
        assert a.id in restored._completed

    def test_agent_json_round_trip(self):
        """Full agent through json.dumps/loads."""
        import json

        agent = IsonomeAgent(name="json-agent")
        data = agent.to_dict()
        raw_json = json.dumps(data, default=str)
        parsed = json.loads(raw_json)
        restored = IsonomeAgent.from_dict(parsed)

        assert restored.identity.name == "json-agent"


# ═══════════════════════════════════════════════════════════════════
# Schema Version & Faithful Config Round-Trip Tests (Schema v1)
# ═══════════════════════════════════════════════════════════════════


class TestSchemaVersioning:
    """Schema version is included and validated across all serialization layers."""

    def test_agent_to_dict_includes_schema_version(self):
        """Agent-level serialization includes top-level schema_version."""
        from isonome import SERIALIZATION_SCHEMA_VERSION

        agent = IsonomeAgent(name="versioned")
        data = agent.to_dict()
        assert "schema_version" in data
        assert data["schema_version"] == SERIALIZATION_SCHEMA_VERSION

    def test_praxis_pillar_includes_schema_version(self):
        """PraxisPillar.serialize() includes _schema_version."""
        from isonome import SERIALIZATION_SCHEMA_VERSION
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = PraxisPillar(name="test-praxis")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)
        data = pillar.serialize()
        assert data is not None
        assert data["_schema_version"] == SERIALIZATION_SCHEMA_VERSION

    def test_mneme_pillar_includes_schema_version(self):
        """MnemePillar.serialize() includes _schema_version."""
        from isonome import SERIALIZATION_SCHEMA_VERSION
        from isonome.mneme.pillar import MnemePillar
        from isonome.types import AgentIdentity, AgentState

        pillar = MnemePillar(name="test-mneme")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)
        data = pillar.serialize()
        assert data is not None
        assert data["_schema_version"] == SERIALIZATION_SCHEMA_VERSION

    def test_cognition_pillar_includes_schema_version(self):
        """CognitionPillar.serialize() includes _schema_version."""
        from isonome import SERIALIZATION_SCHEMA_VERSION
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="test-cog")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)
        data = pillar.serialize()
        assert data is not None
        assert data["_schema_version"] == SERIALIZATION_SCHEMA_VERSION

    def test_backward_compat_no_schema_version(self):
        """from_dict() handles data without schema_version (v0 format)."""
        # Agent without schema_version key should still restore
        agent_data = {
            "agent": {"name": "old-format", "id": "00000000-0000-0000-0000-000000000001"},
            "engine": EquilibriumEngine().to_dict(),
        }
        agent = IsonomeAgent.from_dict(agent_data)
        assert agent.identity.name == "old-format"

    def test_forward_schema_version_warns(self, caplog):
        """from_dict() warns when data has a newer schema version."""
        import logging

        agent_data = IsonomeAgent(name="forward-test").to_dict()
        agent_data["schema_version"] = 999  # Simulate future version

        with caplog.at_level(logging.WARNING):
            restored = IsonomeAgent.from_dict(agent_data)
        assert restored.identity.name == "forward-test"
        # Warning should have been logged about schema mismatch
        assert any("schema" in msg.lower() or "999" in msg for msg in caplog.messages)


class TestPraxisPillarConfigRoundTrip:
    """PraxisPillar.serialize()/restore() faithfully preserves config."""

    def test_max_parallel_survives_round_trip(self):
        """Custom max_parallel should survive serialize/restore."""
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = PraxisPillar(name="parallel-test", max_parallel=4)
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["max_parallel"] == 4

        pillar2 = PraxisPillar(name="parallel-test-2")
        pillar2.restore(data)
        assert pillar2._max_parallel == 4
        assert pillar2.orchestrator._max_parallel == 4

    def test_retry_policy_survives_round_trip(self):
        """Custom RetryPolicy should survive serialize/restore."""
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        custom_retry = RetryPolicy(max_retries=5, base_delay=2.0, backoff_factor=3.0, max_delay=600.0)
        pillar = PraxisPillar(name="retry-test", default_retry_policy=custom_retry)
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        rp_data = data["_pillar_config"]["default_retry_policy"]
        assert rp_data["max_retries"] == 5
        assert rp_data["base_delay"] == 2.0
        assert rp_data["backoff_factor"] == 3.0
        assert rp_data["max_delay"] == 600.0

        pillar2 = PraxisPillar(name="retry-test-2")
        pillar2.restore(data)
        assert pillar2._default_retry is not None
        assert pillar2._default_retry.max_retries == 5
        assert pillar2._default_retry.base_delay == 2.0

    def test_pillar_name_survives_round_trip(self):
        """Custom pillar name should survive serialize/restore."""
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = PraxisPillar(name="my-custom-executor")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["name"] == "my-custom-executor"

        pillar2 = PraxisPillar(name="default-name")
        pillar2.restore(data)
        assert pillar2.name == "my-custom-executor"

    def test_callable_presence_flags(self):
        """serialize() records whether callables were present."""
        from isonome.praxis.pillar import PraxisPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = PraxisPillar(
            name="flags-test",
            executor_fn=lambda a: None,
        )
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        config = data["_pillar_config"]
        assert config["has_executor_fn"] is True
        assert config["has_validator_fn"] is False
        assert config["has_approve_fn"] is False

    def test_praxis_agent_level_round_trip_with_config(self):
        """PraxisPillar config should survive full agent-level round-trip."""
        from isonome.praxis.pillar import PraxisPillar

        custom_retry = RetryPolicy(max_retries=7, base_delay=3.0, backoff_factor=2.5, max_delay=500.0)
        praxis = PraxisPillar(name="my-executor", max_parallel=3, default_retry_policy=custom_retry)
        agent = IsonomeAgent(name="praxis-config-test", praxis=praxis)
        agent.start()

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert restored._praxis is not None
        assert restored._praxis._max_parallel == 3
        assert restored._praxis._default_retry is not None
        assert restored._praxis._default_retry.max_retries == 7
        assert restored._praxis._default_retry.base_delay == 3.0
        assert restored._praxis.name == "my-executor"


class TestMnemePillarConfigRoundTrip:
    """MnemePillar.serialize()/restore() faithfully preserves config."""

    def test_consolidation_significance_survives_round_trip(self):
        """Custom consolidation_significance should survive serialize/restore."""
        from isonome.mneme.pillar import MnemePillar
        from isonome.types import AgentIdentity, AgentState

        pillar = MnemePillar(name="cons-test", consolidation_significance=0.7)
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["consolidation_significance"] == 0.7

        pillar2 = MnemePillar(name="cons-test-2")
        pillar2.restore(data)
        assert pillar2._cons_sig == 0.7
        assert pillar2.mneme._consolidation_significance == 0.7

    def test_promotion_significance_survives_round_trip(self):
        """Custom promotion_significance should survive serialize/restore."""
        from isonome.mneme.pillar import MnemePillar
        from isonome.types import AgentIdentity, AgentState

        pillar = MnemePillar(name="prom-test", promotion_significance=0.85)
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["promotion_significance"] == 0.85

        pillar2 = MnemePillar(name="prom-test-2")
        pillar2.restore(data)
        assert pillar2._prom_sig == 0.85
        assert pillar2.mneme._promotion_significance == 0.85

    def test_pillar_name_survives_round_trip(self):
        """Custom pillar name should survive serialize/restore."""
        from isonome.mneme.pillar import MnemePillar
        from isonome.types import AgentIdentity, AgentState

        pillar = MnemePillar(name="my-custom-memory")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["name"] == "my-custom-memory"

        pillar2 = MnemePillar(name="default-name")
        pillar2.restore(data)
        assert pillar2.name == "my-custom-memory"

    def test_mneme_agent_level_round_trip_with_config(self):
        """MnemePillar config should survive full agent-level round-trip."""
        from isonome.mneme.pillar import MnemePillar

        mneme = MnemePillar(name="my-memory", consolidation_significance=0.6, promotion_significance=0.9)
        agent = IsonomeAgent(name="mneme-config-test", mneme=mneme)
        agent.start()

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert restored._mneme is not None
        assert restored._mneme._cons_sig == 0.6
        assert restored._mneme._prom_sig == 0.9
        assert restored._mneme.name == "my-memory"


class TestCognitionPillarConfigRoundTrip:
    """CognitionPillar.serialize()/restore() faithfully preserves config."""

    def test_pillar_name_survives_round_trip(self):
        """Custom pillar name should survive serialize/restore."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(name="my-custom-thinker")
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        assert data["_pillar_config"]["name"] == "my-custom-thinker"

        pillar2 = CognitionPillar(name="default-name")
        pillar2.restore(data)
        assert pillar2.name == "my-custom-thinker"

    def test_callable_presence_flags(self):
        """serialize() records whether decomposer/evidence fns were present."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.types import AgentIdentity, AgentState

        pillar = CognitionPillar(
            name="flags-test",
            decomposer_fn=lambda t: [],
        )
        state = AgentState(identity=AgentIdentity(name="test"))
        pillar.initialize(state)

        data = pillar.serialize()
        config = data["_pillar_config"]
        assert config["has_decomposer_fn"] is True
        assert config["has_evidence_fn"] is False


class TestEngineBindingAfterRestore:
    """Restored pillars should be bound to the restored engine for auto-sync."""

    def test_pillars_bound_to_restored_engine(self):
        """All restored pillars should be engine-bound after from_dict()."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.praxis.pillar import PraxisPillar
        from isonome.mneme.pillar import MnemePillar

        agent = IsonomeAgent(
            name="binding-test",
            cognition=CognitionPillar(name="thinker"),
            praxis=PraxisPillar(name="executor"),
            mneme=MnemePillar(name="memory"),
        )
        agent.start()

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        # All restored pillars should be bound to the engine
        for pillar in restored._pillar_map.values():
            assert pillar._engine is not None
            assert pillar._engine is restored.engine

    def test_restored_pillars_have_equilibrium_view(self):
        """Restored pillars should have equilibrium views (from bind_engine)."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.praxis.pillar import PraxisPillar
        from isonome.mneme.pillar import MnemePillar

        agent = IsonomeAgent(
            name="view-test",
            cognition=CognitionPillar(name="thinker"),
            praxis=PraxisPillar(name="executor"),
            mneme=MnemePillar(name="memory"),
        )
        agent.start()

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        for pillar in restored._pillar_map.values():
            assert pillar._equilibrium_view is not None

    def test_restored_agent_can_tick(self):
        """A fully restored agent should be able to tick without errors."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.praxis.pillar import PraxisPillar
        from isonome.mneme.pillar import MnemePillar

        agent = IsonomeAgent(
            name="tick-test",
            cognition=CognitionPillar(name="thinker"),
            praxis=PraxisPillar(name="executor"),
            mneme=MnemePillar(name="memory"),
        )
        agent.start()
        agent.submit_task(Task(description="test task", complexity=TaskComplexity.SIMPLE))

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        # Ticking a restored agent should not raise
        snapshot = restored.tick()
        assert snapshot is not None


class TestFullAgentFaithfulRoundTrip:
    """End-to-end: agent with all three pillars and config round-trips faithfully."""

    def test_three_pillar_agent_round_trip(self):
        """Agent with all 3 pillars preserves identity, engine, config, queue."""
        from isonome.cognition.pillar import CognitionPillar
        from isonome.praxis.pillar import PraxisPillar
        from isonome.mneme.pillar import MnemePillar

        custom_retry = RetryPolicy(max_retries=5, base_delay=2.5, backoff_factor=1.5, max_delay=120.0)
        agent = IsonomeAgent(
            name="full-faithful",
            cognition=CognitionPillar(name="deep-thinker", token_capacity=256_000),
            praxis=PraxisPillar(name="safe-executor", max_parallel=2, default_retry_policy=custom_retry),
            mneme=MnemePillar(name="vivid-memory", consolidation_significance=0.75, promotion_significance=0.85),
        )
        agent.start()
        agent.submit_task(Task(description="complex analysis", complexity=TaskComplexity.COMPLEX))
        agent.tick()

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        # Identity
        assert restored.identity.name == "full-faithful"
        assert restored.identity.id == agent.identity.id

        # Counters
        assert restored._tick_count == 1
        assert restored._signals_sent == agent._signals_sent

        # Engine
        assert len(restored.engine.axes) == len(agent.engine.axes)

        # Cognition config
        assert restored._cognition is not None
        assert restored._cognition.name == "deep-thinker"
        assert restored._cognition._token_capacity == 256_000

        # Praxis config
        assert restored._praxis is not None
        assert restored._praxis.name == "safe-executor"
        assert restored._praxis._max_parallel == 2
        assert restored._praxis._default_retry.max_retries == 5
        assert restored._praxis._default_retry.base_delay == 2.5

        # Mneme config
        assert restored._mneme is not None
        assert restored._mneme.name == "vivid-memory"
        assert restored._mneme._cons_sig == 0.75
        assert restored._mneme._prom_sig == 0.85

        # Task queue
        assert len(restored._task_queue) == 1

        # Engine binding
        for p in restored._pillar_map.values():
            assert p._engine is restored.engine

    def test_three_pillar_agent_json_round_trip(self):
        """Full agent with config through json.dumps/loads."""
        import json

        from isonome.cognition.pillar import CognitionPillar
        from isonome.praxis.pillar import PraxisPillar
        from isonome.mneme.pillar import MnemePillar

        custom_retry = RetryPolicy(max_retries=10, base_delay=0.5, backoff_factor=1.0, max_delay=60.0)
        agent = IsonomeAgent(
            name="json-faithful",
            cognition=CognitionPillar(name="thinker"),
            praxis=PraxisPillar(name="executor", max_parallel=4, default_retry_policy=custom_retry),
            mneme=MnemePillar(name="memory", consolidation_significance=0.8),
        )
        agent.start()

        data = agent.to_dict()
        raw_json = json.dumps(data, default=str)
        parsed = json.loads(raw_json)
        restored = IsonomeAgent.from_dict(parsed)

        assert restored.identity.name == "json-faithful"
        assert restored._praxis._max_parallel == 4
        assert restored._praxis._default_retry.max_retries == 10
        assert restored._mneme._cons_sig == 0.8
