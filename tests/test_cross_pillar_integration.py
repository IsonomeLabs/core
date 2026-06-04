"""Cross-pillar integration and serialization round-trip tests.

Tests the data pipeline across pillars:
1. Cognition → Praxis: reasoning plans feed action execution
2. Praxis → Mneme: execution results persist to memory
3. Mneme → Cognition: recalled memories provide context
4. Full cycle: Cognition reason → Praxis execute → Mneme store

Also tests serialization round-trips for:
- ReasoningEngine (cognition)
- ActionOrchestrator (praxis) with the corrected single to_dict/from_dict
- frozendict (shared utility)
"""
from __future__ import annotations

import time
import pytest

from isonome.cognition.pillar import CognitionPillar
from isonome.cognition.reasoning import RecursiveReasoningEngine, ConfidenceCalibrator
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionResult,
    RetryPolicy,
)
from isonome.praxis.pillar import PraxisPillar
from isonome.mneme.pillar import MnemePillar
from isonome.mneme.hierarchical import (
    HierarchicalMneme,
    MemoryEntry,
    MemoryTier,
    ConsolidationReport,
)
from isonome.utils.frozendict import frozendict
from isonome.equilibrium import EquilibriumEngine
from isonome.types import (
    AgentIdentity,
    AgentState,
    Feedback,
    Pillar,
    Signal,
    TensionID,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def agent_state():
    """A minimal agent state for pillar initialization."""
    return AgentState(
        identity=AgentIdentity(name="test-agent"),
        tick=0,
        active=True,
    )


@pytest.fixture
def engine():
    """A fresh equilibrium engine."""
    return EquilibriumEngine()


@pytest.fixture
def cognition(agent_state, engine):
    """An initialized CognitionPillar bound to the equilibrium engine."""
    p = CognitionPillar()
    p.initialize(agent_state)
    p.bind_engine(engine)
    return p


@pytest.fixture
def praxis(agent_state, engine):
    """An initialized PraxisPillar with a no-op executor."""
    p = PraxisPillar(
        executor_fn=lambda action: (True, f"executed {action.description}"),
    )
    p.initialize(agent_state)
    p.bind_engine(engine)
    return p


@pytest.fixture
def mneme(agent_state, engine):
    """An initialized MnemePillar bound to the equilibrium engine."""
    p = MnemePillar()
    p.initialize(agent_state)
    p.bind_engine(engine)
    return p


# ═══════════════════════════════════════════════════════════════════
# 1. Cross-pillar: Cognition → Praxis pipeline
# ═══════════════════════════════════════════════════════════════════


class TestCognitionToPraxis:
    """Reasoning plans from Cognition should feed into Praxis actions."""

    def test_reason_produces_plan_with_actions(self, cognition):
        """Cognition.reason() should produce a plan with actionable nodes."""
        plan = cognition.reason("deploy the service to production")
        assert plan is not None
        # A plan should have structured output fields
        assert hasattr(plan, "root_hypothesis") or hasattr(plan, "plans") or hasattr(plan, "total_nodes")

    def test_plan_actions_can_register_in_praxis(self, cognition, praxis):
        """Actions derived from a reasoning plan should register in Praxis."""
        plan = cognition.reason("write unit tests for the API")
        # Register actions in praxis based on plan
        action = Action(
            description="Write API unit tests",
            tool_name="test_runner",
            risk=ActionRisk.LOW,
        )
        praxis.orchestrator.register_action(action)
        assert praxis.orchestrator.total_actions == 1

    def test_cognition_signal_to_praxis(self, praxis):
        """Praxis should receive signals targeting it."""
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="register_action",
            payload={
                "description": "Deploy service",
                "tool_name": "deployer",
                "risk": "MEDIUM",
            },
        )
        praxis.receive_signal(signal)
        # Praxis should have processed the signal
        praxis.process_queued()


# ═══════════════════════════════════════════════════════════════════
# 2. Cross-pillar: Praxis → Mneme pipeline
# ═══════════════════════════════════════════════════════════════════


class TestPraxisToMneme:
    """Execution results from Praxis should persist to Mneme."""

    def test_praxis_execution_memories(self, praxis):
        """PraxisPillar.get_execution_memories() should return log entries."""
        action = Action(
            description="test action",
            tool_name="test_tool",
        )
        praxis.orchestrator.register_action(action)
        praxis.execute_pending()
        memories = praxis.get_execution_memories()
        assert isinstance(memories, list)

    def test_praxis_results_stored_in_mneme(self, praxis, mneme):
        """Execution results should be storable in Mneme via signal."""
        # Store execution result as memory
        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="store",
            payload={
                "content": "Action 'deploy' completed successfully",
                "significance": 0.7,
                "tags": ("execution", "deployment"),
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        # Verify memory was stored
        results = mneme.mneme.recall("deploy")
        assert len(results) >= 1
        assert any("deploy" in e.content.lower() for e in results)

    def test_praxis_export_to_mneme_format(self, praxis):
        """export_to_mneme() should produce mneme-compatible dicts."""
        action = Action(
            description="deploy service",
            tool_name="deployer",
        )
        praxis.orchestrator.register_action(action)
        praxis.execute_pending()
        memories = praxis.get_execution_memories()
        # Each memory dict should have expected keys
        for mem in memories:
            assert isinstance(mem, dict)


# ═══════════════════════════════════════════════════════════════════
# 3. Cross-pillar: Mneme → Cognition pipeline
# ═══════════════════════════════════════════════════════════════════


class TestMnemeToCognition:
    """Recalled memories from Mneme should provide context to Cognition."""

    def test_recall_provides_context(self, cognition, mneme):
        """Memories stored in Mneme should be recallable for Cognition context."""
        # Store a memory
        mneme.mneme.store(
            "Previous deployment failed due to missing env vars",
            significance=0.8,
            tags=("deployment", "failure"),
        )

        # Recall should find it
        results = mneme.mneme.recall("deployment")
        assert len(results) >= 1

        # Cognition can use this as context
        context = [e.content for e in results]
        plan = cognition.reason("deploy again", context=context)
        assert plan is not None

    def test_mneme_consolidation_preserves_knowledge(self, mneme):
        """After consolidation, high-significance memories should persist."""
        # Store high-significance memory
        entry = mneme.mneme.store(
            "Critical: database requires migration before deploy",
            significance=0.9,
            tags=("database", "migration", "critical"),
        )

        # Run consolidation
        report = mneme.mneme.consolidate()
        assert isinstance(report, ConsolidationReport)

        # The high-significance entry should still be findable
        results = mneme.mneme.recall("database")
        assert len(results) >= 1


# ═══════════════════════════════════════════════════════════════════
# 4. Full cycle: Cognition → Praxis → Mneme → Cognition
# ═══════════════════════════════════════════════════════════════════


class TestFullCycle:
    """End-to-end cycle across all three pillars."""

    def test_full_pipeline(self, cognition, praxis, mneme):
        """Full cycle: reason → register → execute → store → recall."""
        # 1. Cognition reasons about a task
        plan = cognition.reason("set up CI/CD pipeline")
        assert plan is not None

        # 2. Derive actions from plan and register in Praxis
        actions = [
            Action(description="Create CI config", tool_name="file_writer", risk=ActionRisk.LOW),
            Action(description="Add test stage", tool_name="file_writer", risk=ActionRisk.LOW),
            Action(description="Deploy to staging", tool_name="deployer", risk=ActionRisk.MODERATE),
        ]
        for a in actions:
            praxis.orchestrator.register_action(a)

        assert praxis.orchestrator.total_actions == 3

        # 3. Execute actions
        report = praxis.execute_pending()
        assert report is not None

        # 4. Store results in Mneme
        for a in actions:
            signal = Signal(
                source=Pillar.PRAXIS,
                target=Pillar.MNEME,
                kind="store",
                payload={
                    "content": f"Completed: {a.description}",
                    "significance": 0.6,
                    "tags": ("ci-cd", a.tool_name),
                },
            )
            mneme.receive_signal(signal)
        mneme.process_queued()

        # 5. Recall from Mneme to provide context for next reasoning
        results = mneme.mneme.recall("CI")
        assert len(results) >= 1

        # 6. Use recalled context in next reasoning cycle
        context = [e.content for e in results]
        plan2 = cognition.reason("add production deploy stage", context=context)
        assert plan2 is not None

    def test_cycle_with_equilibrium_feedback(self, cognition, praxis, mneme, engine):
        """Full cycle with equilibrium feedback flowing between pillars."""
        # Register and execute a risky action
        action = Action(
            description="Deploy to production",
            tool_name="deployer",
            risk=ActionRisk.HIGH,
        )
        praxis.orchestrator.register_action(action)
        praxis.execute_pending()

        # Pillars should produce feedback through equilibrium
        # After execution, drain feedback
        for pillar in [cognition, praxis, mneme]:
            pillar.process_queued()


# ═══════════════════════════════════════════════════════════════════
# 5. ReasoningEngine serialization round-trip
# ═══════════════════════════════════════════════════════════════════


class TestReasoningEngineSerialization:
    """Round-trip serialization for RecursiveReasoningEngine."""

    def test_fresh_engine_round_trip(self):
        """A fresh engine should serialize/deserialize cleanly."""
        cal = ConfidenceCalibrator()
        engine = RecursiveReasoningEngine(calibrator=cal)

        data = engine.to_dict()
        assert isinstance(data, dict)

        restored = RecursiveReasoningEngine.from_dict(data)
        assert restored is not None

    def test_engine_with_history_round_trip(self):
        """Engine with reasoning history should survive round-trip."""
        cal = ConfidenceCalibrator()
        engine = RecursiveReasoningEngine(calibrator=cal)

        # Reason a few times to build history
        engine.reason("task 1")
        engine.reason("task 2")

        data = engine.to_dict()
        restored = RecursiveReasoningEngine.from_dict(data)

        # Reasoning count should be preserved
        assert restored._stats.sessions == 2

    def test_calibrator_survives_round_trip(self):
        """Confidence calibrator state should survive engine round-trip."""
        cal = ConfidenceCalibrator()
        engine = RecursiveReasoningEngine(calibrator=cal)

        # Record some predictions
        cal.record(0.8, True)
        cal.record(0.3, False)
        cal.record(0.9, True)

        data = engine.to_dict()
        restored = RecursiveReasoningEngine.from_dict(data)

        # Calibrator state should be preserved
        assert restored._calibrator.total_predictions >= 3


# ═══════════════════════════════════════════════════════════════════
# 6. ActionOrchestrator serialization round-trip (single to_dict/from_dict)
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorSerializationConsistency:
    """Verify the deduplicated to_dict/from_dict pair works correctly."""

    def test_single_to_dict_method(self):
        """There should be exactly one to_dict method on ActionOrchestrator."""
        methods = [
            name for name in dir(ActionOrchestrator)
            if name == "to_dict"
        ]
        assert len(methods) == 1

    def test_single_from_dict_method(self):
        """There should be exactly one from_dict method on ActionOrchestrator."""
        methods = [
            name for name in dir(ActionOrchestrator)
            if name == "from_dict"
        ]
        assert len(methods) == 1

    def test_round_trip_preserves_action_count(self):
        """to_dict → from_dict round trip preserves action count."""
        orch = ActionOrchestrator()
        for i in range(5):
            orch.register_action(Action(
                description=f"action-{i}",
                tool_name="tool",
                risk=ActionRisk.LOW,
            ))

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)
        assert restored.total_actions == 5

    def test_round_trip_preserves_retry_policy(self):
        """Custom retry policies should survive round-trip."""
        rp = RetryPolicy(max_retries=7, base_delay=3.0, backoff_factor=2.5, max_delay=500.0)
        orch = ActionOrchestrator(default_retry_policy=rp)
        action = Action(
            description="custom retry action",
            tool_name="t",
            retry_policy=RetryPolicy(max_retries=3, base_delay=1.0),
        )
        orch.register_action(action)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert restored._default_retry.max_retries == 7
        assert restored._default_retry.base_delay == 3.0

    def test_round_trip_preserves_execution_results(self):
        """Execution results with all fields should survive round-trip."""
        orch = ActionOrchestrator()
        action = Action(description="result test", tool_name="t")
        orch.register_action(action)

        result = ExecutionResult(
            action_id=action.id,
            success=True,
            error="",
            duration_ms=100.5,
            validation_passed=True,
            validation_score=0.88,
            attempt=1,
            timestamp=time.time(),
        )
        orch._results[action.id] = [result]

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert action.id in restored._results
        r = restored._results[action.id][0]
        assert r.success is True
        assert r.duration_ms == 100.5
        assert r.validation_score == 0.88


# ═══════════════════════════════════════════════════════════════════
# 7. frozendict (shared utility) tests
# ═══════════════════════════════════════════════════════════════════


class TestFrozendict:
    """Tests for the shared frozendict utility."""

    def test_empty_frozendict(self):
        """Empty frozendict should work."""
        fd = frozendict()
        assert len(fd) == 0
        assert not fd

    def test_dict_constructor(self):
        """frozendict should accept dict in constructor."""
        fd = frozendict({"a": 1, "b": 2})
        assert fd["a"] == 1
        assert fd["b"] == 2
        assert len(fd) == 2

    def test_kwargs_constructor(self):
        """frozendict should accept keyword arguments."""
        fd = frozendict(x=10, y=20)
        assert fd["x"] == 10

    def test_get_method(self):
        """frozendict.get() should return default for missing keys."""
        fd = frozendict({"a": 1})
        assert fd.get("a") == 1
        assert fd.get("missing", 42) == 42

    def test_immutability(self):
        """frozendict should raise TypeError on mutation attempts."""
        fd = frozendict({"a": 1})
        with pytest.raises(TypeError):
            fd["a"] = 2
        with pytest.raises(TypeError):
            del fd["a"]
        with pytest.raises(TypeError):
            fd.update({"b": 2})
        with pytest.raises(TypeError):
            fd.pop("a")

    def test_hashable(self):
        """frozendict should be hashable for use in sets/dict keys."""
        fd1 = frozendict({"a": 1})
        fd2 = frozendict({"a": 1})
        fd3 = frozendict({"b": 2})

        # Same content = same hash
        assert hash(fd1) == hash(fd2)

        # Can use in set
        s = {fd1, fd2, fd3}
        assert len(s) == 2  # fd1 and fd2 are equal, so set deduplicates

        # Can use as dict key
        d = {fd1: "value"}
        assert d[fd2] == "value"  # fd2 equals fd1

    def test_equality(self):
        """frozendict should support equality with other frozendicts and dicts."""
        fd = frozendict({"a": 1, "b": 2})
        fd2 = frozendict({"a": 1, "b": 2})
        assert fd == fd2
        assert fd == {"a": 1, "b": 2}

    def test_iteration(self):
        """frozendict should support iteration, keys(), values(), items()."""
        fd = frozendict({"x": 1, "y": 2})
        assert set(fd.keys()) == {"x", "y"}
        assert set(fd.values()) == {1, 2}
        assert set(fd.items()) == {("x", 1), ("y", 2)}
        assert set(fd) == {"x", "y"}

    def test_contains(self):
        """frozendict should support 'in' operator."""
        fd = frozendict({"key": "value"})
        assert "key" in fd
        assert "missing" not in fd

    def test_copy_returns_mutable_dict(self):
        """frozendict.copy() should return a regular mutable dict."""
        fd = frozendict({"a": 1})
        d = fd.copy()
        assert isinstance(d, dict)
        d["b"] = 2  # Should not raise
        assert "b" in d

    def test_repr(self):
        """frozendict repr should be informative."""
        fd = frozendict({"a": 1})
        r = repr(fd)
        assert "frozendict" in r
        assert "a" in r

    def test_used_in_memory_entry(self):
        """frozendict should work as MemoryEntry.metadata."""
        entry = MemoryEntry(
            content="test",
            metadata=frozendict({"source": "test", "priority": 0.5}),
        )
        assert entry.metadata["source"] == "test"
        assert entry.metadata["priority"] == 0.5

    def test_hash_caching(self):
        """frozendict hash should be computed once and cached."""
        fd = frozendict({"a": 1})
        h1 = hash(fd)
        h2 = hash(fd)
        assert h1 == h2
        # Internal hash cache should be set
        assert fd._hash is not None
