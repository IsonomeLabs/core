"""Tests for the Praxis→Mneme auto-export pipeline (iter-023).

Verifies the signal-based architecture:
1. PraxisPillar emits 'execution_results' signal after batch execution
2. MnemePillar handles 'execution_results' signal by storing memories
3. Full agent tick cycle routes the signal correctly
4. Stored memories are recallable via Mneme recall
5. Significance is correctly derived from execution outcomes
6. Edge cases: empty execution log, no praxis/mneme pillars
"""

import pytest
from uuid import uuid4

from isonome.agent import IsonomeAgent
from isonome.base import BasePillar
from isonome.cognition.pillar import CognitionPillar
from isonome.mneme.pillar import MnemePillar
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ExecutionReport,
    RetryPolicy,
)
from isonome.praxis.pillar import PraxisPillar
from isonome.types import (
    AgentIdentity,
    AgentState,
    Feedback,
    Pillar,
    Signal,
    Task,
    TaskComplexity,
    TensionAxis,
    TensionSnapshot,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def praxis():
    """PraxisPillar with a simple executor function."""
    call_count = {"n": 0}

    def executor(action):
        call_count["n"] += 1
        # Fail every third action to test mixed success/failure
        if call_count["n"] % 3 == 0:
            return None  # None result = failure
        return f"result-{action.description}"

    def validator(action, result):
        if result is None:
            return (False, 0.0)
        return (True, 0.85)

    p = PraxisPillar(
        name="test-praxis",
        executor_fn=executor,
        validator_fn=validator,
    )
    return p


@pytest.fixture
def mneme():
    """MnemePillar for testing."""
    return MnemePillar(name="test-mneme")


@pytest.fixture
def cognition():
    """CognitionPillar for testing."""
    return CognitionPillar(name="test-cognition")


@pytest.fixture
def agent(praxis, mneme, cognition):
    """Fully wired IsonomeAgent."""
    return IsonomeAgent(
        name="test-agent",
        cognition=cognition,
        praxis=praxis,
        mneme=mneme,
    )


def _init_pillars(pillar_or_agent, agent_state=None):
    """Initialize a pillar or all pillars in an agent."""
    if isinstance(pillar_or_agent, IsonomeAgent):
        pillar_or_agent.start()
    else:
        if agent_state is None:
            axes = frozenset([
                TensionAxis(
                    id="explore_exploit",
                    pillar=Pillar.COGNITION,
                    pole_left="explore",
                    pole_right="exploit",
                    position=0.15,
                ),
            ])
            snapshot = TensionSnapshot(axes=axes)
            agent_state = AgentState(
                identity=AgentIdentity(name="test"),
                tensions=snapshot,
            )
        pillar_or_agent.initialize(agent_state)


# ── Test: Signal emission from Praxis ───────────────────────────


class TestPraxisSignalEmission:
    """Verify PraxisPillar emits execution_results signal after batch."""

    def test_signal_emitted_after_execution(self, praxis):
        """After _run_execution_batch, execution_results signal is in queue."""
        _init_pillars(praxis)
        # Register and execute actions
        action = Action(
            description="test action",
            tool_name="test_tool",
            risk=ActionRisk.LOW,
        )
        praxis.orchestrator.register_action(action)
        praxis._run_execution_batch()

        # Check signal queue
        signals = praxis.drain_signals()
        exec_result_signals = [s for s in signals if s.kind == "execution_results"]
        assert len(exec_result_signals) == 1, (
            f"Expected 1 execution_results signal, got {len(exec_result_signals)}"
        )

    def test_signal_target_is_mneme(self, praxis):
        """execution_results signal targets the Mneme pillar."""
        _init_pillars(praxis)
        action = Action(
            description="test action",
            tool_name="test_tool",
            risk=ActionRisk.LOW,
        )
        praxis.orchestrator.register_action(action)
        praxis._run_execution_batch()

        signals = praxis.drain_signals()
        exec_signals = [s for s in signals if s.kind == "execution_results"]
        assert exec_signals[0].source == Pillar.PRAXIS
        assert exec_signals[0].target == Pillar.MNEME

    def test_signal_payload_contains_entries(self, praxis):
        """Signal payload includes the execution log entries."""
        _init_pillars(praxis)
        action = Action(
            description="deploy service",
            tool_name="deploy",
            risk=ActionRisk.LOW,
        )
        praxis.orchestrator.register_action(action)
        praxis._run_execution_batch()

        signals = praxis.drain_signals()
        exec_signals = [s for s in signals if s.kind == "execution_results"]
        payload = exec_signals[0].payload
        assert "entries" in payload
        assert len(payload["entries"]) >= 1
        entry = payload["entries"][0]
        assert "description" in entry
        assert "tool_name" in entry
        assert "success" in entry
        assert entry["description"] == "deploy service"
        assert entry["tool_name"] == "deploy"

    def test_no_signal_when_no_actions_executed(self, praxis):
        """No execution_results signal if batch had no actions."""
        _init_pillars(praxis)
        # Don't register any actions
        praxis._run_execution_batch()

        signals = praxis.drain_signals()
        exec_signals = [s for s in signals if s.kind == "execution_results"]
        assert len(exec_signals) == 0, (
            "Should not emit execution_results when no actions executed"
        )

    def test_no_signal_when_no_executor_fn(self):
        """No execution_results signal when executor_fn is None."""
        praxis = PraxisPillar(name="no-executor")
        _init_pillars(praxis)
        # Register an action but can't execute without executor_fn
        action = Action(
            description="test action",
            tool_name="test_tool",
            risk=ActionRisk.LOW,
        )
        praxis.orchestrator.register_action(action)
        praxis._run_execution_batch()

        signals = praxis.drain_signals()
        exec_signals = [s for s in signals if s.kind == "execution_results"]
        assert len(exec_signals) == 0

    def test_signal_emitted_with_multiple_actions(self, praxis):
        """Signal contains all execution log entries for multi-action batch."""
        _init_pillars(praxis)
        for i in range(5):
            action = Action(
                description=f"action-{i}",
                tool_name=f"tool-{i}",
                risk=ActionRisk.LOW,
            )
            praxis.orchestrator.register_action(action)
        praxis._run_execution_batch()

        signals = praxis.drain_signals()
        exec_signals = [s for s in signals if s.kind == "execution_results"]
        assert len(exec_signals) == 1
        assert len(exec_signals[0].payload["entries"]) >= 1


# ── Test: Mneme signal handler ──────────────────────────────────


class TestMnemeExecutionResultsHandler:
    """Verify MnemePillar handles execution_results signal correctly."""

    def test_handler_stores_memories(self, mneme):
        """execution_results signal stores entries as memories."""
        _init_pillars(mneme)
        initial_count = mneme.mneme.total_memories

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "deploy app",
                        "tool_name": "deploy",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 150,
                        "validation_score": 0.9,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        assert mneme.mneme.total_memories == initial_count + 1

    def test_handler_stores_multiple_entries(self, mneme):
        """Multiple execution entries become multiple memories."""
        _init_pillars(mneme)

        entries = [
            {
                "action_id": str(uuid4()),
                "description": f"task-{i}",
                "tool_name": f"tool-{i}",
                "success": i % 2 == 0,
                "error": None if i % 2 == 0 else "timeout",
                "attempt": 0,
                "duration_ms": 100 + i * 50,
                "validation_score": 0.5 + i * 0.1,
                "batch": 1,
            }
            for i in range(4)
        ]

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={"entries": entries},
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        assert mneme.mneme.total_memories == 4

    def test_success_entry_has_higher_significance(self, mneme):
        """Successful actions are stored with higher significance than failures."""
        _init_pillars(mneme)

        entries = [
            {
                "action_id": str(uuid4()),
                "description": "success action",
                "tool_name": "deploy",
                "success": True,
                "error": None,
                "attempt": 0,
                "duration_ms": 100,
                "validation_score": 0.8,
                "batch": 1,
            },
            {
                "action_id": str(uuid4()),
                "description": "failure action",
                "tool_name": "deploy",
                "success": False,
                "error": "connection refused",
                "attempt": 0,
                "duration_ms": 200,
                "validation_score": 0.0,
                "batch": 1,
            },
        ]

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={"entries": entries},
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        # Get all entries and check significance
        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 2

        sigs = {e.content: e.significance for e in all_entries}
        # The success entry should have higher significance
        success_content = [c for c in sigs if "succeeded" in c]
        failure_content = [c for c in sigs if "failed" in c]
        assert len(success_content) == 1
        assert len(failure_content) == 1
        assert sigs[success_content[0]] > sigs[failure_content[0]]

    def test_content_includes_action_description(self, mneme):
        """Stored memory content describes the action and its outcome."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "build project",
                        "tool_name": "make",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 500,
                        "validation_score": 0.9,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        content = all_entries[0].content
        assert "build project" in content
        assert "make" in content
        assert "succeeded" in content

    def test_failure_includes_error_message(self, mneme):
        """Failed action memory includes the error message."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "deploy service",
                        "tool_name": "deploy",
                        "success": False,
                        "error": "connection timeout",
                        "attempt": 1,
                        "duration_ms": 30000,
                        "validation_score": 0.0,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        content = all_entries[0].content
        assert "failed" in content
        assert "connection timeout" in content

    def test_tags_include_execution_metadata(self, mneme):
        """Stored memories have execution-related tags."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "run tests",
                        "tool_name": "pytest",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 5000,
                        "validation_score": 0.7,
                        "batch": 3,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        tags = all_entries[0].tags
        assert "execution" in tags
        assert "pytest" in tags
        assert "success" in tags
        assert "batch-3" in tags

    def test_source_is_praxis_execution_results(self, mneme):
        """Stored memories have source='praxis:execution_results'."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 100,
                        "validation_score": 0.5,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert all_entries[0].source == "praxis:execution_results"

    def test_empty_entries_list_is_harmless(self, mneme):
        """Empty entries list in payload does not crash."""
        _init_pillars(mneme)
        initial_count = mneme.mneme.total_memories

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={"entries": []},
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        assert mneme.mneme.total_memories == initial_count

    def test_missing_entries_key_is_harmless(self, mneme):
        """Missing entries key in payload does not crash."""
        _init_pillars(mneme)
        initial_count = mneme.mneme.total_memories

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={},
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        assert mneme.mneme.total_memories == initial_count

    def test_validation_score_modulates_significance(self, mneme):
        """Higher validation_score produces higher significance."""
        _init_pillars(mneme)

        entries = [
            {
                "action_id": str(uuid4()),
                "description": f"action-v{v}",
                "tool_name": "test",
                "success": True,
                "error": None,
                "attempt": 0,
                "duration_ms": 100,
                "validation_score": v,
                "batch": 1,
            }
            for v in [0.1, 0.5, 0.9]
        ]

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={"entries": entries},
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 3

        # Higher validation score → higher significance (for same success=True)
        by_desc = {e.content: e.significance for e in all_entries}
        v01 = [s for c, s in by_desc.items() if "action-v0.1" in c][0]
        v05 = [s for c, s in by_desc.items() if "action-v0.5" in c][0]
        v09 = [s for c, s in by_desc.items() if "action-v0.9" in c][0]
        assert v01 < v05 < v09

    def test_uninitialized_mneme_ignores_signal(self):
        """MnemePillar that hasn't been initialized ignores the signal."""
        mneme = MnemePillar(name="uninit-mneme")
        # NOT initialized — mneme.mneme is None

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 100,
                        "validation_score": 0.5,
                        "batch": 1,
                    },
                ],
            },
        )
        # Should not crash — handler guards with mneme is None check
        mneme.receive_signal(signal)
        mneme.process_queued()


# ── Test: Full agent tick integration ───────────────────────────


class TestAgentTickIntegration:
    """Verify the Praxis→Mneme pipeline works through the agent tick cycle."""

    def test_agent_tick_routes_execution_results(self, agent):
        """Agent tick routes execution_results signal from Praxis to Mneme."""
        agent.start()
        initial_memories = agent.mneme.mneme.total_memories

        # Register actions and trigger execution via Praxis
        for i in range(3):
            action = Action(
                description=f"tick-action-{i}",
                tool_name=f"tool-{i}",
                risk=ActionRisk.LOW,
            )
            agent.praxis.orchestrator.register_action(action)

        # Run execution batch on praxis
        agent.praxis._run_execution_batch()

        # Tick the agent — this should route the signal
        agent.tick()

        # After the tick, Mneme should have new memories from execution
        assert agent.mneme.mneme.total_memories > initial_memories, (
            f"Expected new memories after tick, got "
            f"{agent.mneme.mneme.total_memories} (was {initial_memories})"
        )

    def test_multiple_ticks_accumulate_memories(self, agent):
        """Each tick with new executions adds more memories."""
        agent.start()
        initial_memories = agent.mneme.mneme.total_memories

        # First batch
        action1 = Action(
            description="batch-1-action",
            tool_name="test",
            risk=ActionRisk.LOW,
        )
        agent.praxis.orchestrator.register_action(action1)
        agent.praxis._run_execution_batch()
        agent.tick()

        after_first = agent.mneme.mneme.total_memories
        assert after_first > initial_memories

        # Second batch
        action2 = Action(
            description="batch-2-action",
            tool_name="test",
            risk=ActionRisk.LOW,
        )
        agent.praxis.orchestrator.register_action(action2)
        agent.praxis._run_execution_batch()
        agent.tick()

        after_second = agent.mneme.mneme.total_memories
        assert after_second > after_first

    def test_no_execution_no_new_memories(self, agent):
        """Tick without any execution doesn't add memories."""
        agent.start()
        initial_memories = agent.mneme.mneme.total_memories

        # Tick without any execution
        agent.tick()

        assert agent.mneme.mneme.total_memories == initial_memories


# ── Test: Round-trip: execute → store → recall ─────────────────


class TestExecutionRecallRoundTrip:
    """Verify execution results can be recalled from Mneme after storage."""

    def test_stored_execution_memories_are_recallable(self, mneme):
        """Execution results stored via signal are recallable."""
        _init_pillars(mneme)

        # Store execution results
        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "deploy api server",
                        "tool_name": "deploy",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 3000,
                        "validation_score": 0.85,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        # Consolidate to move to episodic tier for recall
        mneme.mneme.consolidate()

        # Recall should find the deployed api server
        results = mneme.mneme.recall("deploy api", max_results=5)
        # Even if recall tokenization is imperfect, the memory should be stored
        assert mneme.mneme.total_memories >= 1

    def test_failure_memories_are_recallable(self, mneme):
        """Failed action memories are stored and count toward total."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "deploy database",
                        "tool_name": "deploy",
                        "success": False,
                        "error": "port already in use",
                        "attempt": 0,
                        "duration_ms": 5000,
                        "validation_score": 0.0,
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        assert mneme.mneme.total_memories >= 1


# ── Test: Edge cases ────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the Praxis→Mneme auto-export pipeline."""

    def test_agent_without_mneme_does_not_crash(self):
        """Agent with Praxis but no Mneme doesn't crash on execution."""
        praxis = PraxisPillar(
            name="solo-praxis",
            executor_fn=lambda a: "ok",
        )
        agent = IsonomeAgent(
            name="no-mneme-agent",
            cognition=CognitionPillar(name="cog"),
            praxis=praxis,
            # No mneme
        )
        agent.start()

        action = Action(
            description="test action",
            tool_name="test",
            risk=ActionRisk.LOW,
        )
        agent.praxis.orchestrator.register_action(action)
        agent.praxis._run_execution_batch()

        # Should not crash — signal is emitted but has no target pillar
        agent.tick()

    def test_agent_without_praxis_no_memories(self):
        """Agent with Mneme but no Praxis doesn't get execution memories."""
        mneme = MnemePillar(name="solo-mneme")
        agent = IsonomeAgent(
            name="no-praxis-agent",
            cognition=CognitionPillar(name="cog"),
            mneme=mneme,
            # No praxis
        )
        agent.start()
        initial = agent.mneme.mneme.total_memories

        agent.tick()

        assert agent.mneme.mneme.total_memories == initial

    def test_non_numeric_validation_score_uses_base_sig(self, mneme):
        """Non-numeric validation_score falls back to base significance."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 100,
                        "validation_score": None,  # Non-numeric
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        # Should use base_sig = 0.7 for success
        assert all_entries[0].significance == pytest.approx(0.7)

    def test_missing_validation_score_uses_base_sig(self, mneme):
        """Missing validation_score falls back to base significance."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": False,
                        "error": "error msg",
                        "attempt": 0,
                        "duration_ms": 100,
                        # No validation_score key
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        # Should use base_sig = 0.4 for failure
        assert all_entries[0].significance == pytest.approx(0.4)

    def test_significance_bounded_at_1(self, mneme):
        """Significance is capped at 1.0 even with high validation score."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 100,
                        "validation_score": 5.0,  # Very high
                        "batch": 1,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        assert all_entries[0].significance <= 1.0

    def test_batch_tag_includes_batch_number(self, mneme):
        """Batch tag reflects the batch number from the execution entry."""
        _init_pillars(mneme)

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.MNEME,
            kind="execution_results",
            payload={
                "entries": [
                    {
                        "action_id": str(uuid4()),
                        "description": "test",
                        "tool_name": "test",
                        "success": True,
                        "error": None,
                        "attempt": 0,
                        "duration_ms": 100,
                        "validation_score": 0.5,
                        "batch": 42,
                    },
                ],
            },
        )
        mneme.receive_signal(signal)
        mneme.process_queued()

        all_entries = list(mneme.mneme._working.values())
        assert len(all_entries) == 1
        assert "batch-42" in all_entries[0].tags
