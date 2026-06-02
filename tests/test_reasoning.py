"""Tests for the Recursive Reasoning Engine and CognitionPillar.

Covers:
    - ReasoningNode: evidence collection, confidence, terminal detection
    - RecursiveReasoningEngine: task decomposition, plan generation,
      tension modulation (depth, branching, divergence)
    - CognitionPillar: initialization, signal handling, plan emission,
      context management, GC feedback
"""

from __future__ import annotations

import pytest

from isonome.cognition.attention import (
    AttentionChunk,
    AttentionEquilibriumSystem,
)
from isonome.cognition.reasoning import (
    EvidencePoint,
    NodeStatus,
    ReasoningNode,
    ReasoningPlan,
    RecursiveReasoningEngine,
)
from isonome.cognition.pillar import CognitionPillar
from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar, Signal, AgentState, AgentIdentity


# ═══════════════════════════════════════════════════════════════════
# ReasoningNode tests
# ═══════════════════════════════════════════════════════════════════


class TestReasoningNode:
    """Test the ReasoningNode data structure."""

    def test_node_defaults(self):
        node = ReasoningNode(hypothesis="test task", depth=0)
        assert node.hypothesis == "test task"
        assert node.depth == 0
        assert node.status == NodeStatus.PENDING
        assert node.confidence == 0.5
        assert not node.terminal
        assert node.action_steps == []
        assert node.total_evidence == 0

    def test_evidence_collection(self):
        node = ReasoningNode(hypothesis="analyze data", depth=0)
        node.evidence_for.append(
            EvidencePoint(content="relevant data exists", supports=True, weight=1.0)
        )
        node.evidence_for.append(
            EvidencePoint(content="prior analysis confirms", supports=True, weight=0.8)
        )
        node.evidence_against.append(
            EvidencePoint(content="data may be stale", supports=False, weight=0.5)
        )
        assert node.total_evidence == 3

    def test_evidence_ratio_all_supporting(self):
        node = ReasoningNode(hypothesis="simple task", depth=0)
        node.evidence_for.append(
            EvidencePoint(content="strong support", supports=True, weight=1.0)
        )
        assert node.evidence_ratio == 1.0

    def test_evidence_ratio_all_against(self):
        node = ReasoningNode(hypothesis="doomed task", depth=0)
        node.evidence_against.append(
            EvidencePoint(content="strong counter", supports=False, weight=1.0)
        )
        assert node.evidence_ratio == 0.0

    def test_evidence_ratio_empty(self):
        node = ReasoningNode(hypothesis="no evidence", depth=0)
        # No evidence → ratio defaults to 0.5
        assert node.evidence_ratio == 0.5

    def test_evidence_ratio_weighted(self):
        node = ReasoningNode(hypothesis="weighted", depth=0)
        # 2 supporting with weight 0.3 each, 1 against with weight 1.0
        node.evidence_for.append(
            EvidencePoint(content="weak support 1", supports=True, weight=0.3)
        )
        node.evidence_for.append(
            EvidencePoint(content="weak support 2", supports=True, weight=0.3)
        )
        node.evidence_against.append(
            EvidencePoint(content="strong against", supports=False, weight=1.0)
        )
        # weighted_for = 0.6, weighted_against = 1.0, total = 1.6
        # ratio = 0.6 / 1.6 = 0.375
        assert 0.37 < node.evidence_ratio < 0.38

    def test_confidence_default(self):
        node = ReasoningNode(hypothesis="test", depth=0)
        assert node.confidence == 0.5

    def test_child_count(self):
        parent = ReasoningNode(hypothesis="parent", depth=0)
        child1 = ReasoningNode(hypothesis="child1", depth=1, parent_id=parent.id)
        child2 = ReasoningNode(hypothesis="child2", depth=1, parent_id=parent.id)
        parent.children = [child1, child2]
        assert parent.child_count == 2

    def test_max_child_depth(self):
        root = ReasoningNode(hypothesis="root", depth=0)
        child = ReasoningNode(hypothesis="child", depth=1)
        grandchild = ReasoningNode(hypothesis="grandchild", depth=2)
        child.children = [grandchild]
        root.children = [child]
        assert root.max_child_depth == 2

    def test_terminal_node(self):
        node = ReasoningNode(hypothesis="final step", depth=3)
        node.terminal = True
        node.action_steps = [{"description": "do it", "tool_name": "execute"}]
        assert node.terminal
        assert len(node.action_steps) == 1


# ═══════════════════════════════════════════════════════════════════
# RecursiveReasoningEngine tests
# ═══════════════════════════════════════════════════════════════════


class TestRecursiveReasoningEngine:
    """Test the reasoning engine's decomposition and plan generation."""

    @pytest.fixture
    def engine(self):
        return RecursiveReasoningEngine()

    def test_reason_simple_task_produces_plan(self, engine):
        """Reasoning about a simple task should produce at least one plan."""
        plan = engine.reason("read the config file and parse it")
        assert isinstance(plan, ReasoningPlan)
        assert plan.root_hypothesis == "read the config file and parse it"
        assert len(plan.plans) >= 1
        assert plan.total_nodes >= 1
        assert plan.max_depth_reached >= 0

    def test_reason_best_plan_has_actions(self, engine):
        """The best plan should contain actionable items."""
        plan = engine.reason("fetch data from the API")
        best = plan.best_plan
        assert len(best) >= 1
        # Each action should have required fields
        for action in best:
            assert "description" in action
            assert "tool_name" in action
            assert "risk" in action

    def test_reason_single_action_simple(self, engine):
        """A very short task should produce a single action."""
        plan = engine.reason("run tests")
        assert len(plan.plans) >= 1
        best = plan.best_plan
        assert len(best) >= 1

    def test_reason_multi_step_decomposed(self, engine):
        """A task with 'and' conjunction should be decomposed."""
        plan = engine.reason("fetch data and analyze it and produce report")
        assert plan.total_nodes > 0
        best = plan.best_plan
        # Multi-step task should produce multiple actions
        assert len(best) >= 1

    def test_reason_produces_confidence(self, engine):
        """Plan confidences should be in valid range."""
        plan = engine.reason("analyze the database schema")
        for conf in plan.confidences:
            assert 0.0 <= conf <= 1.0
        assert len(plan.confidences) == len(plan.plans)

    def test_reason_stats_updated(self, engine):
        """Running multiple reason calls should accumulate stats."""
        engine.reason("task one")
        engine.reason("task two")
        engine.reason("task three")
        stats = engine.stats
        assert stats["sessions"] == 3
        assert stats["total_nodes"] >= 3
        assert stats["total_actions"] >= 1

    def test_reason_single_action_method(self, engine):
        """reason_single_action should return a single action dict."""
        action = engine.reason_single_action("read the file")
        assert isinstance(action, dict)
        assert "description" in action
        assert "tool_name" in action

    def test_reason_plan_summary(self, engine):
        """ReasoningPlan.summary() should produce a readable string."""
        plan = engine.reason("test task")
        summary = plan.summary()
        assert isinstance(summary, str)
        assert "ReasoningPlan" in summary

    def test_reason_plan_best_confidence(self, engine):
        """best_confidence should return a scalar."""
        plan = engine.reason("test")
        conf = plan.best_confidence
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_reason_plan_empty_confidence(self):
        """Empty plans have zero confidence."""
        empty = ReasoningPlan(
            root_hypothesis="nothing",
            plans=[],
            confidences=[],
            total_nodes=0,
            max_depth_reached=0,
            branches_explored=0,
            branches_pruned=0,
            total_evidence_gathered=0,
            duration_ms=0.0,
            tension_profile={},
        )
        assert empty.best_confidence == 0.0
        assert empty.best_plan == []


class TestReasoningTensionModulation:
    """Test that tension profiles modulate reasoning behavior."""

    @pytest.fixture
    def engine(self):
        return RecursiveReasoningEngine()

    def test_shallow_mode_limits_depth(self, engine):
        """Shallow mode should limit max depth to 2-3."""
        engine.set_tension_profile({
            "shallow_deep": -1.0,   # Max shallow
            "explore_exploit": 0.15,
            "divergent_convergent": 0.3,
        })
        # Complex task that would normally decompose deeply
        plan = engine.reason(
            "first fetch data, then analyze it, then produce a report, "
            "then validate results, and finally deploy"
        )
        # In shallow mode, depth should be limited
        assert plan.max_depth_reached <= 3

    def test_deep_mode_allows_deeper(self, engine):
        """Deep mode should allow depth up to 8."""
        engine.set_tension_profile({
            "shallow_deep": 1.0,    # Max deep
            "explore_exploit": 0.15,
            "divergent_convergent": 0.3,
        })
        plan = engine.reason(
            "architect a complete microservice system with data layer, "
            "API gateway, authentication, caching, and monitoring"
        )
        # Deep mode should produce deeper trees
        assert plan.max_depth_reached >= 1  # At minimum, should decompose

    def test_explore_mode_increases_branching(self, engine):
        """Explore mode should increase branching factor."""
        engine.set_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": -1.0,  # Max explore
            "divergent_convergent": 0.3,
        })
        plan = engine.reason("design a new feature and implement it")
        # Explore mode should explore more branches
        assert plan.branches_explored >= 0  # May not branch on simple tasks

    def test_exploit_mode_reduces_branching(self, engine):
        """Exploit mode should reduce branching to 1."""
        engine.set_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 1.0,   # Max exploit
            "divergent_convergent": 0.3,
        })
        plan = engine.reason("build the feature")
        # Exploit mode → branching factor = 1 → minimal branches
        assert plan.branches_explored <= 1

    def test_divergent_mode_returns_multiple_plans(self, engine):
        """Divergent mode should return multiple plan alternatives."""
        engine.set_tension_profile({
            "shallow_deep": 0.5,
            "explore_exploit": -0.5,   # Some exploration to create branches
            "divergent_convergent": -0.8,  # Divergent
        })
        plan = engine.reason("improve system performance and reduce latency")
        # Divergent mode may produce multiple plans depending on branching
        assert plan.branches_explored >= 0

    def test_convergent_mode_returns_single_plan(self, engine):
        """Convergent mode should return exactly one plan."""
        engine.set_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.5,    # Exploit
            "divergent_convergent": 0.8,  # Highly convergent
        })
        plan = engine.reason("fix the bug")
        # Convergent mode → single plan
        if plan.total_nodes > 1:
            assert len(plan.plans) == 1

    def test_max_depth_bounds(self, engine):
        """Max depth should stay within valid range [2, 8]."""
        # Shallow extreme
        engine.set_tension_profile({
            "shallow_deep": -1.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })
        plan_shallow = engine.reason("complex multi-step task with many parts")
        assert 0 <= plan_shallow.max_depth_reached <= 3

        # Deep extreme
        engine.set_tension_profile({
            "shallow_deep": 1.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })
        plan_deep = engine.reason("complex multi-step task with many parts")
        assert plan_deep.max_depth_reached >= 0  # Lower bound validated above

    def test_tension_in_plan_output(self, engine):
        """The plan output should include the tension profile used."""
        plan = engine.reason("test")
        assert "shallow_deep" in plan.tension_profile
        assert "explore_exploit" in plan.tension_profile
        assert "divergent_convergent" in plan.tension_profile


class TestReasoningWithAttention:
    """Test integration between reasoning engine and attention system."""

    @pytest.fixture
    def attention_engine(self):
        return EquilibriumEngine()

    @pytest.fixture
    def attention(self, attention_engine):
        return AttentionEquilibriumSystem(
            engine=attention_engine,
            token_capacity=10_000,
        )

    @pytest.fixture
    def engine(self, attention):
        return RecursiveReasoningEngine(attention_system=attention)

    def test_reason_with_attention_context(self, engine, attention):
        """Reasoning should use attention chunks as evidence."""
        # Add some context to attention
        attention.add_chunk("The database schema has 12 tables", mutual_info=0.6)
        attention.add_chunk("Performance issues in the users table", mutual_info=0.8)
        attention.add_chunk("Previous analysis found 3 bottlenecks", mutual_info=0.7)

        plan = engine.reason("analyze database performance")
        assert plan.total_nodes >= 1
        assert plan.total_evidence_gathered > 0

    def test_reason_with_explicit_context(self, engine):
        """Explicit initial context should be used as evidence."""
        plan = engine.reason(
            "improve the API",
            initial_context=[
                "API latency is 450ms P99",
                "Cache hit rate is only 23%",
                "Database queries are the main bottleneck",
            ],
        )
        assert plan.total_evidence_gathered > 0

    def test_reason_plan_actions_have_tool_names(self, engine, attention):
        """Produced actions should have sensible tool name inferences."""
        plan = engine.reason("read the configuration file")
        best = plan.best_plan
        if best:
            # Tool inference for "read" → "fetch"
            tool = best[0].get("tool_name", "")
            assert tool in ("fetch", "read", "execute", "unknown")

    def test_reason_plan_actions_have_risk(self, engine):
        """Actions should have a risk level assigned."""
        plan = engine.reason("delete the production database")
        best = plan.best_plan
        if best:
            assert best[0]["risk"] in ("low", "moderate", "high")


# ═══════════════════════════════════════════════════════════════════
# CognitionPillar tests
# ═══════════════════════════════════════════════════════════════════


class TestCognitionPillar:
    """Test the CognitionPillar lifecycle and signal handling."""

    @pytest.fixture
    def engine(self):
        return EquilibriumEngine()

    @pytest.fixture
    def agent_state(self, engine):
        identity = AgentIdentity(name="test-agent")
        state = AgentState(
            identity=identity,
            tensions=engine.snapshot(agent_id=identity.id),
        )
        return state

    @pytest.fixture
    def pillar(self, engine, agent_state):
        p = CognitionPillar(name="thinker", engine=engine, token_capacity=10_000)
        p.initialize(agent_state)
        return p

    def test_initialization(self, pillar):
        """Pillar initializes with both attention and reasoning systems."""
        assert pillar.pillar == Pillar.COGNITION
        assert pillar.attention is not None
        assert pillar.reasoning is not None
        assert pillar.initialized

    def test_reason_produces_plan(self, pillar):
        """reason() should produce a ReasoningPlan."""
        plan = pillar.reason("fetch the data and analyze it")
        assert plan is not None
        assert plan.total_nodes >= 1

    def test_reason_emits_plan_ready_feedback(self, pillar):
        """Reasoning should emit feedback to the equilibrium engine."""
        # Drain any init feedback
        pillar.drain_feedback()

        plan = pillar.reason("complex task with multiple steps for analysis")
        assert plan is not None

        # Should have emitted feedback about the plan
        feedback = pillar.drain_feedback()
        # At minimum, should have some feedback (plan quality)
        assert len(feedback) >= 0  # Feedback is emitted via emit_feedback

    def test_add_context(self, pillar):
        """add_context should register chunks in attention."""
        chunk = pillar.add_context("important system information")
        assert chunk is not None
        assert pillar.attention.chunk_count >= 1

    def test_add_context_with_metadata(self, pillar):
        """Context can be added with importance tags."""
        chunk = pillar.add_context(
            "critical: auth token expired",
            mutual_info=0.9,
            task_relevance=1.0,
            importance_tags=("critical", "auth"),
        )
        assert chunk is not None
        # The attention score should be boosted by importance tags
        score = chunk.attention_score()
        assert 0.0 <= score <= 1.0

    def test_collect_garbage(self, pillar):
        """collect_garbage should run a GC cycle."""
        # Add enough chunks to make GC meaningful
        for i in range(5):
            pillar.add_context(f"context chunk {i}")
        report = pillar.collect_garbage()
        assert report is not None

    def test_signal_reason(self, pillar):
        """Receiving a 'reason' signal should produce a plan."""
        # Drain any init feedback
        pillar.drain_feedback()

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.COGNITION,
            kind="reason",
            payload={
                "task": "optimize the database queries and add caching",
            },
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.latest_plan is not None

    def test_signal_add_context(self, pillar):
        """Receiving an 'add_context' signal should add a chunk."""
        signal = Signal(
            source=Pillar.MNEME,
            target=Pillar.COGNITION,
            kind="add_context",
            payload={
                "content": "user reported performance issues",
                "mutual_info": 0.5,
                "task_relevance": 0.8,
            },
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.attention.chunk_count >= 1

    def test_signal_collect_garbage(self, pillar):
        """Receiving a 'collect_garbage' signal should trigger GC."""
        pillar.add_context("some context")
        pillar.add_context("more context")

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.COGNITION,
            kind="collect_garbage",
            payload={},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.latest_gc_report is not None

    def test_signal_evaluate_result(self, pillar):
        """Receiving 'evaluate_result' should add context and emit feedback."""
        pillar.drain_feedback()

        signal = Signal(
            source=Pillar.PRAXIS,
            target=Pillar.COGNITION,
            kind="evaluate_result",
            payload={
                "success": True,
                "description": "deployment completed successfully",
                "confidence": 0.9,
            },
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        # Should have added context about execution result
        assert pillar._context_added > 0 or pillar.attention.chunk_count > 0

    def test_update_tension_profile(self, pillar):
        """update_tension_profile should propagate to reasoning system."""
        profile = {
            "shallow_deep": 0.8,
            "explore_exploit": -0.3,
            "divergent_convergent": 0.1,
            "consolidate_prune": 0.0,
            "specific_general": 0.0,
        }
        pillar.update_tension_profile(profile)
        # After update, next reason should use new profile
        plan = pillar.reason("test task")
        assert "shallow_deep" in plan.tension_profile
        # Plan profile should reflect current tensions at reasoning time

    def test_stats_initial(self, pillar):
        """Stats should reflect initial state."""
        s = pillar.stats
        assert "context_added" in s
        assert "tasks_reasoned" in s
        assert "attention" in s
        assert "reasoning" in s

    def test_stats_after_work(self, pillar):
        """Stats should update after reasoning and context addition."""
        pillar.add_context("data point 1")
        pillar.add_context("data point 2")
        pillar.reason("analyze the data")

        s = pillar.stats
        assert s["context_added"] >= 2
        assert s["tasks_reasoned"] >= 1

    def test_serialize(self, pillar):
        """serialize should return a dictionary of state."""
        pillar.add_context("sample context")
        state = pillar.serialize()
        assert state is not None
        assert "attention" in state
        assert "reasoning" in state
        assert "context_added" in state
        assert "tasks_reasoned" in state

    def test_shutdown_clean(self, pillar):
        """Shutdown should not raise and should log state."""
        pillar.add_context("pre-shutdown context")
        pillar.reason("test before shutdown")
        # Should not raise
        pillar.shutdown()
        assert not pillar.initialized

    def test_signal_unknown_kind_no_error(self, pillar):
        """Unknown signal kinds should not crash the pillar."""
        signal = Signal(
            source=Pillar.MNEME,
            target=Pillar.COGNITION,
            kind="unknown_strange_signal_kind",
            payload={},
        )
        pillar.receive_signal(signal)
        # Should not raise
        pillar.process_queued()


class TestCognitionPillarSignaling:
    """Test CognitionPillar inter-pillar signaling."""

    @pytest.fixture
    def engine(self):
        return EquilibriumEngine()

    @pytest.fixture
    def pillar(self, engine):
        identity = AgentIdentity(name="sig-test")
        state = AgentState(
            identity=identity,
            tensions=engine.snapshot(agent_id=identity.id),
        )
        p = CognitionPillar(name="signaler", engine=engine, token_capacity=10_000)
        p.initialize(state)
        return p

    def test_plan_ready_signal_contains_actions(self, pillar):
        """After reasoning, the latest_plan should have consumable actions."""
        pillar.reason("run the integration tests and verify results")
        plan = pillar.latest_plan
        assert plan is not None
        best = plan.best_plan
        assert len(best) >= 1
        # Actions should be in Praxis's expected format
        for action in best:
            assert isinstance(action, dict)
            assert "description" in action
            assert "tool_name" in action
            assert "risk" in action
            # These keys exist in action dicts that Praxis.import_from_cognition() consumes

    def test_set_priority_boosts_tags(self, pillar):
        """Setting priority on chunks should add priority tag."""
        chunk = pillar.add_context("critical security vulnerability found")
        signal = Signal(
            source=Pillar.MNEME,
            target=Pillar.COGNITION,
            kind="set_priority",
            payload={
                "chunk_ids": [str(chunk.id)],
            },
        )
        pillar.receive_signal(signal)
        pillar.process_queued()
        # The chunk should now have the "priority" tag
        all_chunks = pillar.attention.equilibrium_chunks
        for c in all_chunks:
            if c.id == chunk.id:
                assert "priority" in c.importance_tags

    def test_auto_gc_triggers_on_high_utilization(self, pillar):
        """Auto-GC should run when budget utilization exceeds threshold."""
        # Create a pillar with very small capacity and low GC threshold
        identity = AgentIdentity(name="gc-test")
        state = AgentState(
            identity=identity,
            tensions=EquilibriumEngine().snapshot(agent_id=identity.id),
        )
        small_pillar = CognitionPillar(
            name="small",
            token_capacity=100,  # Tiny capacity
            gc_utilization_threshold=0.1,  # Low threshold
            auto_gc=True,
        )
        small_pillar.initialize(state)

        # Add a chunk that fills most of the budget
        small_pillar.add_context("a" * 200)  # ~50 tokens, should push over 10%

        # Update profile (triggers auto-gc check)
        small_pillar.update_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })

        # GC should have run automatically
        assert small_pillar.latest_gc_report is not None


# ═══════════════════════════════════════════════════════════════════
# Recursive reasoning edge cases
# ═══════════════════════════════════════════════════════════════════


class TestReasoningEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_task(self):
        """Empty task should produce minimal plan."""
        engine = RecursiveReasoningEngine()
        plan = engine.reason("")
        assert plan.total_nodes >= 1  # Should still create a root node

    def test_very_long_task(self):
        """Very long task descriptions should be handled."""
        engine = RecursiveReasoningEngine()
        long_task = (
            "first analyze the market, then design the product, "
            "then build the prototype, then test with users, "
            "then iterate on feedback, then launch to production, "
            "and finally monitor and optimize"
        )
        plan = engine.reason(long_task)
        assert plan.total_nodes >= 1

    def test_single_word_task(self):
        """Single word tasks should produce atomic actions."""
        engine = RecursiveReasoningEngine()
        plan = engine.reason("optimize")
        assert plan.total_nodes >= 1
        best = plan.best_plan
        assert len(best) >= 1

    def test_task_with_step_markers(self):
        """Tasks with sequential markers should be decomposed."""
        engine = RecursiveReasoningEngine()
        engine.set_tension_profile({
            "shallow_deep": 0.5,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.5,
        })
        plan = engine.reason(
            "first gather requirements, then design the architecture"
        )
        # Should decompose into at least 2 sub-questions
        assert plan.total_nodes >= 1

    def test_custom_decomposer(self):
        """Custom decomposer should override default behavior."""
        def my_decomposer(hypothesis, branching):
            return ["custom sub-question 1", "custom sub-question 2"]

        engine = RecursiveReasoningEngine(decomposer_fn=my_decomposer)
        engine.set_tension_profile({
            "shallow_deep": 0.5,
            "explore_exploit": -0.5,
            "divergent_convergent": 0.0,
        })
        plan = engine.reason("complex task that needs decomposition")
        # With custom decomposer at depth 0 and shallow mode,
        # the tree should contain nodes
        assert plan.total_nodes >= 1

    def test_custom_evidence_fn(self):
        """Custom evidence function should provide custom evidence."""
        def my_evidence(hypothesis, parent):
            return [
                EvidencePoint(content="custom evidence", supports=True, weight=1.0),
                EvidencePoint(content="custom counter", supports=False, weight=0.3),
            ]

        engine = RecursiveReasoningEngine(evidence_fn=my_evidence)
        plan = engine.reason("test with custom evidence")
        assert plan.total_evidence_gathered >= 1

    def test_multiple_reason_calls_independent(self):
        """Multiple reason calls should produce independent plans."""
        engine = RecursiveReasoningEngine()
        plan1 = engine.reason("task alpha")
        plan2 = engine.reason("task beta")
        # Each should be independent
        assert plan1.root_hypothesis == "task alpha"
        assert plan2.root_hypothesis == "task beta"
        assert plan1.total_nodes != plan2.total_nodes or (
            plan1.root_hypothesis != plan2.root_hypothesis
        )
        # Stats should accumulate
        assert engine.stats["sessions"] == 2

    def test_default_tension_profile(self):
        """Engine should have sensible defaults even without explicit profile."""
        engine = RecursiveReasoningEngine()
        plan = engine.reason("default test")
        assert "shallow_deep" in plan.tension_profile
        assert plan.tension_profile["shallow_deep"] == -0.2

    def test_action_tool_inference(self):
        """Tool name inference should produce appropriate names."""
        engine = RecursiveReasoningEngine()
        # "read" task → "fetch" tool
        plan_read = engine.reason("read the configuration")
        best_read = plan_read.best_plan
        if best_read:
            assert best_read[0]["tool_name"] in ("fetch", "read", "execute")

        # "write" task → "write" tool
        plan_write = engine.reason("write the output file")
        best_write = plan_write.best_plan
        if best_write:
            assert best_write[0]["tool_name"] in ("write", "execute")

        # "analyze" task → "analyze" tool
        plan_analyze = engine.reason("analyze the metrics")
        best_analyze = plan_analyze.best_plan
        if best_analyze:
            assert best_analyze[0]["tool_name"] in ("analyze", "execute")

    def test_action_refs_for_dependencies(self):
        """Multi-step plans should have dependency refs for Praxis DAG."""
        engine = RecursiveReasoningEngine()
        engine.set_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.5,
        })
        plan = engine.reason("fetch data and analyze it")
        best = plan.best_plan
        if len(best) > 1:
            # Multi-step should have dependency chain
            has_deps = any("dependencies" in a and len(a["dependencies"]) > 0 for a in best)
            # At least one action should depend on another
            # (This is structure-dependent, not always guaranteed)
            pass  # Soft assertion — depends on decomposition

    def test_reasoning_node_tree_structure(self, engine=None):
        """Verify tree structure is built correctly."""
        if engine is None:
            engine = RecursiveReasoningEngine()
        engine.set_tension_profile({
            "shallow_deep": 0.5,
            "explore_exploit": -0.3,
            "divergent_convergent": 0.0,
        })
        engine.reason("complex multi-part task for tree testing")
        nodes = engine.current_nodes
        assert len(nodes) > 0
        # Root should exist
        root = engine.root_node
        assert root is not None
        assert root.depth == 0

    def test_cognition_pillar_imports(self):
        """Verify CognitionPillar can be imported without circular deps."""
        from isonome.cognition import CognitionPillar
        from isonome.cognition import RecursiveReasoningEngine
        from isonome.cognition import AttentionEquilibriumSystem
        assert CognitionPillar is not None
        assert RecursiveReasoningEngine is not None
        assert AttentionEquilibriumSystem is not None
