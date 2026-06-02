"""Tests for the Action Orchestrator (Praxis pillar).

Covers:
    - Action: creation, dependency checking, retry policy
    - RetryPolicy: exponential backoff calculation
    - ActionOrchestrator: registration, DAG levels, execution, safety gating,
      parallelism, validation, retry, serialization round-trip
    - ActionOrchestrator: import_from_cognition, export_to_mneme
    - PraxisPillar: pillar lifecycle integration, signal handling, feedback emission
"""

import time
from uuid import UUID, uuid4

import pytest

from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionReport,
    ExecutionResult,
    RetryPolicy,
)
from isonome.praxis.pillar import PraxisPillar
from isonome.types import AgentIdentity, AgentState, Feedback, Pillar, Signal


# ═══════════════════════════════════════════════════════════════════
# RetryPolicy tests
# ═══════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    def test_default_policy(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.backoff_factor == 2.0

    def test_delay_attempt_zero(self):
        policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0)
        assert policy.delay_for_attempt(0) == 1.0  # 1 * 2^0 = 1

    def test_delay_attempt_one(self):
        policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0)
        assert policy.delay_for_attempt(1) == 2.0  # 1 * 2^1 = 2

    def test_delay_attempt_three(self):
        policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0)
        assert policy.delay_for_attempt(3) == 8.0  # 1 * 2^3 = 8

    def test_delay_capped_at_max(self):
        policy = RetryPolicy(
            base_delay=1.0,
            backoff_factor=10.0,
            max_delay=5.0,
        )
        # 1 * 10^1 = 10, capped at 5
        assert policy.delay_for_attempt(1) == 5.0

    def test_delay_negative_attempt_zero(self):
        policy = RetryPolicy()
        assert policy.delay_for_attempt(-1) == 0.0

    def test_custom_policy(self):
        policy = RetryPolicy(max_retries=5, base_delay=0.5, backoff_factor=3.0)
        assert policy.max_retries == 5
        assert policy.delay_for_attempt(2) == pytest.approx(4.5)  # 0.5 * 9


# ═══════════════════════════════════════════════════════════════════
# Action tests
# ═══════════════════════════════════════════════════════════════════


class TestAction:
    def test_creation_defaults(self):
        action = Action(description="test", tool_name="echo")
        assert action.description == "test"
        assert action.tool_name == "echo"
        assert action.risk == ActionRisk.LOW
        assert action.dependency_count() == 0
        assert action.params == {}
        assert action.tags == ()

    def test_with_dependencies(self):
        dep1, dep2 = uuid4(), uuid4()
        action = Action(
            description="dependent",
            tool_name="write",
            dependencies=(dep1, dep2),
        )
        assert action.dependency_count() == 2
        assert dep1 in action.dependencies
        assert not action.is_ready(frozenset())

    def test_is_ready_all_deps_completed(self):
        dep1, dep2 = uuid4(), uuid4()
        action = Action(
            description="test",
            tool_name="test",
            dependencies=(dep1, dep2),
        )
        assert action.is_ready(frozenset({dep1, dep2}))
        assert not action.is_ready(frozenset({dep1}))

    def test_no_deps_always_ready(self):
        action = Action(description="test", tool_name="test")
        assert action.is_ready(frozenset())
        assert action.is_ready(frozenset({uuid4()}))

    def test_risk_levels(self):
        trivial = Action(
            description="read", tool_name="read", risk=ActionRisk.TRIVIAL
        )
        critical = Action(
            description="delete", tool_name="rm", risk=ActionRisk.CRITICAL
        )
        assert trivial.risk == ActionRisk.TRIVIAL
        assert critical.risk == ActionRisk.CRITICAL

    def test_with_retry_policy(self):
        policy = RetryPolicy(max_retries=2, base_delay=0.5)
        action = Action(
            description="flaky",
            tool_name="flaky",
            retry_policy=policy,
        )
        assert action.retry_policy.max_retries == 2

    def test_with_tags(self):
        action = Action(
            description="tagged",
            tool_name="tagged",
            tags=("critical", "deploy"),
        )
        assert "deploy" in action.tags

    def test_with_metadata(self):
        action = Action(
            description="meta",
            tool_name="meta",
            metadata={"priority": 10, "owner": "cognition"},
        )
        assert action.metadata["priority"] == 10


# ═══════════════════════════════════════════════════════════════════
# ExecutionResult tests
# ═══════════════════════════════════════════════════════════════════


class TestExecutionResult:
    def test_success_result(self):
        result = ExecutionResult(
            action_id=uuid4(),
            success=True,
            output="done",
            duration_ms=42.0,
            validation_passed=True,
            validation_score=0.9,
            attempt=0,
        )
        assert result.success
        assert result.output == "done"
        assert result.duration_ms == pytest.approx(42.0)
        assert result.validation_passed

    def test_failure_result(self):
        result = ExecutionResult(
            action_id=uuid4(),
            success=False,
            error="timeout",
            attempt=3,
        )
        assert not result.success
        assert result.error == "timeout"
        assert result.attempt == 3


# ═══════════════════════════════════════════════════════════════════
# ExecutionReport tests
# ═══════════════════════════════════════════════════════════════════


class TestExecutionReport:
    def test_report_fields(self):
        report = ExecutionReport(
            actions_total=5,
            actions_completed=4,
            actions_failed=1,
            actions_blocked=0,
            actions_retried=1,
            total_duration_ms=150.0,
            success_rate=0.8,
            avg_validation_score=0.75,
            parallelism_level=2,
            gate_blocks=0,
            tension_profile={"autonomy_safety": -0.4},
        )
        assert report.actions_total == 5
        assert report.success_rate == 0.8
        assert report.parallelism_level == 2
        assert report.gate_blocks == 0


# ═══════════════════════════════════════════════════════════════════
# ActionOrchestrator tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorRegistration:
    def test_register_single_action(self):
        orch = ActionOrchestrator()
        aid = orch.register_action(
            Action(description="test", tool_name="echo")
        )
        assert isinstance(aid, UUID)
        assert orch.total_actions == 1
        assert orch.dag_depth == 0

    def test_register_batch(self):
        orch = ActionOrchestrator()
        actions = [
            Action(description=f"task_{i}", tool_name="echo")
            for i in range(5)
        ]
        ids = orch.register_batch(actions)
        assert len(ids) == 5
        assert orch.total_actions == 5

    def test_dag_depth_with_dependencies(self):
        orch = ActionOrchestrator()
        # A → B → C
        a = Action(description="A", tool_name="a")
        b = Action(description="B", tool_name="b", dependencies=(a.id,))
        c = Action(description="C", tool_name="c", dependencies=(b.id,))
        orch.register_batch([a, b, c])
        assert orch.dag_depth == 2

    def test_dag_depth_independent(self):
        orch = ActionOrchestrator()
        # Three independent actions at level 0
        actions = [
            Action(description=f"ind_{i}", tool_name="echo")
            for i in range(3)
        ]
        orch.register_batch(actions)
        assert orch.dag_depth == 0

    def test_dag_depth_diamond(self):
        orch = ActionOrchestrator()
        a = Action(description="A", tool_name="a")
        b = Action(description="B", tool_name="b", dependencies=(a.id,))
        c = Action(description="C", tool_name="c", dependencies=(a.id,))
        d = Action(description="D", tool_name="d", dependencies=(b.id, c.id))
        orch.register_batch([a, b, c, d])
        # A(0) → B(1),C(1) → D(2)
        assert orch.dag_depth == 2


class TestOrchestratorExecution:
    def _echo_executor(self, action: Action):
        """Simple executor that returns the action params."""
        return {"echo": action.description, "params": action.params}

    def test_execute_single_action(self):
        orch = ActionOrchestrator()
        orch.register_action(
            Action(description="hello world", tool_name="echo")
        )
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.actions_total == 1
        assert report.actions_completed == 1
        assert report.success_rate == 1.0
        assert orch.total_actions == 1
        assert len(orch.completed_actions) == 1

    def test_execute_multiple_independent(self):
        orch = ActionOrchestrator()
        orch.register_batch([
            Action(description=f"task_{i}", tool_name="echo")
            for i in range(5)
        ])
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.actions_completed == 5
        assert report.success_rate == 1.0

    def test_execute_respects_dependencies(self):
        orch = ActionOrchestrator()
        results_log = []

        def tracking_executor(action: Action):
            results_log.append(action.description)
            return {"ok": True}

        a = Action(description="first", tool_name="a")
        b = Action(description="second", tool_name="b", dependencies=(a.id,))
        orch.register_batch([a, b])

        report = orch.execute_batch(executor_fn=tracking_executor)
        assert report.actions_completed == 2
        # first must execute before second
        idx_a = results_log.index("first")
        idx_b = results_log.index("second")
        assert idx_a < idx_b

    def test_execute_with_failure(self):
        orch = ActionOrchestrator()
        call_count = {"count": 0}

        def failing_executor(action: Action):
            call_count["count"] += 1
            if action.description == "fail":
                raise RuntimeError("expected failure")
            return {"ok": True}

        orch.register_batch([
            Action(description="good", tool_name="a"),
            Action(
                description="fail",
                tool_name="b",
                retry_policy=RetryPolicy(max_retries=2, base_delay=0.01),
            ),
        ])
        report = orch.execute_batch(executor_fn=failing_executor)
        assert report.actions_completed == 1
        assert report.actions_failed == 1
        assert len(orch.completed_actions) == 1
        assert len(orch.failed_actions) == 1

    def test_execute_retry_succeeds(self):
        orch = ActionOrchestrator()
        call_count = {"count": 0}

        def flaky_executor(action: Action):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise RuntimeError("transient error")
            return {"ok": True}

        orch.register_action(
            Action(
                description="flaky",
                tool_name="flaky",
                retry_policy=RetryPolicy(max_retries=3, base_delay=0.01),
            )
        )
        report = orch.execute_batch(executor_fn=flaky_executor)
        assert report.actions_completed == 1
        assert report.actions_retried >= 1  # At least one retry needed

    def test_execute_retry_exhausted(self):
        orch = ActionOrchestrator()
        call_count = {"count": 0}

        def always_fails(action: Action):
            call_count["count"] += 1
            raise RuntimeError("permanent error")

        orch.register_action(
            Action(
                description="doomed",
                tool_name="doomed",
                retry_policy=RetryPolicy(max_retries=2, base_delay=0.01),
            )
        )
        report = orch.execute_batch(executor_fn=always_fails)
        assert report.actions_completed == 0
        assert report.actions_failed == 1
        assert call_count["count"] == 3  # 1 initial + 2 retries


class TestOrchestratorSafetyGating:
    def _echo_executor(self, action: Action):
        return {"echo": action.description}

    def test_trivial_action_not_blocked(self):
        """Even in safe mode, TRIVIAL actions should execute."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": -0.9})  # very safe
        orch.register_action(
            Action(
                description="read file",
                tool_name="read",
                risk=ActionRisk.TRIVIAL,
            )
        )
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.gate_blocks == 0
        assert report.actions_completed == 1

    def test_critical_action_blocked_in_safe_mode(self):
        """CRITICAL actions should be blocked when in safe mode."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": -0.9})  # very safe
        orch.register_action(
            Action(
                description="delete database",
                tool_name="rm",
                risk=ActionRisk.CRITICAL,
            )
        )
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.gate_blocks == 1
        assert report.actions_completed == 0

    def test_critical_action_allowed_with_approval(self):
        """CRITICAL actions pass when explicitly approved."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": -0.9})  # very safe
        orch.register_action(
            Action(
                description="delete database",
                tool_name="rm",
                risk=ActionRisk.CRITICAL,
            )
        )
        report = orch.execute_batch(
            executor_fn=self._echo_executor,
            approve_fn=lambda a: True,  # approve everything
        )
        assert report.gate_blocks == 0
        assert report.actions_completed == 1

    def test_high_autonomy_executes_anything(self):
        """In highly autonomous mode, HIGH-risk actions execute freely."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": 0.9})  # very autonomous
        orch.register_action(
            Action(
                description="deploy to staging",
                tool_name="deploy",
                risk=ActionRisk.HIGH,  # HIGH passes at τ=0.95 (q=0.75 ≤ 0.95)
            )
        )
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.gate_blocks == 0
        assert report.actions_completed == 1

    def test_max_autonomy_executes_critical(self):
        """At max autonomy (1.0), even CRITICAL actions execute."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": 1.0})
        orch.register_action(
            Action(
                description="delete everything",
                tool_name="rm -rf",
                risk=ActionRisk.CRITICAL,
            )
        )
        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.gate_blocks == 0
        assert report.actions_completed == 1


class TestOrchestratorValidation:
    def _echo_executor(self, action: Action):
        return {"result": action.description}

    def test_verify_heavy_passes_validation(self):
        """In verify_heavy mode, validation must pass."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"verify_execute": -0.8})  # verify_heavy

        def validator(action, output):
            return (True, 0.95)  # Good validation

        orch.register_action(
            Action(description="validate me", tool_name="test")
        )
        report = orch.execute_batch(
            executor_fn=self._echo_executor,
            validator_fn=validator,
        )
        assert report.actions_completed == 1
        assert report.avg_validation_score > 0.8

    def test_verify_heavy_fails_bad_validation(self):
        """In verify_heavy mode, poor validation should cause failure."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"verify_execute": -0.8})  # verify_heavy

        def strict_validator(action, output):
            return (False, 0.1)  # Purposefully bad validation

        orch.register_action(
            Action(
                description="bad output",
                tool_name="test",
                retry_policy=RetryPolicy(max_retries=1, base_delay=0.01),
            )
        )
        report = orch.execute_batch(
            executor_fn=self._echo_executor,
            validator_fn=strict_validator,
        )
        assert report.actions_completed == 0

    def test_execute_fast_skips_validation(self):
        """In execute_fast mode, validation is skipped regardless of output."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"verify_execute": 0.8})  # execute_fast

        def harsh_validator(action, output):
            return (False, 0.0)  # Would fail if checked

        orch.register_action(
            Action(description="fast execution", tool_name="test")
        )
        report = orch.execute_batch(
            executor_fn=self._echo_executor,
            validator_fn=harsh_validator,
        )
        # In execute_fast mode, validation is skipped entirely
        assert report.actions_completed == 1


class TestOrchestratorParallelism:
    def _echo_executor(self, action: Action):
        return {"echo": action.description}

    def test_sequential_mode_one_at_a_time(self):
        """In sequential mode, parallelism level should be 1."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"sequential_parallel": -0.9})  # sequential

        for i in range(5):
            orch.register_action(
                Action(description=f"task_{i}", tool_name="echo")
            )

        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.parallelism_level == 1
        assert report.actions_completed == 5

    def test_parallel_mode_higher_concurrency(self):
        """In parallel mode, max_concurrent should be > 1."""
        orch = ActionOrchestrator()
        orch.set_tension_profile({"sequential_parallel": 0.9})  # parallel

        for i in range(5):
            orch.register_action(
                Action(description=f"task_{i}", tool_name="echo")
            )

        report = orch.execute_batch(executor_fn=self._echo_executor)
        assert report.parallelism_level > 1


class TestOrchestratorProperties:
    def test_pending_actions(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="pending", tool_name="echo"))
        assert len(orch.pending_actions) == 1

    def test_completed_actions_after_execution(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="done", tool_name="echo"))
        orch.execute_batch(executor_fn=lambda a: "ok")
        assert len(orch.completed_actions) == 1
        assert len(orch.pending_actions) == 0

    def test_blocked_actions(self):
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": -0.95})
        orch.register_action(
            Action(
                description="blocked",
                tool_name="rm",
                risk=ActionRisk.CRITICAL,
            )
        )
        orch.execute_batch(executor_fn=lambda a: "ok")
        assert len(orch.blocked_actions) == 1

    def test_action_states(self):
        orch = ActionOrchestrator()
        aid = orch.register_action(
            Action(description="test", tool_name="echo")
        )
        states = orch.action_states
        assert states[aid] == ActionState.PENDING

    def test_total_actions(self):
        orch = ActionOrchestrator()
        assert orch.total_actions == 0
        orch.register_action(Action(description="1", tool_name="a"))
        orch.register_action(Action(description="2", tool_name="b"))
        assert orch.total_actions == 2

    def test_stats(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="s1", tool_name="a"))
        orch.register_action(Action(description="s2", tool_name="b"))
        orch.execute_batch(executor_fn=lambda a: "ok")
        stats = orch.stats
        assert stats["total_actions"] == 2
        assert stats["total_executed"] == 2
        assert stats["total_completed"] == 2
        assert stats["success_rate"] == 1.0

    def test_execution_log_after_execution(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="logged", tool_name="log"))
        orch.execute_batch(executor_fn=lambda a: "ok")
        log = orch.execution_log
        assert len(log) >= 1
        assert log[0]["description"] == "logged"
        assert log[0]["success"] is True


# ═══════════════════════════════════════════════════════════════════
# Import/Export (cross-pillar pipeline) tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorImportExport:
    def _echo_executor(self, action: Action):
        return {"echo": action.description}

    def test_import_from_cognition_single(self):
        orch = ActionOrchestrator()
        tasks = [{"description": "plan step 1", "tool_name": "write"}]
        ids = orch.import_from_cognition(tasks)
        assert len(ids) == 1
        assert orch.total_actions == 1

    def test_import_from_cognition_with_risk(self):
        orch = ActionOrchestrator()
        tasks = [
            {
                "description": "deploy production",
                "tool_name": "deploy",
                "risk": "CRITICAL",
            }
        ]
        ids = orch.import_from_cognition(tasks)
        action = orch.actions[0]
        assert action.risk == ActionRisk.CRITICAL

    def test_import_with_dependencies(self):
        orch = ActionOrchestrator()
        tasks = [
            {"description": "step A", "tool_name": "a", "ref": "a"},
            {"description": "step B", "tool_name": "b", "ref": "b"},
            {
                "description": "step C",
                "tool_name": "c",
                "ref": "c",
                "dependencies": ("a", "b"),
            },
        ]
        ids = orch.import_from_cognition(tasks)
        # Find action C
        c_action = None
        for aid in ids:
            a = orch.actions
            for act in a:
                if act.description == "step C":
                    c_action = act
        assert c_action is not None
        assert c_action.dependency_count() == 2

    def test_export_to_mneme_empty(self):
        orch = ActionOrchestrator()
        entries = orch.export_to_mneme()
        assert entries == []

    def test_export_to_mneme_after_execution(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="test", tool_name="test"))
        orch.execute_batch(executor_fn=self._echo_executor)
        entries = orch.export_to_mneme()
        assert len(entries) >= 1
        assert entries[0]["tool_name"] == "test"
        assert entries[0]["success"] is True


# ═══════════════════════════════════════════════════════════════════
# Serialization tests
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorSerialization:
    def _echo_executor(self, action: Action):
        return {"echo": action.description}

    def test_round_trip_empty(self):
        orch = ActionOrchestrator()
        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)
        assert restored.total_actions == 0

    def test_round_trip_with_actions(self):
        orch = ActionOrchestrator()
        a = Action(description="step 1", tool_name="a")
        b = Action(description="step 2", tool_name="b", dependencies=(a.id,))
        orch.register_batch([a, b])
        orch.execute_batch(executor_fn=self._echo_executor)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert restored.total_actions == 2
        assert restored.dag_depth == orch.dag_depth
        assert restored.stats["total_executed"] == orch.stats["total_executed"]

    def test_round_trip_preserves_risk_levels(self):
        orch = ActionOrchestrator()
        orch.register_action(
            Action(
                description="dangerous",
                tool_name="rm",
                risk=ActionRisk.CRITICAL,
            )
        )
        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)
        assert restored.actions[0].risk == ActionRisk.CRITICAL

    def test_round_trip_preserves_completed_state(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="done", tool_name="echo"))
        orch.execute_batch(executor_fn=self._echo_executor)

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        assert len(restored.completed_actions) == 1
        assert restored.actions[0].id in {
            a.id for a in restored.completed_actions
        }

    def test_serialization_has_expected_keys(self):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="test", tool_name="test"))
        data = orch.to_dict()
        assert "actions" in data
        assert "states" in data
        assert "completed" in data
        assert "stats" in data
        assert "execution_log" in data

    def test_from_dict_recomputes_levels(self):
        """After deserialization, topological levels should be recomputed."""
        orch = ActionOrchestrator()
        a = Action(description="A", tool_name="a")
        b = Action(description="B", tool_name="b", dependencies=(a.id,))
        c = Action(description="C", tool_name="c", dependencies=(b.id,))
        orch.register_batch([a, b, c])
        assert orch.dag_depth == 2

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)
        assert restored.dag_depth == 2


# ═══════════════════════════════════════════════════════════════════
# PraxisPillar integration tests
# ═══════════════════════════════════════════════════════════════════


class TestPraxisPillar:
    @pytest.fixture
    def agent_state(self):
        return AgentState(identity=AgentIdentity(name="test-agent"))

    @pytest.fixture
    def echo_executor(self):
        def executor(action: Action):
            return {"echo": action.description}
        return executor

    def test_initialization(self, agent_state):
        pillar = PraxisPillar(name="executor")
        pillar.initialize(agent_state)
        assert pillar.orchestrator is not None
        assert pillar.pillar == Pillar.PRAXIS
        assert pillar.initialized

    def test_import_plan_signal(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        tasks = [
            {
                "description": "deploy app",
                "tool_name": "deploy",
                "risk": "MODERATE",
            }
        ]
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="import_plan",
            payload={"tasks": tasks},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.orchestrator.total_actions == 1

    def test_execute_plan_signal(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        tasks = [
            {"description": "step 1", "tool_name": "step1"},
            {"description": "step 2", "tool_name": "step2"},
        ]
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="execute_plan",
            payload={"tasks": tasks},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.orchestrator.total_actions == 2
        assert len(pillar.orchestrator.completed_actions) == 2

    def test_execute_pending_signal(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        # Register actions first
        pillar.orchestrator.register_action(
            Action(description="pending 1", tool_name="a")
        )
        pillar.orchestrator.register_action(
            Action(description="pending 2", tool_name="b")
        )

        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="execute_pending",
            payload={},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert len(pillar.orchestrator.completed_actions) == 2

    def test_execution_emits_feedback(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        tasks = [{"description": "simple task", "tool_name": "simple"}]
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="execute_plan",
            payload={"tasks": tasks},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        feedback = pillar.drain_feedback()
        assert len(feedback) == 3  # One for each Praxis tension axis

        axis_ids = {fb.tension_axis_id for fb in feedback}
        assert "autonomy_safety" in axis_ids
        assert "sequential_parallel" in axis_ids
        assert "verify_execute" in axis_ids

    def test_get_execution_memories(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        pillar.orchestrator.register_action(
            Action(description="memorable", tool_name="mem")
        )
        pillar.execute_pending()

        memories = pillar.get_execution_memories()
        assert len(memories) >= 1
        assert memories[0]["description"] == "memorable"

    def test_serialize_restore(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        pillar.orchestrator.register_action(
            Action(description="serialize me", tool_name="ser")
        )
        pillar.orchestrator.register_action(
            Action(description="serialize me too", tool_name="ser2")
        )

        data = pillar.serialize()
        assert data is not None
        assert len(data["actions"]) == 2

        new_pillar = PraxisPillar(name="restored")
        new_pillar.restore(data)
        assert new_pillar.orchestrator is not None
        assert new_pillar.orchestrator.total_actions == 2

    def test_cancel_action(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        aid = pillar.orchestrator.register_action(
            Action(description="doomed", tool_name="doom")
        )

        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="cancel_action",
            payload={"action_id": str(aid)},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()

        assert pillar.orchestrator.action_states[aid] == ActionState.CANCELLED

    def test_update_tension_profile(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        pillar.update_tension_profile({
            "autonomy_safety": 1.0,  # Maximum autonomy
            "sequential_parallel": 0.8,
            "verify_execute": -0.3,
        })

        # Register a CRITICAL action — should not be blocked with high autonomy
        pillar.orchestrator.register_action(
            Action(
                description="risky",
                tool_name="rm",
                risk=ActionRisk.CRITICAL,
            )
        )
        report = pillar.execute_pending()
        assert report is not None
        assert report.gate_blocks == 0
        assert report.parallelism_level > 1  # parallel mode

    def test_no_executor_warns(self, agent_state):
        """Without executor_fn, execution should be a no-op with warning."""
        pillar = PraxisPillar(name="executor")
        pillar.initialize(agent_state)

        tasks = [{"description": "will fail", "tool_name": "nope"}]
        signal = Signal(
            source=Pillar.COGNITION,
            target=Pillar.PRAXIS,
            kind="execute_plan",
            payload={"tasks": tasks},
        )
        pillar.receive_signal(signal)
        pillar.process_queued()
        # Should not crash; imported but not executed
        assert pillar.orchestrator.total_actions == 1

    def test_last_report_after_execution(self, agent_state, echo_executor):
        pillar = PraxisPillar(name="executor", executor_fn=echo_executor)
        pillar.initialize(agent_state)

        pillar.orchestrator.register_action(
            Action(description="report test", tool_name="test")
        )
        report = pillar.execute_pending()
        assert report is not None
        assert report.actions_completed == 1
        assert pillar.last_report is not None
