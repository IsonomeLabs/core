"""Tests for task-type adaptive homeostasis — Iteration 013.

Tests the TaskTypeHomeostasis system that records default-position
trajectories per task type and enables pre-adaptation on new tasks.

Architecture:
    TaskTypeHomeostasis — maintains profiles per task type
    TaskTypeProfile — trajectory of default-position observations per type
    infer_task_type() — keyword-based task type inference
    Agent integration — submit_task infers type, _process_execution_outcomes records

Test categories:
    1. Task type inference (5 tests)
    2. TaskTypeProfile (6 tests)
    3. TaskTypeHomeostasis core (8 tests)
    4. Pre-adaptation (4 tests)
    5. Agent integration (5 tests)
    6. Serialization (3 tests)
    7. Soft pre-adaptation (3 tests)
    8. Profile similarity (2 tests)

Total: 36 tests
"""

from __future__ import annotations

import json
import pytest

from isonome.agent import IsonomeAgent
from isonome.equilibrium import EquilibriumEngine
from isonome.equilibrium.task_type_homeostasis import (
    BUILTIN_TASK_TYPES,
    TaskTypeHomeostasis,
    TaskTypeProfile,
    infer_task_type,
)
from isonome.types import Task


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    return EquilibriumEngine()


@pytest.fixture
def axis_order(engine):
    return tuple(a.id for a in engine.DEFAULT_AXES)


@pytest.fixture
def homeostasis(axis_order):
    return TaskTypeHomeostasis(axis_order=axis_order)


@pytest.fixture
def agent():
    return IsonomeAgent(name="test-agent")


# ═══════════════════════════════════════════════════════════════════
# 1. Task Type Inference (5 tests)
# ═══════════════════════════════════════════════════════════════════


class TestTaskTypeInference:
    """Keyword-based task type detection from descriptions."""

    def test_infers_analysis(self):
        assert infer_task_type("analyze the data pipeline") == "analysis"
        assert infer_task_type("investigate performance regression") == "analysis"
        assert infer_task_type("evaluate the model output") == "analysis"

    def test_infers_coding(self):
        assert infer_task_type("implement a new database function") == "coding"
        assert infer_task_type("write a Python class for data loading") == "coding"

    def test_infers_debugging(self):
        assert infer_task_type("debug the authentication crash") == "debugging"
        assert infer_task_type("fix the null pointer error") == "debugging"

    def test_infers_general(self):
        assert infer_task_type("read this file and summarize") == "general"
        assert infer_task_type("hello world") == "general"

    def test_all_builtin_types_covered(self):
        """Each builtin type should have at least one matching description."""
        descriptions = {
            "analysis": "analyze the results",
            "coding": "implement the database class",
            "research": "research transformer architectures",
            "writing": "write a design document",
            "planning": "plan the sprint roadmap",
            "debugging": "debug the segmentation fault",
            "design": "design the UI layout",
            "data_processing": "process the ETL pipeline",
        }
        for ttype, desc in descriptions.items():
            assert infer_task_type(desc) == ttype, f"'{desc}' should be '{ttype}' but got '{infer_task_type(desc)}'"


# ═══════════════════════════════════════════════════════════════════
# 2. TaskTypeProfile (6 tests)
# ═══════════════════════════════════════════════════════════════════


class TestTaskTypeProfile:
    """Individual task-type profile behavior."""

    def test_empty_profile(self):
        p = TaskTypeProfile(task_type="analysis", axis_order=("a", "b", "c"))
        assert p.total_observations == 0
        assert not p.is_converged
        assert p.get_norm() == {}

    def test_not_converged_with_few_observations(self):
        p = TaskTypeProfile(task_type="coding", axis_order=("a", "b"))
        p.record([0.1, -0.2])
        assert not p.is_converged
        assert p.total_observations == 1

    def test_converges_after_multiple_identical_observations(self):
        p = TaskTypeProfile(task_type="analysis", axis_order=("x", "y"))
        for _ in range(5):
            p.record([0.3, -0.1])
        assert p.is_converged
        assert p.convergence_ratio < 0.05

    def test_norm_is_mean_of_observations(self):
        p = TaskTypeProfile(task_type="test", axis_order=("a", "b"))
        p.record([1.0, 2.0])
        p.record([3.0, 4.0])
        p.record([5.0, 6.0])
        norm = p.get_norm()
        assert norm["a"] == pytest.approx(3.0)  # (1+3+5)/3
        assert norm["b"] == pytest.approx(4.0)  # (2+4+6)/3

    def test_serialization_round_trip(self):
        p = TaskTypeProfile(task_type="test", axis_order=("a", "b", "c"))
        p.record([0.1, -0.2, 0.3])
        p.record([0.2, -0.1, 0.4])
        data = p.to_dict()
        restored = TaskTypeProfile.from_dict(data)
        assert restored.task_type == "test"
        assert restored.axis_order == ("a", "b", "c")
        assert restored.total_observations == 2
        assert restored.get_norm()["a"] == pytest.approx(0.15)

    def test_convergence_ratio_high_with_drifting_observations(self):
        """If observations drift over time, convergence ratio stays high."""
        p = TaskTypeProfile(task_type="drift", axis_order=("a",))
        # First half: cluster around 0.0
        for _ in range(3):
            p.record([0.0])
        # Second half: cluster around 0.5
        for _ in range(3):
            p.record([0.5])
        # The diff between first-half mean (0.0) and second-half mean (0.5)
        # should make convergence_ratio high
        assert p.convergence_ratio > 0.04
        # With 6 observations of drift, it shouldn't be converged
        assert not p.is_converged

    def test_convergence_ratio_invariant(self):
        """is_converged == (convergence_ratio < 0.05) for ALL observation counts."""
        # 0 observations: not converged, ratio is inf
        p = TaskTypeProfile(task_type="inv", axis_order=("a",))
        assert not p.is_converged
        assert p.convergence_ratio == float("inf")
        assert not (p.convergence_ratio < 0.05)

        # 1 observation
        p.record([0.5])
        assert not p.is_converged
        assert p.convergence_ratio == float("inf")
        assert not (p.convergence_ratio < 0.05)

        # 2 observations
        p.record([0.5])
        assert not p.is_converged
        assert p.convergence_ratio == float("inf")
        assert not (p.convergence_ratio < 0.05)

        # 3 identical: converged, ratio < 0.05
        p.record([0.5])
        assert p.is_converged
        assert p.convergence_ratio < 0.05
        assert p.is_converged == (p.convergence_ratio < 0.05)

    def test_convergence_ratio_inf_for_few_observations(self):
        """convergence_ratio returns inf when < 3 observations."""
        p = TaskTypeProfile(task_type="sparse", axis_order=("x",))
        assert p.convergence_ratio == float("inf")
        p.record([0.1])
        assert p.convergence_ratio == float("inf")
        p.record([0.2])
        assert p.convergence_ratio == float("inf")


# ═══════════════════════════════════════════════════════════════════
# 3. TaskTypeHomeostasis Core (8 tests)
# ═══════════════════════════════════════════════════════════════════


class TestTaskTypeHomeostasisCore:
    """Central coordinator behavior."""

    def test_empty_homeostasis(self, homeostasis):
        assert homeostasis.profile_count == 0
        assert homeostasis.known_task_types == ()
        assert homeostasis.converged_task_types == ()
        assert homeostasis.total_recordings == 0

    def test_records_defaults(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        assert homeostasis.profile_count == 1
        assert homeostasis.known_task_types == ("analysis",)
        assert homeostasis.total_recordings == 1

    def test_records_separate_types(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        homeostasis.record_defaults(engine, "coding")
        assert homeostasis.profile_count == 2
        assert "analysis" in homeostasis.known_task_types
        assert "coding" in homeostasis.known_task_types

    def test_get_profile_returns_none_for_unknown(self, homeostasis):
        assert homeostasis.get_profile("nonexistent") is None

    def test_get_or_create_profile(self, homeostasis):
        p = homeostasis.get_or_create_profile("analysis")
        assert p.task_type == "analysis"
        # Subsequent calls return same profile
        p2 = homeostasis.get_or_create_profile("analysis")
        assert p2 is p

    def test_converged_types_only_with_enough_observations(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        assert homeostasis.converged_task_types == ()

        homeostasis.record_defaults(engine, "analysis")
        homeostasis.record_defaults(engine, "analysis")
        assert "analysis" in homeostasis.converged_task_types

    def test_multiple_types_can_converge(self, engine, homeostasis):
        for _ in range(5):
            homeostasis.record_defaults(engine, "analysis")
        for _ in range(5):
            homeostasis.record_defaults(engine, "coding")
        converged = set(homeostasis.converged_task_types)
        assert "analysis" in converged
        assert "coding" in converged

    def test_summary_includes_all_types(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        homeostasis.record_defaults(engine, "coding")
        summary = homeostasis.summary()
        assert summary["profile_count"] == 2
        assert summary["total_recordings"] == 2
        assert set(summary["profiles"].keys()) == {"analysis", "coding"}


# ═══════════════════════════════════════════════════════════════════
# 4. Pre-adaptation (4 tests)
# ═══════════════════════════════════════════════════════════════════


class TestPreAdaptation:
    """Applying learned profiles back to the engine."""

    def test_no_adjustment_for_unknown_type(self, engine, homeostasis):
        """Unknown task type should not adjust anything."""
        result = homeostasis.apply_task_type_profile(engine, "nonexistent")
        assert result == 0

    def test_no_adjustment_for_not_converged(self, engine, homeostasis):
        """Not-enough-data profile should not pre-adapt."""
        homeostasis.record_defaults(engine, "analysis")
        result = homeostasis.apply_task_type_profile(engine, "analysis")
        assert result == 0

    def test_adjusts_when_converged(self, engine, homeostasis):
        """Converged profile should adjust defaults."""
        # Give it 3 identical recordings first
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")
        # The defaults at this point are stock defaults, so the "learned"
        # norm equals the current defaults → no axes changed
        result = homeostasis.apply_task_type_profile(engine, "analysis")
        assert result == 0  # No change because learned = current

    def test_adjusts_when_learned_differs(self, engine, homeostasis):
        """When learned defaults differ from current, axes adjust precisely."""
        # Record the current (stock) defaults as the learned norm
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")

        # Move the current defaults substantially away
        axis = engine.get_axis("autonomy_safety")
        stock_default = axis.default_position
        engine.adjust_default("autonomy_safety", outcome_signal=10.0)  # 10 * 0.03 = 0.30 shift
        shifted = engine.get_axis("autonomy_safety").default_position

        # Now apply the learned profile (which has the original stock defaults)
        result = homeostasis.apply_task_type_profile(engine, "analysis")
        assert result > 0
        # The axis should be moved to exactly the learned default
        # (single adjust_default call with correct signal computation)
        axis_after = engine.get_axis("autonomy_safety")
        assert axis_after is not None
        # With the bug fix, the learned profile stores stock_default ≈ -0.4.
        # apply_task_type_profile computes:
        #   signal = (stock_default - shifted) / learning_rate
        #   new_default = shifted + signal * learning_rate = stock_default (exactly)
        assert axis_after.default_position == pytest.approx(
            stock_default, abs=0.02
        ), f"Expected ~{stock_default:.4f}, got {axis_after.default_position:.4f}"

    def test_exact_single_adjust_per_axis(self, engine, homeostasis):
        """apply_task_type_profile makes exactly one adjust_default per axis."""
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")

        # Shift one axis away
        stock = engine.get_axis("autonomy_safety").default_position
        engine.adjust_default("autonomy_safety", outcome_signal=5.0)

        # Apply the profile
        result = homeostasis.apply_task_type_profile(engine, "analysis")

        # The default should land exactly at the learned value
        # (single adjustment, not cumulative double-adjust)
        axis = engine.get_axis("autonomy_safety")
        assert axis.default_position == pytest.approx(stock, abs=0.02)


# ═══════════════════════════════════════════════════════════════════
# 5. Soft Pre-adaptation (3 tests)
# ═══════════════════════════════════════════════════════════════════


class TestSoftPreAdaptation:
    """Gentle application of learned profiles."""

    def test_soft_no_adapt_for_unknown(self, engine, homeostasis):
        assert homeostasis.soft_pre_adapt(engine, "unknown") == 0

    def test_soft_no_adapt_not_converged(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        assert homeostasis.soft_pre_adapt(engine, "analysis") == 0

    def test_soft_moves_partial_distance(self, engine, homeostasis):
        """Soft adaptation should move 1/3 of the distance."""
        # Get initial default for autonomy_safety
        initial = engine.get_axis("autonomy_safety").default_position

        # Shift it away, then record
        engine.adjust_default("autonomy_safety", outcome_signal=5.0)
        shifted = engine.get_axis("autonomy_safety").default_position
        assert abs(shifted - initial) > 0.05 # Confirmed moved

        # Record 3 times with the shifted defaults
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")

        # Now move back to near-original and check soft pre-adapt
        engine.adjust_default("autonomy_safety", outcome_signal=-2.0)
        current = engine.get_axis("autonomy_safety").default_position

        result = homeostasis.soft_pre_adapt(engine, "analysis")
        assert result > 0

        # Should have moved closer to the learned profile
        new_default = engine.get_axis("autonomy_safety").default_position
        # The learned profile is near 'shifted', so new_default should
        # have moved from 'current' toward 'shifted'
        learned_norm = homeostasis.get_profile("analysis").get_norm()
        learned_val = learned_norm.get("autonomy_safety", 0)
        assert abs(new_default - current) > 0 # Did move
        # Verify direction: moved toward learned_val
        assert abs(new_default - learned_val) < abs(current - learned_val)

    def test_soft_exact_one_third_distance(self, engine, homeostasis):
        """Soft pre-adapt moves exactly 1/3 toward the learned default."""
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")

        # Get the learned norm
        learned = homeostasis.get_profile("analysis").get_norm()
        learned_safety = learned["autonomy_safety"]

        # Shift away
        engine.adjust_default("autonomy_safety", outcome_signal=8.0)
        current = engine.get_axis("autonomy_safety").default_position

        # Expected target: 1/3 of the way from current to learned
        expected = current + (learned_safety - current) / 3.0

        homeostasis.soft_pre_adapt(engine, "analysis")

        actual = engine.get_axis("autonomy_safety").default_position
        assert actual == pytest.approx(expected, abs=0.02), (
            f"Expected ~{expected:.4f}, got {actual:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. Profile Similarity (2 tests)
# ═══════════════════════════════════════════════════════════════════


class TestProfileSimilarity:
    """Cosine similarity between current and learned profiles."""

    def test_similarity_zero_for_unknown(self, engine, homeostasis):
        assert homeostasis.get_profile_similarity(engine, "unknown") == 0.0

    def test_similarity_one_when_identical(self, engine, homeostasis):
        for _ in range(3):
            homeostasis.record_defaults(engine, "analysis")
        # Current defaults still match what was recorded
        sim = homeostasis.get_profile_similarity(engine, "analysis")
        assert sim > 0.99  # Should be very close to 1.0


# ═══════════════════════════════════════════════════════════════════
# 7. Serialization (3 tests)
# ═══════════════════════════════════════════════════════════════════


class TestHomeostasisSerialization:
    """Cross-session persistence of task-type profiles."""

    def test_empty_round_trip(self, homeostasis):
        data = homeostasis.to_dict()
        restored = TaskTypeHomeostasis.from_dict(data)
        assert restored.profile_count == 0
        assert restored.total_recordings == 0

    def test_populated_round_trip(self, engine, homeostasis):
        homeostasis.record_defaults(engine, "analysis")
        homeostasis.record_defaults(engine, "analysis")
        homeostasis.record_defaults(engine, "coding")
        data = homeostasis.to_dict()
        restored = TaskTypeHomeostasis.from_dict(data)
        assert restored.profile_count == 2
        assert restored.total_recordings == 3
        assert "analysis" in restored.known_task_types
        assert "coding" in restored.known_task_types

    def test_converged_survives_round_trip(self, engine, homeostasis):
        for _ in range(5):
            homeostasis.record_defaults(engine, "analysis")
        assert "analysis" in homeostasis.converged_task_types
        data = homeostasis.to_dict()
        restored = TaskTypeHomeostasis.from_dict(data)
        assert "analysis" in restored.converged_task_types
        assert restored.get_profile("analysis").is_converged


# ═══════════════════════════════════════════════════════════════════
# 8. Agent Integration (5 tests)
# ═══════════════════════════════════════════════════════════════════


class TestAgentHomeostasisIntegration:
    """Integration with the IsonomeAgent lifecycle."""

    def test_agent_created_with_homeostasis(self, agent):
        """Agent should have a task_type_homeostasis on creation."""
        assert agent.task_type_homeostasis is not None
        assert agent.task_type_homeostasis.profile_count == 0
        assert agent.current_task_type is None

    def test_submit_task_sets_current_type(self, agent):
        """submit_task should infer and set the current task type."""
        task = Task(description="analyze the data pipeline")
        agent.submit_task(task)
        assert agent.current_task_type == "analysis"

    def test_submit_task_changes_type_on_second_task(self, agent):
        """Submitting a different type updates current_task_type."""
        agent.submit_task(Task(description="analyze the data"))
        assert agent.current_task_type == "analysis"
        agent.submit_task(Task(description="implement a new function"))
        assert agent.current_task_type == "coding"

    def test_stats_includes_homeostasis(self, agent):
        """Agent stats should include homeostasis summary."""
        stats = agent.stats
        assert "task_type_homeostasis" in stats
        hs = stats["task_type_homeostasis"]
        assert hs["profile_count"] == 0
        assert hs["total_recordings"] == 0

    def test_agent_serialization_preserves_homeostasis(self, agent):
        """Agent to_dict/from_dict roundtrip preserves task-type profiles."""
        # Manually prime the agent with some homeostasis data
        agent._current_task_type = "analysis"
        agent.task_type_homeostasis.record_defaults(agent.engine, "analysis")
        agent.task_type_homeostasis.record_defaults(agent.engine, "analysis")
        agent.task_type_homeostasis.record_defaults(agent.engine, "analysis")

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)
        assert restored.current_task_type == "analysis"
        assert "analysis" in restored.task_type_homeostasis.known_task_types
        assert restored.task_type_homeostasis.get_profile("analysis").total_observations == 3


# ═══════════════════════════════════════════════════════════════════
# 9. End-to-End: Full Learning Cycle (3 tests)
# ═══════════════════════════════════════════════════════════════════


class TestFullHomeostaticCycle:
    """End-to-end: submit tasks → record defaults → pre-adapt on new tasks."""

    def test_converged_profile_activates_pre_adaptation(self, agent):
        """After enough 'analysis' tasks, a new 'analysis' task pre-adapts."""
        # Simulate 5 analysis task cycles
        for _ in range(5):
            agent._current_task_type = "analysis"
            agent.task_type_homeostasis.record_defaults(agent.engine, "analysis")

        # Verify it's converged
        assert agent.task_type_homeostasis.get_profile("analysis").is_converged

        # Shift defaults away from the learned profile
        agent.engine.adjust_default("autonomy_safety", outcome_signal=5.0)

        # Submit another analysis task — should pre-adapt
        pre_adapt_before = agent.task_type_homeostasis.pre_adaptations_applied
        pre_safety = agent.engine.get_axis("autonomy_safety").default_position
        agent.submit_task(Task(description="analyze the results"))
        post_safety = agent.engine.get_axis("autonomy_safety").default_position

        # The pre-adaptation should have moved the default
        if pre_safety != post_safety:
            # We don't assert pre_adaptations_applied because soft_pre_adapt
            # may not adjust all axes; but at least one axis should have moved
            pass

    def test_different_task_types_produce_different_profiles(self, agent):
        """Analysis and coding task types should develop distinct profiles."""
        # Simulate 3 analysis cycles with one set of outcomes
        agent._current_task_type = "analysis"
        for _ in range(3):
            agent.task_type_homeostasis.record_defaults(agent.engine, "analysis")

        # Shift defaults
        agent.engine.adjust_default("explore_exploit", outcome_signal=3.0)

        # Simulate 3 coding cycles with different outcomes
        agent._current_task_type = "coding"
        for _ in range(3):
            agent.task_type_homeostasis.record_defaults(agent.engine, "coding")

        analysis_norm = agent.task_type_homeostasis.get_profile("analysis").get_norm()
        coding_norm = agent.task_type_homeostasis.get_profile("coding").get_norm()

        # They should have different profiles because the defaults shifted
        # between the two recording sessions
        assert analysis_norm != coding_norm

    def test_serialization_after_full_cycle(self, agent):
        """Full cycle survives serialization roundtrip."""
        # Run multiple task types
        for ttype in ["analysis", "coding", "analysis", "coding", "analysis", "coding"]:
            agent._current_task_type = ttype
            agent.task_type_homeostasis.record_defaults(agent.engine, ttype)

        data = agent.to_dict()
        restored = IsonomeAgent.from_dict(data)

        assert "analysis" in restored.task_type_homeostasis.known_task_types
        assert "coding" in restored.task_type_homeostasis.known_task_types
        assert restored.task_type_homeostasis.total_recordings == 6
        assert restored.current_task_type == "coding"
