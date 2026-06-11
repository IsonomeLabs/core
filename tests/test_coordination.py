"""Tests for Chamber 3: FSM Compiler & Action Merger.

Closes architecture gap #2 — previously ``Agent.tick()`` was a single
linear pipeline for one body with no multi-agent composition, no FSM,
and no merge strategy.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from isonome.core.config import AppConfig, BridgeConfig, SomaConfig
from isonome.core.coordination import (
    ActionMerger,
    Coordinator,
    FSMCompiler,
    FSMContext,
    FSMExecutor,
    NullspaceMerger,
    PriorityMerger,
    SubAgentSlot,
    WeightedAverageMerger,
)
from isonome.core.coordination.fsm import FSMDefinition
from isonome.core.state import FullAction, MergeStrategy, PartialAction


TEST_URDF = Path(__file__).parent.parent / "examples" / "robot_arm.urdf"


# =============================================================================
# FSM Compiler
# =============================================================================


def test_fsm_compiler_basic() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    assert isinstance(fsm, FSMDefinition)
    assert fsm.initial == "idle"
    assert len(fsm.transitions) == 1


def test_fsm_compiler_multiple_transitions() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_state("run")
        .add_transition("idle", "walk", event="start")
        .add_transition("walk", "run", event="sprint")
        .add_transition("run", "idle", event="stop")
        .compile()
    )
    assert len(fsm.states) == 3
    assert len(fsm.transitions) == 3


def test_fsm_compiler_duplicate_state_raises() -> None:
    compiler = FSMCompiler().add_state("idle", initial=True)
    with pytest.raises(ValueError, match="already defined"):
        compiler.add_state("idle")


def test_fsm_compiler_no_initial_raises() -> None:
    compiler = FSMCompiler().add_state("idle")
    with pytest.raises(ValueError, match="No initial state"):
        compiler.compile()


def test_fsm_compiler_invalid_transition_source_raises() -> None:
    compiler = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_transition("idle", "walk")
    )
    with pytest.raises(ValueError, match="not defined"):
        compiler.compile()


def test_fsm_compiler_guard_and_event() -> None:
    def can_walk(ctx: FSMContext) -> bool:
        return ctx.data.get("battery", 0) > 20

    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start", guard=can_walk)
        .compile()
    )
    assert fsm.transitions[0].event == "start"
    assert fsm.transitions[0].guard is can_walk


# =============================================================================
# FSM Executor
# =============================================================================


def test_fsm_executor_starts_in_initial_state() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    assert exec_.current_state == "idle"


def test_fsm_executor_tick_with_event() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    exec_.inject_event("start")
    state = exec_.tick()
    assert state == "walk"
    assert exec_.current_state == "walk"


def test_fsm_executor_no_transition_without_event() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    state = exec_.tick()
    assert state == "idle"


def test_fsm_executor_guard_blocks_transition() -> None:
    def battery_ok(ctx: FSMContext) -> bool:
        return ctx.data.get("battery", 0) > 20

    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start", guard=battery_ok)
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    exec_.inject_event("start")
    # battery is 0 by default → guard fails
    state = exec_.tick()
    assert state == "idle"

    # Now set battery and try again
    exec_.context.data["battery"] = 50
    exec_.inject_event("start")
    state = exec_.tick()
    assert state == "walk"


def test_fsm_executor_entry_exit_during_called() -> None:
    log: list[str] = []

    def on_entry(ctx: FSMContext) -> None:
        log.append("enter_idle")

    def on_exit(ctx: FSMContext) -> None:
        log.append("exit_idle")

    def on_during(ctx: FSMContext) -> None:
        log.append("during_idle")

    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True, entry=on_entry, exit=on_exit, during=on_during)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    exec_.tick()  # entry + during on first tick
    assert "enter_idle" in log
    assert "during_idle" in log

    exec_.inject_event("start")
    exec_.tick()  # exit idle, enter walk
    assert "exit_idle" in log


def test_fsm_executor_reset() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    exec_.inject_event("start")
    exec_.tick()
    assert exec_.current_state == "walk"

    exec_.reset()
    assert exec_.current_state == "idle"


def test_fsm_executor_snapshot() -> None:
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="start")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    exec_.tick()
    snap = exec_.snapshot()
    assert snap.current_state == "idle"
    assert snap.tick_count == 1


# =============================================================================
# Action Merger — Priority
# =============================================================================


def test_priority_merge_basic() -> None:
    merger = PriorityMerger()
    partials = [
        PartialAction(
            agent_id="agent_a",
            commands=torch.tensor([1.0, 2.0, 3.0]),
            dof_slice=slice(0, 3),
            priority=2,
        ),
        PartialAction(
            agent_id="agent_b",
            commands=torch.tensor([10.0, 20.0, 30.0]),
            dof_slice=slice(0, 3),
            priority=1,
        ),
    ]
    full = merger.merge(partials, total_dof=3)
    # Higher priority (agent_a) wins for overlapping DOFs
    assert torch.allclose(full.commands, torch.tensor([1.0, 2.0, 3.0]))
    assert full.strategy == MergeStrategy.PRIORITY


def test_priority_merge_non_overlapping() -> None:
    merger = PriorityMerger()
    partials = [
        PartialAction(
            agent_id="locomotion",
            commands=torch.tensor([1.0, 2.0]),
            dof_slice=slice(0, 2),
            priority=1,
        ),
        PartialAction(
            agent_id="arm",
            commands=torch.tensor([3.0, 4.0, 5.0]),
            dof_slice=slice(2, 5),
            priority=2,
        ),
    ]
    full = merger.merge(partials, total_dof=5)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert torch.allclose(full.commands, expected)
    assert set(full.merged_from) == {"locomotion", "arm"}


def test_priority_merge_inactive_ignored() -> None:
    merger = PriorityMerger()
    partials = [
        PartialAction(
            agent_id="active_agent",
            commands=torch.tensor([5.0]),
            dof_slice=slice(0, 1),
            priority=1,
            active=True,
        ),
        PartialAction(
            agent_id="inactive_agent",
            commands=torch.tensor([99.0]),
            dof_slice=slice(0, 1),
            priority=999,
            active=False,
        ),
    ]
    full = merger.merge(partials, total_dof=1)
    assert torch.allclose(full.commands, torch.tensor([5.0]))


def test_priority_merge_empty_returns_zeros() -> None:
    merger = PriorityMerger()
    full = merger.merge([], total_dof=4)
    assert torch.allclose(full.commands, torch.zeros(4))


# =============================================================================
# Action Merger — Weighted Average
# =============================================================================


def test_weighted_average_merge_basic() -> None:
    merger = WeightedAverageMerger()
    partials = [
        PartialAction(
            agent_id="a",
            commands=torch.tensor([0.0, 0.0]),
            dof_slice=slice(0, 2),
            weight=1.0,
        ),
        PartialAction(
            agent_id="b",
            commands=torch.tensor([10.0, 10.0]),
            dof_slice=slice(0, 2),
            weight=1.0,
        ),
    ]
    full = merger.merge(partials, total_dof=2)
    # Equal weights → average
    assert torch.allclose(full.commands, torch.tensor([5.0, 5.0]))


def test_weighted_average_merge_different_weights() -> None:
    merger = WeightedAverageMerger()
    partials = [
        PartialAction(
            agent_id="a",
            commands=torch.tensor([0.0]),
            dof_slice=slice(0, 1),
            weight=3.0,
        ),
        PartialAction(
            agent_id="b",
            commands=torch.tensor([10.0]),
            dof_slice=slice(0, 1),
            weight=1.0,
        ),
    ]
    full = merger.merge(partials, total_dof=1)
    # (0*3 + 10*1) / 4 = 2.5
    assert torch.allclose(full.commands, torch.tensor([2.5]))


def test_weighted_average_non_overlapping() -> None:
    merger = WeightedAverageMerger()
    partials = [
        PartialAction(
            agent_id="a", commands=torch.tensor([1.0]), dof_slice=slice(0, 1), weight=1.0
        ),
        PartialAction(
            agent_id="b", commands=torch.tensor([2.0]), dof_slice=slice(1, 2), weight=1.0
        ),
    ]
    full = merger.merge(partials, total_dof=2)
    assert torch.allclose(full.commands, torch.tensor([1.0, 2.0]))


# =============================================================================
# Action Merger — Nullspace
# =============================================================================


def test_nullspace_merge_lower_blocked() -> None:
    merger = NullspaceMerger()
    partials = [
        PartialAction(
            agent_id="high",
            commands=torch.tensor([1.0, 2.0]),
            dof_slice=slice(0, 2),
            priority=10,
        ),
        PartialAction(
            agent_id="low",
            commands=torch.tensor([99.0, 99.0]),
            dof_slice=slice(0, 2),
            priority=1,
        ),
    ]
    full = merger.merge(partials, total_dof=2)
    # High priority claims all DOFs → low is fully blocked
    assert torch.allclose(full.commands, torch.tensor([1.0, 2.0]))
    assert "high" in full.merged_from
    assert "low" not in full.merged_from


def test_nullspace_merge_partial_overlap() -> None:
    merger = NullspaceMerger()
    partials = [
        PartialAction(
            agent_id="high",
            commands=torch.tensor([1.0, 2.0]),
            dof_slice=slice(0, 2),
            priority=10,
        ),
        PartialAction(
            agent_id="low",
            commands=torch.tensor([3.0]),
            dof_slice=slice(2, 3),
            priority=1,
        ),
    ]
    full = merger.merge(partials, total_dof=3)
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(full.commands, expected)
    assert set(full.merged_from) == {"high", "low"}


# =============================================================================
# ActionMerger factory
# =============================================================================


def test_merger_factory_priority() -> None:
    m = ActionMerger.create(MergeStrategy.PRIORITY)
    assert isinstance(m, PriorityMerger)


def test_merger_factory_weighted_average() -> None:
    m = ActionMerger.create(MergeStrategy.WEIGHTED_AVERAGE)
    assert isinstance(m, WeightedAverageMerger)


def test_merger_factory_nullspace() -> None:
    m = ActionMerger.create(MergeStrategy.NULLSPACE)
    assert isinstance(m, NullspaceMerger)


def test_merger_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ActionMerger.create("invalid")  # type: ignore[arg-type]


# =============================================================================
# Coordinator (integration)
# =============================================================================


def _make_test_agent(agent_name: str) -> "Agent":
    from isonome.core.agent import Agent

    cfg = AppConfig(
        agent_name=agent_name,
        soma=SomaConfig(urdf_path=str(TEST_URDF)),
        bridge=BridgeConfig(engine="none"),
    )
    return Agent(cfg)


@pytest.mark.asyncio
async def test_coordinator_boot_shutdown() -> None:
    agent = _make_test_agent("sub_a")
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    merger = PriorityMerger()

    coord = Coordinator(
        slots=[SubAgentSlot(agent=agent, agent_id="sub_a", dof_slice=slice(0, 7))],
        fsm_executor=exec_,
        merger=merger,
        total_dof=7,
    )

    await coord.boot()
    assert coord.is_running
    assert coord.current_phase == "idle"

    await coord.shutdown()
    assert not coord.is_running


@pytest.mark.asyncio
async def test_coordinator_tick_produces_full_action() -> None:
    agent = _make_test_agent("sub_a")
    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    merger = PriorityMerger()

    coord = Coordinator(
        slots=[SubAgentSlot(agent=agent, agent_id="sub_a", dof_slice=slice(0, 7))],
        fsm_executor=exec_,
        merger=merger,
        total_dof=7,
    )

    await coord.boot()
    full_action = await coord.tick()
    assert isinstance(full_action, FullAction)
    assert full_action.commands.shape[-1] == 7
    await coord.shutdown()


@pytest.mark.asyncio
async def test_coordinator_phase_filter() -> None:
    agent_a = _make_test_agent("agent_a")
    agent_b = _make_test_agent("agent_b")

    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk")
        .add_transition("idle", "walk", event="go")
        .compile()
    )
    exec_ = FSMExecutor(fsm)
    merger = PriorityMerger()

    coord = Coordinator(
        slots=[
            SubAgentSlot(
                agent=agent_a,
                agent_id="a",
                dof_slice=slice(0, 3),
                priority=1,
                phase_filter={"idle"},
            ),
            SubAgentSlot(
                agent=agent_b,
                agent_id="b",
                dof_slice=slice(3, 7),
                priority=2,
                phase_filter={"walk"},
            ),
        ],
        fsm_executor=exec_,
        merger=merger,
        total_dof=7,
    )

    await coord.boot()

    # In "idle", only agent_a is active
    full = await coord.tick()
    assert "a" in full.merged_from
    assert "b" not in full.merged_from

    # Transition to "walk"
    exec_.inject_event("go")
    full = await coord.tick()
    assert "b" in full.merged_from
    assert "a" not in full.merged_from

    await coord.shutdown()


@pytest.mark.asyncio
async def test_coordinator_duplicate_agent_id_raises() -> None:
    agent = _make_test_agent("dup")
    fsm = FSMCompiler().add_state("idle", initial=True).compile()
    exec_ = FSMExecutor(fsm)
    with pytest.raises(ValueError, match="Duplicate agent_ids"):
        Coordinator(
            slots=[
                SubAgentSlot(agent=agent, agent_id="same", dof_slice=slice(0, 3)),
                SubAgentSlot(agent=agent, agent_id="same", dof_slice=slice(3, 6)),
            ],
            fsm_executor=exec_,
            merger=PriorityMerger(),
            total_dof=6,
        )


@pytest.mark.asyncio
async def test_coordinator_tick_not_running_raises() -> None:
    agent = _make_test_agent("sub")
    fsm = FSMCompiler().add_state("idle", initial=True).compile()
    exec_ = FSMExecutor(fsm)
    coord = Coordinator(
        slots=[SubAgentSlot(agent=agent, agent_id="sub", dof_slice=slice(0, 7))],
        fsm_executor=exec_,
        merger=PriorityMerger(),
        total_dof=7,
    )
    with pytest.raises(RuntimeError, match="not running"):
        await coord.tick()


@pytest.mark.asyncio
async def test_coordinator_run_with_duration() -> None:
    agent = _make_test_agent("sub")
    fsm = FSMCompiler().add_state("idle", initial=True).compile()
    exec_ = FSMExecutor(fsm)
    coord = Coordinator(
        slots=[SubAgentSlot(agent=agent, agent_id="sub", dof_slice=slice(0, 7))],
        fsm_executor=exec_,
        merger=PriorityMerger(),
        total_dof=7,
    )
    await coord.run(duration_s=0.05)
    assert not coord.is_running


# =============================================================================
# End-to-end: FSM + Merger + Coordinator
# =============================================================================


@pytest.mark.asyncio
async def test_end_to_end_walk_cycle() -> None:
    """Simulate a simple walk cycle with two agents and an FSM."""
    log: list[str] = []

    def on_walk_entry(ctx: FSMContext) -> None:
        log.append("walk_enter")

    def on_walk_during(ctx: FSMContext) -> None:
        log.append("walk_during")

    fsm = (
        FSMCompiler()
        .add_state("idle", initial=True)
        .add_state("walk", entry=on_walk_entry, during=on_walk_during)
        .add_state("stop")
        .add_transition("idle", "walk", event="start")
        .add_transition("walk", "stop", event="halt")
        .compile()
    )
    exec_ = FSMExecutor(fsm)

    agent = _make_test_agent("walker")
    merger = WeightedAverageMerger()
    coord = Coordinator(
        slots=[
            SubAgentSlot(
                agent=agent,
                agent_id="walker",
                dof_slice=slice(0, 7),
                priority=1,
                weight=1.0,
            )
        ],
        fsm_executor=exec_,
        merger=merger,
        total_dof=7,
    )

    await coord.boot()

    # idle tick
    await coord.tick()
    assert coord.current_phase == "idle"
    assert "walk_enter" not in log

    # start walking
    exec_.inject_event("start")
    await coord.tick()
    assert coord.current_phase == "walk"
    assert "walk_enter" in log

    # halt
    exec_.inject_event("halt")
    await coord.tick()
    assert coord.current_phase == "stop"

    await coord.shutdown()
