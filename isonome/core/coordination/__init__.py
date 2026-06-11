"""Chamber 3: Operational Coordination Engine.

FSM Compiler + Action Merger — closes architecture gap #2.

  FSMCompiler / FSMExecutor  →  finite-state machine with guards & events
  ActionMerger               →  Priority | Weighted Average | Nullspace
  Coordinator                →  multi-agent composition layer
"""
from __future__ import annotations

from isonome.core.coordination.fsm import FSMCompiler, FSMContext, FSMExecutor
from isonome.core.coordination.merger import (
    ActionMerger,
    NullspaceMerger,
    PriorityMerger,
    WeightedAverageMerger,
)
from isonome.core.coordination.coordinator import Coordinator, SubAgentSlot

__all__ = [
    "FSMCompiler",
    "FSMContext",
    "FSMExecutor",
    "ActionMerger",
    "PriorityMerger",
    "WeightedAverageMerger",
    "NullspaceMerger",
    "Coordinator",
    "SubAgentSlot",
]
