"""Action Merger — Chamber 3 merge strategies.

Supports three strategies from the architecture:
  PRIORITY         → highest-priority agent wins for each DOF
  WEIGHTED_AVERAGE → blend by per-agent weight
  NULLSPACE        → lower-priority agents act only on unclaimed DOFs
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch

from isonome.core.state import FullAction, MergeStrategy, PartialAction
from isonome.utils.logging import get_layer_logger


class ActionMerger(ABC):
    """Abstract action merger."""

    def __init__(self) -> None:
        self._logger = get_layer_logger("coordination.merger")

    @property
    @abstractmethod
    def strategy(self) -> MergeStrategy:
        ...

    @abstractmethod
    def merge(self, partials: List[PartialAction], total_dof: int) -> FullAction:
        """Combine partial actions into a single full-action command tensor.

        Args:
            partials: active partial actions from sub-agents.
            total_dof: size of the full robot joint vector.

        Returns:
            ``FullAction`` with ``commands`` of shape ``[..., total_dof]``.
        """
        ...

    # -- factory -------------------------------------------------------------

    @classmethod
    def create(cls, strategy: MergeStrategy) -> "ActionMerger":
        if strategy == MergeStrategy.PRIORITY:
            return PriorityMerger()
        if strategy == MergeStrategy.WEIGHTED_AVERAGE:
            return WeightedAverageMerger()
        if strategy == MergeStrategy.NULLSPACE:
            return NullspaceMerger()
        raise ValueError(f"Unknown merge strategy: {strategy}")


# ---------------------------------------------------------------------------
# Priority Merger
# ---------------------------------------------------------------------------

class PriorityMerger(ActionMerger):
    """Highest-priority agent wins for every DOF it claims.

    If two agents claim overlapping DOFs, the one with the larger
    ``priority`` value owns those indices; lower-priority agents are
    skipped for already-claimed DOFs.  Non-overlapping DOFs are
    concatenated into the full vector.
    """

    @property
    def strategy(self) -> MergeStrategy:
        return MergeStrategy.PRIORITY

    def merge(self, partials: List[PartialAction], total_dof: int) -> FullAction:
        active = [p for p in partials if p.active]
        if not active:
            return FullAction(
                commands=torch.zeros(total_dof),
                source_map={},
                merged_from=[],
                strategy=self.strategy,
            )

        # Determine batch shape from the first partial
        sample = active[0].commands
        if sample.ndim == 1:
            batch_shape = (total_dof,)
        else:
            batch_shape = (*sample.shape[:-1], total_dof)

        full = torch.zeros(batch_shape, dtype=sample.dtype, device=sample.device)
        claimed = torch.zeros(batch_shape, dtype=torch.bool, device=sample.device)
        source_map: dict[str, slice] = {}
        merged_from: list[str] = []

        # Sort by priority descending — highest priority claims first
        sorted_partials = sorted(active, key=lambda p: p.priority, reverse=True)

        for p in sorted_partials:
            s = p.dof_slice
            # Clamp slice to valid range
            start = max(0, s.start if s.start is not None else 0)
            stop = min(total_dof, s.stop if s.stop is not None else total_dof)
            if start >= stop:
                continue

            cmd = p.commands
            # Ensure cmd is at least 1-D
            if cmd.ndim == 0:
                cmd = cmd.unsqueeze(0)

            idx = slice(start, stop)
            # Only write to unclaimed DOFs in this slice
            mask = ~claimed[..., idx]
            if not mask.any():
                continue

            segment = cmd[..., : stop - start]
            try:
                full[..., idx] = torch.where(mask, segment, full[..., idx])
            except RuntimeError as e:
                self._logger.warning(
                    "priority_merge_shape_mismatch",
                    extra={
                        "agent_id": p.agent_id,
                        "expected": list(full[..., idx].shape),
                        "got": list(segment.shape),
                        "error": str(e),
                    },
                )
                continue

            claimed[..., idx] = True
            source_map[p.agent_id] = idx
            merged_from.append(p.agent_id)

        return FullAction(
            commands=full,
            source_map=source_map,
            merged_from=merged_from,
            strategy=self.strategy,
        )


# ---------------------------------------------------------------------------
# Weighted Average Merger
# ---------------------------------------------------------------------------

class WeightedAverageMerger(ActionMerger):
    """Blend overlapping DOFs by per-agent weight.

    For each DOF index, the final value is the weighted average of all
    agents that claim that index.  Weights are normalised per-index.
    """

    @property
    def strategy(self) -> MergeStrategy:
        return MergeStrategy.WEIGHTED_AVERAGE

    def merge(self, partials: List[PartialAction], total_dof: int) -> FullAction:
        active = [p for p in partials if p.active]
        if not active:
            return FullAction(
                commands=torch.zeros(total_dof),
                source_map={},
                merged_from=[],
                strategy=self.strategy,
            )

        sample = active[0].commands
        if sample.ndim == 1:
            batch_shape = (total_dof,)
        else:
            batch_shape = (*sample.shape[:-1], total_dof)

        device = sample.device
        dtype = sample.dtype

        # Accumulators: weighted sum and total weight per DOF
        weighted_sum = torch.zeros(batch_shape, dtype=dtype, device=device)
        weight_sum = torch.zeros(batch_shape, dtype=dtype, device=device)
        source_map: dict[str, slice] = {}
        merged_from: list[str] = []

        for p in active:
            s = p.dof_slice
            start = max(0, s.start if s.start is not None else 0)
            stop = min(total_dof, s.stop if s.stop is not None else total_dof)
            if start >= stop:
                continue

            cmd = p.commands
            if cmd.ndim == 0:
                cmd = cmd.unsqueeze(0)

            idx = slice(start, stop)
            segment = cmd[..., : stop - start]

            # Expand weight to match batch dims
            w = torch.tensor(p.weight, dtype=dtype, device=device)
            for _ in range(segment.ndim - 1):
                w = w.unsqueeze(0)

            weighted_sum[..., idx] += segment * w
            weight_sum[..., idx] += w

            source_map[p.agent_id] = idx
            merged_from.append(p.agent_id)

        # Normalise; avoid divide-by-zero
        full = torch.where(
            weight_sum > 0,
            weighted_sum / weight_sum,
            torch.zeros_like(weighted_sum),
        )

        return FullAction(
            commands=full,
            source_map=source_map,
            merged_from=merged_from,
            strategy=self.strategy,
        )


# ---------------------------------------------------------------------------
# Nullspace Merger
# ---------------------------------------------------------------------------

class NullspaceMerger(ActionMerger):
    """Joint-space nullspace merger.

    Higher-priority agents claim their DOFs fully.  Lower-priority agents
    may only influence DOFs that are **not claimed** by any higher-priority
    agent.  Within the unclaimed set, lower-priority agents blend via
    weighted average.

    .. note::
        This is a *joint-space* approximation.  True task-space nullspace
        projection requires a Jacobian, which is not yet available in the
        v0.2 state model.  When Jacobian support lands, this class can be
        upgraded without changing the ``ActionMerger`` interface.
    """

    @property
    def strategy(self) -> MergeStrategy:
        return MergeStrategy.NULLSPACE

    def merge(self, partials: List[PartialAction], total_dof: int) -> FullAction:
        active = [p for p in partials if p.active]
        if not active:
            return FullAction(
                commands=torch.zeros(total_dof),
                source_map={},
                merged_from=[],
                strategy=self.strategy,
            )

        sample = active[0].commands
        if sample.ndim == 1:
            batch_shape = (total_dof,)
        else:
            batch_shape = (*sample.shape[:-1], total_dof)

        device = sample.device
        dtype = sample.dtype
        full = torch.zeros(batch_shape, dtype=dtype, device=device)
        claimed = torch.zeros(batch_shape, dtype=torch.bool, device=device)

        source_map: dict[str, slice] = {}
        merged_from: list[str] = []

        # Sort by priority descending
        sorted_partials = sorted(active, key=lambda p: p.priority, reverse=True)

        for p in sorted_partials:
            s = p.dof_slice
            start = max(0, s.start if s.start is not None else 0)
            stop = min(total_dof, s.stop if s.stop is not None else total_dof)
            if start >= stop:
                continue

            cmd = p.commands
            if cmd.ndim == 0:
                cmd = cmd.unsqueeze(0)

            idx = slice(start, stop)
            segment = cmd[..., : stop - start]

            # Build mask for unclaimed DOFs in this slice
            mask = ~claimed[..., idx]
            if not mask.any():
                self._logger.debug(
                    "nullspace_fully_blocked",
                    extra={"agent_id": p.agent_id, "slice": [start, stop]},
                )
                continue

            # Write only where unclaimed
            full[..., idx] = torch.where(mask, segment, full[..., idx])
            claimed[..., idx] = True

            source_map[p.agent_id] = idx
            merged_from.append(p.agent_id)

        return FullAction(
            commands=full,
            source_map=source_map,
            merged_from=merged_from,
            strategy=self.strategy,
        )
