"""Recursive Reasoning Engine — the planning heart of the Cognition pillar.

A chain-of-thought reasoning system that recursively decomposes tasks
into sub-questions, evaluates evidence at each depth, and converges
on executable action plans. Governed by all three Cognition tensions.

Architecture:
    Task → decompose into sub-questions → for each, gather evidence →
    branch if explore mode → recurse to terminal depth → produce actions.

Core insight: Reasoning is modeled as a tree where each node represents
a hypothesis about what to do. The equilibrium engine modulates:
    - How deep to go (shallow_deep)
    - How many alternatives to consider (explore_exploit)
    - Whether to produce one plan or multiple options (divergent_convergent)

Mathematical foundation:
    - Confidence propagation: C(parent) = weighted avg of children's C
    - Branching factor B = ceil(3 × (1 - p_exploit))
    - Max depth D = 2 + ceil(6 × (1 - p_shallow))  [range: 2-8]
    - Plan convergence: when divergent, return top-K plans by confidence

Cross-pillar integration:
    - Uses AttentionEquilibriumSystem for context retrieval per reasoning step
    - Produces action dicts consumable by Praxis.import_from_cognition()
    - Reasoning traces can be stored in Mneme for pattern learning
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence
from uuid import UUID, uuid4

from isonome.types import TensionID


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


class NodeStatus(Enum):
    """Lifecycle of a reasoning node in the recursive decomposition."""

    PENDING = auto()       # Awaiting decomposition
    DECOMPOSING = auto()   # Currently being broken into sub-questions
    EVALUATING = auto()    # Gathering evidence for/against
    CONVERGED = auto()     # Reached terminal depth, actions produced
    PRUNED = auto()        # Branch was abandoned (low confidence / exploit mode)
    ERROR = auto()         # Decomposition failed


@dataclass(frozen=True, slots=True)
class EvidencePoint:
    """A single piece of evidence for or against a hypothesis.

    Evidence can come from the attention system (context chunks),
    prior reasoning steps, or external signals.
    """

    content: str = field(repr=False)
    supports: bool  # True = evidence FOR, False = AGAINST
    weight: float = 1.0  # Relevance weight [0, 1]
    source: str = "attention"  # Where this evidence came from
    id: UUID = field(default_factory=uuid4)


@dataclass
class ReasoningNode:
    """A single node in the reasoning tree.

    Each node represents a hypothesis about the task. Terminal nodes
    produce concrete action steps. Non-terminal nodes recursively
    decompose into sub-questions.
    """

    hypothesis: str
    depth: int
    id: UUID = field(default_factory=uuid4)
    status: NodeStatus = NodeStatus.PENDING
    evidence_for: list[EvidencePoint] = field(default_factory=list)
    evidence_against: list[EvidencePoint] = field(default_factory=list)
    children: list[ReasoningNode] = field(default_factory=list)
    confidence: float = 0.5  # Aggregate confidence [0, 1]
    terminal: bool = False
    action_steps: list[dict[str, Any]] = field(default_factory=list)
    parent_id: UUID | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def total_evidence(self) -> int:
        return len(self.evidence_for) + len(self.evidence_against)

    @property
    def evidence_ratio(self) -> float:
        """Ratio of supporting to total evidence. >0.5 = net supportive."""
        total = self.total_evidence
        if total == 0:
            return 0.5
        weighted_for = sum(e.weight for e in self.evidence_for)
        weighted_against = sum(e.weight for e in self.evidence_against)
        total_weight = weighted_for + weighted_against
        if total_weight == 0:
            return 0.5
        return weighted_for / total_weight

    @property
    def child_count(self) -> int:
        return len(self.children)

    @property
    def max_child_depth(self) -> int:
        """Maximum depth of any descendant (for tree statistics)."""
        if not self.children:
            return self.depth
        return max(c.max_child_depth for c in self.children)


@dataclass
class ReasoningPlan:
    """The output of a reasoning session — a set of action plans.

    When divergent mode is active, multiple plan alternatives are
    returned. When convergent, only the single best plan.
    """

    root_hypothesis: str
    plans: list[list[dict[str, Any]]]  # Each inner list is one complete plan
    confidences: list[float]  # Confidence per plan
    total_nodes: int
    max_depth_reached: int
    branches_explored: int
    branches_pruned: int
    total_evidence_gathered: int
    duration_ms: float
    tension_profile: dict[TensionID, float]

    @property
    def best_plan(self) -> list[dict[str, Any]]:
        """The highest-confidence plan."""
        if not self.plans:
            return []
        best_idx = max(range(len(self.plans)), key=lambda i: self.confidences[i])
        return self.plans[best_idx]

    @property
    def best_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return max(self.confidences)

    def summary(self) -> str:
        return (
            f"ReasoningPlan: {len(self.plans)} plan(s), "
            f"best confidence={self.best_confidence:.2f}, "
            f"nodes={self.total_nodes}, depth={self.max_depth_reached}, "
            f"branches={self.branches_explored}, pruned={self.branches_pruned}, "
            f"{self.duration_ms:.1f}ms"
        )


@dataclass
class ReasoningStats:
    """Accumulated statistics across reasoning sessions."""

    sessions: int = 0
    total_nodes_created: int = 0
    total_actions_produced: int = 0
    total_branches_explored: int = 0
    total_branches_pruned: int = 0
    total_evidence_points: int = 0
    avg_plan_confidence: float = 0.0
    avg_depth_reached: float = 0.0
    avg_duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# The Recursive Reasoning Engine
# ═══════════════════════════════════════════════════════════════════


class RecursiveReasoningEngine:
    """Recursively decomposes tasks into executable action plans.

    This is THE planning system within the Cognition pillar. It models
    reasoning as a tree-structured process where each node represents
    a hypothesis about what the agent should do, decomposing recursively
    until terminal depth where concrete actions are produced.

    Tension modulation:
    - shallow_deep < 0 (Shallow): max_depth = 2-3, fast planning
    - shallow_deep > 0 (Deep):    max_depth = 5-8, thorough analysis
    - explore_exploit < 0 (Explore): high branching factor (3-4 alternatives)
    - explore_exploit > 0 (Exploit): low branching (1-2, commit early)
    - divergent_convergent < 0 (Divergent): return multiple plan options
    - divergent_convergent > 0 (Convergent): return single best plan

    Mathematical foundation:
        Given tension positions p_s, p_e, p_d ∈ [-1, 1]:
        - Max depth D = 2 + ⌈6 × (1 + p_s)/2⌉       [range: 2-8]
        - Branching B = max(1, ⌈3 × (1 - p_e)⌉)     [range: 1-6]
        - Confidence(c) = evidence_ratio(c) × 0.7 + mean(children's C) × 0.3
        - Terminal when depth ≥ D or hypothesis is atomic

    Usage:
        engine = RecursiveReasoningEngine(attention_system)
        plan = engine.reason("analyze data and produce report")
        # plan.best_plan → list of action dicts ready for Praxis
    """

    # Default tension profile
    _DEFAULT_PROFILE: dict[TensionID, float] = {
        "shallow_deep": -0.2,
        "explore_exploit": 0.15,
        "divergent_convergent": 0.3,
    }

    # Decomposition granularity thresholds
    MIN_HYPOTHESIS_LENGTH_FOR_DECOMPOSITION = 20  # chars
    MAX_TERMINAL_ACTIONS = 8

    def __init__(
        self,
        attention_system: Any = None,  # AttentionEquilibriumSystem for context
        *,
        decomposer_fn: Any = None,  # Callable[[str, list[EvidencePoint]], list[str]]
        evidence_fn: Any = None,    # Callable[[str, list[str]], list[EvidencePoint]]
        action_composer_fn: Any = None,  # Callable[[str, list[EvidencePoint]], list[dict]]
    ):
        """Initialize the reasoning engine.

        Args:
            attention_system: Optional AttentionEquilibriumSystem for context.
            decomposer_fn: Optional custom hypothesis decomposition function.
            evidence_fn: Optional custom evidence gathering function.
            action_composer_fn: Optional custom action composition function.
        """
        self._attention = attention_system
        self._decomposer_fn = decomposer_fn
        self._evidence_fn = evidence_fn
        self._action_composer_fn = action_composer_fn

        # Tension profile cache (set by pillar wrapper each tick)
        self._current_profile: dict[TensionID, float] = dict(self._DEFAULT_PROFILE)

        # Session tracking
        self._nodes: dict[UUID, ReasoningNode] = {}
        self._root_id: UUID | None = None
        self._branches_explored: int = 0
        self._branches_pruned: int = 0

        # Accumulated statistics
        self._stats = ReasoningStats()

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    def reason(
        self,
        task: str,
        *,
        initial_context: list[str] | None = None,
    ) -> ReasoningPlan:
        """Execute the full recursive reasoning pipeline on a task.

        This is THE entry point. It:
        1. Creates a root ReasoningNode with the task as hypothesis
        2. Recursively decomposes into sub-questions
        3. Gathers evidence at each node
        4. Branches alternatives in explore mode
        5. At terminal depth, composes action steps
        6. Collapses the tree into a ReasoningPlan

        Args:
            task: The task description to reason about.
            initial_context: Optional initial context strings to seed evidence.

        Returns:
            A ReasoningPlan with one or more action plan alternatives.
        """
        t_start = time.time()
        self._nodes.clear()
        self._branches_explored = 0
        self._branches_pruned = 0

        # Compute modulation parameters from tensions
        max_depth = self._compute_max_depth()
        branching_factor = self._compute_branching_factor()
        is_divergent = self._is_divergent()

        # Seed initial evidence from context
        initial_evidence: list[EvidencePoint] = []
        if initial_context:
            for ctx in initial_context:
                initial_evidence.append(
                    EvidencePoint(content=ctx, supports=True, weight=0.5, source="context")
                )
        if self._attention is not None:
            try:
                top_chunks = self._attention.get_top_chunks(n=5)
                for chunk in top_chunks:
                    initial_evidence.append(
                        EvidencePoint(
                            content=chunk.content,
                            supports=True,
                            weight=chunk.attention_score() * 0.7,
                            source="attention",
                        )
                    )
            except Exception:
                pass

        # Phase 1: Build the reasoning tree
        root = ReasoningNode(
            hypothesis=task,
            depth=0,
            evidence_for=list(initial_evidence),
        )
        self._nodes[root.id] = root
        self._root_id = root.id

        self._decompose_recursive(root, max_depth, branching_factor)

        # Phase 2: Collapse tree into plan(s)
        plans, confidences = self._collapse_into_plans(root, is_divergent)

        t_end = time.time()
        duration_ms = (t_end - t_start) * 1000

        # Update stats
        self._stats.sessions += 1
        self._stats.total_nodes_created += len(self._nodes)
        total_actions = sum(len(p) for p in plans)
        self._stats.total_actions_produced += total_actions
        self._stats.total_branches_explored += self._branches_explored
        self._stats.total_branches_pruned += self._branches_pruned
        evidence_count = sum(
            n.total_evidence for n in self._nodes.values()
        )
        self._stats.total_evidence_points += evidence_count

        # Rolling averages
        if confidences:
            self._stats.avg_plan_confidence = (
                self._stats.avg_plan_confidence * (self._stats.sessions - 1)
                + self.best_confidence(confidences)
            ) / self._stats.sessions
        self._stats.avg_depth_reached = (
            self._stats.avg_depth_reached * (self._stats.sessions - 1)
            + self._max_depth_used()
        ) / self._stats.sessions
        self._stats.avg_duration_ms = (
            self._stats.avg_duration_ms * (self._stats.sessions - 1)
            + duration_ms
        ) / self._stats.sessions

        return ReasoningPlan(
            root_hypothesis=task,
            plans=plans,
            confidences=confidences,
            total_nodes=len(self._nodes),
            max_depth_reached=self._max_depth_used(),
            branches_explored=self._branches_explored,
            branches_pruned=self._branches_pruned,
            total_evidence_gathered=evidence_count,
            duration_ms=duration_ms,
            tension_profile=dict(self._current_profile),
        )

    def reason_single_action(self, task: str) -> dict[str, Any]:
        """Quick reasoning for simple tasks — returns a single action dict.

        For trivial tasks that don't need recursive decomposition.
        """
        plan = self.reason(task)
        best = plan.best_plan
        if best:
            return best[0]
        # Fallback: create a direct action from the task
        return {
            "description": task,
            "tool_name": "reason",
            "risk": "low",
            "params": {},
        }

    # ══════════════════════════════════════════════════════════════
    # Tension integration
    # ══════════════════════════════════════════════════════════════

    def set_tension_profile(self, profile: dict[TensionID, float]) -> None:
        """Update the tension profile (called each tick by pillar wrapper)."""
        self._current_profile = dict(profile)

    # ══════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "sessions": self._stats.sessions,
            "total_nodes": self._stats.total_nodes_created,
            "total_actions": self._stats.total_actions_produced,
            "total_branches": self._stats.total_branches_explored,
            "branches_pruned": self._stats.total_branches_pruned,
            "total_evidence": self._stats.total_evidence_points,
            "avg_confidence": round(self._stats.avg_plan_confidence, 4),
            "avg_depth": round(self._stats.avg_depth_reached, 2),
            "avg_duration_ms": round(self._stats.avg_duration_ms, 1),
        }

    @property
    def current_nodes(self) -> tuple[ReasoningNode, ...]:
        """All nodes in the current reasoning tree (immutable view)."""
        return tuple(self._nodes.values())

    @property
    def root_node(self) -> ReasoningNode | None:
        """The root of the current reasoning tree, or None."""
        if self._root_id is None:
            return None
        return self._nodes.get(self._root_id)

    # ══════════════════════════════════════════════════════════════
    # Internal: recursive decomposition
    # ══════════════════════════════════════════════════════════════

    def _decompose_recursive(
        self,
        node: ReasoningNode,
        max_depth: int,
        branching: int,
    ) -> None:
        """Recursively decompose a reasoning node.

        The core algorithm:
        1. Check if terminal (at max depth or hypothesis is atomic)
        2. If terminal: compose action steps, mark converged
        3. Else: decompose into sub-questions (up to `branching` alternatives)
        4. For each sub-question, create a child node and recurse
        5. Propagate confidence from children to parent
        """
        # Terminal condition: max depth reached OR hypothesis is atomic
        if node.depth >= max_depth or self._is_atomic(node.hypothesis):
            node.terminal = True
            node.status = NodeStatus.CONVERGED
            node.action_steps = self._compose_actions(node)
            node.confidence = self._evaluate_confidence(node)
            node.completed_at = time.time()
            return

        # Decompose the hypothesis into sub-questions
        node.status = NodeStatus.DECOMPOSING
        sub_questions = self._decompose_hypothesis(node.hypothesis, branching)

        if not sub_questions:
            # Can't decompose further — treat as terminal
            node.terminal = True
            node.status = NodeStatus.CONVERGED
            node.action_steps = self._compose_actions(node)
            node.confidence = self._evaluate_confidence(node)
            node.completed_at = time.time()
            return

        # Create child nodes for each sub-question
        for sq in sub_questions[:branching]:
            child = ReasoningNode(
                hypothesis=sq,
                depth=node.depth + 1,
                parent_id=node.id,
            )
            self._nodes[child.id] = child
            node.children.append(child)

            if len(sub_questions) > 1:
                self._branches_explored += 1

            # Gather evidence for this sub-question
            evidence = self._gather_evidence(child.hypothesis, node)
            child.evidence_for = [e for e in evidence if e.supports]
            child.evidence_against = [e for e in evidence if not e.supports]

            # Recurse
            self._decompose_recursive(child, max_depth, min(branching, 3))

        # Prune low-confidence branches in exploit mode
        exploit = self._current_profile.get("explore_exploit", 0.15)
        if exploit > 0.2 and len(node.children) > 1:
            self._prune_low_confidence_children(node)

        # Propagate confidence from children to parent
        node.confidence = self._evaluate_confidence(node)
        node.status = NodeStatus.CONVERGED
        node.completed_at = time.time()

    def _is_atomic(self, hypothesis: str) -> bool:
        """Determine if a hypothesis is atomic (cannot be decomposed further).

        Atomic hypotheses are short, single-step instructions that
        map directly to executable actions.
        """
        # Very short hypotheses are atomic
        if len(hypothesis) < self.MIN_HYPOTHESIS_LENGTH_FOR_DECOMPOSITION:
            return True
        # Already looks like a single concrete action
        action_indicators = ["run ", "call ", "execute ", "fetch ", "read ", "write "]
        if any(hypothesis.lower().startswith(ai) for ai in action_indicators):
            return True
        return False

    def _decompose_hypothesis(self, hypothesis: str, branching: int) -> list[str]:
        """Break a hypothesis into sub-questions.

        Uses structural decomposition heuristics when no custom
        decomposer_fn is provided. The decomposition strategy varies
        with the task structure.

        Args:
            hypothesis: The hypothesis to decompose.
            branching: Target number of sub-questions.

        Returns:
            List of sub-question strings.
        """
        # Use custom decomposer if available
        if self._decomposer_fn is not None:
            try:
                return self._decomposer_fn(hypothesis, branching)
            except Exception:
                pass

        # ── Structural decomposition heuristics ──────────────────
        sub_questions: list[str] = []

        # Strategy 1: "and" conjunction splitting
        if " and " in hypothesis:
            parts = hypothesis.split(" and ")
            for part in parts:
                part = part.strip().rstrip(".,;")
                if len(part) > 5 and part not in sub_questions:
                    sub_questions.append(part)
            if len(sub_questions) >= 2:
                return sub_questions[:branching]

        # Strategy 2: Sequential step decomposition
        step_markers = [
            "first", "then", "next", "finally", "after",
            "before", "subsequently", "lastly",
        ]
        for marker in step_markers:
            idx = hypothesis.lower().find(marker)
            if idx > 5:  # Not at the very start
                before = hypothesis[:idx].strip().rstrip(".,;")
                after = hypothesis[idx:].strip()
                if len(before) > 10:
                    sub_questions.append(before)
                if len(after) > 10:
                    sub_questions.append(after)
                if len(sub_questions) >= 2:
                    return sub_questions[:branching]

        # Strategy 3: WH-question decomposition (what, how, why, where)
        wh_patterns = [
            ("what ", "What is "),
            ("how ", "How to "),
            ("why ", "Why "),
            ("which ", "Which "),
        ]
        if any(hypothesis.lower().startswith(w) for w, _ in wh_patterns):
            # Complex question → break into factual sub-questions
            return [
                f"Gather relevant information for: {hypothesis[:60]}",
                f"Analyze and structure the information",
                f"Produce final output for: {hypothesis[:60]}",
            ][:branching]

        # Strategy 4: Generic multi-step breakdown
        return [
            f"Analyze requirements for: {hypothesis[:60]}",
            f"Plan the approach for: {hypothesis[:60]}",
            f"Execute the plan for: {hypothesis[:60]}",
        ][:branching]

    def _gather_evidence(
        self,
        hypothesis: str,
        parent: ReasoningNode,
    ) -> list[EvidencePoint]:
        """Gather evidence for a hypothesis.

        Evidence can come from:
        - Parent node's evidence (inheritance)
        - Attention system context
        - Custom evidence_fn
        """
        evidence: list[EvidencePoint] = []

        # Use custom evidence function if available
        if self._evidence_fn is not None:
            try:
                return self._evidence_fn(hypothesis, parent)
            except Exception:
                pass

        # Inherit relevant evidence from parent
        inherited = 0
        for ep in parent.evidence_for:
            if self._is_relevant(ep.content, hypothesis) and inherited < 3:
                evidence.append(EvidencePoint(
                    content=ep.content,
                    supports=True,
                    weight=ep.weight * 0.6,  # Discounted inheritance
                    source="inherited",
                ))
                inherited += 1

        # Pull from attention system
        if self._attention is not None:
            try:
                chunks = self._attention.get_top_chunks(n=3)
                for chunk in chunks[:2]:
                    evidence.append(EvidencePoint(
                        content=chunk.content,
                        supports=True,
                        weight=chunk.attention_score() * 0.5,
                        source="attention",
                    ))
            except Exception:
                pass

        # Ensure at least one evidence point
        if not evidence:
            evidence.append(EvidencePoint(
                content=f"Hypothesis: {hypothesis[:80]}",
                supports=True,
                weight=0.5,
                source="generated",
            ))

        return evidence

    def _compose_actions(self, node: ReasoningNode) -> list[dict[str, Any]]:
        """Compose concrete action steps for a terminal node.

        Converts the hypothesis and gathered evidence into actionable
        steps consumable by Praxis.import_from_cognition().

        Returns a list of action dicts with:
            {description, tool_name, risk, params, dependencies, tags}
        """
        # Use custom composer if available
        if self._action_composer_fn is not None:
            try:
                return self._action_composer_fn(node)
            except Exception:
                pass

        actions: list[dict[str, Any]] = []

        # Determine risk from confidence and hypothesis complexity
        risk = "low"
        if node.confidence < 0.4:
            risk = "moderate"
        elif node.confidence < 0.2:
            risk = "high"

        # Generate actions from the hypothesis structure
        hyp = node.hypothesis

        if " and " in hyp:
            # Multi-step: each conjunction is an action
            parts = hyp.split(" and ")
            for i, part in enumerate(parts):
                part = part.strip().rstrip(".,;")
                if len(part) > 5:
                    deps = [f"step_{i-1}"] if i > 0 else []
                    actions.append({
                        "description": part,
                        "tool_name": self._infer_tool(part),
                        "risk": risk,
                        "params": {},
                        "ref": f"step_{i}",
                        "dependencies": deps,
                        "tags": ["reasoning", f"depth_{node.depth}"],
                    })
        else:
            # Single action
            actions.append({
                "description": hyp,
                "tool_name": self._infer_tool(hyp),
                "risk": risk,
                "params": {},
                "ref": "step_0",
                "dependencies": [],
                "tags": ["reasoning", f"depth_{node.depth}"],
            })

        return actions[:self.MAX_TERMINAL_ACTIONS]

    def _infer_tool(self, description: str) -> str:
        """Heuristically infer the tool name from an action description."""
        desc_lower = description.lower()
        if "read" in desc_lower or "fetch" in desc_lower or "get " in desc_lower:
            return "fetch"
        elif "write" in desc_lower or "create" in desc_lower or "save" in desc_lower:
            return "write"
        elif "analyze" in desc_lower or "compute" in desc_lower or "calculate" in desc_lower:
            return "analyze"
        elif "search" in desc_lower or "find" in desc_lower or "lookup" in desc_lower:
            return "search"
        elif "deploy" in desc_lower or "run" in desc_lower or "execute" in desc_lower:
            return "execute"
        elif "test" in desc_lower or "verify" in desc_lower or "validate" in desc_lower:
            return "validate"
        elif "summarize" in desc_lower or "report" in desc_lower:
            return "report"
        return "execute"

    # ══════════════════════════════════════════════════════════════
    # Internal: tree collapsing
    # ══════════════════════════════════════════════════════════════

    def _collapse_into_plans(
        self,
        root: ReasoningNode,
        is_divergent: bool,
    ) -> tuple[list[list[dict[str, Any]]], list[float]]:
        """Collapse the reasoning tree into linear action plans.

        In convergent mode: return the single best path through the tree.
        In divergent mode: return multiple alternative paths.

        A "path" is a walk from root to terminal leaves, collecting
        all action steps along the way.
        """
        if root.terminal:
            # Root itself is the only node
            return ([root.action_steps], [root.confidence])

        # Collect all terminal-node paths
        all_paths = self._collect_terminal_paths(root)

        if not all_paths:
            return ([], [])

        if is_divergent:
            # Return all distinct paths, sorted by confidence
            all_paths.sort(key=lambda x: x[1], reverse=True)
            return (
                [p for p, _ in all_paths],
                [c for _, c in all_paths],
            )
        else:
            # Return only the best path
            best = max(all_paths, key=lambda x: x[1])
            return ([best[0]], [best[1]])

    def _collect_terminal_paths(
        self,
        node: ReasoningNode,
    ) -> list[tuple[list[dict[str, Any]], float]]:
        """Collect all paths from this node to terminal leaves.

        Returns list of (action_list, confidence) tuples.
        """
        paths: list[tuple[list[dict[str, Any]], float]] = []

        if node.terminal or not node.children:
            # Leaf node: return its own actions
            if node.action_steps:
                paths.append((list(node.action_steps), node.confidence))
            return paths

        # Internal node: combine with children's paths
        for child in node.children:
            if child.status == NodeStatus.PRUNED:
                continue
            child_paths = self._collect_terminal_paths(child)
            for child_actions, child_conf in child_paths:
                combined_actions = node.action_steps + child_actions
                # Confidence: weighted average (parent 0.3, child 0.7)
                combined_conf = node.confidence * 0.3 + child_conf * 0.7
                paths.append((combined_actions, combined_conf))

        return paths

    # ══════════════════════════════════════════════════════════════
    # Internal: tension modulation
    # ══════════════════════════════════════════════════════════════

    def _compute_max_depth(self) -> int:
        """Compute max reasoning depth from shallow_deep tension.

        D = 2 + ⌈6 × (1 + p_shallow)/2⌉
        p_shallow = -1 → D = 2  (shallow mode)
        p_shallow = +1 → D = 8  (deep mode)
        """
        p_shallow = self._current_profile.get("shallow_deep", -0.2)
        # Map [-1,1] → [0,6] via (1 + p) / 2 * 6
        depth_range = (1.0 + p_shallow) / 2.0 * 6.0
        return 2 + math.ceil(depth_range)

    def _compute_branching_factor(self) -> int:
        """Compute branching factor from explore_exploit tension.

        B = max(1, ⌈3 × (1 - p_exploit)⌉)
        p_exploit = -1 (explore) → B = 6
        p_exploit = +1 (exploit)  → B = 1
        """
        p_exploit = self._current_profile.get("explore_exploit", 0.15)
        raw = 3.0 * (1.0 - p_exploit)
        return max(1, math.ceil(raw))

    def _is_divergent(self) -> bool:
        """Check if divergent mode is active.

        Divergent when divergent_convergent < 0.
        Convergent when divergent_convergent >= 0.
        """
        p_diverge = self._current_profile.get("divergent_convergent", 0.3)
        return p_diverge < 0.0

    # ══════════════════════════════════════════════════════════════
    # Internal: confidence & pruning
    # ══════════════════════════════════════════════════════════════

    def _evaluate_confidence(self, node: ReasoningNode) -> float:
        """Compute aggregate confidence for a reasoning node.

        C(node) = evidence_ratio × 0.7 + mean(children_confidences) × 0.3

        Terminal nodes rely more on evidence; internal nodes
        propagate child confidence upward.
        """
        evidence_ratio = node.evidence_ratio

        if not node.children:
            # Terminal: pure evidence-based
            return evidence_ratio

        # Internal: blend evidence with children's confidence
        child_confs = [
            c.confidence for c in node.children
            if c.status != NodeStatus.PRUNED
        ]
        if not child_confs:
            return evidence_ratio

        mean_child_conf = sum(child_confs) / len(child_confs)
        return evidence_ratio * 0.7 + mean_child_conf * 0.3

    def _prune_low_confidence_children(self, node: ReasoningNode) -> None:
        """Remove children with confidence significantly below the best.

        In exploit mode, keep only the best branch and prune the rest.
        This prevents wasteful exploration of unpromising alternatives.
        """
        if not node.children:
            return

        best_conf = max(c.confidence for c in node.children)
        threshold = best_conf * 0.6  # Prune if < 60% of best

        for child in node.children:
            if child.confidence < threshold and child.status != NodeStatus.PRUNED:
                child.status = NodeStatus.PRUNED
                self._branches_pruned += 1

    # ══════════════════════════════════════════════════════════════
    # Internal: utilities
    # ══════════════════════════════════════════════════════════════

    def _is_relevant(self, content: str, hypothesis: str) -> bool:
        """Quick relevance check via token overlap."""
        hyp_tokens = set(hypothesis.lower().split())
        content_tokens = set(content.lower().split())
        if not hyp_tokens:
            return False
        overlap = len(hyp_tokens & content_tokens)
        return overlap >= 2 or (overlap >= 1 and len(hyp_tokens) <= 5)

    def _max_depth_used(self) -> int:
        """Maximum depth of any node in the current tree."""
        if not self._nodes:
            return 0
        return max(n.depth for n in self._nodes.values())

    @staticmethod
    def best_confidence(confidences: list[float]) -> float:
        if not confidences:
            return 0.0
        return max(confidences)
