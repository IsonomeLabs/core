"""Attention Equilibrium System — dynamic context window management.

An information-theoretic context window manager that treats the
agent's attention as a finite resource to be continuously balanced.

Core insight: The context window is an information channel with
limited capacity C. We must allocate attention bits to maximize
expected utility while respecting the channel constraint.

Mathematical foundation:
    - Surprisal: I(x) = -log₂ P(x)          (Shannon information content)
    - Mutual Information: I(X;Y) = H(X) - H(X|Y)
    - Attention Score: A(x) = α·I(x) + β·MI(x;task) + γ·recency(x)

The system operates in two modes governed by the 'shallow_deep'
tension axis:
    - Shallow (<0): Aggressive pruning, high compression, fast throughput
    - Deep (>0):   Conservative retention, low compression, thorough analysis

Connected tension axes:
    - explore_exploit:  Explore → keep diverse content; Exploit → focus on relevant
    - divergent_convergent: Divergent → keep alternatives; Convergent → narrow focus
    - consolidate_prune: Consolidate → summarize old content; Prune → delete
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heappop, heappush
from uuid import UUID, uuid4


from isonome.equilibrium import EquilibriumEngine
from isonome.types import TensionID

# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════


class RetentionDecision(Enum):
    """What to do with a context chunk."""

    KEEP_FULL = auto()  # Retain verbatim — high attention score
    COMPRESS = auto()  # Summarize to ~20% original size
    PRUNE = auto()  # Remove entirely — low attention score


class BudgetEnforcementPolicy(Enum):
    """How to handle chunks when the context window budget is full.

    - REJECT: Silently drop new chunks that don't fit. No auto-GC.
    - AUTO_GC: Automatically run garbage collection to free space.
      If GC doesn't free enough, reject the chunk.
    - AUTO_COMPRESS: Like AUTO_GC, but also compress the incoming
      chunk if it still doesn't fit after GC.
    """

    REJECT = "reject"
    AUTO_GC = "auto_gc"
    AUTO_COMPRESS = "auto_compress"



class ChunkPriorityQueue:
    """Priority queue for chunks rejected by budget enforcement.

    When a chunk is rejected (doesn't fit in the budget), it's buffered
    here for retry after the next GC cycle frees space. Chunks are
    ordered by effective priority (task_relevance + importance tag boost),
    so the most important rejected chunks get retried first.

    Uses a min-heap internally (negated priority for max-heap behavior)
    with a sequence counter for stable FIFO tiebreaking.

    Attributes:
        max_size: Maximum number of chunks the queue can hold.
    """

    def __init__(self, max_size: int = 64) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self._heap: list[tuple[float, int, AttentionChunk]] = []
        self._seq: int = 0  # Tiebreaker for FIFO ordering
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_evicted: int = 0
        self._total_dropped: int = 0

    def _effective_priority(self, chunk: AttentionChunk) -> float:
        """Compute effective priority for a chunk.

        Priority = task_relevance + importance tag boost.
        Higher = more important = dequeued first.
        """
        tag_boost = min(0.1, 0.05 * len(chunk.importance_tags))
        return chunk.task_relevance + tag_boost

    def enqueue(self, chunk: AttentionChunk) -> int:
        """Add a chunk to the priority queue.

        If the queue is at capacity, the lowest-priority entry is removed.
        If the newcomer has lower priority than all entries, it is dropped
        instead (no eviction needed).

        Returns:
            The position in the queue after insertion, or -1 if dropped.
        """
        self._total_enqueued += 1
        priority = self._effective_priority(chunk)

        if len(self._heap) >= self.max_size:
            # In our negated min-heap, heap[0] is the most-negative value
            # = HIGHEST actual priority (dequeued first).
            # We need the LOWEST actual priority entry (closest to 0
            # negated value = smallest actual priority) to decide eviction.
            # Key: -neg_p gives the actual priority, so min(-neg_p)
            # gives the smallest actual priority.
            lowest_neg_p, lowest_seq, lowest_chunk = min(
                self._heap, key=lambda x: -x[0]
            )
            lowest_priority = self._effective_priority(lowest_chunk)

            if priority <= lowest_priority:
                # Newcomer can't displace anyone - drop it
                self._total_dropped += 1
                return -1

            # Evict the lowest-priority entry by removing it and re-heapifying
            self._heap = [
                item for item in self._heap
                if not (item[0] == lowest_neg_p and item[1] == lowest_seq)
            ]
            heapq.heapify(self._heap)
            self._total_evicted += 1

        # Negate priority for max-heap behavior via min-heap
        heappush(self._heap, (-priority, self._seq, chunk))
        self._seq += 1
        return len(self._heap) - 1

    def dequeue(self) -> AttentionChunk | None:
        """Remove and return the highest-priority chunk, or None if empty."""
        if not self._heap:
            return None
        _, _, chunk = heappop(self._heap)
        self._total_dequeued += 1
        return chunk

    def peek(self) -> AttentionChunk | None:
        """Return the highest-priority chunk without removing it."""
        if not self._heap:
            return None
        _, _, chunk = self._heap[0]
        return chunk

    def clear(self) -> None:
        """Remove all chunks from the queue."""
        self._heap.clear()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self):
        """Iterate over chunks in priority order (highest first)."""
        for _, _, chunk in sorted(self._heap):
            yield chunk

    @property
    def stats(self) -> dict:
        """Queue statistics."""
        return {
            "current_size": len(self._heap),
            "max_size": self.max_size,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_evicted": self._total_evicted,
            "total_dropped": self._total_dropped,
        }

@dataclass(frozen=True, slots=True)
class AttentionChunk:
    """A single chunk of context being managed by the attention system.

    Each chunk carries its own attention metadata, allowing the
    equilibrium engine to make fine-grained retention decisions.
    """

    content: str = field(repr=False)
    token_count: int
    surprisal: float = 0.0  # -log₂ P(content | prior context)
    mutual_info: float = 0.0  # I(content; task_outcome)
    recency: float = 0.0  # Normalized recency [0, 1] where 1 = newest
    task_relevance: float = 0.5  # Estimated relevance to current task [0, 1]
    importance_tags: tuple[str, ...] = field(default_factory=tuple)
    id: UUID = field(default_factory=uuid4)

    def attention_score(
        self,
        alpha: float = 0.35,
        beta: float = 0.40,
        gamma: float = 0.15,
        delta: float = 0.10,
    ) -> float:
        """Compute the composite attention score.

        Args:
            alpha: Weight for surprisal (information content)
            beta:  Weight for mutual information (task relevance)
            gamma: Weight for recency
            delta: Weight for explicit importance tags

        Returns:
            Score in [0, 1] where higher = more attention-worthy.

        The weights are dynamically modulated by the equilibrium
        engine's tension positions. E.g., when 'shallow_deep' is < 0,
        gamma (recency) increases and beta (task relevance) decreases,
        favoring fresh-but-shallow over deep-relevance.
        """
        # Normalize surprisal: tanh squashes large values to [0,1]
        norm_surprisal = math.tanh(self.surprisal / 10.0)

        # Mutual information is already in a reasonable range
        norm_mi = min(1.0, max(0.0, self.mutual_info))

        # Importance boost from tags (linear combination)
        importance_boost = min(1.0, 0.1 * len(self.importance_tags))

        return (
            alpha * norm_surprisal
            + beta * norm_mi
            + gamma * self.recency
            + delta * importance_boost
        )


@dataclass
class AttentionBudget:
    """The finite attention capacity of the agent.

    Modeled as an information budget: how many tokens (and their
    information content) can fit in the context window.
    """

    token_capacity: int  # Max tokens the context window can hold
    tokens_used: int = 0  # Currently occupied tokens
    information_capacity: float = 0.0  # Total information bits (H_max)

    def __post_init__(self):
        # Estimate information capacity: ~1 bit per token as upper bound
        # (Shannon's estimate for English is ~1.0-1.5 bits per character,
        #  or ~4-6 bits per word; we use 8 bits/token as a rough proxy)
        self.information_capacity = self.token_capacity * 8.0

    @property
    def utilization(self) -> float:
        """Fraction of token capacity used [0, 1]."""
        if self.token_capacity == 0:
            return 0.0
        return self.tokens_used / self.token_capacity

    @property
    def headroom(self) -> int:
        """Remaining token capacity."""
        return max(0, self.token_capacity - self.tokens_used)


# ═══════════════════════════════════════════════════════════════════
# The Attention Equilibrium System
# ═══════════════════════════════════════════════════════════════════


class AttentionEquilibriumSystem:
    """Dynamically manages context window contents via equilibrium tension.

    This is THE key system within the Cognition pillar. It decides
    what to keep, compress, or evict from the agent's context window,
    using the equilibrium engine's tension positions to modulate
    the decision thresholds.

    The system maintains an internal registry of AttentionChunks
    and periodically performs a "garbage collection" pass that:
    1. Scores all chunks
    2. Sorts by attention score
    3. Applies retention decisions based on tension-modulated thresholds
    4. Compresses mid-tier chunks
    5. Prunes low-tier chunks
    """

    # Default scoring weights (modulated by tensions at runtime)
    DEFAULT_ALPHA = 0.35  # surprisal weight
    DEFAULT_BETA = 0.40  # mutual information weight
    DEFAULT_GAMMA = 0.15  # recency weight
    DEFAULT_DELTA = 0.10  # importance tags weight

    def __init__(
        self,
        engine: EquilibriumEngine,
        token_capacity: int = 128_000,
        *,
        compress_ratio: float = 0.20,
        keep_threshold: float = 0.65,
        prune_threshold: float = 0.25,
        enforcement_policy: BudgetEnforcementPolicy = BudgetEnforcementPolicy.AUTO_GC,
        enforcement_threshold: float = 0.85,
        rejected_queue_capacity: int = 64,
    ):
        """Initialize the attention equilibrium system.

        Args:
            engine: The equilibrium engine for tension-driven modulation.
            token_capacity: Max tokens the context window can hold.
            compress_ratio: Target compression ratio (fraction of original).
            keep_threshold: Minimum score to retain verbatim.
            prune_threshold: Below this, chunks are pruned.
            enforcement_policy: How to handle budget overflow when adding chunks.
            enforcement_threshold: Utilization fraction [0.1, 1.0] at which
                enforce_budget() triggers auto-GC. Default 0.85 (85% full).
            rejected_queue_capacity: Max chunks in the rejected-chunk buffer.
                Set to 0 to disable rejected-chunk buffering. Default 64.
        """
        self._engine = engine
        self._budget = AttentionBudget(token_capacity=token_capacity)
        self._chunks: dict[UUID, AttentionChunk] = {}
        self._compress_ratio = compress_ratio
        self._keep_threshold = keep_threshold
        self._prune_threshold = prune_threshold

        # Budget enforcement
        self._enforcement_policy = enforcement_policy
        self._enforcement_threshold = max(0.1, min(1.0, enforcement_threshold))

        # Rejected-chunk priority queue (iter-026)
        self._rejected_queue: ChunkPriorityQueue | None = None
        if rejected_queue_capacity > 0:
            self._rejected_queue = ChunkPriorityQueue(
                max_size=rejected_queue_capacity
            )

        # Enforcement statistics
        self._enforcement_auto_gc_triggered: int = 0
        self._enforcement_rejections: int = 0
        self._enforcement_auto_compressions: int = 0
        self._enforcement_oversized_rejections: int = 0
        self._enforcement_post_gc_rejections: int = 0

        # Statistics
        self._total_pruned: int = 0
        self._total_compressed: int = 0
        self._total_kept: int = 0
        self._gc_cycles: int = 0

        # Entropy tracking for surprisal computation
        self._token_frequencies: dict[str, int] = {}
        self._total_tokens_seen: int = 0

        # Calibration state — pushed from CognitionPillar.update_tension_profile()
        # When the calibrator detects poor calibration, attention should
        # retain MORE context (less aggressive GC, slower recency decay).
        self._calibration_ece: float = 0.0
        self._calibration_bias: float = 0.0
        self._calibration_overconfident: bool = False
        self._calibration_underconfident: bool = False
        self._calibration_predictions: int = 0
        self._calibration_active: bool = False  # True once calibration data flows

    # ── Public API ───────────────────────────────────────────────

    def add_chunk(
        self,
        content: str,
        *,
        token_count: int | None = None,
        mutual_info: float = 0.0,
        task_relevance: float = 0.5,
        importance_tags: tuple[str, ...] = (),
    ) -> AttentionChunk | None:
        """Register a new chunk of context.

        Automatically computes surprisal based on prior context and
        assigns recency = 1.0 (newest). Budget-aware: if the chunk
        would exceed capacity, the enforcement policy determines
        the response (auto-GC, reject, or auto-compress).

        Args:
            content: The text content of the chunk.
            token_count: Number of tokens (estimated if None).
            mutual_info: Estimated mutual information with the task.
            task_relevance: Estimated relevance to current task [0, 1].
            importance_tags: Tags marking explicit importance.

        Returns:
            The created AttentionChunk, or None if rejected by
            budget enforcement.
        """
        if token_count is None:
            token_count = self._estimate_tokens(content)

        # ── Budget enforcement ──
        # 1. Oversized: chunk exceeds total capacity → always reject
        if token_count > self._budget.token_capacity:
            self._enforcement_oversized_rejections += 1
            return None

        # 2. Check if there's room (or if enforcement threshold is exceeded)
        needs_enforcement = (
            self._budget.tokens_used + token_count > self._budget.token_capacity
            or self._budget.utilization >= self._enforcement_threshold
        )

        if needs_enforcement and token_count > 0:
            policy = self._enforcement_policy

            if policy == BudgetEnforcementPolicy.REJECT:
                # Check if chunk would exceed capacity
                if self._budget.tokens_used + token_count > self._budget.token_capacity:
                    self._enforcement_rejections += 1
                    self._buffer_rejected(content, token_count, mutual_info, task_relevance, importance_tags)
                    return None
                # Under threshold: no enforcement needed yet
                # but above threshold: still reject if over capacity
                if self._budget.utilization >= self._enforcement_threshold:
                    if self._budget.tokens_used + token_count > self._budget.token_capacity:
                        self._enforcement_rejections += 1
                        self._buffer_rejected(content, token_count, mutual_info, task_relevance, importance_tags)
                        return None

            elif policy in (BudgetEnforcementPolicy.AUTO_GC, BudgetEnforcementPolicy.AUTO_COMPRESS):
                # Try to free space via auto-GC
                if self._budget.utilization >= self._enforcement_threshold:
                    self._enforcement_auto_gc_triggered += 1
                    self.collect_garbage()

                # Check if GC freed enough space
                if self._budget.tokens_used + token_count > self._budget.token_capacity:
                    if policy == BudgetEnforcementPolicy.AUTO_COMPRESS:
                        # Try to compress the incoming chunk
                        compressed_tokens = int(token_count * self._compress_ratio)
                        if self._budget.tokens_used + compressed_tokens <= self._budget.token_capacity:
                            token_count = compressed_tokens
                            self._enforcement_auto_compressions += 1
                        else:
                            # Even compressed doesn't fit
                            self._enforcement_post_gc_rejections += 1
                            self._buffer_rejected(content, token_count, mutual_info, task_relevance, importance_tags)
                            return None
                    else:
                        # AUTO_GC: reject if still no room after GC
                        self._enforcement_post_gc_rejections += 1
                        self._buffer_rejected(content, token_count, mutual_info, task_relevance, importance_tags)
                        return None

        # Update frequencies FIRST so repeated content gets low surprisal
        self._update_token_frequencies(content)
        surprisal = self._compute_surprisal(content)

        chunk = AttentionChunk(
            content=content,
            token_count=token_count,
            surprisal=surprisal,
            mutual_info=mutual_info,
            recency=1.0,
            task_relevance=task_relevance,
            importance_tags=importance_tags,
        )

        self._chunks[chunk.id] = chunk
        self._budget.tokens_used += token_count

        return chunk

    def _buffer_rejected(
        self,
        content: str,
        token_count: int,
        mutual_info: float,
        task_relevance: float,
        importance_tags: tuple[str, ...],
    ) -> None:
        """Buffer a rejected chunk in the priority queue for later retry.

        Only buffers if the rejected queue is enabled (capacity > 0).
        Oversized chunks (larger than total capacity) should NOT be
        buffered — they would never fit. This method is called only
        for budget-based rejections, not oversized ones.
        """
        if self._rejected_queue is None:
            return
        chunk = AttentionChunk(
            content=content,
            token_count=token_count,
            surprisal=0.0,
            mutual_info=mutual_info,
            recency=1.0,
            task_relevance=task_relevance,
            importance_tags=importance_tags,
        )
        self._rejected_queue.enqueue(chunk)

    def retry_rejected(self) -> AttentionChunk | None:
        """Try to admit the highest-priority rejected chunk.

        After GC frees space, call this to retry buffered chunks in
        priority order. Returns the admitted chunk, or None if no
        chunk could be admitted (queue empty or still no room).

        Returns:
            The admitted AttentionChunk, or None if no chunk could fit.
        """
        if self._rejected_queue is None or not self._rejected_queue:
            return None

        chunk = self._rejected_queue.peek()
        if chunk is None:
            return None

        # Check if the highest-priority chunk fits
        if self._budget.tokens_used + chunk.token_count > self._budget.token_capacity:
            return None

        # Dequeue and admit
        chunk = self._rejected_queue.dequeue()
        # Recompute surprisal now that we have more context
        self._update_token_frequencies(chunk.content)
        surprisal = self._compute_surprisal(chunk.content)

        admitted = AttentionChunk(
            content=chunk.content,
            token_count=chunk.token_count,
            surprisal=surprisal,
            mutual_info=chunk.mutual_info,
            recency=1.0,  # Freshly admitted = newest
            task_relevance=chunk.task_relevance,
            importance_tags=chunk.importance_tags,
        )
        self._chunks[admitted.id] = admitted
        self._budget.tokens_used += admitted.token_count
        return admitted

    @property
    def rejected_queue(self) -> ChunkPriorityQueue:
        """The rejected-chunk priority queue.

        Returns the queue even if buffering is disabled (returns
        a zero-capacity queue in that case for API compatibility).
        """
        if self._rejected_queue is None:
            # Return a shared sentinel zero-capacity queue
            if not hasattr(self.__class__, '_sentinel_queue'):
                self.__class__._sentinel_queue = ChunkPriorityQueue(max_size=1)
            return self.__class__._sentinel_queue
        return self._rejected_queue

    def set_calibration_state(
        self,
        ece: float,
        bias: float,
        *,
        is_overconfident: bool = False,
        is_underconfident: bool = False,
        total_predictions: int = 0,
    ) -> None:
        """Push calibration metrics from the Cognition pillar.

        Called by CognitionPillar.update_tension_profile() each tick.
        When calibration quality is poor, the attention system adjusts:
        - Retention thresholds are lowered (keep MORE context)
        - GC aggressiveness is reduced
        - Recency decay is slowed
        - Scoring weights are rebalanced (iter-016):
          Overconfident → shift β→α (seek novelty, distrust relevance)
          Underconfident → shift α→β (trust relevance, avoid distraction)

        The system requires >= 10 predictions before activation
        (matches the calibrator's minimum-data guard).

        Args:
            ece: Expected Calibration Error [0, ~0.3+].
            bias: Weighted confidence-accuracy gap [-1, +1].
            is_overconfident: True if systematically overconfident.
            is_underconfident: True if systematically underconfident.
            total_predictions: Total prediction-outcome pairs recorded.
        """
        self._calibration_ece = max(0.0, ece)
        self._calibration_bias = bias
        self._calibration_overconfident = is_overconfident
        self._calibration_underconfident = is_underconfident
        self._calibration_predictions = total_predictions
        self._calibration_active = total_predictions >= 10

    def _compute_calibration_retention_modifier(self) -> float:
        """Compute a GC threshold adjustment from calibration quality.

        When calibration is poor, the system should retain MORE context
        (lower thresholds = easier to keep). When well-calibrated,
        thresholds return to nominal.

        Mathematical foundation:
            modifier = -(kappa * ECE * (1 + |bias|) * overconfidence_bonus)

            kappa = 1.5 (retention sensitivity)
            modifier is negative (lowers thresholds) and bounded to [-0.30, 0.0]

        When calibration is good:   modifier =  0.00  (nominal thresholds)
        When ECE = 0.10, bias 0.05: modifier = -0.17  (mild retention)
        When ECE = 0.20, bias 0.10: modifier = -0.30  (max retention)
        When ECE = 0.30, bias 0.15, overconfident: modifier = -0.30 (floor)

        Returns:
            Threshold adjustment in [-0.30, 0.0]. Negative = keep more.
        """
        if not self._calibration_active:
            return 0.0

        ece = self._calibration_ece
        bias = abs(self._calibration_bias)

        kappa = 1.5  # Attention retention sensitivity

        # Overconfidence bonus: systematic overconfidence is more dangerous
        overconfidence_bonus = 1.2 if self._calibration_overconfident else 1.0

        # Compute the retention modifier (negative -> keep more)
        modifier = -(kappa * ece * (1.0 + bias) * overconfidence_bonus)

        # Bound to [-0.30, 0.0] -- max retention bonus of 0.30 on thresholds
        return max(-0.30, min(0.0, modifier))

    def _compute_calibration_decay_modifier(self) -> float:
        """Compute a recency decay rate modifier from calibration quality.

        When calibration is poor, the system should slow recency decay
        so older context chunks don't fade as quickly -- the agent needs
        broader temporal context when it's uncertain.

        Mathematical foundation:
            modifier = kappa * ECE * (1 + |bias|)

            kappa = 0.5 (decay sensitivity)
            modifier reduces the decay rate multiplicatively

        Well-calibrated:        modifier = 0.00 -> decay_rate * 1.00 (nominal)
        ECE = 0.10, bias 0.05:  modifier = 0.05 -> decay_rate * 0.95 (slower)
        ECE = 0.20, bias 0.10:  modifier = 0.11 -> decay_rate * 0.89
        ECE = 0.30, bias 0.15:  modifier = 0.17 -> decay_rate * 0.83
        Max:                     modifier = 0.20 -> decay_rate * 0.80

        Returns:
            Decay reduction factor in [0.0, 0.20]. Higher = slower decay.
        """
        if not self._calibration_active:
            return 0.0

        ece = self._calibration_ece
        bias = abs(self._calibration_bias)

        kappa = 0.5  # Decay sensitivity factor

        modifier = kappa * ece * (1.0 + bias)
        return min(0.20, max(0.0, modifier))

    def _compute_calibration_weight_rebalance(
        self,
    ) -> tuple[float, float]:
        """Compute scoring weight adjustments from calibration quality.

        When the system is poorly calibrated, its confidence in its own
        task-relevance judgments is unreliable. This method shifts weight
        between α (surprisal) and β (mutual information / task relevance):

        - **Overconfident**: shift β → α. The system over-trusts its
          relevance scoring, so it should seek novel/surprising content
          instead of relying on potentially wrong task-relevance judgments.
          Intuition: "I'm wrong about what's important → seek the unexpected."

        - **Underconfident**: shift α → β. The system undervalues its
          own relevance judgments, so it should trust them more and avoid
          being distracted by novel but irrelevant content.
          Intuition: "I'm actually better at judging relevance than I think
          → trust my task focus."

        - **Well-calibrated / moderate**: no shift. The system's weight
          allocation is already trustworthy.

        This mechanism is ADDITIVE with tension modulation (same pattern
        as iter-008/009 retention/decay modifiers). It does NOT compose
        with discrete calibration gates (iter-010 pattern).

        Mathematical foundation:
            ECE threshold for activation: 0.15 (matches iter-010 moderate tier)

            Overconfident shift:
                Δ = η × ECE × (1 + |bias|) × 1.2  (overconfidence bonus)
                α += Δ,  β -= Δ

            Underconfident shift:
                Δ = η × ECE × (1 + |bias|)
                α -= Δ,  β += Δ

            η = 0.50 (weight rebalance sensitivity)
            Bounded: |Δ| ≤ 0.12 (max 12% weight shift, conservative)

        Returns:
            (alpha_delta, beta_delta) — additive adjustments to α and β.
            Both sum to zero (weight is redistributed, not created/destroyed).
        """
        if not self._calibration_active:
            return (0.0, 0.0)

        ece = self._calibration_ece
        bias = abs(self._calibration_bias)

        # Require significant miscalibration (ECE > 0.15) before rebalancing.
        # Below this threshold, the system's confidence is reliable enough.
        if ece < 0.15:
            return (0.0, 0.0)

        eta = 0.50  # Weight rebalance sensitivity
        max_shift = 0.12  # Cap at 12% weight transfer

        if self._calibration_overconfident:
            # Overconfident: shift β → α (seek novelty, distrust relevance)
            overconfidence_bonus = 1.2
            delta = eta * ece * (1.0 + bias) * overconfidence_bonus
            delta = min(delta, max_shift)
            return (delta, -delta)

        elif self._calibration_underconfident:
            # Underconfident: shift α → β (trust relevance, avoid distraction)
            delta = eta * ece * (1.0 + bias)
            delta = min(delta, max_shift)
            return (-delta, delta)

        # Moderate miscalibration (ECE > 0.15 but neither over/underconfident)
        return (0.0, 0.0)

    def collect_garbage(self) -> GarbageCollectionReport:
        """Run one garbage collection cycle.

        Scores all chunks, applies retention decisions, and returns
        a report of what was done. The decision thresholds are
        modulated by the current tension profile.

        This should be called periodically (every N ticks) or when
        the budget utilization exceeds a threshold.

        Returns:
            A report detailing what was kept, compressed, or pruned.
        """
        self._gc_cycles += 1

        # Get current tension profile for modulation
        profile = self._engine.get_behavior_profile()

        # Modulate scoring weights based on tensions
        alpha, beta, gamma, delta = self._modulate_weights(profile)

        # Modulate thresholds based on tensions
        keep_thresh, prune_thresh = self._modulate_thresholds(profile)

        # ── Calibration-aware retention ──
        # When poorly calibrated, lower thresholds to retain MORE context.
        # The calibration modifier is negative (wider retention), tension
        # modulation is additive, and they compose independently.
        cal_mod = self._compute_calibration_retention_modifier()
        keep_thresh += cal_mod
        prune_thresh += cal_mod * 0.8 # Proportional (prune is more aggressive)
        # Re-clamp after calibration adjustment
        keep_thresh = max(0.1, min(0.95, keep_thresh))
        prune_thresh = max(0.05, min(0.90, prune_thresh))

        # ── Calibration weight rebalance ──
        # Computed above during _modulate_weights; retrieve for reporting
        cal_alpha_delta, cal_beta_delta = self._compute_calibration_weight_rebalance()

        # Score all chunks
        scored: list[tuple[float, AttentionChunk]] = []
        for chunk in self._chunks.values():
            score = chunk.attention_score(
                alpha=alpha, beta=beta, gamma=gamma, delta=delta
            )
            scored.append((score, chunk))

        # Sort descending by attention score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply retention decisions
        kept: list[UUID] = []
        compressed: list[UUID] = []
        pruned: list[UUID] = []
        tokens_freed: int = 0

        for score, chunk in scored:
            decision = self._decide_retention(score, keep_thresh, prune_thresh)

            if decision == RetentionDecision.KEEP_FULL:
                kept.append(chunk.id)
                self._total_kept += 1
            elif decision == RetentionDecision.COMPRESS:
                # Compression: keep but reduce token count
                compressed.append(chunk.id)
                old_tokens = chunk.token_count
                new_tokens = int(old_tokens * self._compress_ratio)
                tokens_freed += old_tokens - new_tokens
                # Update the chunk in place with compressed token count
                self._chunks[chunk.id] = AttentionChunk(
                    id=chunk.id,
                    content=chunk.content,  # Content unchanged; compression is logical
                    token_count=new_tokens,
                    surprisal=chunk.surprisal * 0.7,  # Less surprisal post-compression
                    mutual_info=chunk.mutual_info * 0.8,
                    recency=chunk.recency,
                    task_relevance=chunk.task_relevance,
                    importance_tags=chunk.importance_tags,
                )
                self._total_compressed += 1
            else:
                pruned.append(chunk.id)
                tokens_freed += chunk.token_count
                self._total_pruned += 1
                del self._chunks[chunk.id]

        # Update budget
        self._budget.tokens_used -= tokens_freed

        return GarbageCollectionReport(
        	gc_cycle=self._gc_cycles,
        	chunks_before=len(scored),
        	chunks_after=len(kept) + len(compressed),
        	kept_count=len(kept),
        	compressed_count=len(compressed),
        	pruned_count=len(pruned),
        	tokens_freed=tokens_freed,
        	budget_utilization_before=max(0.0, self._budget.utilization),
        	budget_utilization_after=max(0.0, self._budget.utilization),
        	keep_threshold=keep_thresh,
        	prune_threshold=prune_thresh,
        	tension_profile=profile,
        	alpha=alpha,
        	beta=beta,
        	gamma=gamma,
        	delta=delta,
        	calibration_active=self._calibration_active,
        	calibration_ece=round(self._calibration_ece, 4),
        	calibration_modifier=round(cal_mod, 4),
        	calibration_weight_rebalance_alpha=round(cal_alpha_delta, 4),
        	calibration_weight_rebalance_beta=round(cal_beta_delta, 4),
        )

    def apply_recency_decay(self, decay_rate: float = 0.05) -> None:
        """Apply exponential decay to all chunks' recency scores.

        Called each tick to model the natural decay of attention over
        time. Newer content has higher recency.

        Args:
            decay_rate: Multiplicative decay factor per tick.
        """
        # ── Calibration-aware decay ──
        # When poorly calibrated, slow recency decay so older chunks
        # persist longer — the agent needs broader temporal context
        # when it's uncertain.
        cal_decay_mod = self._compute_calibration_decay_modifier()
        effective_rate = decay_rate * (1.0 - cal_decay_mod)

        for chunk_id, chunk in self._chunks.items():
            new_recency = chunk.recency * (1.0 - effective_rate)
            # Re-frozen dataclass: replace in dict
            self._chunks[chunk_id] = AttentionChunk(
                id=chunk.id,
                content=chunk.content,
                token_count=chunk.token_count,
                surprisal=chunk.surprisal,
                mutual_info=chunk.mutual_info,
                recency=new_recency,
                task_relevance=chunk.task_relevance,
                importance_tags=chunk.importance_tags,
            )

    def get_top_chunks(self, n: int = 10) -> list[AttentionChunk]:
        """Return the top N chunks by attention score (current profile)."""
        profile = self._engine.get_behavior_profile()
        alpha, beta, gamma, delta = self._modulate_weights(profile)

        scored = [
            (c.attention_score(alpha, beta, gamma, delta), c)
            for c in self._chunks.values()
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:n]]

    # ── Properties ───────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def budget(self) -> AttentionBudget:
        return self._budget

    @property
    def stats(self) -> dict:
        return {
            "chunks_active": len(self._chunks),
            "tokens_used": self._budget.tokens_used,
            "token_capacity": self._budget.token_capacity,
            "utilization": round(self._budget.utilization, 4),
            "gc_cycles": self._gc_cycles,
            "total_kept": self._total_kept,
            "total_compressed": self._total_compressed,
            "total_pruned": self._total_pruned,
            "entropy_estimate": round(self._estimate_entropy(), 4),
            "enforcement": {
                "policy": self._enforcement_policy.value,
                "threshold": self._enforcement_threshold,
                "auto_gc_triggered": self._enforcement_auto_gc_triggered,
                "rejections": self._enforcement_rejections,
                "auto_compressions": self._enforcement_auto_compressions,
                "oversized_rejections": self._enforcement_oversized_rejections,
                "post_gc_rejections": self._enforcement_post_gc_rejections,
            },
            "rejected_queue": self.rejected_queue.stats,
        }

    @property
    def equilibrium_chunks(self) -> tuple[AttentionChunk, ...]:
        """All current chunks (immutable view)."""
        return tuple(self._chunks.values())

    # ── Budget enforcement ──────────────────────────────────────

    def enforce_budget(self) -> GarbageCollectionReport | None:
        """Enforce the context window budget by triggering GC if needed.

        If the current utilization exceeds the enforcement threshold
        and the policy allows auto-GC (AUTO_GC or AUTO_COMPRESS),
        runs a garbage collection cycle to free space.

        Returns:
            A GarbageCollectionReport if GC was triggered, else None.
        """
        if self._budget.utilization < self._enforcement_threshold:
            return None

        if self._enforcement_policy == BudgetEnforcementPolicy.REJECT:
            return None

        # AUTO_GC or AUTO_COMPRESS: trigger GC
        self._enforcement_auto_gc_triggered += 1
        return self.collect_garbage()

    @property
    def enforcement_policy(self) -> BudgetEnforcementPolicy:
        """Current budget enforcement policy."""
        return self._enforcement_policy

    @property
    def enforcement_threshold(self) -> float:
        """Utilization threshold at which auto-GC triggers."""
        return self._enforcement_threshold

    @property
    def enforcement_stats(self) -> dict:
        """Enforcement-specific statistics."""
        return {
            "policy": self._enforcement_policy.value,
            "threshold": self._enforcement_threshold,
            "auto_gc_triggered": self._enforcement_auto_gc_triggered,
            "rejections": self._enforcement_rejections,
            "auto_compressions": self._enforcement_auto_compressions,
            "oversized_rejections": self._enforcement_oversized_rejections,
            "post_gc_rejections": self._enforcement_post_gc_rejections,
        }

    # ── Tension modulation ──────────────────────────────────────

    def _modulate_weights(
        self, profile: dict[TensionID, float]
    ) -> tuple[float, float, float, float]:
        """Modulate attention score weights based on tension positions.

        - 'shallow_deep': Shallow → increase gamma (recency), decrease beta (MI)
                         Deep   → increase beta, decrease gamma
        - 'explore_exploit': Explore → increase alpha (surprisal)
                            Exploit → increase beta (task MI)
        - 'divergent_convergent': Divergent → increase alpha (keep diverse)
                                 Convergent → increase beta (focus)
        """
        shallow = profile.get("shallow_deep", 0.0)  # >0 = deep, <0 = shallow
        explore = profile.get("explore_exploit", 0.0)  # >0 = exploit, <0 = explore
        diverge = profile.get("divergent_convergent", 0.0)  # >0 = convergent

        alpha = self.DEFAULT_ALPHA
        beta = self.DEFAULT_BETA
        gamma = self.DEFAULT_GAMMA
        delta = self.DEFAULT_DELTA

        # Shallow/Deep modulation (±30% swing)
        if shallow < 0:  # Shallow mode: favor recency
            gamma += 0.12
            beta -= 0.12
        else:  # Deep mode: favor task relevance
            beta += 0.12
            gamma -= 0.08

        # Explore/Exploit modulation (±15% swing)
        if explore < 0:  # Explore: favor surprisal (novelty)
            alpha += 0.10
        else:  # Exploit: favor task relevance
            beta += 0.08

        # Divergent/Convergent modulation (±10% swing)
        if diverge < 0: # Divergent: favor surprisal (diversity)
        	alpha += 0.06
        else: # Convergent: favor task relevance
        	beta += 0.06

        # Calibration-driven weight rebalance (iter-016)
        # When miscalibrated, shift weight between α (surprisal) and β (MI).
        # Overconfident → β→α (seek novelty), Underconfident → α→β (trust relevance)
        cal_alpha_delta, cal_beta_delta = self._compute_calibration_weight_rebalance()
        alpha += cal_alpha_delta
        beta += cal_beta_delta

        # Normalize to sum to 1.0
        total = alpha + beta + gamma + delta
        return (alpha / total, beta / total, gamma / total, delta / total)

    def _modulate_thresholds(
        self, profile: dict[TensionID, float]
    ) -> tuple[float, float]:
        """Modulate retention decision thresholds based on tensions.

        - 'shallow_deep': Shallow → raise thresholds (more aggressive pruning)
                         Deep   → lower thresholds (keep more)
                         Proportional: deeper negative = higher thresholds
        - 'consolidate_prune': Consolidate → lower prune threshold
                               Prune      → raise prune threshold
        """
        shallow = profile.get("shallow_deep", 0.0)
        consolidate = profile.get("consolidate_prune", 0.0)

        keep_thresh = self._keep_threshold
        prune_thresh = self._prune_threshold

        # Proportional shallow/deep modulation: -1.0 → +0.15, +1.0 → -0.10
        keep_thresh += -shallow * 0.10  # Negative shallow → higher threshold
        prune_thresh += -shallow * 0.08

        # Prune mode: raise thresholds
        if consolidate > 0:
            prune_thresh += consolidate * 0.10

        return (
            max(0.1, min(0.95, keep_thresh)),
            max(0.05, min(0.90, prune_thresh)),
        )

    # ── Information-theoretic computation ────────────────────────

    def _compute_surprisal(self, content: str) -> float:
        """Estimate surprisal: how unexpected is this content?

        Uses token-level n-gram frequencies to estimate P(content)
        and returns -log₂ P(content). Higher = more surprising.

        For efficiency, we use a unigram model with Laplace smoothing.
        """
        if self._total_tokens_seen == 0:
            # No prior data — all tokens are maximally surprising
            return 3.0  # Reasonable default for novel tokens

        tokens = content.lower().split()
        if not tokens:
            return 0.0

        total_surprisal = 0.0
        vocab_size = len(self._token_frequencies) + 1  # +1 for smoothing

        for token in tokens:
            freq = self._token_frequencies.get(token, 0)
            # Laplace-smoothed probability
            prob = (freq + 1) / (self._total_tokens_seen + vocab_size)
            total_surprisal += -math.log2(max(prob, 1e-15))

        return total_surprisal / len(tokens)  # Average surprisal per token

    def _update_token_frequencies(self, content: str) -> None:
        """Update frequency counts for surprisal computation."""
        for token in content.lower().split():
            self._token_frequencies[token] = (
                self._token_frequencies.get(token, 0) + 1
            )
            self._total_tokens_seen += 1

    def _estimate_entropy(self) -> float:
        """Estimate the entropy of the observed token distribution."""
        if self._total_tokens_seen == 0:
            return 0.0
        entropy = 0.0
        for freq in self._token_frequencies.values():
            p = freq / self._total_tokens_seen
            entropy -= p * math.log2(p)
        return entropy

    def _estimate_tokens(self, content: str) -> int:
        """Quick token count estimate (~4 chars per token for English)."""
        return max(1, len(content) // 4)

    def _decide_retention(
        self, score: float, keep_thresh: float, prune_thresh: float
    ) -> RetentionDecision:
        """Map attention score to retention decision."""
        if score >= keep_thresh:
            return RetentionDecision.KEEP_FULL
        elif score >= prune_thresh:
            return RetentionDecision.COMPRESS
        else:
            return RetentionDecision.PRUNE


@dataclass
class GarbageCollectionReport:
    """Detailed report of a garbage collection cycle."""

    gc_cycle: int
    chunks_before: int
    chunks_after: int
    kept_count: int
    compressed_count: int
    pruned_count: int
    tokens_freed: int
    budget_utilization_before: float
    budget_utilization_after: float
    keep_threshold: float
    prune_threshold: float
    tension_profile: dict[TensionID, float]
    alpha: float
    beta: float
    gamma: float
    delta: float
    calibration_active: bool = False
    calibration_ece: float = 0.0
    calibration_modifier: float = 0.0
    calibration_weight_rebalance_alpha: float = 0.0
    calibration_weight_rebalance_beta: float = 0.0

    def summary(self) -> str:
    	base = (
    		f"GC#{self.gc_cycle}: {self.chunks_before}→{self.chunks_after} chunks "
    		f"(kept={self.kept_count}, comp={self.compressed_count}, "
    		f"pruned={self.pruned_count}) | "
    		f"freed {self.tokens_freed} tokens | "
    		f"util {self.budget_utilization_before:.1%}→{self.budget_utilization_after:.1%} | "
    		f"thresholds k={self.keep_threshold:.2f} p={self.prune_threshold:.2f}"
    	)
    	if self.calibration_active and abs(self.calibration_modifier) > 0.001:
    		base += f" | calΔ={self.calibration_modifier:+.3f} (ECE={self.calibration_ece:.3f})"
    	if self.calibration_active and (
    		abs(self.calibration_weight_rebalance_alpha) > 0.001
    		or abs(self.calibration_weight_rebalance_beta) > 0.001
    	):
    		base += (
    			f" | calWΔ α={self.calibration_weight_rebalance_alpha:+.4f}"
    			f" β={self.calibration_weight_rebalance_beta:+.4f}"
    		)
    	return base
