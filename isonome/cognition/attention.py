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

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence
from uuid import UUID, uuid4

import numpy as np
from scipy.special import softmax  # type: ignore[import-untyped]

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
    ):
        """Initialize the attention equilibrium system.

        Args:
            engine: The equilibrium engine for tension-driven modulation.
            token_capacity: Max tokens the context window can hold.
            compress_ratio: Target compression ratio (fraction of original).
            keep_threshold: Minimum score to retain verbatim.
            prune_threshold: Below this, chunks are pruned.
        """
        self._engine = engine
        self._budget = AttentionBudget(token_capacity=token_capacity)
        self._chunks: dict[UUID, AttentionChunk] = {}
        self._compress_ratio = compress_ratio
        self._keep_threshold = keep_threshold
        self._prune_threshold = prune_threshold

        # Statistics
        self._total_pruned: int = 0
        self._total_compressed: int = 0
        self._total_kept: int = 0
        self._gc_cycles: int = 0

        # Entropy tracking for surprisal computation
        self._token_frequencies: dict[str, int] = {}
        self._total_tokens_seen: int = 0

    # ── Public API ───────────────────────────────────────────────

    def add_chunk(
        self,
        content: str,
        *,
        token_count: int | None = None,
        mutual_info: float = 0.0,
        task_relevance: float = 0.5,
        importance_tags: tuple[str, ...] = (),
    ) -> AttentionChunk:
        """Register a new chunk of context.

        Automatically computes surprisal based on prior context and
        assigns recency = 1.0 (newest).

        Args:
            content: The text content of the chunk.
            token_count: Number of tokens (estimated if None).
            mutual_info: Estimated mutual information with the task.
            task_relevance: Estimated relevance to current task [0, 1].
            importance_tags: Tags marking explicit importance.

        Returns:
            The created AttentionChunk.
        """
        if token_count is None:
            token_count = self._estimate_tokens(content)

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
        )

    def apply_recency_decay(self, decay_rate: float = 0.05) -> None:
        """Apply exponential decay to all chunks' recency scores.

        Called each tick to model the natural decay of attention over
        time. Newer content has higher recency.

        Args:
            decay_rate: Multiplicative decay factor per tick.
        """
        for chunk_id, chunk in self._chunks.items():
            new_recency = chunk.recency * (1.0 - decay_rate)
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
        }

    @property
    def equilibrium_chunks(self) -> tuple[AttentionChunk, ...]:
        """All current chunks (immutable view)."""
        return tuple(self._chunks.values())

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
        if diverge < 0:  # Divergent: favor surprisal (diversity)
            alpha += 0.06
        else:  # Convergent: favor task relevance
            beta += 0.06

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

    def summary(self) -> str:
        return (
            f"GC#{self.gc_cycle}: {self.chunks_before}→{self.chunks_after} chunks "
            f"(kept={self.kept_count}, comp={self.compressed_count}, "
            f"pruned={self.pruned_count}) | "
            f"freed {self.tokens_freed} tokens | "
            f"util {self.budget_utilization_before:.1%}→{self.budget_utilization_after:.1%} | "
            f"thresholds k={self.keep_threshold:.2f} p={self.prune_threshold:.2f}"
        )
