"""μνήμη (Mneme) — Hierarchical memory, learning, and knowledge persistence.

A three-tier memory architecture bridging transient context to durable knowledge:

    Working  (seconds–minutes)  →  high-fidelity, capacity-bounded, LRU eviction
    Episodic (hours–days)       →  timestamped experience records with decay
    Semantic (persistent)       →  consolidated abstractions, patterns, facts

The system is governed by two equilibrium tensions:
    - consolidate_prune:  Consolidate → aggressively move WM→Episodic→Semantic
                          Prune       → aggressively evict low-significance entries
    - specific_general:   Specific    → retain verbatim details
                          General     → extract patterns, discard specifics

Core mechanisms:
    1. Ebbinghaus forgetting curves with spaced repetition
       R(t) = e^(-t/S) where S = strength after reinforcement
    2. Significance gating: only memories above threshold consolidate
    3. Frequency-weighted pattern extraction for semantic abstraction
    4. Multi-level recall modulated by tension positions
    5. Full serialization for cross-session persistence

Pillar integration:
    - Receives GarbageCollectionReports from Attention (νοῦς) — pruned chunks
      are evaluated for long-term storage before final deletion
    - Emits Feedback(signal, confidence) to modulate consolidation pace
    - Responds to Signal(e.g., 'consolidate_now', 'recall:<query>') from other pillars
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence
from uuid import UUID, uuid4

from isonome.types import (
    Feedback,
    TensionID,
)
from isonome.utils.frozendict import frozendict

# ═══════════════════════════════════════════════════════════════════
# Memory tiers
# ═══════════════════════════════════════════════════════════════════


class MemoryTier(Enum):
    """The three tiers of the hierarchical memory system."""

    WORKING = auto()  # Transient, high-detail, capacity-bounded
    EPISODIC = auto()  # Timestamped experience records
    SEMANTIC = auto()  # Consolidated abstractions, facts, patterns


# ═══════════════════════════════════════════════════════════════════
# Helper types
# ═══════════════════════════════════════════════════════════════════


# Immutable dict for MemoryEntry metadata — must be defined before MemoryEntry
# frozendict imported from isonome.utils (see utils/frozendict.py)


# ═══════════════════════════════════════════════════════════════════
# Core data structures
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single memory item at any tier.

    Memory entries are frozen for safe sharing across concurrent
    pillar operations. Mutations produce new entries.
    """

    content: str = field(repr=False)
    id: UUID = field(default_factory=uuid4)
    tier: MemoryTier = MemoryTier.WORKING
    strength: float = 1.0  # Memory strength [0, 1] — decays over time
    significance: float = 0.5  # Estimated importance [0, 1]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_rehearsed: float = field(default_factory=time.time)
    rehearsal_count: int = 0
    access_count: int = 0
    source: str | None = None  # e.g., "attention_prune", "user_input"
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: frozendict = field(default_factory=frozendict)  # type: ignore[valid-type]

    # ── Forgetting curve parameters ────────────────────────────
    # After consolidation, the forgetting half-life extends.
    # Base half-life: how long (seconds) until strength drops to 0.5.
    base_half_life: float = 3600.0  # 1 hour for working memory

    def forget(self, current_time: float | None = None) -> MemoryEntry:
        """Apply Ebbinghaus decay: R(t) = e^(-t·ln(2)/S).

        Where S is the effective half-life (extended by rehearsal).
        """
        t = current_time if current_time is not None else time.time()
        elapsed = t - self.last_rehearsed
        effective_hl = self._effective_half_life()
        if effective_hl <= 0:
            new_strength = 0.0
        else:
            decay = math.exp(-elapsed * math.log(2) / effective_hl)
            new_strength = self.strength * decay
            new_strength = max(0.0, min(1.0, new_strength))

        return object.__new__(MemoryEntry) if False else MemoryEntry(
            id=self.id,
            content=self.content,
            tier=self.tier,
            strength=new_strength,
            significance=self.significance,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            last_rehearsed=self.last_rehearsed,
            rehearsal_count=self.rehearsal_count,
            access_count=self.access_count,
            source=self.source,
            tags=self.tags,
            metadata=self.metadata,
            base_half_life=self.base_half_life,
        )

    def rehearse(self, boost: float = 0.15, current_time: float | None = None) -> MemoryEntry:
        """Strengthen memory via rehearsal (spaced repetition boost).

        Each rehearsal:
        1. Boosts current strength
        2. Increments rehearsal_count (half-life extension is in _effective_half_life)
        3. Extends the effective half-life (spacing effect)
        """
        t = current_time if current_time is not None else time.time()
        new_strength = min(1.0, self.strength + boost)

        return MemoryEntry(
            id=self.id,
            content=self.content,
            tier=self.tier,
            strength=new_strength,
            significance=self.significance,
            created_at=self.created_at,
            last_accessed=t,
            last_rehearsed=t,
            rehearsal_count=self.rehearsal_count + 1,
            access_count=self.access_count + 1,
            source=self.source,
            tags=self.tags,
            metadata=self.metadata,
            base_half_life=self.base_half_life,  # Base stays fixed; rehearsal_count extends in _effective_half_life
        )

    def access(self, current_time: float | None = None) -> MemoryEntry:
        """Record an access (retrieval) — slight rehearsal boost."""
        t = current_time if current_time is not None else time.time()
        # Access provides a tiny boost (retrieval practice effect)
        new_strength = min(1.0, self.strength + 0.03)
        return MemoryEntry(
            id=self.id,
            content=self.content,
            tier=self.tier,
            strength=new_strength,
            significance=self.significance,
            created_at=self.created_at,
            last_accessed=t,
            last_rehearsed=self.last_rehearsed,
            rehearsal_count=self.rehearsal_count,
            access_count=self.access_count + 1,
            source=self.source,
            tags=self.tags,
            metadata=self.metadata,
            base_half_life=self.base_half_life,
        )

    def promote(self, to_tier: MemoryTier) -> MemoryEntry:
        """Promote this entry to a higher memory tier."""
        # On promotion, boost half-life significantly
        tier_multiplier = {
            MemoryTier.WORKING: 1.0,
            MemoryTier.EPISODIC: 24.0,  # 24× longer half-life
            MemoryTier.SEMANTIC: 720.0,  # 720× longer (essentially permanent)
        }
        return MemoryEntry(
            id=self.id,
            content=self.content,
            tier=to_tier,
            strength=self.strength,
            significance=self.significance,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            last_rehearsed=self.last_rehearsed,
            rehearsal_count=self.rehearsal_count,
            access_count=self.access_count,
            source=self.source,
            tags=self.tags,
            metadata=self.metadata,
            base_half_life=self.base_half_life * tier_multiplier.get(to_tier, 1.0),
        )

    def _effective_half_life(self) -> float:
        """Compute the effective half-life including rehearsal extensions.

        The spacing effect: each rehearsal extends the forgetting curve.
        After n rehearsals, effective_HL = base_HL × 1.5ⁿ
        """
        return self.base_half_life * (1.5 ** self.rehearsal_count)

    def is_forgotten(self, threshold: float = 0.05) -> bool:
        """Whether this memory has decayed below the retention threshold."""
        return self.strength < threshold

    def content_hash(self) -> str:
        """Stable content hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════
# Hierarchical Mneme System
# ═══════════════════════════════════════════════════════════════════


@dataclass
class ConsolidationEvent:
    """Record of a memory consolidation action."""

    entry_id: UUID
    from_tier: MemoryTier
    to_tier: MemoryTier
    significance: float
    strength_at_consolidation: float
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class MnemeStats:
    """Aggregate statistics for the Mneme system."""

    working_count: int = 0
    episodic_count: int = 0
    semantic_count: int = 0
    total_consolidations: int = 0
    total_pruned: int = 0
    total_rehearsals: int = 0
    total_retrievals: int = 0
    consolidation_events: list[ConsolidationEvent] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "working_memories": self.working_count,
            "episodic_memories": self.episodic_count,
            "semantic_memories": self.semantic_count,
            "total_consolidations": self.total_consolidations,
            "total_pruned": self.total_pruned,
            "total_rehearsals": self.total_rehearsals,
            "total_retrievals": self.total_retrievals,
        }


class HierarchicalMneme:
    """Three-tier memory with Ebbinghaus decay and spaced repetition.

    This is THE Mneme pillar implementation. It manages:
    - WorkingMemory: LRU-bounded, high-fidelity, short-lived
    - EpisodicMemory: timestamped records with medium decay
    - SemanticMemory: abstracted patterns/knowledge with near-permanent retention

    Tension modulation:
    - consolidate_prune < 0 (Consolidate):
        → Lower significance threshold for consolidation
        → More aggressive WM → Episodic → Semantic promotion
    - consolidate_prune > 0 (Prune):
        → Raise significance threshold
        → Aggressively evict low-strength entries
    - specific_general < 0 (Specific):
        → Keep verbatim content during consolidation
    - specific_general > 0 (General):
        → Abstract patterns, discard specifics during consolidation
    """

    # ── Capacity limits ─────────────────────────────────────────

    WORKING_CAPACITY = 7  # Miller's Law: ~7±2 items in working memory
    EPISODIC_CAPACITY = 1000
    SEMANTIC_CAPACITY = 10000

    # ── Consolidation thresholds (modulated by tensions) ─────────

    DEFAULT_CONSOLIDATION_SIGNIFICANCE = 0.5
    DEFAULT_PROMOTION_SIGNIFICANCE = 0.7
    DEFAULT_PATTERN_COUNT_THRESHOLD = 3  # Episodic→Semantic needs ≥3 related events

    # ── Forgetting thresholds ────────────────────────────────────

    FORGET_THRESHOLD = 0.05  # Strength below this → candidate for eviction
    DEFAULT_REHEARSAL_BOOST = 0.15

    def __init__(
        self,
        *,
        consolidation_significance: float | None = None,
        promotion_significance: float | None = None,
        pattern_count_threshold: int | None = None,
        rehearsal_boost: float | None = None,
    ):
        self._consolidation_sig = (
            consolidation_significance or self.DEFAULT_CONSOLIDATION_SIGNIFICANCE
        )
        self._promotion_sig = (
            promotion_significance or self.DEFAULT_PROMOTION_SIGNIFICANCE
        )
        self._pattern_threshold = (
            pattern_count_threshold or self.DEFAULT_PATTERN_COUNT_THRESHOLD
        )
        self._rehearsal_boost = rehearsal_boost or self.DEFAULT_REHEARSAL_BOOST

        # Ordered access for LRU eviction in working memory
        self._working: OrderedDict[UUID, MemoryEntry] = OrderedDict()
        self._episodic: OrderedDict[UUID, MemoryEntry] = OrderedDict()
        self._semantic: dict[UUID, MemoryEntry] = {}

        # Pattern extraction state
        self._pattern_frequencies: dict[str, int] = {}  # n-gram → count
        self._tag_cooccurrence: dict[frozenset[str], int] = {}  # tag pair → count

        # Stats
        self._stats = MnemeStats()
        self._consolidation_log: deque[ConsolidationEvent] = deque(maxlen=1000)

        # Feedback accumulator
        self._pending_feedback: list[Feedback] = []

        # Tension profile cache (set by the pillar wrapper)
        self._current_profile: dict[TensionID, float] = {}
        
        # Calibration state (set by CognitionPillar each tick)
        self._calibration_ece: float = 0.0
        self._calibration_bias: float = 0.0
        self._calibration_overconfident: bool = False
        self._calibration_underconfident: bool = False
        self._calibration_total_predictions: int = 0

    # ══════════════════════════════════════════════════════════════
    # Public API — Storage
    # ══════════════════════════════════════════════════════════════

    def store(
        self,
        content: str,
        *,
        significance: float = 0.5,
        source: str | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Store new content in working memory.

        If working memory is full, the lowest-strength entry is evicted
        (after evaluating for episodic consolidation).

        Args:
            content: The text content to store.
            significance: Estimated importance [0, 1]. Higher = more likely to consolidate.
            source: Where this memory came from (e.g., 'attention_prune').
            tags: Categorical tags for retrieval and pattern extraction.
            metadata: Optional structured metadata.

        Returns:
            The created MemoryEntry.
        """
        # Evict if at capacity
        if len(self._working) >= self.WORKING_CAPACITY:
            self._evict_working()

        entry = MemoryEntry(
            content=content,
            tier=MemoryTier.WORKING,
            significance=significance,
            source=source,
            tags=tags,
            metadata=frozendict(metadata or {}),
        )
        self._working[entry.id] = entry
        self._update_patterns(content, tags)
        return entry

    def store_batch(
        self,
        items: Sequence[tuple[str, float]],
        *,
        source: str | None = None,
    ) -> list[MemoryEntry]:
        """Store multiple items efficiently.

        Each item is (content, significance).
        """
        results = []
        for content, sig in items:
            results.append(self.store(content, significance=sig, source=source))
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text for recall matching, stripping punctuation.

        Splits on whitespace and strips leading/trailing punctuation
        so that quoted words like 'deploy' match unquoted 'deploy'.
        """
        return {re.sub(r'^\W+|\W+$', '', t) for t in text.lower().split() if re.sub(r'^\W+|\W+$', '', t)}

    # ══════════════════════════════════════════════════════════════
    # Public API — Retrieval
    # ══════════════════════════════════════════════════════════════

    def recall(
        self,
        query: str,
        *,
        max_results: int = 10,
        tier_filter: MemoryTier | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching a query across all tiers.

        Uses a multi-strategy relevance scoring:
        1. Text overlap (Jaccard on token sets)
        2. Tag match
        3. Strength-weighted rank

        Tension modulation:
        - specific_general < 0 (Specific): favor exact matches
        - specific_general > 0 (General): favor semantic overlap

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            tier_filter: Optional tier to restrict search to.

        Returns:
            Sorted list of matching MemoryEntry items (strongest first).
        """
        query_tokens = self._tokenize(query)
        specific_bias = self._current_profile.get("specific_general", 0.0)

        candidates: list[tuple[float, MemoryEntry]] = []

        tiers = [tier_filter] if tier_filter else list(MemoryTier)
        tier_collections = {
            MemoryTier.WORKING: self._working,
            MemoryTier.EPISODIC: self._episodic,
            MemoryTier.SEMANTIC: self._semantic,
        }

        for tier in tiers:
            for entry in tier_collections.get(tier, {}).values():
                if entry.is_forgotten(self.FORGET_THRESHOLD):
                    continue

                score = self._relevance_score(
                    entry, query_tokens, specific_bias=specific_bias
                )
                if score > 0:
                    candidates.append((score, entry))

        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Return top results, marking them as accessed
        results = []
        for score, entry in candidates[:max_results]:
            accessed = entry.access()
            self._update_in_tier(accessed)
            results.append(accessed)
            self._stats.total_retrievals += 1

        return results

    def recall_by_tags(
        self,
        tags: frozenset[str],
        *,
        max_results: int = 10,
        match_all: bool = False,
    ) -> list[MemoryEntry]:
        """Retrieve memories by exact tag matching.

        Args:
            tags: Set of tags to match.
            max_results: Maximum results.
            match_all: If True, all tags must match. If False, any tag matches.

        Returns:
            Sorted list by strength (strongest first).
        """
        results: list[MemoryEntry] = []
        all_entries = list(self._working.values()) + list(self._episodic.values()) + list(self._semantic.values())

        for entry in all_entries:
            if entry.is_forgotten(self.FORGET_THRESHOLD):
                continue
            entry_tags = set(entry.tags)
            if match_all:
                if tags.issubset(entry_tags):
                    results.append(entry)
            else:
                if tags & entry_tags:
                    results.append(entry)

        results.sort(key=lambda e: e.strength, reverse=True)
        return results[:max_results]

    # ══════════════════════════════════════════════════════════════
    # Public API — Consolidation
    # ══════════════════════════════════════════════════════════════

    def consolidate(self) -> ConsolidationReport:
        """Run one full consolidation cycle.

        This is the central maintenance operation:
        1. Apply forgetting curves to all entries
        2. Evaluate WM entries for episodic promotion (significance-gated)
        3. Evaluate episodic entries for semantic promotion (pattern-gated)
        4. Prune forgotten entries
        5. Extract semantic patterns from episodic clusters

        The consolidation and promotion thresholds are modulated by
        the current tension profile.

        Returns:
            A ConsolidationReport with detailed metrics.
        """
        current_time = time.time()

        # Modulate thresholds based on tensions
        cons_sig, prom_sig = self._modulate_thresholds()

        # Phase 1: Apply forgetting
        _forgotten_working = 0  # noqa: F841
        _forgotten_episodic = 0  # noqa: F841
        _forgotten_semantic = 0  # noqa: F841

        for tier, collection, counter_attr in [
            (MemoryTier.WORKING, self._working, "forgotten_working"),
            (MemoryTier.EPISODIC, self._episodic, "forgotten_episodic"),
            (MemoryTier.SEMANTIC, self._semantic, "forgotten_semantic"),
        ]:
            for eid, entry in list(collection.items()):
                decayed = entry.forget(current_time)
                if decayed.strength != entry.strength:
                    collection[eid] = decayed

        # Phase 2: WM → Episodic (significance-gated)
        wm_to_episodic = 0
        for eid, entry in list(self._working.items()):
            if entry.significance >= cons_sig:
                promoted = entry.promote(MemoryTier.EPISODIC).rehearse(
                    boost=self._rehearsal_boost, current_time=current_time
                )
                self._episodic[eid] = promoted
                del self._working[eid]
                wm_to_episodic += 1
                self._stats.total_consolidations += 1
                self._consolidation_log.append(
                    ConsolidationEvent(
                        entry_id=eid,
                        from_tier=MemoryTier.WORKING,
                        to_tier=MemoryTier.EPISODIC,
                        significance=entry.significance,
                        strength_at_consolidation=entry.strength,
                        reason=f"significance {entry.significance:.2f} >= {cons_sig:.2f}",
                    )
                )

        # Phase 3: Episodic → Semantic (pattern-gated)
        ep_to_semantic = 0
        for eid, entry in list(self._episodic.items()):
            if entry.significance >= prom_sig and self._has_pattern_support(entry):
                promoted = entry.promote(MemoryTier.SEMANTIC).rehearse(
                    boost=self._rehearsal_boost * 2, current_time=current_time
                )
                self._semantic[eid] = promoted
                del self._episodic[eid]
                ep_to_semantic += 1
                self._stats.total_consolidations += 1
                self._consolidation_log.append(
                    ConsolidationEvent(
                        entry_id=eid,
                        from_tier=MemoryTier.EPISODIC,
                        to_tier=MemoryTier.SEMANTIC,
                        significance=entry.significance,
                        strength_at_consolidation=entry.strength,
                        reason=f"significance {entry.significance:.2f} >= {prom_sig:.2f} + pattern support",
                    )
                )

        # Phase 4: Prune forgotten
        pruned = 0
        for collection, counter_attr in [
            (self._working, "forgotten_working"),
            (self._episodic, "forgotten_episodic"),
            (self._semantic, "forgotten_semantic"),
        ]:
            to_remove = [
                eid
                for eid, entry in collection.items()
                if entry.is_forgotten(self.FORGET_THRESHOLD)
            ]
            for eid in to_remove:
                del collection[eid]
                pruned += 1
            if counter_attr == "forgotten_working":
                pass  # count tracked via pruned counter
            elif counter_attr == "forgotten_episodic":
                pass  # count tracked via pruned counter
            else:
                pass  # count tracked via pruned counter

        # ── Calibration-aware pruning sensitivity ──
        # When the agent is overconfident, it underestimates how much
        # memorized content may still be needed. Reduce pruning rate
        # proportionally to how overconfident the system is.
        #   λ = 1 - min(overconfident_bonus, 0.30)
        #   pruned_effective = int(pruned × λ)
        if self._calibration_total_predictions >= 10 and self._calibration_overconfident and pruned > 0:
            ece = self._calibration_ece
            overconfident_prune_discount = min(0.30, ece * 2.0)  # Up to 30% reduction
            reduced_prune = int(pruned * (1.0 - overconfident_prune_discount))
            spared = pruned - reduced_prune
            pruned = reduced_prune
            # Re-add spared entries to working memory (they get a second chance)
            # This creates a reserve pool — entries that OVERCONFIDENCE would have
            # discarded but CALIBRATION saved.
            if spared > 0 and counter_attr == "forgotten_working":
                # Re-register spared entries into working memory at reduced strength
                re_added = 0
                for eid in list(self._working.keys()):
                    if re_added >= spared:
                        break
                    entry = self._working.get(eid)
                    if entry is not None and entry.strength < self.FORGET_THRESHOLD * 3:
                        # Give it a small reprieve — boost just above the threshold
                        self._working[eid] = MemoryEntry(
                            id=entry.id,
                            content=entry.content,
                            tier=entry.tier,
                            strength=self.FORGET_THRESHOLD * 1.5,
                            significance=entry.significance,
                            created_at=entry.created_at,
                            last_accessed=entry.last_accessed,
                            last_rehearsed=entry.last_rehearsed,
                            rehearsal_count=entry.rehearsal_count,
                            access_count=entry.access_count,
                            source=entry.source,
                            tags=entry.tags,
                            metadata=entry.metadata,
                            base_half_life=entry.base_half_life,
                        )
                        re_added += 1
                if re_added > 0:
                    calibration_prune_saved = re_added
                else:
                    calibration_prune_saved = 0
            else:
                calibration_prune_saved = 0
        else:
            overconfident_prune_discount = 0.0
            calibration_prune_saved = 0

        self._stats.total_pruned += pruned

        # Phase 5: Enforce capacity limits
        self._enforce_capacities()

        # Update stats
        self._stats.working_count = len(self._working)
        self._stats.episodic_count = len(self._episodic)
        self._stats.semantic_count = len(self._semantic)

        calibration_active = self._calibration_total_predictions >= 10

        return ConsolidationReport(
            working_count=len(self._working),
            episodic_count=len(self._episodic),
            semantic_count=len(self._semantic),
            wm_to_episodic=wm_to_episodic,
            ep_to_semantic=ep_to_semantic,
            pruned=pruned,
            thresholds=(cons_sig, prom_sig),
            tension_profile=dict(self._current_profile),
            calibration_ece=round(self._calibration_ece, 4) if calibration_active else 0.0,
            calibration_active=calibration_active,
            calibration_prune_saved=calibration_prune_saved,
        )

    def import_from_attention(
        self,
        content: str,
        attention_score: float,
        *,
        tags: tuple[str, ...] = (),
    ) -> MemoryEntry | None:
        """Handle a chunk pruned from the Attention Equilibrium System.

        Converts the attention score to a significance estimate and
        stores it in working memory if it meets the minimum threshold.

        This is the primary cross-pillar integration point: νοῦς → μνήμη.

        Args:
            content: The pruned chunk content.
            attention_score: The attention score at the time of pruning.
            tags: Optional tags from the attention system.

        Returns:
            The MemoryEntry if stored, None if rejected.
        """
        # Map attention score [0,1] to significance [0,1] with a sigmoid
        # Low attention score → low significance (probably noise)
        # High attention score → potentially worth remembering
        significance = 1.0 / (1.0 + math.exp(-8.0 * (attention_score - 0.35)))

        # Determine minimum significance floor — calibration-aware
        # Well-calibrated: standard 0.15 floor (trust attention scores)
        # Underconfident: lower floor to 0.08 (compensate for undervaluation)
        # Overconfident: raise floor to 0.20 (need stronger evidence)
        min_sig = 0.15
        if self._calibration_total_predictions >= 10:
            if self._calibration_underconfident and self._calibration_ece > 0.15:
                min_sig = 0.08  # More permissive — system undervalues everything
            elif self._calibration_overconfident and self._calibration_ece > 0.15:
                min_sig = 0.20  # Stricter — system thinks marginal content is relevant

        if significance < min_sig:
            return None

        return self.store(
            content,
            significance=significance,
            source="attention_prune",
            tags=tags,
        )

    # ══════════════════════════════════════════════════════════════
    # Public API — Rehearsal
    # ══════════════════════════════════════════════════════════════

    def rehearse(
        self,
        entry_id: UUID,
        *,
        boost: float | None = None,
    ) -> MemoryEntry | None:
        """Manually rehearse a memory to strengthen it.

        This can be triggered by the equilibrium engine when an agent
        revisits a task or context similar to a stored memory.
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            return None

        rehearse_boost = boost or self._rehearsal_boost
        strengthened = entry.rehearse(boost=rehearse_boost)
        self._update_in_tier(strengthened)
        self._stats.total_rehearsals += 1
        return strengthened

    def rehearse_by_tags(
        self,
        tags: frozenset[str],
        *,
        boost: float | None = None,
    ) -> int:
        """Rehearse all memories matching given tags.

        Calibration-aware rehearsal prioritization:
        - Well-calibrated (ECE ≤ 0.05): boost high-significance entries
          (~10% bonus), standard boost for medium-significance, skip
          low-significance. Trust significance judgments.
        - Moderate miscalibration (ECE 0.05-0.15): standard equal boost
          for all matched entries. Neutral — no correction needed.
        - Overconfident (ECE > 0.15, overconfident flag):
          distributed rehearsal: spread the total rehearsal budget across
          MORE entries with a SMALLER per-entry boost. Prevents the
          overconfident system from over-investing in what it thinks is
          important, while still refreshing borderline memories.
        - Underconfident (ECE > 0.15, underconfident flag):
          uniform boost across all entries, slightly higher than default.
          Compensates for the system undervaluing its own judgments.

        Returns the count of memories rehearsed.
        """
        entries = self.recall_by_tags(tags, max_results=100, match_all=False)
        if not entries:
            return 0

        calibration_active = self._calibration_total_predictions >= 10

        if calibration_active and self._calibration_overconfident and self._calibration_ece > 0.15:
            # Overconfident: distributed rehearsal — more entries, smaller per-entry boost
            # Total rehearsal budget = len(entries) * base_boost
            # Distribute: all entries get reduced boost (~50% of normal)
            base_boost = boost or self._rehearsal_boost
            reduced_boost = base_boost * 0.5
            count = 0
            for entry in entries:
                if self.rehearse(entry.id, boost=reduced_boost):
                    count += 1
            return count

        elif calibration_active and self._calibration_underconfident and self._calibration_ece > 0.15:
            # Underconfident: uniform boost, slightly elevated
            # Compensates for undervaluing — give everything a fair rehearsal
            base_boost = boost or self._rehearsal_boost
            elevated_boost = base_boost * 1.3
            count = 0
            for entry in entries:
                if self.rehearse(entry.id, boost=elevated_boost):
                    count += 1
            return count

        elif calibration_active and self._calibration_ece <= 0.05:
            # Well-calibrated: significance-ranked prioritization
            # High-significance entries get a bonus, low-significance get skipped
            base_boost = boost or self._rehearsal_boost
            count = 0
            for entry in entries:
                if entry.significance >= 0.7:
                    # High-value: bonus boost
                    if self.rehearse(entry.id, boost=base_boost * 1.1):
                        count += 1
                elif entry.significance >= 0.35:
                    # Medium: standard boost
                    if self.rehearse(entry.id, boost=base_boost):
                        count += 1
                # Below 0.35: skipped — well-calibrated significance judgments are trustworthy
            return count

        # Default: standard uniform rehearsal
        base_boost = boost or self._rehearsal_boost
        count = 0
        for entry in entries:
            if self.rehearse(entry.id, boost=base_boost):
                count += 1
        return count

    # ══════════════════════════════════════════════════════════════
    # Public API — Tension Integration
    # ══════════════════════════════════════════════════════════════

    def set_calibration_state(
        self,
        ece: float,
        bias: float,
        is_overconfident: bool,
        is_underconfident: bool,
        total_predictions: int,
    ) -> None:
        """Update calibration metrics from the reasoning engine.

        Called by the pillar wrapper each tick. Poor calibration
        modulates consolidation thresholds: when the agent cannot
        accurately judge confidence, it should consolidate more
        cautiously (higher thresholds) and prune less aggressively.

        Args:
            ece: Expected Calibration Error from ConfidenceCalibrator.
            bias: Signed bias (positive = overconfident).
            is_overconfident: Whether the calibrator is overconfident.
            is_underconfident: Whether the calibrator is underconfident.
            total_predictions: How many predictions the calibrator has.
        """
        self._calibration_ece = ece
        self._calibration_bias = bias
        self._calibration_overconfident = is_overconfident
        self._calibration_underconfident = is_underconfident
        self._calibration_total_predictions = total_predictions

    def set_tension_profile(self, profile: dict[TensionID, float]) -> None:
        """Update the tension profile from the equilibrium engine.

        Called by the pillar wrapper on each tick.
        """
        self._current_profile = profile

    def drain_feedback(self) -> list[Feedback]:
        """Return and clear pending feedback for the equilibrium engine."""
        result = self._pending_feedback[:]
        self._pending_feedback.clear()
        return result



    # ══════════════════════════════════════════════════════════════
    # Properties
    # ══════════════════════════════════════════════════════════════

    @property
    def stats(self) -> dict:
        return self._stats.summary()

    @property
    def working_memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._working.values())

    @property
    def episodic_memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._episodic.values())

    @property
    def semantic_memory(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._semantic.values())

    @property
    def total_memories(self) -> int:
        return len(self._working) + len(self._episodic) + len(self._semantic)

    @property
    def consolidation_log(self) -> tuple[ConsolidationEvent, ...]:
        return tuple(self._consolidation_log)

    # ══════════════════════════════════════════════════════════════
    # Internal
    # ══════════════════════════════════════════════════════════════

    def _evict_working(self) -> MemoryEntry | None:
        """Evict the weakest entry from working memory.

        Before eviction, evaluate for episodic promotion.
        """
        if not self._working:
            return None

        # Find the weakest entry
        weakest_id = min(
            self._working.keys(),
            key=lambda eid: self._working[eid].strength,
        )
        entry = self._working[weakest_id]
        del self._working[weakest_id]

        # Before discarding, check if it's worth promoting
        cons_sig, _ = self._modulate_thresholds()
        if entry.significance >= cons_sig:
            promoted = entry.promote(MemoryTier.EPISODIC)
            self._episodic[promoted.id] = promoted
            self._stats.total_consolidations += 1

        self._stats.total_pruned += 1
        return entry

    def _update_in_tier(self, entry: MemoryEntry) -> None:
        """Update an entry in its tier's collection."""
        collections = {
            MemoryTier.WORKING: self._working,
            MemoryTier.EPISODIC: self._episodic,
            MemoryTier.SEMANTIC: self._semantic,
        }
        collection = collections.get(entry.tier)
        if collection is not None and entry.id in collection:
            collection[entry.id] = entry

    def _find_entry(self, entry_id: UUID) -> MemoryEntry | None:
        """Find an entry across all tiers."""
        for collection in (self._working, self._episodic, self._semantic):
            if entry_id in collection:
                return collection[entry_id]
        return None

    def _modulate_thresholds(self) -> tuple[float, float]:
        """Modulate consolidation and promotion thresholds from tensions and calibration.

        Tension modulation:
        - consolidate_prune < 0 (Consolidate):
            Lower thresholds → more consolidation
        - consolidate_prune > 0 (Prune):
            Raise thresholds → less consolidation

        Calibration modulation (additive with tension):
        - High ECE (poorly calibrated) → raise thresholds (consolidate cautiously)
        - Low ECE (well calibrated) → slightly lower thresholds (consolidate confidently)
        - Overconfidence adds a bonus → higher thresholds
        """
        consolidate = self._current_profile.get("consolidate_prune", 0.0)
        specific_gen = self._current_profile.get("specific_general", 0.0)

        cons_thresh = self._consolidation_sig
        prom_thresh = self._promotion_sig

        # Proportional consolidation modulation: ±0.20 range
        # Negative consolidate → lower threshold (more consolidation)
        cons_thresh += consolidate * 0.20
        prom_thresh += consolidate * 0.15

        # General mode slightly lowers the bar (more abstraction)
        if specific_gen > 0:
            cons_thresh -= 0.05
            prom_thresh -= 0.05

        # Calibration modulation: additive with tension-based thresholds
        # Requires ≥10 predictions to activate (avoids startup noise)
        if self._calibration_total_predictions >= 10:
            ece = self._calibration_ece
            # bias magnitude available as abs(self._calibration_bias) if needed
            
            # Core ECE modulation: high ECE → raise thresholds
            #   Δ_cal = ECE × 0.30  (at ECE=0.20: +Δ 0.06)
            cal_mod = ece * 0.30
            
            # Overconfidence bonus: systematic overconfidence needs
            # extra caution when deciding what's worth remembering
            if self._calibration_overconfident:
                cal_mod += ece * 0.20  # +50% more caution
            
            # Apply calibration modulation
            cons_thresh += cal_mod
            prom_thresh += cal_mod * 0.75  # Promotion slightly less sensitive
        
        return (
            max(0.15, min(0.95, cons_thresh)),
            max(0.30, min(0.98, prom_thresh)),
        )

    def _relevance_score(
        self,
        entry: MemoryEntry,
        query_tokens: set[str],
        *,
        specific_bias: float = 0.0,
    ) -> float:
        """Compute relevance of a memory entry to a query.

        Multi-strategy scoring:
        - Jaccard similarity on token overlap (specific_bias > 0 boosts this)
        - Tag overlap
        - Strength weighting
        """
        if not query_tokens:
            return 0.0

        entry_tokens = self._tokenize(entry.content)
        if not entry_tokens:
            return 0.0

        # Jaccard similarity
        intersection = query_tokens & entry_tokens
        union = query_tokens | entry_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # Weight Jaccard more when specific mode is active
        if specific_bias < 0:  # Specific
            text_score = jaccard * 1.5  # Boost exact matching
        else:
            text_score = jaccard

        # Tag overlap
        tag_overlap = 0.0
        if entry.tags:
            matching_tags = sum(1 for t in entry.tags if t in query_tokens)
            tag_overlap = matching_tags / len(entry.tags)

        # Combine with strength weighting
        return (0.6 * text_score + 0.4 * tag_overlap) * entry.strength

    def _has_pattern_support(self, entry: MemoryEntry) -> bool:
        """Check if an episodic entry has enough supporting patterns
        for semantic promotion.

        Uses n-gram frequency overlap: if the entry shares significant
        n-grams with other episodic or semantic entries, it likely
        represents a recurring pattern worth abstracting.

        Calibration-aware support threshold:
        - Well-calibrated (ECE ≤ 0.05): standard 30% threshold.
          Trust pattern detection; allow normal abstraction.
        - Overconfident (ECE > 0.15): RAISE threshold to 40%.
          Systematically overconfident agents think their pattern
          matches are stronger than they are. Require MORE evidence
          before abstracting.
        - Underconfident (ECE > 0.15): LOWER threshold to 20%.
          Systematically underconfident agents miss real patterns;
          lower the bar to compensate.
        - Moderate miscalibration (ECE 0.05-0.15): 30% standard.
        """
        if not self._pattern_frequencies:
            return False

        tokens = [re.sub(r"^\W+|\W+$", "", t) for t in entry.content.lower().split() if re.sub(r"^\W+|\W+$", "", t)]
        if not tokens:
            return False

        # Determine calibration-adjusted threshold
        pattern_threshold = self._pattern_threshold  # Default: 3

        calibration_active = self._calibration_total_predictions >= 10

        # Check bigram and trigram overlap with known patterns
        pattern_hits = 0
        total_grams = 0

        # Bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            total_grams += 1
            if self._pattern_frequencies.get(bigram, 0) >= pattern_threshold:
                pattern_hits += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            total_grams += 1
            if self._pattern_frequencies.get(trigram, 0) >= pattern_threshold:
                pattern_hits += 1

        if total_grams == 0:
            return False

        raw_ratio = pattern_hits / total_grams

        # Apply calibration modulation to the required ratio
        if calibration_active:
            if self._calibration_overconfident and self._calibration_ece > 0.15:
                # Overconfident: require MORE pattern evidence
                required_ratio = 0.40
            elif self._calibration_underconfident and self._calibration_ece > 0.15:
                # Underconfident: require LESS pattern evidence (compensate)
                required_ratio = 0.20
            else:
                # Well-calibrated or moderate: standard threshold
                required_ratio = 0.30
        else:
            required_ratio = 0.30

        return raw_ratio >= required_ratio

    def _update_patterns(self, content: str, tags: tuple[str, ...]) -> None:
        """Update n-gram frequencies and tag co-occurrence for pattern extraction."""
        tokens = [re.sub(r"^\W+|\W+$", "", t) for t in content.lower().split() if re.sub(r"^\W+|\W+$", "", t)]
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            self._pattern_frequencies[bigram] = (
                self._pattern_frequencies.get(bigram, 0) + 1
            )
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            self._pattern_frequencies[trigram] = (
                self._pattern_frequencies.get(trigram, 0) + 1
            )

        # Tag co-occurrence
        if len(tags) >= 2:
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair = frozenset([tags[i], tags[j]])
                    self._tag_cooccurrence[pair] = (
                        self._tag_cooccurrence.get(pair, 0) + 1
                    )

    def _enforce_capacities(self) -> None:
        """Enforce capacity limits on each tier via LRU+strength eviction."""
        # Episodic capacity
        while len(self._episodic) > self.EPISODIC_CAPACITY:
            weakest = min(self._episodic.keys(),
                          key=lambda eid: self._episodic[eid].strength)
            del self._episodic[weakest]

        # Semantic capacity
        while len(self._semantic) > self.SEMANTIC_CAPACITY:
            weakest = min(self._semantic.keys(),
                          key=lambda eid: self._semantic[eid].strength)
            del self._semantic[weakest]


    # ── Serialization ───────────────────────────────────────────

    def _memory_entry_to_minimal(self, entry) -> dict:
        """Serialize a single MemoryEntry for JSON-safe persistence."""
        return {
            "id": str(entry.id),
            "content": entry.content,
            "tier": entry.tier.name,
            "strength": entry.strength,
            "significance": entry.significance,
            "created_at": entry.created_at,
            "last_accessed": entry.last_accessed,
            "last_rehearsed": entry.last_rehearsed,
            "rehearsal_count": entry.rehearsal_count,
            "access_count": entry.access_count,
            "source": entry.source,
            "tags": list(entry.tags),
            "metadata": dict(entry.metadata),
            "base_half_life": entry.base_half_life,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full HierarchicalMneme state for cross-session persistence.

        Saves all three memory tiers (WM, Episodic, Semantic) with full
        metadata, pattern frequencies, tag co-occurrence, calibration
        state, configuration parameters, and consolidation history.
        The agent can resume with identical memory state.

        Returns:
            A JSON-serializable dict of all mneme state.
        """
        return {
            "working": [
                self._memory_entry_to_minimal(e)
                for e in self._working.values()
            ],
            "episodic": [
                self._memory_entry_to_minimal(e)
                for e in self._episodic.values()
            ],
            "semantic": [
                self._memory_entry_to_minimal(e)
                for e in self._semantic.values()
            ],
            "pattern_frequencies": dict(self._pattern_frequencies),
            "tag_cooccurrence": {
                "|".join(sorted(pair)): count
                for pair, count in self._tag_cooccurrence.items()
            },
            "consolidation_sig": self._consolidation_sig,
            "promotion_sig": self._promotion_sig,
            "pattern_threshold": self._pattern_threshold,
            "rehearsal_boost": self._rehearsal_boost,
            "calibration_ece": self._calibration_ece,
            "calibration_bias": self._calibration_bias,
            "calibration_overconfident": self._calibration_overconfident,
            "calibration_underconfident": self._calibration_underconfident,
            "calibration_total_predictions": self._calibration_total_predictions,
            "stats": self._stats.summary(),
            "consolidation_log": [
                {
                    "entry_id": str(ce.entry_id),
                    "from_tier": ce.from_tier.name,
                    "to_tier": ce.to_tier.name,
                    "significance": ce.significance,
                    "strength_at_consolidation": ce.strength_at_consolidation,
                    "timestamp": ce.timestamp,
                    "reason": ce.reason,
                }
                for ce in self._consolidation_log
            ],
            "tension_profile": dict(self._current_profile),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalMneme:
        """Deserialize the full HierarchicalMneme state.

        Reconstructs MemoryEntry objects for all three tiers with full
        metadata, restores pattern frequencies and tag co-occurrence,
        and rebuilds stats counters from saved values.

        Args:
            data: A dict produced by to_dict().

        Returns:
            A reconstructed HierarchicalMneme with full memory state.
        """
        from uuid import UUID

        # Reconstruct from the data
        mneme = cls(
            consolidation_significance=data.get("consolidation_sig"),
            promotion_significance=data.get("promotion_sig"),
            pattern_count_threshold=data.get("pattern_threshold"),
            rehearsal_boost=data.get("rehearsal_boost"),
        )

        # Restore working memory
        for e_data in data.get("working", []):
            entry = MemoryEntry(
                id=UUID(e_data["id"]),
                content=e_data["content"],
                tier=MemoryTier[e_data.get("tier", "WORKING")],
                strength=e_data.get("strength", 1.0),
                significance=e_data.get("significance", 0.5),
                created_at=e_data.get("created_at", 0.0),
                last_accessed=e_data.get("last_accessed", 0.0),
                last_rehearsed=e_data.get("last_rehearsed", 0.0),
                rehearsal_count=e_data.get("rehearsal_count", 0),
                access_count=e_data.get("access_count", 0),
                source=e_data.get("source"),
                tags=tuple(e_data.get("tags", [])),
                metadata=frozendict(e_data.get("metadata", {})),
                base_half_life=e_data.get("base_half_life", 3600.0),
            )
            mneme._working[entry.id] = entry

        # Restore episodic memory
        for e_data in data.get("episodic", []):
            entry = MemoryEntry(
                id=UUID(e_data["id"]),
                content=e_data["content"],
                tier=MemoryTier[e_data.get("tier", "EPISODIC")],
                strength=e_data.get("strength", 1.0),
                significance=e_data.get("significance", 0.5),
                created_at=e_data.get("created_at", 0.0),
                last_accessed=e_data.get("last_accessed", 0.0),
                last_rehearsed=e_data.get("last_rehearsed", 0.0),
                rehearsal_count=e_data.get("rehearsal_count", 0),
                access_count=e_data.get("access_count", 0),
                source=e_data.get("source"),
                tags=tuple(e_data.get("tags", [])),
                metadata=frozendict(e_data.get("metadata", {})),
                base_half_life=e_data.get("base_half_life", 86400.0),
            )
            mneme._episodic[entry.id] = entry

        # Restore semantic memory
        for e_data in data.get("semantic", []):
            entry = MemoryEntry(
                id=UUID(e_data["id"]),
                content=e_data["content"],
                tier=MemoryTier[e_data.get("tier", "SEMANTIC")],
                strength=e_data.get("strength", 1.0),
                significance=e_data.get("significance", 0.5),
                created_at=e_data.get("created_at", 0.0),
                last_accessed=e_data.get("last_accessed", 0.0),
                last_rehearsed=e_data.get("last_rehearsed", 0.0),
                rehearsal_count=e_data.get("rehearsal_count", 0),
                access_count=e_data.get("access_count", 0),
                source=e_data.get("source"),
                tags=tuple(e_data.get("tags", [])),
                metadata=frozendict(e_data.get("metadata", {})),
                base_half_life=e_data.get("base_half_life", 2592000.0),
            )
            mneme._semantic[entry.id] = entry

        # Restore pattern frequencies
        mneme._pattern_frequencies.clear()
        for key, count in data.get("pattern_frequencies", {}).items():
            mneme._pattern_frequencies[key] = int(count)

        # Restore tag co-occurrence
        mneme._tag_cooccurrence.clear()
        for pair_str, count in data.get("tag_cooccurrence", {}).items():
            pair = frozenset(pair_str.split("|"))
            mneme._tag_cooccurrence[pair] = int(count)

        # Restore calibration state
        mneme._calibration_ece = float(data.get("calibration_ece", 0.0))
        mneme._calibration_bias = float(data.get("calibration_bias", 0.0))
        mneme._calibration_overconfident = bool(data.get("calibration_overconfident", False))
        mneme._calibration_underconfident = bool(data.get("calibration_underconfident", False))
        mneme._calibration_total_predictions = int(data.get("calibration_total_predictions", 0))

        # Restore consolidation log
        mneme._consolidation_log.clear()
        for ce_data in data.get("consolidation_log", []):
            event = ConsolidationEvent(
                entry_id=UUID(ce_data["entry_id"]),
                from_tier=MemoryTier[ce_data.get("from_tier", "WORKING")],
                to_tier=MemoryTier[ce_data.get("to_tier", "WORKING")],
                significance=ce_data.get("significance", 0.0),
                strength_at_consolidation=ce_data.get("strength_at_consolidation", 0.0),
                timestamp=ce_data.get("timestamp", 0.0),
                reason=ce_data.get("reason", ""),
            )
            mneme._consolidation_log.append(event)

        # Restore stats (rebuild counts from actual collections)
        saved_stats = data.get("stats", {})
        mneme._stats.working_count = len(mneme._working)
        mneme._stats.episodic_count = len(mneme._episodic)
        mneme._stats.semantic_count = len(mneme._semantic)
        mneme._stats.total_consolidations = max(
            len(mneme._consolidation_log),
            saved_stats.get("total_consolidations", 0),
        )
        mneme._stats.total_pruned = int(saved_stats.get("total_pruned", 0))
        mneme._stats.total_rehearsals = int(saved_stats.get("total_rehearsals", 0))
        mneme._stats.total_retrievals = int(saved_stats.get("total_retrievals", 0))

        # Restore tension profile
        mneme._current_profile.update(data.get("tension_profile", {}))

        return mneme


@dataclass
class ConsolidationReport:
    """Detailed report of a consolidation cycle."""

    working_count: int
    episodic_count: int
    semantic_count: int
    wm_to_episodic: int
    ep_to_semantic: int
    pruned: int
    thresholds: tuple[float, float]
    tension_profile: dict[TensionID, float]
    calibration_ece: float = 0.0
    calibration_active: bool = False
    calibration_prune_saved: int = 0

    def summary(self) -> str:
        parts = [
            f"Consolidation: WM→Ep={self.wm_to_episodic}, Ep→Sem={self.ep_to_semantic}, "
            f"pruned={self.pruned} | "
            f"tiers: WM={self.working_count}, Ep={self.episodic_count}, Sem={self.semantic_count} | "
            f"thresholds: cons={self.thresholds[0]:.2f}, prom={self.thresholds[1]:.2f}"
        ]
        if self.calibration_active:
            parts.append(
                f"cal: ECE={self.calibration_ece:.3f} saved={self.calibration_prune_saved}"
            )
        return " ".join(parts)
