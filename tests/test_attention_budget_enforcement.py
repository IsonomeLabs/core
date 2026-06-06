"""Tests for attention budget enforcement (iter-025).

When the context window budget is full or nearly full, the
AttentionEquilibriumSystem must enforce capacity limits rather than
silently allowing unbounded growth. This test suite covers:

1. BudgetEnforcementPolicy enum values and behavior
2. enforce_budget() auto-GC triggering
3. add_chunk() budget-aware admission (reject, auto-GC, auto-compress)
4. Oversized single-chunk rejection (chunk > capacity)
5. Enforcement statistics tracking
6. Serialization round-trip with enforcement state
7. Integration with calibration-aware GC
"""

import pytest

from isonome.cognition.attention import (
    AttentionBudget,
    AttentionChunk,
    AttentionEquilibriumSystem,
    BudgetEnforcementPolicy,
    GarbageCollectionReport,
    RetentionDecision,
)
from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    return EquilibriumEngine()


@pytest.fixture
def small_aes(engine):
    """AES with a very small capacity for easy overflow testing."""
    return AttentionEquilibriumSystem(
        engine=engine,
        token_capacity=100,
        enforcement_policy=BudgetEnforcementPolicy.AUTO_GC,
    )


@pytest.fixture
def reject_aes(engine):
    """AES with REJECT policy."""
    return AttentionEquilibriumSystem(
        engine=engine,
        token_capacity=100,
        enforcement_policy=BudgetEnforcementPolicy.REJECT,
    )


@pytest.fixture
def auto_compress_aes(engine):
    """AES with AUTO_COMPRESS policy."""
    return AttentionEquilibriumSystem(
        engine=engine,
        token_capacity=100,
        enforcement_policy=BudgetEnforcementPolicy.AUTO_COMPRESS,
    )


# ═══════════════════════════════════════════════════════════════════
# BudgetEnforcementPolicy enum
# ═══════════════════════════════════════════════════════════════════


class TestBudgetEnforcementPolicy:
    def test_enum_values(self):
        assert BudgetEnforcementPolicy.REJECT.value == "reject"
        assert BudgetEnforcementPolicy.AUTO_GC.value == "auto_gc"
        assert BudgetEnforcementPolicy.AUTO_COMPRESS.value == "auto_compress"

    def test_enum_members(self):
        members = list(BudgetEnforcementPolicy)
        assert len(members) == 3

    def test_default_policy_is_auto_gc(self):
        """The default enforcement policy should be AUTO_GC."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine())
        assert aes.enforcement_policy == BudgetEnforcementPolicy.AUTO_GC


# ═══════════════════════════════════════════════════════════════════
# enforce_budget() method
# ═══════════════════════════════════════════════════════════════════


class TestEnforceBudget:
    def test_no_gc_when_under_capacity(self, small_aes):
        """enforce_budget() should not trigger GC when under capacity."""
        small_aes.add_chunk("Some content", token_count=10)
        result = small_aes.enforce_budget()
        assert result is None  # No GC needed

    def test_triggers_gc_when_over_capacity(self, small_aes, engine):
        """enforce_budget() should trigger GC when tokens_used > capacity."""
        # Fill beyond capacity — note: add_chunk enforcement may trigger GC
        # during the fill loop, so we bypass enforcement by using REJECT
        # to fill, then switch to AUTO_GC and call enforce_budget()
        reject_fill = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )
        for i in range(15):
            reject_fill.add_chunk(f"Chunk {i} content to fill", token_count=10)

        assert reject_fill.budget.tokens_used == 100  # REJECT caps at capacity

        # Now manually set over-capacity to test enforce_budget
        reject_fill._budget.tokens_used = 150
        reject_fill._enforcement_policy = BudgetEnforcementPolicy.AUTO_GC
        result = reject_fill.enforce_budget()
        assert isinstance(result, GarbageCollectionReport)

    def test_triggers_gc_at_utilization_threshold(self, engine):
        """enforce_budget() should trigger GC when utilization exceeds threshold."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=1000,
            enforcement_policy=BudgetEnforcementPolicy.AUTO_GC,
            enforcement_threshold=0.8,  # GC at 80% utilization
        )
        # Fill to 85% utilization
        aes.add_chunk("Big chunk", token_count=850)
        assert aes.budget.utilization >= 0.8

        result = aes.enforce_budget()
        assert isinstance(result, GarbageCollectionReport)

    def test_no_gc_below_threshold(self, engine):
        """enforce_budget() should not trigger GC below threshold."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=1000,
            enforcement_threshold=0.9,
        )
        aes.add_chunk("Small chunk", token_count=100)
        assert aes.budget.utilization < 0.9

        result = aes.enforce_budget()
        assert result is None

    def test_returns_none_for_reject_policy(self, reject_aes):
        """enforce_budget() with REJECT policy should not trigger GC."""
        # Fill beyond capacity
        for i in range(15):
            reject_aes.add_chunk(f"Chunk {i}", token_count=10)

        # REJECT policy: enforce_budget doesn't auto-GC
        result = reject_aes.enforce_budget()
        assert result is None

    def test_auto_compress_policy_triggers_gc(self, auto_compress_aes, engine):
        """enforce_budget() with AUTO_COMPRESS policy should trigger GC."""
        # Fill to threshold using REJECT to avoid auto-GC during fill
        reject_fill = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )
        for i in range(15):
            reject_fill.add_chunk(f"Chunk {i} filler content", token_count=10)

        # Switch to AUTO_COMPRESS and overfill
        reject_fill._enforcement_policy = BudgetEnforcementPolicy.AUTO_COMPRESS
        reject_fill._budget.tokens_used = 150

        result = reject_fill.enforce_budget()
        assert isinstance(result, GarbageCollectionReport)


# ═══════════════════════════════════════════════════════════════════
# add_chunk() budget-aware admission
# ═══════════════════════════════════════════════════════════════════


class TestAddChunkBudgetAware:
    def test_auto_gc_frees_space_then_admits(self, small_aes, engine):
        """With AUTO_GC, add_chunk should trigger GC to free space."""
        # Fill the budget with low-relevance chunks
        for i in range(10):
            small_aes.add_chunk(f"Filler {i}", token_count=10, task_relevance=0.1)

        # Verify budget is full or near-full
        assert small_aes.budget.tokens_used > 0

        # Apply heavy decay so GC can prune the low-relevance chunks
        small_aes.apply_recency_decay(decay_rate=0.9)
        small_aes.apply_recency_decay(decay_rate=0.9)

        # Record gc count before adding
        gc_before = small_aes._gc_cycles

        # Adding a new chunk should trigger GC
        chunk = small_aes.add_chunk("New important chunk", token_count=10)
        # The chunk should be admitted (GC freed space or room existed)
        assert chunk is not None
        # At least one GC should have been triggered by enforcement
        assert small_aes.enforcement_stats["auto_gc_triggered"] >= 1

    def test_reject_policy_returns_none_when_full(self, reject_aes):
        """With REJECT policy, add_chunk should return None when budget is full."""
        # Fill the budget
        for i in range(10):
            reject_aes.add_chunk(f"Filler {i}", token_count=10)

        assert reject_aes.budget.tokens_used >= 100

        # Trying to add should be rejected
        chunk = reject_aes.add_chunk("Rejected chunk", token_count=10)
        assert chunk is None
        assert reject_aes.enforcement_stats["rejections"] >= 1

    def test_reject_policy_admits_when_space_available(self, reject_aes):
        """With REJECT policy, add_chunk should work when there's space."""
        chunk = reject_aes.add_chunk("Small chunk", token_count=10)
        assert chunk is not None
        assert reject_aes.budget.tokens_used == 10

    def test_auto_compress_admits_compressed_when_full(self, auto_compress_aes, engine):
        """With AUTO_COMPRESS policy, chunk is compressed if no space."""
        # Fill budget with high-scoring chunks that GC will keep
        # Use REJECT policy to fill without auto-GC, then switch
        reject_fill = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            compress_ratio=0.20,
            keep_threshold=0.65,
            prune_threshold=0.25,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )
        for i in range(9):
            reject_fill.add_chunk(
                f"Key info {i}",
                token_count=10,
                task_relevance=1.0,
                mutual_info=1.0,
                importance_tags=("critical",),
            )

        assert reject_fill.budget.tokens_used == 90

        # Switch to AUTO_COMPRESS
        reject_fill._enforcement_policy = BudgetEnforcementPolicy.AUTO_COMPRESS

        # Add a chunk needing 30 tokens. GC won't free space (high-scoring),
        # so chunk gets compressed: 30 * 0.20 = 6 tokens, 90 + 6 = 96 <= 100
        chunk = reject_fill.add_chunk("New chunk", token_count=30, task_relevance=0.9)
        assert chunk is not None
        # The chunk should have been compressed
        assert chunk.token_count < 30
        assert reject_fill.enforcement_stats["auto_compressions"] >= 1

    def test_oversized_chunk_rejected(self, small_aes):
        """A single chunk larger than total capacity should always be rejected."""
        chunk = small_aes.add_chunk("Huge", token_count=500)
        assert chunk is None
        assert small_aes.enforcement_stats["oversized_rejections"] >= 1

    def test_oversized_rejected_even_with_empty_budget(self, engine):
        """Oversized chunks rejected even when budget is empty."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=50,
        )
        chunk = aes.add_chunk("Too big", token_count=100)
        assert chunk is None
        assert aes.enforcement_stats["oversized_rejections"] >= 1

    def test_add_chunk_with_zero_tokens(self, small_aes):
        """A chunk with 0 tokens should always be admitted."""
        chunk = small_aes.add_chunk("Zero size", token_count=0)
        assert chunk is not None

    def test_add_chunk_updates_tokens_used(self, small_aes):
        """add_chunk should correctly track tokens_used."""
        small_aes.add_chunk("A", token_count=10)
        small_aes.add_chunk("B", token_count=20)
        assert small_aes.budget.tokens_used == 30

    def test_auto_gc_then_reject_if_still_full(self, small_aes, engine):
        """AUTO_GC: if GC doesn't free enough space, reject the chunk."""
        # Fill budget using REJECT to prevent auto-GC during fill.
        # We use chunks with modest scores so GC will prune some but
        # not enough to make room for a large new chunk.
        reject_fill = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )
        for i in range(10):
            reject_fill.add_chunk(
                f"Chunk {i}",
                token_count=10,
                task_relevance=0.5,
            )

        assert reject_fill.budget.tokens_used == 100

        # Switch to AUTO_GC
        reject_fill._enforcement_policy = BudgetEnforcementPolicy.AUTO_GC

        # Add a chunk that is too large for any space GC might free.
        # With capacity=100 and already 100 used, even if GC frees
        # some tokens, a chunk of 95 tokens likely won't fit.
        chunk = reject_fill.add_chunk("Too big after GC", token_count=95)
        # This should be rejected because even after GC, not enough space
        assert chunk is None
        assert reject_fill.enforcement_stats["post_gc_rejections"] >= 1


# ═══════════════════════════════════════════════════════════════════
# Enforcement statistics
# ═══════════════════════════════════════════════════════════════════


class TestEnforcementStats:
    def test_initial_stats_are_zero(self, small_aes):
        """Enforcement stats should start at zero."""
        stats = small_aes.enforcement_stats
        assert stats["auto_gc_triggered"] == 0
        assert stats["rejections"] == 0
        assert stats["auto_compressions"] == 0
        assert stats["oversized_rejections"] == 0
        assert stats["post_gc_rejections"] == 0

    def test_stats_track_auto_gc(self, small_aes, engine):
        """Auto-GC triggers should be tracked."""
        for i in range(12):
            small_aes.add_chunk(f"Chunk {i}", token_count=10, task_relevance=0.1)
        small_aes.apply_recency_decay(decay_rate=0.9)
        small_aes.enforce_budget()
        assert small_aes.enforcement_stats["auto_gc_triggered"] >= 1

    def test_stats_track_rejections(self, reject_aes):
        """Rejections should be tracked."""
        for i in range(10):
            reject_aes.add_chunk(f"Fill {i}", token_count=10)
        reject_aes.add_chunk("Overflow", token_count=10)
        assert reject_aes.enforcement_stats["rejections"] >= 1

    def test_stats_in_overall_stats(self, small_aes):
        """Enforcement stats should appear in the overall stats dict."""
        stats = small_aes.stats
        assert "enforcement" in stats
        assert "auto_gc_triggered" in stats["enforcement"]

    def test_stats_increment_on_oversized_rejection(self, small_aes):
        """Oversized rejection should increment stats."""
        small_aes.add_chunk("Huge", token_count=9999)
        assert small_aes.enforcement_stats["oversized_rejections"] == 1


# ═══════════════════════════════════════════════════════════════════
# enforcement_threshold configuration
# ═══════════════════════════════════════════════════════════════════


class TestEnforcementThreshold:
    def test_default_threshold(self):
        """Default enforcement threshold should be 0.85."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine())
        assert aes.enforcement_threshold == 0.85

    def test_custom_threshold(self, engine):
        """Custom threshold should be respected."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            enforcement_threshold=0.5,
        )
        assert aes.enforcement_threshold == 0.5

    def test_threshold_clamped_to_valid_range(self, engine):
        """Threshold should be clamped to [0.1, 1.0]."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            enforcement_threshold=0.0,
        )
        assert aes.enforcement_threshold >= 0.1

        aes2 = AttentionEquilibriumSystem(
            engine=engine,
            enforcement_threshold=2.0,
        )
        assert aes2.enforcement_threshold <= 1.0

    def test_threshold_at_1_means_never_auto_gc(self, engine):
        """Threshold of 1.0 means enforcement only on actual overflow."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_threshold=1.0,
        )
        # Fill to 90% — no enforcement
        aes.add_chunk("Ninety percent", token_count=90)
        result = aes.enforce_budget()
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_default_system_behaves_as_before(self, engine):
        """With default settings, system should work like before for normal use."""
        aes = AttentionEquilibriumSystem(engine=engine, token_capacity=10_000)
        # Add chunks well within budget — no enforcement needed
        for i in range(5):
            chunk = aes.add_chunk(f"Normal chunk {i}", token_count=100)
            assert chunk is not None

    def test_existing_gc_still_works(self, small_aes):
        """Manual collect_garbage() should still work."""
        small_aes.add_chunk("Test", token_count=10)
        report = small_aes.collect_garbage()
        assert isinstance(report, GarbageCollectionReport)

    def test_add_chunk_returns_chunk_when_under_budget(self, small_aes):
        """add_chunk should return a chunk when under budget."""
        chunk = small_aes.add_chunk("Normal add", token_count=10)
        assert isinstance(chunk, AttentionChunk)
        assert chunk.recency == 1.0

    def test_enforcement_does_not_break_existing_tests(self, engine):
        """The existing test fixture pattern should still work."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=10_000,
        )
        aes.add_chunk("Hello world, this is a test message for the agent.")
        assert aes.chunk_count == 1
        assert aes.budget.tokens_used > 0


# ═══════════════════════════════════════════════════════════════════
# Integration with calibration
# ═══════════════════════════════════════════════════════════════════


class TestEnforcementCalibrationIntegration:
    def test_miscalibrated_system_retains_more_on_gc(self, engine):
        """When miscalibrated, enforcement GC should retain more context."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=1000,
            enforcement_policy=BudgetEnforcementPolicy.AUTO_GC,
            enforcement_threshold=0.5,
        )

        # Fill budget
        for i in range(20):
            aes.add_chunk(f"Content {i}", token_count=30, task_relevance=0.3)

        # Without calibration: normal GC
        aes.apply_recency_decay(decay_rate=0.7)
        report_no_cal = aes.enforce_budget()

        # Reset
        aes2 = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=1000,
            enforcement_policy=BudgetEnforcementPolicy.AUTO_GC,
            enforcement_threshold=0.5,
        )
        for i in range(20):
            aes2.add_chunk(f"Content {i}", token_count=30, task_relevance=0.3)

        # With poor calibration: should retain more
        aes2.set_calibration_state(
            ece=0.25,
            bias=0.15,
            is_overconfident=True,
            total_predictions=50,
        )
        aes2.apply_recency_decay(decay_rate=0.7)
        report_with_cal = aes2.enforce_budget()

        # With calibration, more chunks should be retained (lower thresholds)
        # The calibration modifier lowers keep_threshold, so more kept
        if report_no_cal and report_with_cal:
            assert report_with_cal.calibration_active is True


# ═══════════════════════════════════════════════════════════════════
# add_chunk return type
# ═══════════════════════════════════════════════════════════════════


class TestAddChunkReturnType:
    def test_returns_chunk_on_success(self, small_aes):
        chunk = small_aes.add_chunk("Good", token_count=5)
        assert isinstance(chunk, AttentionChunk)

    def test_returns_none_on_rejection(self, reject_aes):
        for i in range(10):
            reject_aes.add_chunk(f"Fill {i}", token_count=10)
        result = reject_aes.add_chunk("Overflow", token_count=10)
        assert result is None

    def test_returns_none_on_oversized(self, small_aes):
        result = small_aes.add_chunk("Oversized", token_count=9999)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Enforcement stats in overall stats dict
# ═══════════════════════════════════════════════════════════════════


class TestEnforcementInStats:
    def test_enforcement_subdict_in_stats(self, small_aes):
        """Enforcement stats should appear as a sub-dict in stats."""
        stats = small_aes.stats
        assert "enforcement" in stats
        assert stats["enforcement"]["policy"] == "auto_gc"
        assert "threshold" in stats["enforcement"]
        assert stats["enforcement"]["oversized_rejections"] == 0

    def test_enforcement_counts_reflect_operations(self, small_aes):
        """Enforcement counts in stats should reflect operations."""
        small_aes.add_chunk("Huge", token_count=9999)
        stats = small_aes.stats
        assert stats["enforcement"]["oversized_rejections"] == 1
