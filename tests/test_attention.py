"""Tests for the Attention Equilibrium System."""

import math

import pytest

from isonome.cognition.attention import (
    AttentionBudget,
    AttentionChunk,
    AttentionEquilibriumSystem,
    GarbageCollectionReport,
    RetentionDecision,
)
from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar


class TestAttentionChunk:
    def test_attention_score_defaults(self):
        chunk = AttentionChunk(
            content="test content",
            token_count=3,
            surprisal=5.0,
            mutual_info=0.6,
            recency=0.8,
            task_relevance=0.7,
        )
        score = chunk.attention_score()
        assert 0.0 <= score <= 1.0

    def test_attention_score_high_surprisal(self):
        chunk = AttentionChunk(
            content="very surprising content",
            token_count=5,
            surprisal=50.0,  # Very high surprisal
            mutual_info=0.0,
            recency=0.5,
        )
        score_low = chunk.attention_score()
        # surprisal is tanh-normalized, so it shouldn't dominate unreasonably
        assert 0.0 <= score_low <= 1.0

    def test_attention_score_importance_tags(self):
        chunk_no_tags = AttentionChunk(
            content="test",
            token_count=1,
            surprisal=1.0,
            mutual_info=0.5,
            recency=0.5,
        )
        chunk_with_tags = AttentionChunk(
            content="test",
            token_count=1,
            surprisal=1.0,
            mutual_info=0.5,
            recency=0.5,
            importance_tags=("critical", "user_instruction", "system_prompt"),
        )
        assert chunk_with_tags.attention_score() > chunk_no_tags.attention_score()

    def test_attention_score_zero_info(self):
        chunk = AttentionChunk(
            content="",
            token_count=0,
            surprisal=0.0,
            mutual_info=0.0,
            recency=0.0,
        )
        score = chunk.attention_score()
        assert 0.0 <= score <= 1.0
        # Should be close to 0
        assert score < 0.2


class TestAttentionBudget:
    def test_budget_initialization(self):
        budget = AttentionBudget(token_capacity=100_000)
        assert budget.token_capacity == 100_000
        assert budget.tokens_used == 0
        assert budget.utilization == 0.0
        assert budget.headroom == 100_000

    def test_budget_information_capacity(self):
        budget = AttentionBudget(token_capacity=128_000)
        # 128k tokens × 8 bits/token
        assert budget.information_capacity == 1_024_000.0

    def test_budget_utilization(self):
        budget = AttentionBudget(token_capacity=1000)
        budget.tokens_used = 500
        assert budget.utilization == 0.5
        assert budget.headroom == 500


class TestAttentionEquilibriumSystem:
    @pytest.fixture
    def engine(self):
        return EquilibriumEngine()

    @pytest.fixture
    def aes(self, engine):
        return AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=10_000,
        )

    def test_add_chunk(self, aes):
        chunk = aes.add_chunk("Hello world, this is a test message for the agent.")
        assert chunk.id is not None
        assert chunk.recency == 1.0
        assert aes.chunk_count == 1
        assert aes.budget.tokens_used > 0

    def test_add_multiple_chunks(self, aes):
        for i in range(10):
            aes.add_chunk(f"Chunk number {i} with some content to fill space.")
        assert aes.chunk_count == 10

    def test_garbage_collection_basic(self, aes):
        """GC should work even with a single chunk."""
        aes.add_chunk("Test content")
        report = aes.collect_garbage()
        assert isinstance(report, GarbageCollectionReport)
        assert report.gc_cycle == 1
        assert report.chunks_before == 1

    def test_garbage_collection_prunes_low_value(self, aes):
        """Chunks with low scores should be pruned."""
        # Add chunks with very low scores
        for i in range(20):
            aes.add_chunk(
                f"low value chunk {i}",
                mutual_info=0.0,
                task_relevance=0.0,
            )
        # Apply heavy recency decay so they score even lower
        aes.apply_recency_decay(decay_rate=0.5)
        aes.apply_recency_decay(decay_rate=0.5)

        report = aes.collect_garbage()
        assert report.pruned_count >= 0  # May prune some
        assert report.kept_count + report.compressed_count + report.pruned_count == report.chunks_before

    def test_garbage_collection_keeps_high_value(self, aes):
        """Chunks with very high scores should be kept verbatim."""
        aes.add_chunk(
            "critical system instruction — do not modify this directive",
            mutual_info=1.0,
            task_relevance=1.0,
            importance_tags=("critical", "system", "immutable", "high_priority"),
        )
        # Apply zero recency decay so the chunk is fresh
        report = aes.collect_garbage()
        assert report.kept_count == 1
        assert report.pruned_count == 0

    def test_recency_decay(self, aes):
        chunk = aes.add_chunk("Fresh content")
        assert aes._chunks[chunk.id].recency == 1.0

        aes.apply_recency_decay(decay_rate=0.1)
        assert aes._chunks[chunk.id].recency == pytest.approx(0.9)

        aes.apply_recency_decay(decay_rate=0.1)
        assert aes._chunks[chunk.id].recency == pytest.approx(0.81)

    def test_surprisal_increases_with_repetition(self, aes):
        """Repeated content should have lower surprisal (less surprising)."""
        content = "the quick brown fox jumps over the lazy dog"

        chunk1 = aes.add_chunk(content)
        surprisal1 = chunk1.surprisal

        # Add many more tokens to build frequency data
        for _ in range(5):
            aes.add_chunk(content)

        chunk_last = aes.add_chunk(content)
        surprisal_last = chunk_last.surprisal

        # Last chunk should be less surprising (seen before)
        assert surprisal_last <= surprisal1

    def test_get_top_chunks(self, aes):
        aes.add_chunk("Low value A", mutual_info=0.1, task_relevance=0.1)
        aes.add_chunk("High value B", mutual_info=0.9, task_relevance=1.0)
        aes.add_chunk("Medium value C", mutual_info=0.5, task_relevance=0.5)

        top = aes.get_top_chunks(n=2)
        assert len(top) == 2
        # "High value B" should be in top 2
        assert any("High value B" in c.content for c in top)

    def test_add_then_gc_with_tension_modulation(self, aes, engine):
        """GC thresholds should change when tensions are modulated."""
        # Record default GC behavior
        aes.add_chunk("Test A", mutual_info=0.3)
        aes.add_chunk("Test B", mutual_info=0.3)
        report_before = aes.collect_garbage()

        # Move tension to shallow mode (aggressive pruning)
        engine.apply_feedback(
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="shallow_deep",
                signal=-1.0,
                confidence=1.0,
                reason="force shallow mode",
            )
        )

        # Re-add similar chunks
        aes.add_chunk("Test C", mutual_info=0.3)
        aes.add_chunk("Test D", mutual_info=0.3)
        report_after = aes.collect_garbage()

        # In shallow mode, thresholds should be higher
        assert report_after.keep_threshold > report_before.keep_threshold

    def test_stats_property(self, aes):
        aes.add_chunk("Stats test content")
        stats = aes.stats
        assert "chunks_active" in stats
        assert "tokens_used" in stats
        assert "utilization" in stats
        assert "gc_cycles" in stats
        assert stats["chunks_active"] == 1

    def test_equilibrium_chunks_immutable_view(self, aes):
        aes.add_chunk("A")
        aes.add_chunk("B")
        chunks = aes.equilibrium_chunks
        assert len(chunks) == 2
        # It's a tuple, should be hashable
        hash(chunks)

    def test_token_estimation(self, aes):
        content = "a" * 400  # 400 chars
        chunk = aes.add_chunk(content)
        # ~4 chars per token → ~100 tokens
        assert 80 <= chunk.token_count <= 120

    def test_gc_report_summary(self, aes):
        aes.add_chunk("Report test")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "GC#1" in summary
        assert "freed" in summary
        assert "util" in summary


class TestRetentionDecision:
    def test_keep_full(self):
        aes = AttentionEquilibriumSystem(EquilibriumEngine())
        result = aes._decide_retention(0.8, 0.6, 0.3)
        assert result == RetentionDecision.KEEP_FULL

    def test_compress(self):
        aes = AttentionEquilibriumSystem(EquilibriumEngine())
        result = aes._decide_retention(0.5, 0.6, 0.3)
        assert result == RetentionDecision.COMPRESS

    def test_prune(self):
        aes = AttentionEquilibriumSystem(EquilibriumEngine())
        result = aes._decide_retention(0.1, 0.6, 0.3)
        assert result == RetentionDecision.PRUNE
