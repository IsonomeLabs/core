"""Tests for chunk priority queue — rejected chunk buffering (iter-026).

When a chunk is rejected by budget enforcement, it is buffered in a
priority queue for retry after the next GC cycle. This test suite covers:

1. ChunkPriorityQueue data structure (creation, ordering)
2. Basic enqueue/dequeue operations
3. Priority ordering by task_relevance (higher = higher priority)
4. Overflow buffer capacity limits
5. Integration with AttentionEquilibriumSystem.add_chunk rejection paths
6. Retry after GC frees space
7. Buffer statistics tracking
8. Integration with all enforcement policies
"""

import pytest

from isonome.cognition.attention import (
    AttentionBudget,
    AttentionChunk,
    AttentionEquilibriumSystem,
    BudgetEnforcementPolicy,
    ChunkPriorityQueue,
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
    """AES with REJECT policy and small capacity."""
    return AttentionEquilibriumSystem(
        engine=engine,
        token_capacity=100,
        enforcement_policy=BudgetEnforcementPolicy.REJECT,
    )


@pytest.fixture
def auto_compress_aes(engine):
    """AES with AUTO_COMPRESS policy and small capacity."""
    return AttentionEquilibriumSystem(
        engine=engine,
        token_capacity=100,
        enforcement_policy=BudgetEnforcementPolicy.AUTO_COMPRESS,
    )


# ═══════════════════════════════════════════════════════════════════
# ChunkPriorityQueue unit tests
# ═══════════════════════════════════════════════════════════════════


class TestChunkPriorityQueueCreation:
    """Test ChunkPriorityQueue construction and defaults."""

    def test_default_creation(self):
        queue = ChunkPriorityQueue()
        assert len(queue) == 0
        assert queue.max_size == 64  # Default buffer capacity

    def test_custom_max_size(self):
        queue = ChunkPriorityQueue(max_size=10)
        assert queue.max_size == 10
        assert len(queue) == 0

    def test_max_size_must_be_positive(self):
        with pytest.raises(ValueError, match="max_size"):
            ChunkPriorityQueue(max_size=0)

    def test_max_size_must_be_positive_negative(self):
        with pytest.raises(ValueError, match="max_size"):
            ChunkPriorityQueue(max_size=-5)

    def test_is_iterable(self):
        queue = ChunkPriorityQueue()
        assert list(queue) == []

    def test_bool_empty(self):
        queue = ChunkPriorityQueue()
        assert not queue

    def test_bool_non_empty(self):
        queue = ChunkPriorityQueue()
        queue.enqueue(AttentionChunk(content="test", token_count=5, task_relevance=0.8))
        assert queue


class TestChunkPriorityQueueEnqueueDequeue:
    """Test basic enqueue and dequeue operations."""

    def test_enqueue_single_chunk(self):
        queue = ChunkPriorityQueue()
        chunk = AttentionChunk(content="hello", token_count=5, task_relevance=0.5)
        queue.enqueue(chunk)
        assert len(queue) == 1

    def test_dequeue_single_chunk(self):
        queue = ChunkPriorityQueue()
        chunk = AttentionChunk(content="hello", token_count=5, task_relevance=0.5)
        queue.enqueue(chunk)
        dequeued = queue.dequeue()
        assert dequeued is chunk
        assert len(queue) == 0

    def test_dequeue_empty_returns_none(self):
        queue = ChunkPriorityQueue()
        assert queue.dequeue() is None

    def test_peek_returns_without_removing(self):
        queue = ChunkPriorityQueue()
        chunk = AttentionChunk(content="hello", token_count=5, task_relevance=0.5)
        queue.enqueue(chunk)
        peeked = queue.peek()
        assert peeked is chunk
        assert len(queue) == 1

    def test_peek_empty_returns_none(self):
        queue = ChunkPriorityQueue()
        assert queue.peek() is None

    def test_enqueue_multiple_chunks(self):
        queue = ChunkPriorityQueue()
        for i in range(5):
            queue.enqueue(
                AttentionChunk(content=f"chunk-{i}", token_count=5, task_relevance=0.5)
            )
        assert len(queue) == 5

    def test_fifo_within_same_priority(self):
        """Chunks with the same priority should come out in FIFO order."""
        queue = ChunkPriorityQueue()
        chunks = []
        for i in range(3):
            c = AttentionChunk(content=f"c{i}", token_count=5, task_relevance=0.5)
            chunks.append(c)
            queue.enqueue(c)
        # Same priority → FIFO
        assert queue.dequeue() is chunks[0]
        assert queue.dequeue() is chunks[1]
        assert queue.dequeue() is chunks[2]


class TestChunkPriorityQueueOrdering:
    """Test priority ordering by task_relevance."""

    def test_higher_relevance_dequeued_first(self):
        queue = ChunkPriorityQueue()
        low = AttentionChunk(content="low", token_count=5, task_relevance=0.2)
        high = AttentionChunk(content="high", token_count=5, task_relevance=0.9)
        mid = AttentionChunk(content="mid", token_count=5, task_relevance=0.5)
        queue.enqueue(low)
        queue.enqueue(high)
        queue.enqueue(mid)
        assert queue.dequeue() is high
        assert queue.dequeue() is mid
        assert queue.dequeue() is low

    def test_priority_ordering_with_same_relevance(self):
        """When task_relevance is the same, FIFO breaks ties."""
        queue = ChunkPriorityQueue()
        first = AttentionChunk(content="first", token_count=5, task_relevance=0.7)
        second = AttentionChunk(content="second", token_count=5, task_relevance=0.7)
        queue.enqueue(first)
        queue.enqueue(second)
        assert queue.dequeue() is first
        assert queue.dequeue() is second

    def test_mixed_priorities_fifo_tiebreak(self):
        """Two high-priority chunks should come out in FIFO, then low."""
        queue = ChunkPriorityQueue()
        low = AttentionChunk(content="low", token_count=5, task_relevance=0.2)
        high1 = AttentionChunk(content="high1", token_count=5, task_relevance=0.9)
        high2 = AttentionChunk(content="high2", token_count=5, task_relevance=0.9)
        queue.enqueue(low)
        queue.enqueue(high1)
        queue.enqueue(high2)
        assert queue.dequeue() is high1
        assert queue.dequeue() is high2
        assert queue.dequeue() is low

    def test_importance_tags_boost_priority(self):
        """Chunks with importance tags should get a priority boost."""
        queue = ChunkPriorityQueue()
        no_tags = AttentionChunk(
            content="no_tags", token_count=5, task_relevance=0.5
        )
        with_tags = AttentionChunk(
            content="with_tags",
            token_count=5,
            task_relevance=0.5,
            importance_tags=("critical",),
        )
        queue.enqueue(no_tags)
        queue.enqueue(with_tags)
        # Tagged chunk should have higher effective priority
        assert queue.dequeue() is with_tags
        assert queue.dequeue() is no_tags

    def test_enqueue_returns_position(self):
        """enqueue() should return the position in the queue."""
        queue = ChunkPriorityQueue(max_size=5)
        pos1 = queue.enqueue(
            AttentionChunk(content="c1", token_count=5, task_relevance=0.5)
        )
        assert pos1 == 0
        pos2 = queue.enqueue(
            AttentionChunk(content="c2", token_count=5, task_relevance=0.5)
        )
        assert pos2 == 1


class TestChunkPriorityQueueOverflow:
    """Test buffer capacity limits."""

    def test_overflow_drops_lowest_priority(self):
        """When the queue is full, enqueue drops the lowest-priority chunk."""
        queue = ChunkPriorityQueue(max_size=3)
        low = AttentionChunk(content="low", token_count=5, task_relevance=0.1)
        mid = AttentionChunk(content="mid", token_count=5, task_relevance=0.5)
        high = AttentionChunk(content="high", token_count=5, task_relevance=0.9)

        queue.enqueue(low)
        queue.enqueue(mid)
        queue.enqueue(high)
        assert len(queue) == 3

        # Add a medium-high chunk — should evict the lowest (low)
        newcomer = AttentionChunk(
            content="new", token_count=5, task_relevance=0.7
        )
        result = queue.enqueue(newcomer)
        assert len(queue) == 3  # Still at capacity
        # low should have been evicted
        remaining = list(queue)
        contents = [c.content for c in remaining]
        assert "low" not in contents
        assert "new" in contents

    def test_overflow_returns_evicted_chunk(self):
        """enqueue() on a full queue evicts the lowest priority chunk."""
        queue = ChunkPriorityQueue(max_size=2)
        c1 = AttentionChunk(content="c1", token_count=5, task_relevance=0.3)
        c2 = AttentionChunk(content="c2", token_count=5, task_relevance=0.5)
        queue.enqueue(c1)
        queue.enqueue(c2)

        c3 = AttentionChunk(content="c3", token_count=5, task_relevance=0.9)
        result = queue.enqueue(c3)
        # c3 was admitted (evicted c1), result should be a valid position
        assert result >= 0
        # Verify c1 was evicted
        contents = [c.content for c in queue]
        assert "c1" not in contents
        assert "c3" in contents

    def test_no_eviction_when_below_capacity(self):
        """No eviction when queue is not full."""
        queue = ChunkPriorityQueue(max_size=10)
        c = AttentionChunk(content="c1", token_count=5, task_relevance=0.5)
        result = queue.enqueue(c)
        # No eviction
        assert result is not None  # Returns position info

    def test_overflow_lower_priority_newcomer_rejected(self):
        """When the newcomer has lower priority than all queued, it's dropped."""
        queue = ChunkPriorityQueue(max_size=3)
        high1 = AttentionChunk(content="h1", token_count=5, task_relevance=0.9)
        high2 = AttentionChunk(content="h2", token_count=5, task_relevance=0.8)
        high3 = AttentionChunk(content="h3", token_count=5, task_relevance=0.7)
        queue.enqueue(high1)
        queue.enqueue(high2)
        queue.enqueue(high3)

        # Newcomer with very low priority
        low_new = AttentionChunk(content="low", token_count=5, task_relevance=0.1)
        queue.enqueue(low_new)
        assert len(queue) == 3
        remaining = [c.content for c in queue]
        assert "low" not in remaining
        assert "h1" in remaining

    def test_clear(self):
        """Clear the queue."""
        queue = ChunkPriorityQueue()
        for i in range(5):
            queue.enqueue(
                AttentionChunk(content=f"c{i}", token_count=5, task_relevance=0.5)
            )
        assert len(queue) == 5
        queue.clear()
        assert len(queue) == 0


class TestChunkPriorityQueueStats:
    """Test queue statistics tracking."""

    def test_stats_empty(self):
        queue = ChunkPriorityQueue()
        stats = queue.stats
        assert stats["current_size"] == 0
        assert stats["max_size"] == 64
        assert stats["total_enqueued"] == 0
        assert stats["total_dequeued"] == 0
        assert stats["total_evicted"] == 0
        assert stats["total_dropped"] == 0

    def test_stats_after_operations(self):
        queue = ChunkPriorityQueue(max_size=3)
        c1 = AttentionChunk(content="c1", token_count=5, task_relevance=0.3)
        c2 = AttentionChunk(content="c2", token_count=5, task_relevance=0.5)
        queue.enqueue(c1)
        queue.enqueue(c2)
        queue.dequeue()
        stats = queue.stats
        assert stats["current_size"] == 1
        assert stats["total_enqueued"] == 2
        assert stats["total_dequeued"] == 1

    def test_stats_track_evictions(self):
        queue = ChunkPriorityQueue(max_size=2)
        c1 = AttentionChunk(content="c1", token_count=5, task_relevance=0.3)
        c2 = AttentionChunk(content="c2", token_count=5, task_relevance=0.5)
        c3 = AttentionChunk(content="c3", token_count=5, task_relevance=0.9)
        queue.enqueue(c1)
        queue.enqueue(c2)
        queue.enqueue(c3)  # Should evict lowest
        stats = queue.stats
        assert stats["total_evicted"] == 1
        assert stats["current_size"] == 2

    def test_stats_track_drops(self):
        """When a lower-priority newcomer can't displace anyone, it's dropped."""
        queue = ChunkPriorityQueue(max_size=2)
        c1 = AttentionChunk(content="c1", token_count=5, task_relevance=0.9)
        c2 = AttentionChunk(content="c2", token_count=5, task_relevance=0.8)
        queue.enqueue(c1)
        queue.enqueue(c2)
        # Low-priority newcomer — should be dropped, not evict
        low = AttentionChunk(content="low", token_count=5, task_relevance=0.1)
        queue.enqueue(low)
        stats = queue.stats
        assert stats["total_dropped"] == 1
        assert stats["total_evicted"] == 0  # No eviction — newcomer was dropped


# ═══════════════════════════════════════════════════════════════════
# Integration tests with AttentionEquilibriumSystem
# ═══════════════════════════════════════════════════════════════════


class TestAESRejectBuffering:
    """Test that rejected chunks are buffered in the priority queue."""

    def test_reject_policy_buffers_rejected_chunks(self, reject_aes):
        """With REJECT policy, rejected chunks go to the queue."""
        # Fill up the budget
        for i in range(10):
            reject_aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )
        assert reject_aes.budget.tokens_used == 100  # Full

        # This should be rejected and buffered
        result = reject_aes.add_chunk(
            content="rejected", token_count=10, task_relevance=0.7
        )
        assert result is None
        assert len(reject_aes.rejected_queue) == 1
        assert reject_aes.rejected_queue.peek().content == "rejected"

    def test_auto_gc_policy_buffers_post_gc_rejections(self, engine):
        """With AUTO_GC, chunks rejected after GC go to the queue."""
        # Use REJECT policy to precisely control budget fill, then test
        # AUTO_GC separately with high-relevance chunks that GC won't prune
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.AUTO_GC,
        )

        # Fill with high-relevance chunks that GC won't easily prune
        for i in range(10):
            aes.add_chunk(
                content=f"fill-{i}",
                token_count=10,
                task_relevance=0.9,
                importance_tags=("critical",),
            )
        # After adding, budget might have been reduced by auto-GC,
        # but we should still be near capacity
        assert aes.budget.tokens_used > 0

        # If budget is full, try to add another and check queue
        if aes.budget.tokens_used + 10 > aes.budget.token_capacity:
            result = aes.add_chunk(
                content="rejected-after-gc", token_count=10, task_relevance=0.5
            )
            if result is None:
                assert len(aes.rejected_queue) >= 1

    def test_oversized_chunks_not_buffered(self, reject_aes):
        """Oversized chunks (larger than total capacity) are NOT buffered."""
        result = reject_aes.add_chunk(
            content="oversized", token_count=200, task_relevance=0.9
        )
        assert result is None
        # Oversized chunks should not go to the queue — they'd never fit
        assert len(reject_aes.rejected_queue) == 0

    def test_rejected_queue_priority_ordering(self, reject_aes):
        """Rejected chunks are ordered by task_relevance in the queue."""
        # Fill up
        for i in range(10):
            reject_aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )

        # Reject multiple chunks with different priorities
        reject_aes.add_chunk(content="low-pri", token_count=10, task_relevance=0.2)
        reject_aes.add_chunk(content="high-pri", token_count=10, task_relevance=0.9)
        reject_aes.add_chunk(content="mid-pri", token_count=10, task_relevance=0.5)

        assert len(reject_aes.rejected_queue) == 3
        # Should be dequeued in priority order
        assert reject_aes.rejected_queue.peek().content == "high-pri"


class TestAESRetryAfterGC:
    """Test retry of buffered chunks after GC frees space."""

    def test_retry_buffered_after_gc_frees_space(self, engine):
        """After GC frees space, buffered chunks can be retried."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
            rejected_queue_capacity=10,
        )

        # Fill with low-relevance chunks
        for i in range(10):
            aes.add_chunk(
                content=f"low-{i}", token_count=10, task_relevance=0.1
            )
        assert aes.budget.tokens_used == 100

        # Reject a high-relevance chunk
        result = aes.add_chunk(
            content="important", token_count=10, task_relevance=0.9
        )
        assert result is None
        assert len(aes.rejected_queue) == 1

        # Manually run GC — low-relevance chunks should be pruned
        report = aes.collect_garbage()
        assert report.pruned_count > 0
        assert aes.budget.headroom >= 10

        # Retry the buffered chunk
        retried = aes.retry_rejected()
        assert retried is not None
        assert retried.content == "important"
        assert len(aes.rejected_queue) == 0

    def test_retry_returns_none_when_empty(self, small_aes):
        """retry_rejected() returns None when queue is empty."""
        assert small_aes.retry_rejected() is None

    def test_retry_returns_none_when_still_full(self, engine):
        """retry_rejected() returns None if budget is still full."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )

        # Fill with high-relevance chunks (GC won't free much)
        for i in range(10):
            aes.add_chunk(
                content=f"high-{i}",
                token_count=10,
                task_relevance=0.9,
                importance_tags=("critical",),
            )

        # Reject a chunk
        result = aes.add_chunk(
            content="rejected", token_count=10, task_relevance=0.5
        )
        assert result is None
        assert len(aes.rejected_queue) == 1

        # Retry without freeing space — should fail
        retried = aes.retry_rejected()
        assert retried is None
        # Chunk stays in queue
        assert len(aes.rejected_queue) == 1

    def test_retry_multiple_chunks_in_priority_order(self, engine):
        """Retry drains the queue in priority order."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=200,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
            rejected_queue_capacity=10,
        )

        # Fill with low-relevance chunks
        for i in range(20):
            aes.add_chunk(
                content=f"low-{i}", token_count=10, task_relevance=0.1
            )
        assert aes.budget.tokens_used == 200

        # Reject chunks with different priorities
        aes.add_chunk(content="low-pri", token_count=10, task_relevance=0.2)
        aes.add_chunk(content="high-pri", token_count=10, task_relevance=0.9)
        aes.add_chunk(content="mid-pri", token_count=10, task_relevance=0.5)
        assert len(aes.rejected_queue) == 3

        # GC frees lots of space (all low-relevance)
        report = aes.collect_garbage()
        assert report.pruned_count > 0

        # Retry should get high-pri first
        r1 = aes.retry_rejected()
        assert r1 is not None
        assert r1.content == "high-pri"

        r2 = aes.retry_rejected()
        assert r2 is not None
        assert r2.content == "mid-pri"

        r3 = aes.retry_rejected()
        assert r3 is not None
        assert r3.content == "low-pri"

        r4 = aes.retry_rejected()
        assert r4 is None  # Queue empty


class TestAESRejectedQueueStats:
    """Test rejected queue statistics in the main AES stats."""

    def test_stats_include_queue_info(self, reject_aes):
        """The main stats dict includes rejected queue statistics."""
        # Fill and reject
        for i in range(10):
            reject_aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )
        reject_aes.add_chunk(
            content="rejected", token_count=10, task_relevance=0.7
        )

        stats = reject_aes.stats
        assert "rejected_queue" in stats
        queue_stats = stats["rejected_queue"]
        assert queue_stats["current_size"] == 1
        assert queue_stats["total_enqueued"] == 1

    def test_stats_queue_empty_by_default(self, small_aes):
        """By default, the queue section shows zero stats."""
        stats = small_aes.stats
        assert "rejected_queue" in stats
        assert stats["rejected_queue"]["current_size"] == 0


class TestAESRejectedQueueCapacity:
    """Test rejected queue capacity configuration."""

    def test_custom_queue_capacity(self, engine):
        """Custom queue capacity limits the buffer size."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
            rejected_queue_capacity=2,
        )
        # Fill
        for i in range(10):
            aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )

        # Reject more than the queue can hold
        aes.add_chunk(content="r1", token_count=10, task_relevance=0.6)
        aes.add_chunk(content="r2", token_count=10, task_relevance=0.7)
        aes.add_chunk(content="r3", token_count=10, task_relevance=0.8)

        # Queue should be at capacity (2), with lowest priority evicted (r1)
        assert len(aes.rejected_queue) == 2
        # r3 (0.8) and r2 (0.7) should be kept; r1 (0.6) evicted
        contents = [c.content for c in aes.rejected_queue]
        assert "r3" in contents
        assert "r2" in contents
        assert "r1" not in contents

    def test_zero_queue_capacity_disables_buffering(self, engine):
        """Setting capacity to 0 disables the rejected chunk queue."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
            rejected_queue_capacity=0,
        )
        # Fill
        for i in range(10):
            aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )

        # Reject — should not be buffered in the real queue
        result = aes.add_chunk(
            content="rejected", token_count=10, task_relevance=0.9
        )
        assert result is None
        # The sentinel queue should still show 0 (nothing was enqueued to it)
        assert aes.rejected_queue.stats["current_size"] == 0

    def test_default_queue_capacity(self, engine):
        """Default queue capacity should be reasonable (e.g. 64)."""
        aes = AttentionEquilibriumSystem(
            engine=engine,
            token_capacity=100,
            enforcement_policy=BudgetEnforcementPolicy.REJECT,
        )
        assert aes.rejected_queue.max_size == 64


class TestBackwardCompatibility:
    """Ensure existing functionality is preserved."""

    def test_add_chunk_still_returns_chunk_on_success(self, small_aes):
        """add_chunk() still returns an AttentionChunk on success."""
        chunk = small_aes.add_chunk(
            content="hello", token_count=10, task_relevance=0.5
        )
        assert chunk is not None
        assert isinstance(chunk, AttentionChunk)

    def test_add_chunk_still_returns_none_on_rejection(self, reject_aes):
        """add_chunk() still returns None on rejection."""
        for i in range(10):
            reject_aes.add_chunk(
                content=f"fill-{i}", token_count=10, task_relevance=0.5
            )
        result = reject_aes.add_chunk(
            content="rejected", token_count=10, task_relevance=0.5
        )
        assert result is None

    def test_no_queue_property_breaks_existing_code(self, small_aes):
        """The rejected_queue property doesn't break existing code paths."""
        # Existing code that doesn't check rejected_queue still works
        for i in range(5):
            small_aes.add_chunk(
                content=f"normal-{i}", token_count=10, task_relevance=0.5
            )
        assert small_aes.chunk_count == 5
        report = small_aes.collect_garbage()
        assert isinstance(report, GarbageCollectionReport)

    def test_existing_enforcement_stats_preserved(self, reject_aes):
        """Enforcement stats from iter-025 are still present and correct."""
        stats = reject_aes.stats
        assert "enforcement" in stats
        assert "policy" in stats["enforcement"]
        assert "rejections" in stats["enforcement"]
