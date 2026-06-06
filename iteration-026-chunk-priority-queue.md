# Iteration 026: Chunk Priority Queue for Rejected Chunks

## Problem

When the `AttentionEquilibriumSystem` rejects a chunk due to budget enforcement
(iteration-025), the chunk is silently dropped. There is no mechanism to buffer
rejected chunks for potential re-admission after a GC cycle frees space. This
means:

1. **Lost information**: High-priority chunks rejected during a temporary budget
   spike are gone forever, even if space becomes available moments later.
2. **No retry semantics**: Callers have no way to say "try this again later."
3. **Wasted computation**: Expensive chunk generation that is rejected cannot be
   recovered.

## Solution

Added `ChunkPriorityQueue` — a bounded priority queue that buffers chunks
rejected by budget enforcement, ordered by a composite priority score derived
from `task_relevance` and `mutual_info` (matching the existing retention
scoring weights: 0.7 * task_relevance + 0.3 * mutual_info).

### Key design decisions

1. **Negated min-heap**: Uses `heapq` with negated priorities so `heappop`
   returns the highest-priority chunk first, while preserving stable FIFO
   ordering via a monotonic sequence counter as a tiebreaker.

2. **Overflow eviction**: When the queue is full, the lowest-priority entry is
   evicted to make room. If the new chunk has lower priority than the current
   minimum, it is dropped instead (enqueue returns -1). This ensures the queue
   always retains the most valuable rejected chunks.

3. **Sentinel queue (capacity=0)**: When `rejected_queue_capacity=0`, a
   sentinel queue is created that accepts `enqueue()` calls (returning -1) but
   silently drops everything. This provides a clean "disable buffering" option
   without requiring conditional checks throughout the code.

4. **Oversized chunks not buffered**: Chunks whose `token_count` exceeds the
   total system capacity are never buffered — they can never be admitted, so
   buffering them would waste queue slots.

5. **Integration with `add_chunk()` rejection paths**:
   - `REJECT` policy: buffers rejected chunks
   - `AUTO_GC` policy: buffers chunks rejected *after* GC runs
   - `AUTO_COMPRESS` policy: same as AUTO_GC for buffering

6. **`retry_rejected()` method**: After GC frees space, `retry_rejected()`
   dequeues chunks from the rejected queue and re-attempts `add_chunk()`,
   returning a list of successfully re-admitted chunks.

### Changes

- `isonome/cognition/attention.py`:
  - `ChunkPriorityQueue` class (~100 lines):
    - `__init__(max_size)`, `enqueue(chunk)`, `dequeue()`, `peek()`, `__len__`, `__iter__`
    - `stats` property: `current_size`, `max_size`, `total_enqueued`, `total_dequeued`, `total_evicted`, `total_dropped`
    - `_effective_priority(chunk)` helper
  - `AttentionEquilibriumSystem.__init__()`: added `rejected_queue_capacity` param (default=64)
  - `AttentionEquilibriumSystem.add_chunk()`: buffers rejected chunks via `_buffer_rejected()`
  - `AttentionEquilibriumSystem._buffer_rejected()`: helper that enqueues to rejected queue
  - `AttentionEquilibriumSystem.rejected_queue`: property exposing the queue
  - `AttentionEquilibriumSystem.retry_rejected()`: dequeues and re-admits chunks
  - `AttentionEquilibriumSystem.stats`: includes `rejected_queue` stats

- `isonome/cognition/__init__.py`: Export `ChunkPriorityQueue`

- `tests/test_chunk_priority_queue.py`: 45 tests covering:
  - Basic enqueue/dequeue operations
  - Priority ordering (higher relevance = higher priority)
  - FIFO tiebreaking for equal priorities
  - Overflow eviction (lowest priority evicted)
  - Empty queue behavior
  - Iteration over queue contents
  - Stats tracking (enqueued, dequeued, evicted, dropped counts)
  - Sentinel queue (capacity=0) behavior
  - AES integration with REJECT policy
  - AES integration with AUTO_GC policy
  - Post-GC retry via `retry_rejected()`
  - Custom queue capacity configuration

## Test Results

All 1330 tests pass (1285 existing + 45 new).

## Next Iteration Candidates

1. **Budget-aware chunk splitting**: Break large chunks into smaller pieces
   that fit within remaining capacity, avoiding rejection entirely for
   partitionable content.

2. **Adaptive enforcement threshold**: Adjust the enforcement threshold based
   on GC effectiveness — if GC consistently frees >50% capacity, raise the
   threshold to delay enforcement.

3. **Attention compression with semantic summarization**: Current compression
   is purely numerical (token_count * ratio). Real compression would summarize
   content while preserving key information.

4. **Rejected queue persistence**: Serialize the rejected queue state so it
   survives across session boundaries, enabling long-lived retry semantics.
