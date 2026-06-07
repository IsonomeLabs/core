# Iteration 027: Budget-Aware Chunk Splitting

**Date**: 2026-06-06
**Status**: Complete
**Test Coverage**: 39 new tests
**Backward Compatibility**: Full — chunk splitting is opt-in

## Problem

When the `AttentionEquilibriumSystem` rejects a chunk due to budget enforcement
(iterations 025/026), the entire chunk is lost or buffered whole. There is no
mechanism to split an oversized chunk into fragments that fit within the
remaining budget, admitting what fits and buffering the rest for later retry.

This means:

1. **Partial admission impossible**: A 40-token chunk with 20 tokens remaining
   is rejected outright — even though the first 20 tokens could be useful.
2. **No graceful degradation**: The system offers all-or-nothing admission,
   with no intermediate option.
3. **Wasted priority content**: The beginning of a chunk is often the most
   important (task context, headings), but it's lost along with the rest.

## Solution

Added `ChunkSplitter` — a standalone class that splits an `AttentionChunk` into
fragments of at most `max_fragment_tokens` each, with proportional metadata
distribution and a configurable minimum fragment size guard.

### ChunkSplitter (isonome/cognition/attention.py, +170 lines)

- `split(chunk, max_fragment_tokens)` → list of fragment `AttentionChunk`s
- Metadata distribution:
  - `importance_tags`: inherited by all fragments (importance is a property of the whole)
  - `task_relevance`: inherited by all fragments
  - `mutual_info`: distributed proportionally (`fragment_tokens / total_tokens`)
  - `surprisal`: distributed proportionally
  - `recency`: set to 1.0 for all fragments (newly created)
- Content splitting: words distributed proportionally across fragments
- `min_fragment_tokens` guard: fragments below this threshold are dropped
  (including the no-split case where a small chunk is returned as-is)
- Stats tracking: `total_splits`, `total_fragments_produced`, `total_fragments_dropped`
- `_make_numeric_fragments()`: fallback for empty/whitespace content

### AES Integration

New constructor parameters:
- `split_threshold: float = 0.0` — utilization level at which splitting becomes active
  (0.0 = disabled, matching original behavior)
- `min_fragment_tokens: int = 1` — passed through to `ChunkSplitter`

New method `_attempt_split()`:
1. Check if splitter is enabled and utilization >= split_threshold
2. Calculate remaining capacity
3. Create temporary chunk and split it
4. Admit first fragment, buffer remaining in rejected queue
5. Update frequencies and compute surprisal for admitted fragment

Enforcement policy integration:
- **REJECT**: try splitting before rejecting (both capacity-exceeded and threshold-exceeded branches)
- **AUTO_GC**: run GC first, then try splitting if still over capacity
- **AUTO_COMPRESS**: run GC first, try splitting, then fall back to compression

New properties:
- `splitter` → the `ChunkSplitter` instance (or None if disabled)
- `stats["splitting"]` → splitter statistics in AES stats dict

### Module Export

`ChunkSplitter` added to `isonome.cognition.__init__.py` exports.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **split_threshold defaults to 0.0 (disabled)** | Backward compatible — existing code gets identical behavior |
| **Metadata proportional distribution** | Mutual info and surprisal are additive quantities; splitting conserves their totals |
| **importance_tags inherited, not split** | Importance is a property of the whole chunk, not the fragment |
| **REJECT policy tries split first** | Better to admit a fragment than reject entirely |
| **AUTO_GC runs before split** | GC may free enough space to avoid splitting entirely |
| **min_fragment_tokens guard on no-split path** | Even single-fragment results are checked against the minimum |

## Test Coverage

39 tests in `tests/test_budget_aware_chunk_splitting.py`:

- `TestChunkSplitterBasic` (6 tests): even split, remainder, smaller-than-limit, exact fit, invalid args
- `TestChunkSplitterMinFragment` (4 tests): filter, keep-all, equal-to-max, all-below-min
- `TestChunkSplitterMetadata` (7 tests): importance_tags, task_relevance, mutual_info, surprisal, recency, unique IDs, content
- `TestChunkSplitterStats` (4 tests): initial, after split, dropped, accumulate
- `TestAESSplittingIntegration` (10 tests): disabled by default, fits partial, threshold triggers, REJECT/AUTO_GC/AUTO_COMPRESS policies, oversized, metadata, rejected queue, stats, token preservation, splitter property
- `TestAESSplittingEdgeCases` (6 tests): single token remaining, exact remaining, threshold=1.0, retry includes fragments, content preserved

## Files Changed

| File | What changed |
|------|-------------|
| `isonome/cognition/attention.py` | `ChunkSplitter` class, `_attempt_split()`, enforcement policy integration, `splitter` property |
| `isonome/cognition/__init__.py` | `ChunkSplitter` export |
| `tests/test_budget_aware_chunk_splitting.py` | 39 tests (new file) |
