# Iteration 032: Unified Calibration Cache — Bugfixes for On-Disk Cleanup

**Date**: 2026-06-15
**Status**: Complete
**Test Coverage**: 58 tests (all passing)
**Backward Compatibility**: Full — no API changes; bugfixes only.

---

## Problem

Two bugs in `UnifiedCalibrationCache` (from the iteration-030 unification of
gap #6) caused `remove()` and FIFO max-size eviction to only clean up
in-memory entries while leaving on-disk files intact. Because `has()` and
`get()` fall through to `_load_from_disk()` on an in-memory miss, removed or
evicted entries would "come back from the dead" on subsequent lookups.

### Bug 1: `remove()` doesn't clean on-disk entries

`remove()` called `_remove_entry(ik)` which correctly cleans the in-memory
store but leaves the `entry.json` file on disk. A subsequent `has(key)` call
finds no in-memory entry, falls through to the disk, and reloads it — making
the entry reappear after being "removed."

### Bug 2: `_evict_oldest()` doesn't clean on-disk entries

Same pattern: `_evict_oldest()` cleaned up in-memory dicts (`_store`,
`_keys`, `_namespaces`, `_topology_vectors`, `_insertion_order`) but left the
on-disk `entry.json` untouched. With `max_size=2`, after inserting 3 entries,
the oldest (k1) was evicted from memory but still present on disk, so
`has(k1)` returned `True`.

## Solution

### Fix 1: `remove()` — delete on-disk file alongside in-memory cleanup

After calling `_remove_entry(ik)`, the method now checks if persistence is
enabled and, if so, deletes the `entry.json` file and attempts to rmdir the
(now-empty) entry directory.

### Fix 2: `_evict_oldest()` — delete on-disk file alongside in-memory cleanup

The method now captures the namespace and key *before* removing them from
their dicts (by using `dict.pop()` instead of `del` for `_namespaces` and
`_keys`). After the in-memory cleanup, it deletes the on-disk file and
attempts to rmdir the entry directory.

## Test Coverage

All 58 existing tests in `tests/test_unified_calibration_cache.py` now pass
(previously 2 failed, 56 passed):

| Test | What it covers |
|---|---|
| `test_remove` | Verify `remove()` fully cleans both in-memory and on-disk |
| `test_max_size_evicts_oldest` | Verify FIFO eviction also cleans on-disk |

Full suite: 2148 passed, 4 skipped.

## Files Changed

- `isonome/core/unified_calibration_cache.py` — Fix `remove()` and
  `_evict_oldest()` to delete on-disk entries alongside in-memory cleanup.
- `iteration-032-unified-calibration-cache.md` — **NEW**: this document.

## Next Steps

- **Gap #6** is now fully closed. The unified cache provides in-memory hot
  path + on-disk persistence + near-match search + certification filtering
  + TTL + max-size eviction, all consistent across memory and disk.
- **Gap #4** (Simulation Backends Mismatch): Add Isaac Lab / MuJoCo MJX
  backend adapters.
- **Gap #10** (`run` / `deploy` CLI): Implement hardware bridge boot and
  deploy.
