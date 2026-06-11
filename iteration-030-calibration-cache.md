# Iteration 030: Calibration Cache — Topology-Aware Policy Package Cache

**Date**: 2026-06-11
**Status**: Complete
**Test Coverage**: 16 new tests (all passing)
**Backward Compatibility**: Full — no existing interfaces changed; new module only.

---

## Problem

Architecture gap #6: the architecture specifies a **Calibration Cache** keyed by `SHA256(topology_hash + task_type + vla_version)` that stores certified policy packages. The only cache in the codebase was `SemanticCache` in `isonome/llm/cache.py` — a generic string TTL cache for Cortex advice strings with no topology awareness, no namespaces, and no support for policy packages.

---

## Solution

### `isonome/praxis/calibration_cache.py`

New module implementing the full calibration cache spec (PRD FR-5.1–FR-5.5):

**`CacheKey`** (frozen dataclass)
- Fields: `topology_hash`, `task_type`, `vla_version`
- Canonical key: `SHA256("topology_hash:task_type:vla_version")`
- Deterministic, hashable, serializable to/from dict

**`CertifiedPolicyPackage`** (dataclass)
- Mirrors PRD FR-4.9 artifact list:
  - `manifest`, `agent_configs`, `coordinator_config`, `reflex_gains`
  - `sim_metrics`, `policy_package_path`, `certification_video_path`, `launcher_path`
- Loose schema (dicts + optional paths) so future calibration pipelines can store arbitrary metadata without cache migrations.

**`CalibrationCache`**
- On-disk storage under `~/.isonome/cache/{namespace}/{sha256_key}/`
- Per-entry files: `meta.json` (key + timestamp + topology vector) and `package.json` (package data)
- **`put(key, package, namespace, topology_vector)`** — stores package, optionally with a 32-D topology vector for near-match search.
- **`get(key, namespace)`** — exact-match retrieval.
- **`exists(key, namespace)`** — boolean check.
- **`list_keys(namespace)`** — enumerate all keys in a namespace.
- **`find_near_matches(key, topology_vector, epsilon, namespace)`** — L2 distance search over stored topology vectors, filtered by matching `task_type` and `vla_version`. Returns `(distance, package)` tuples sorted by distance.
- **`clear(namespace)`** — bulk remove all entries in a namespace.
- **Namespaces**: `public` (default) and `private` directories. Open-source runtime stores both as plain JSON; enterprise deployments are expected to layer encryption on top of `private`.
- **Persistence**: cache entries survive process restarts because they are plain files on disk.

### `isonome/cli.py` — Cache CLI Subcommands

Added `cache` Typer subcommand group:

| Command | Purpose |
|---|---|
| `isonome cache put <topology_hash> <task_type> <vla_version> <package.json>` | Store a package in the cache. Optional `--topology-vector` JSON file for near-match support. |
| `isonome cache lookup <topology_hash> <task_type> <vla_version>` | Exact-match lookup. Optional `--near-match` / `--epsilon` to include topology-near results. |
| `isonome cache list` | List all keys in a namespace. |

### `isonome/praxis/__init__.py`

Exports `CacheKey`, `CalibrationCache`, `CertifiedPolicyPackage` at the package level.

### `architecture-gaps.md`

Updated to mark gaps #1, #5 as closed and #10 as partially closed. Summary table now lists gap #6 as the **next gap**.

---

## Test Coverage

16 tests in `tests/test_calibration_cache.py`:

| Test | What it covers |
|---|---|
| `test_cache_key_sha256_is_stable_and_unique` | Determinism and collision resistance |
| `test_cache_key_to_dict_roundtrip` | Serialization |
| `test_put_and_get` | Basic storage and retrieval |
| `test_get_missing_returns_none` | Missing key handling |
| `test_exists` | Boolean existence check |
| `test_namespace_isolation` | Public/private separation |
| `test_default_namespace` | Default namespace fallback |
| `test_near_match_finds_similar_topology` | L2 distance search, sorting |
| `test_near_match_filters_task_and_vla_version` | Correct filtering by task/VLA |
| `test_near_match_without_stored_vector_is_ignored` | Graceful handling of missing vectors |
| `test_list_keys` | Key enumeration per namespace |
| `test_clear` | Bulk deletion |
| `test_persistence_across_instances` | On-disk durability |
| `test_cli_cache_put_and_lookup` | End-to-end CLI roundtrip |
| `test_cli_cache_lookup_missing` | CLI error handling |
| `test_cli_cache_list` | CLI list command |

All 1971 tests in the full suite pass (16 new + 1955 existing).

---

## Architectural Impact

This iteration closes **architecture gap #6** (Calibration Cache vs LLM Cache). It enables:

1. **Semantic cache keys**: `SHA256(topology + task_type + vla_version)` as specified in Diagram 1.
2. **Near-match interpolation**: Find certified packages for morphologically similar robots when no exact match exists.
3. **Namespace separation**: Open-source public cache vs enterprise private cache.
4. **Certified policy package storage**: The runtime can now cache and retrieve complete calibration artifacts (manifests, configs, gains, metrics) instead of just raw kernel weights.
5. **Foundation for future training pipeline**: When gap #3 (CMA-ES / 256 envs / auto-adjustment) is implemented, its output can be stored directly in this cache and retrieved via the existing CLI/SDK.

---

## Files Changed

- `isonome/praxis/calibration_cache.py` — **NEW**
- `isonome/praxis/__init__.py` — Export new classes
- `isonome/cli.py` — Add `cache` subcommand group (`put`, `lookup`, `list`)
- `tests/test_calibration_cache.py` — **NEW**: 16 tests
- `architecture-gaps.md` — Mark #1, #5 closed; #10 partially closed

---

## Next Steps

- **Gap #3** (Calibration / Training Pipeline): Connect `CalibrationCache.put()` to the future CMA-ES / composition validation loop so successful calibrations are automatically cached.
- **Gap #10** (`run` / `deploy` CLI): Implement hardware bridge boot and deploy commands.
