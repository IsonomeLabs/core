"""Tests for CalibrationCache — morphology-aware kernel/policy cache.

Addresses architecture gap #6: the architecture specifies a Calibration Cache
keyed by SHA256(topology + task_type + vla_version) that stores certified
policy packages. The existing SemanticCache is a generic string TTL cache
with no topology awareness.

The CalibrationCache composes the TopologyVector.topology_hash with task_type
and vla_version to produce deterministic, morphology-aware cache keys.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from isonome.core.calibration_cache import (
    CalibrationCache,
    CalibrationCacheKey,
    CalibrationCacheEntry,
    CalibrationCacheStats,
)
from isonome.utils.morphology import TopologyVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_topology_vector(seed: int = 0) -> TopologyVector:
    """Create a deterministic 32-D TopologyVector for testing."""
    torch.manual_seed(seed)
    return TopologyVector(features=torch.randn(32))


def _make_cache_key(
    topology_hash: str = "abc123",
    task_type: str = "reach",
    vla_version: str = "openvla-7b-v1",
) -> CalibrationCacheKey:
    return CalibrationCacheKey(
        topology_hash=topology_hash,
        task_type=task_type,
        vla_version=vla_version,
    )


# ===========================================================================
# CalibrationCacheKey
# ===========================================================================


class TestCalibrationCacheKey:
    """Tests for the composite cache key model."""

    def test_key_attributes(self) -> None:
        key = _make_cache_key()
        assert key.topology_hash == "abc123"
        assert key.task_type == "reach"
        assert key.vla_version == "openvla-7b-v1"

    def test_key_deterministic_hash(self) -> None:
        """Same inputs → same composite hash."""
        k1 = _make_cache_key(topology_hash="aaa", task_type="pick", vla_version="v2")
        k2 = _make_cache_key(topology_hash="aaa", task_type="pick", vla_version="v2")
        assert k1.composite_hash() == k2.composite_hash()

    def test_key_different_topology_different_hash(self) -> None:
        k1 = _make_cache_key(topology_hash="aaa")
        k2 = _make_cache_key(topology_hash="bbb")
        assert k1.composite_hash() != k2.composite_hash()

    def test_key_different_task_different_hash(self) -> None:
        k1 = _make_cache_key(task_type="reach")
        k2 = _make_cache_key(task_type="pick")
        assert k1.composite_hash() != k2.composite_hash()

    def test_key_different_vla_different_hash(self) -> None:
        k1 = _make_cache_key(vla_version="v1")
        k2 = _make_cache_key(vla_version="v2")
        assert k1.composite_hash() != k2.composite_hash()

    def test_key_composite_hash_format(self) -> None:
        """Composite hash should be a SHA-256 hex digest (64 chars)."""
        key = _make_cache_key()
        h = key.composite_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_key_composite_hash_uses_sha256(self) -> None:
        """Verify the hash matches manual SHA-256 computation."""
        key = _make_cache_key(topology_hash="aaa", task_type="pick", vla_version="v2")
        expected_input = "aaa:pick:v2"
        expected = hashlib.sha256(expected_input.encode()).hexdigest()
        assert key.composite_hash() == expected

    def test_key_equality(self) -> None:
        k1 = _make_cache_key(topology_hash="x", task_type="y", vla_version="z")
        k2 = _make_cache_key(topology_hash="x", task_type="y", vla_version="z")
        assert k1 == k2

    def test_key_inequality(self) -> None:
        k1 = _make_cache_key(topology_hash="x")
        k2 = _make_cache_key(topology_hash="y")
        assert k1 != k2

    def test_key_hashable(self) -> None:
        """Keys must be usable in sets and dicts."""
        k1 = _make_cache_key(topology_hash="x", task_type="y", vla_version="z")
        k2 = _make_cache_key(topology_hash="x", task_type="y", vla_version="z")
        s = {k1, k2}
        assert len(s) == 1

    def test_key_from_topology_vector(self) -> None:
        """Factory method: build key from TopologyVector + task + version."""
        tv = _make_topology_vector(seed=42)
        key = CalibrationCacheKey.from_topology_vector(
            topology_vector=tv, task_type="reach", vla_version="openvla-v1"
        )
        assert key.topology_hash == tv.topology_hash
        assert key.task_type == "reach"
        assert key.vla_version == "openvla-v1"

    def test_key_from_topology_vector_deterministic(self) -> None:
        """Same topology vector → same hash in key."""
        tv1 = _make_topology_vector(seed=42)
        tv2 = _make_topology_vector(seed=42)
        k1 = CalibrationCacheKey.from_topology_vector(tv1, "reach", "v1")
        k2 = CalibrationCacheKey.from_topology_vector(tv2, "reach", "v1")
        assert k1.composite_hash() == k2.composite_hash()


# ===========================================================================
# CalibrationCacheEntry
# ===========================================================================


class TestCalibrationCacheEntry:
    """Tests for cache entry model."""

    def test_entry_attributes(self) -> None:
        entry = CalibrationCacheEntry(
            kernel_path="/path/to/kernel.pt",
            metadata={"episodes": 1000, "success_rate": 0.99},
        )
        assert entry.kernel_path == "/path/to/kernel.pt"
        assert entry.metadata["episodes"] == 1000

    def test_entry_default_metadata(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt")
        assert entry.metadata == {}

    def test_entry_timestamp_auto(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt")
        assert entry.cached_at > 0

    def test_entry_certified_flag(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt", certified=True)
        assert entry.certified is True

    def test_entry_default_not_certified(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt")
        assert entry.certified is False

    def test_entry_is_expired(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=0.0)
        # With TTL=0, should be expired immediately
        time.sleep(0.01)
        assert entry.is_expired

    def test_entry_not_expired(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=3600.0)
        assert not entry.is_expired

    def test_entry_no_ttl_never_expires(self) -> None:
        """Calibration entries typically have no TTL — they persist."""
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=None)
        assert not entry.is_expired


# ===========================================================================
# CalibrationCache — Core Operations
# ===========================================================================


class TestCalibrationCacheCore:
    """Tests for basic cache operations: put, get, has, evict."""

    def test_put_and_get(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        entry = CalibrationCacheEntry(kernel_path="/k.pt", certified=True)
        cache.put(key, entry)
        result = cache.get(key)
        assert result is not None
        assert result.kernel_path == "/k.pt"
        assert result.certified is True

    def test_get_missing_returns_none(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        assert cache.get(key) is None

    def test_has_key(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        assert not cache.has(key)
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.has(key)

    def test_remove(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.has(key)
        removed = cache.remove(key)
        assert removed is not None
        assert not cache.has(key)

    def test_remove_missing_returns_none(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        assert cache.remove(key) is None

    def test_put_overwrites(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/v1.pt"))
        cache.put(key, CalibrationCacheEntry(kernel_path="/v2.pt", certified=True))
        result = cache.get(key)
        assert result is not None
        assert result.kernel_path == "/v2.pt"
        assert result.certified is True

    def test_len(self) -> None:
        cache = CalibrationCache()
        assert len(cache) == 0
        cache.put(_make_cache_key(topology_hash="a"), CalibrationCacheEntry(kernel_path="/a.pt"))
        assert len(cache) == 1
        cache.put(_make_cache_key(topology_hash="b"), CalibrationCacheEntry(kernel_path="/b.pt"))
        assert len(cache) == 2

    def test_clear(self) -> None:
        cache = CalibrationCache()
        cache.put(_make_cache_key(topology_hash="a"), CalibrationCacheEntry(kernel_path="/a.pt"))
        cache.put(_make_cache_key(topology_hash="b"), CalibrationCacheEntry(kernel_path="/b.pt"))
        cache.clear()
        assert len(cache) == 0


# ===========================================================================
# CalibrationCache — Topology-Aware Keying
# ===========================================================================


class TestCalibrationCacheTopologyKeying:
    """Tests verifying that topology hash correctly differentiates entries."""

    def test_same_robot_different_tasks(self) -> None:
        """Same robot (topology), different tasks → different cache entries."""
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=10)

        key_reach = CalibrationCacheKey.from_topology_vector(tv, "reach", "v1")
        key_pick = CalibrationCacheKey.from_topology_vector(tv, "pick", "v1")

        cache.put(key_reach, CalibrationCacheEntry(kernel_path="/reach.pt"))
        cache.put(key_pick, CalibrationCacheEntry(kernel_path="/pick.pt"))

        assert cache.get(key_reach).kernel_path == "/reach.pt"
        assert cache.get(key_pick).kernel_path == "/pick.pt"

    def test_same_robot_different_vla(self) -> None:
        """Same robot, same task, different VLA → different entries."""
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=10)

        key_v1 = CalibrationCacheKey.from_topology_vector(tv, "reach", "openvla-v1")
        key_v2 = CalibrationCacheKey.from_topology_vector(tv, "reach", "openvla-v2")

        cache.put(key_v1, CalibrationCacheEntry(kernel_path="/v1.pt"))
        cache.put(key_v2, CalibrationCacheEntry(kernel_path="/v2.pt"))

        assert cache.get(key_v1).kernel_path == "/v1.pt"
        assert cache.get(key_v2).kernel_path == "/v2.pt"

    def test_different_robots_same_task(self) -> None:
        """Different robots (topology), same task → different entries."""
        cache = CalibrationCache()
        tv_a = _make_topology_vector(seed=10)
        tv_b = _make_topology_vector(seed=20)

        key_a = CalibrationCacheKey.from_topology_vector(tv_a, "reach", "v1")
        key_b = CalibrationCacheKey.from_topology_vector(tv_b, "reach", "v1")

        cache.put(key_a, CalibrationCacheEntry(kernel_path="/robot_a.pt"))
        cache.put(key_b, CalibrationCacheEntry(kernel_path="/robot_b.pt"))

        assert cache.get(key_a).kernel_path == "/robot_a.pt"
        assert cache.get(key_b).kernel_path == "/robot_b.pt"

    def test_convenience_lookup_by_topology(self) -> None:
        """Convenience method: get() accepts topology_vector + task + version."""
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=42)

        key = CalibrationCacheKey.from_topology_vector(tv, "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))

        # Convenience lookup
        result = cache.get_by_topology(tv, "reach", "v1")
        assert result is not None
        assert result.kernel_path == "/k.pt"
        assert result.certified is True

    def test_convenience_has_by_topology(self) -> None:
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=42)

        assert not cache.has_by_topology(tv, "reach", "v1")

        key = CalibrationCacheKey.from_topology_vector(tv, "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))

        assert cache.has_by_topology(tv, "reach", "v1")


# ===========================================================================
# CalibrationCache — TTL & Eviction
# ===========================================================================


class TestCalibrationCacheEviction:
    """Tests for TTL-based eviction and max size enforcement."""

    def test_expired_entry_not_returned(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=0.0)
        cache.put(key, entry)
        time.sleep(0.01)
        assert cache.get(key) is None

    def test_expired_entry_removed_on_get(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=0.0)
        cache.put(key, entry)
        time.sleep(0.01)
        cache.get(key)  # triggers lazy eviction
        assert len(cache) == 0

    def test_max_size_evicts_oldest(self) -> None:
        """When max_size is reached, evict the oldest entry."""
        cache = CalibrationCache(max_size=2)
        k1 = _make_cache_key(topology_hash="a")
        k2 = _make_cache_key(topology_hash="b")
        k3 = _make_cache_key(topology_hash="c")

        cache.put(k1, CalibrationCacheEntry(kernel_path="/1.pt"))
        cache.put(k2, CalibrationCacheEntry(kernel_path="/2.pt"))
        cache.put(k3, CalibrationCacheEntry(kernel_path="/3.pt"))

        # k1 should have been evicted (oldest)
        assert not cache.has(k1)
        assert cache.has(k2)
        assert cache.has(k3)

    def test_evict_expired_removes_only_expired(self) -> None:
        cache = CalibrationCache()
        k1 = _make_cache_key(topology_hash="exp")
        k2 = _make_cache_key(topology_hash="fresh")

        cache.put(k1, CalibrationCacheEntry(kernel_path="/e.pt", ttl=0.0))
        cache.put(k2, CalibrationCacheEntry(kernel_path="/f.pt", ttl=3600.0))

        time.sleep(0.01)
        removed = cache.evict_expired()
        assert removed == 1
        assert not cache.has(k1)
        assert cache.has(k2)

    def test_no_ttl_entries_not_evicted(self) -> None:
        """Calibration entries (ttl=None) should never expire."""
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", ttl=None))
        removed = cache.evict_expired()
        assert removed == 0
        assert cache.has(key)


# ===========================================================================
# CalibrationCache — Statistics
# ===========================================================================


class TestCalibrationCacheStats:
    """Tests for cache statistics tracking."""

    def test_stats_initial(self) -> None:
        cache = CalibrationCache()
        stats = cache.stats
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.puts == 0

    def test_stats_tracks_puts(self) -> None:
        cache = CalibrationCache()
        cache.put(_make_cache_key(), CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.stats.puts == 1

    def test_stats_tracks_hits(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        cache.get(key)
        assert cache.stats.hits == 1

    def test_stats_tracks_misses(self) -> None:
        cache = CalibrationCache()
        cache.get(_make_cache_key())
        assert cache.stats.misses == 1

    def test_stats_tracks_evictions(self) -> None:
        cache = CalibrationCache(max_size=1)
        k1 = _make_cache_key(topology_hash="a")
        k2 = _make_cache_key(topology_hash="b")
        cache.put(k1, CalibrationCacheEntry(kernel_path="/1.pt"))
        cache.put(k2, CalibrationCacheEntry(kernel_path="/2.pt"))
        assert cache.stats.evictions == 1

    def test_stats_hit_rate(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        cache.get(key)  # hit
        cache.get(_make_cache_key(topology_hash="miss"))  # miss
        assert cache.stats.hit_rate == 0.5

    def test_stats_hit_rate_zero_when_no_accesses(self) -> None:
        cache = CalibrationCache()
        assert cache.stats.hit_rate == 0.0

    def test_stats_resets(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        cache.get(key)
        cache.get(_make_cache_key(topology_hash="miss"))
        cache.stats.reset()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.puts == 0
        assert cache.stats.evictions == 0


# ===========================================================================
# CalibrationCache — Certification Filtering
# ===========================================================================


class TestCalibrationCacheCertification:
    """Tests for certified-only lookups."""

    def test_get_certified_only_true(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))
        result = cache.get(key, certified_only=True)
        assert result is not None
        assert result.certified is True

    def test_get_certified_only_filters_uncertified(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=False))
        result = cache.get(key, certified_only=True)
        assert result is None

    def test_get_certified_only_false_returns_any(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=False))
        result = cache.get(key, certified_only=False)
        assert result is not None

    def test_get_by_topology_certified_only(self) -> None:
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=5)
        key = CalibrationCacheKey.from_topology_vector(tv, "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))

        result = cache.get_by_topology(tv, "reach", "v1", certified_only=True)
        assert result is not None
        assert result.certified is True


# ===========================================================================
# CalibrationCache — Integration with PlasticityLayer
# ===========================================================================


class TestCalibrationCachePlasticityIntegration:
    """Tests that CalibrationCache integrates with the existing kernel path scheme."""

    def test_kernel_path_uses_composite_hash(self) -> None:
        """Cache should be able to produce a kernel filename from its key."""
        key = _make_cache_key(topology_hash="abc", task_type="reach", vla_version="v1")
        # The composite hash should be usable as a kernel filename
        filename = f"{key.composite_hash()[:16]}.pt"
        assert filename.endswith(".pt")
        assert len(filename) == 19  # 16 hex chars + ".pt"

    def test_entries_for_different_robots_independent(self) -> None:
        """Two different robots with same task/version get independent entries."""
        cache = CalibrationCache()
        tv_arm = _make_topology_vector(seed=1)
        tv_leg = _make_topology_vector(seed=2)

        key_arm = CalibrationCacheKey.from_topology_vector(tv_arm, "reach", "v1")
        key_leg = CalibrationCacheKey.from_topology_vector(tv_leg, "reach", "v1")

        cache.put(key_arm, CalibrationCacheEntry(
            kernel_path=f"/kernels/{key_arm.composite_hash()[:16]}.pt",
            metadata={"joints": 7},
            certified=True,
        ))
        cache.put(key_leg, CalibrationCacheEntry(
            kernel_path=f"/kernels/{key_leg.composite_hash()[:16]}.pt",
            metadata={"joints": 12},
            certified=True,
        ))

        arm_entry = cache.get(key_arm)
        leg_entry = cache.get(key_leg)

        assert arm_entry.metadata["joints"] == 7
        assert leg_entry.metadata["joints"] == 12
        assert arm_entry.kernel_path != leg_entry.kernel_path

    def test_backward_compatible_with_robot_hash(self) -> None:
        """The old SomaLayer._robot_hash() (16-char hex) can still be used
        as the topology_hash field for backward compatibility."""
        old_hash = "a1b2c3d4e5f6a7b8"  # 16-char hex from _robot_hash()
        key = _make_cache_key(topology_hash=old_hash)
        cache = CalibrationCache()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.has(key)

    def test_lookup_by_robot_hash_shortcut(self) -> None:
        """Convenience method for looking up by raw robot hash string."""
        cache = CalibrationCache()
        robot_hash = "a1b2c3d4e5f6a7b8"
        key = CalibrationCacheKey(
            topology_hash=robot_hash, task_type="reach", vla_version="v1"
        )
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))

        result = cache.get_by_robot_hash(robot_hash, "reach", "v1")
        assert result is not None
        assert result.kernel_path == "/k.pt"


# ===========================================================================
# CalibrationCache — Serialization
# ===========================================================================


class TestCalibrationCacheSerialization:
    """Tests for serializing/deserializing the cache state."""

    def test_to_dict(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(
            kernel_path="/k.pt",
            metadata={"episodes": 500},
            certified=True,
        ))
        d = cache.to_dict()
        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_from_dict_roundtrip(self) -> None:
        cache = CalibrationCache()
        key = _make_cache_key()
        cache.put(key, CalibrationCacheEntry(
            kernel_path="/k.pt",
            metadata={"episodes": 500},
            certified=True,
        ))

        serialized = cache.to_dict()
        restored = CalibrationCache.from_dict(serialized)

        assert len(restored) == 1
        result = restored.get(key)
        assert result is not None
        assert result.kernel_path == "/k.pt"
        assert result.certified is True
        assert result.metadata["episodes"] == 500

    def test_empty_cache_serialization(self) -> None:
        cache = CalibrationCache()
        d = cache.to_dict()
        restored = CalibrationCache.from_dict(d)
        assert len(restored) == 0
