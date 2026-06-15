"""Tests for UnifiedCalibrationCache — architecture gap #6 unification.

The unified cache combines:
- In-memory hot path with stats tracking and certification filtering (from core)
- On-disk persistence, namespaces, and topology-vector near-match search (from praxis)

These tests verify that a single cache provides all features.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from isonome.utils.morphology import TopologyVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_topology_vector(seed: int = 0) -> TopologyVector:
    """Create a deterministic 32-D TopologyVector for testing."""
    torch.manual_seed(seed)
    return TopologyVector(features=torch.randn(32))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "unified_cache"


@pytest.fixture
def unified_cache(tmp_cache_dir: Path):
    from isonome.core.unified_calibration_cache import (
        UnifiedCalibrationCache,
    )
    return UnifiedCalibrationCache(root_dir=tmp_cache_dir)


@pytest.fixture
def sample_key():
    from isonome.core.unified_calibration_cache import UnifiedCacheKey
    return UnifiedCacheKey(
        topology_hash="a1b2c3d4e5f6",
        task_type="reach",
        vla_version="openvla-7b",
    )


@pytest.fixture
def sample_entry():
    from isonome.core.unified_calibration_cache import UnifiedCacheEntry
    return UnifiedCacheEntry(
        kernel_path="/path/to/kernel.pt",
        metadata={"episodes": 1000, "success_rate": 0.99},
        certified=True,
    )


@pytest.fixture
def sample_package():
    from isonome.core.unified_calibration_cache import CertifiedPolicyPackage
    return CertifiedPolicyPackage(
        manifest={"task": "reach red cube"},
        agent_configs={"arm": {"dof": 7}},
        coordinator_config={"strategy": "priority"},
        reflex_gains={"kp": 1.0},
        sim_metrics={"success_rate": 0.99},
        policy_package_path="/tmp/pkg.zip",
    )


# ===========================================================================
# UnifiedCacheKey
# ===========================================================================


class TestUnifiedCacheKey:
    def test_key_attributes(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        key = UnifiedCacheKey("abc", "reach", "v1")
        assert key.topology_hash == "abc"
        assert key.task_type == "reach"
        assert key.vla_version == "v1"

    def test_key_sha256_deterministic(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        k1 = UnifiedCacheKey("aaa", "pick", "v2")
        k2 = UnifiedCacheKey("aaa", "pick", "v2")
        assert k1.composite_hash() == k2.composite_hash()

    def test_key_different_inputs_different_hash(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        k1 = UnifiedCacheKey("aaa", "reach", "v1")
        k2 = UnifiedCacheKey("bbb", "reach", "v1")
        k3 = UnifiedCacheKey("aaa", "grasp", "v1")
        k4 = UnifiedCacheKey("aaa", "reach", "v2")
        hashes = [k.composite_hash() for k in [k1, k2, k3, k4]]
        assert len(set(hashes)) == 4

    def test_key_composite_hash_uses_sha256(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        key = UnifiedCacheKey("aaa", "pick", "v2")
        expected = hashlib.sha256("aaa:pick:v2".encode()).hexdigest()
        assert key.composite_hash() == expected

    def test_key_from_topology_vector(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        tv = _make_topology_vector(seed=42)
        key = UnifiedCacheKey.from_topology_vector(tv, "reach", "openvla-v1")
        assert key.topology_hash == tv.topology_hash
        assert key.task_type == "reach"
        assert key.vla_version == "openvla-v1"

    def test_key_serialization_roundtrip(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        key = UnifiedCacheKey("topo", "task", "vla")
        d = key.to_dict()
        restored = UnifiedCacheKey.from_dict(d)
        assert restored == key
        assert restored.composite_hash() == key.composite_hash()


# ===========================================================================
# UnifiedCacheEntry
# ===========================================================================


class TestUnifiedCacheEntry:
    def test_entry_attributes(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(
            kernel_path="/path/to/kernel.pt",
            metadata={"episodes": 1000},
            certified=True,
        )
        assert entry.kernel_path == "/path/to/kernel.pt"
        assert entry.metadata["episodes"] == 1000
        assert entry.certified is True

    def test_entry_default_metadata(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(kernel_path="/k.pt")
        assert entry.metadata == {}

    def test_entry_default_not_certified(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(kernel_path="/k.pt")
        assert entry.certified is False

    def test_entry_is_expired_with_ttl(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(kernel_path="/k.pt", ttl=0.0)
        time.sleep(0.01)
        assert entry.is_expired

    def test_entry_not_expired_no_ttl(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(kernel_path="/k.pt", ttl=None)
        assert not entry.is_expired

    def test_entry_serialization_roundtrip(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        entry = UnifiedCacheEntry(
            kernel_path="/k.pt",
            metadata={"episodes": 500},
            certified=True,
        )
        d = entry.to_dict()
        restored = UnifiedCacheEntry.from_dict(d)
        assert restored.kernel_path == entry.kernel_path
        assert restored.certified == entry.certified
        assert restored.metadata == entry.metadata


# ===========================================================================
# CertifiedPolicyPackage
# ===========================================================================


class TestCertifiedPolicyPackage:
    def test_package_attributes(self) -> None:
        from isonome.core.unified_calibration_cache import CertifiedPolicyPackage
        pkg = CertifiedPolicyPackage(
            manifest={"task": "reach"},
            agent_configs={"arm": {"dof": 7}},
        )
        assert pkg.manifest == {"task": "reach"}
        assert pkg.agent_configs == {"arm": {"dof": 7}}

    def test_package_defaults(self) -> None:
        from isonome.core.unified_calibration_cache import CertifiedPolicyPackage
        pkg = CertifiedPolicyPackage()
        assert pkg.manifest == {}
        assert pkg.policy_package_path is None

    def test_package_serialization_roundtrip(self) -> None:
        from isonome.core.unified_calibration_cache import CertifiedPolicyPackage
        pkg = CertifiedPolicyPackage(
            manifest={"task": "reach"},
            policy_package_path="/tmp/pkg.zip",
        )
        d = pkg.to_dict()
        restored = CertifiedPolicyPackage.from_dict(d)
        assert restored.manifest == pkg.manifest
        assert restored.policy_package_path == pkg.policy_package_path


# ===========================================================================
# UnifiedCalibrationCache — Core Operations
# ===========================================================================


class TestUnifiedCacheCore:
    def test_put_and_get_in_memory(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        result = unified_cache.get(sample_key)
        assert result is not None
        assert result.kernel_path == sample_entry.kernel_path
        assert result.certified is True

    def test_put_and_get_with_package(self, unified_cache, sample_key, sample_package) -> None:
        unified_cache.put(sample_key, sample_package)
        result = unified_cache.get(sample_key)
        # When a CertifiedPolicyPackage is put, it should be retrievable
        assert result is not None

    def test_get_missing_returns_none(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        result = unified_cache.get(UnifiedCacheKey("x", "y", "z"))
        assert result is None

    def test_has_key(self, unified_cache, sample_key, sample_entry) -> None:
        assert not unified_cache.has(sample_key)
        unified_cache.put(sample_key, sample_entry)
        assert unified_cache.has(sample_key)

    def test_remove(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        removed = unified_cache.remove(sample_key)
        assert removed is not None
        assert not unified_cache.has(sample_key)

    def test_remove_missing_returns_none(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        assert unified_cache.remove(UnifiedCacheKey("x", "y", "z")) is None

    def test_put_overwrites(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/v1.pt", certified=False))
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/v2.pt", certified=True))
        result = unified_cache.get(sample_key)
        assert result is not None
        assert result.kernel_path == "/v2.pt"
        assert result.certified is True

    def test_len(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        assert len(unified_cache) == 0
        unified_cache.put(UnifiedCacheKey("a", "reach", "v1"), UnifiedCacheEntry(kernel_path="/a.pt"))
        assert len(unified_cache) == 1

    def test_clear(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        unified_cache.clear()
        assert len(unified_cache) == 0


# ===========================================================================
# UnifiedCalibrationCache — TTL & Eviction
# ===========================================================================


class TestUnifiedCacheEviction:
    def test_expired_entry_not_returned(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/k.pt", ttl=0.0))
        time.sleep(0.01)
        assert unified_cache.get(sample_key) is None

    def test_expired_entry_removed_on_get(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/k.pt", ttl=0.0))
        time.sleep(0.01)
        unified_cache.get(sample_key)
        assert len(unified_cache) == 0

    def test_max_size_evicts_oldest(self, tmp_cache_dir) -> None:
        from isonome.core.unified_calibration_cache import (
            UnifiedCacheEntry,
            UnifiedCacheKey,
            UnifiedCalibrationCache,
        )
        cache = UnifiedCalibrationCache(root_dir=tmp_cache_dir, max_size=2)
        k1 = UnifiedCacheKey("a", "reach", "v1")
        k2 = UnifiedCacheKey("b", "reach", "v1")
        k3 = UnifiedCacheKey("c", "reach", "v1")
        cache.put(k1, UnifiedCacheEntry(kernel_path="/1.pt"))
        cache.put(k2, UnifiedCacheEntry(kernel_path="/2.pt"))
        cache.put(k3, UnifiedCacheEntry(kernel_path="/3.pt"))
        assert not cache.has(k1)
        assert cache.has(k2)
        assert cache.has(k3)

    def test_evict_expired(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        k1 = UnifiedCacheKey("exp", "reach", "v1")
        k2 = UnifiedCacheKey("fresh", "reach", "v1")
        unified_cache.put(k1, UnifiedCacheEntry(kernel_path="/e.pt", ttl=0.0))
        unified_cache.put(k2, UnifiedCacheEntry(kernel_path="/f.pt", ttl=3600.0))
        time.sleep(0.01)
        removed = unified_cache.evict_expired()
        assert removed == 1
        assert not unified_cache.has(k1)
        assert unified_cache.has(k2)


# ===========================================================================
# UnifiedCalibrationCache — Statistics
# ===========================================================================


class TestUnifiedCacheStats:
    def test_stats_initial(self, unified_cache) -> None:
        assert unified_cache.stats.hits == 0
        assert unified_cache.stats.misses == 0
        assert unified_cache.stats.puts == 0
        assert unified_cache.stats.evictions == 0

    def test_stats_tracks_puts(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        assert unified_cache.stats.puts == 1

    def test_stats_tracks_hits(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        unified_cache.get(sample_key)
        assert unified_cache.stats.hits == 1

    def test_stats_tracks_misses(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        unified_cache.get(UnifiedCacheKey("x", "y", "z"))
        assert unified_cache.stats.misses == 1

    def test_hit_rate_calculation(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        unified_cache.get(sample_key)  # hit
        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        unified_cache.get(UnifiedCacheKey("x", "y", "z"))  # miss
        assert unified_cache.stats.hit_rate == 0.5

    def test_stats_reset(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        unified_cache.get(sample_key)
        unified_cache.stats.reset()
        assert unified_cache.stats.hits == 0
        assert unified_cache.stats.puts == 0


# ===========================================================================
# UnifiedCalibrationCache — Certification Filtering
# ===========================================================================


class TestUnifiedCacheCertification:
    def test_get_certified_only_true(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/k.pt", certified=True))
        assert unified_cache.get(sample_key, certified_only=True) is not None

    def test_get_certified_only_filters_uncertified(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(kernel_path="/k.pt", certified=False))
        assert unified_cache.get(sample_key, certified_only=True) is None

    def test_get_by_topology_certified_only(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        tv = _make_topology_vector(seed=5)
        key = UnifiedCacheKey.from_topology_vector(tv, "reach", "v1")
        unified_cache.put(key, UnifiedCacheEntry(kernel_path="/k.pt", certified=True))
        result = unified_cache.get_by_topology(tv, "reach", "v1", certified_only=True)
        assert result is not None


# ===========================================================================
# UnifiedCalibrationCache — On-Disk Persistence
# ===========================================================================


class TestUnifiedCachePersistence:
    def test_entry_persisted_to_disk(self, unified_cache, sample_key, sample_entry, tmp_cache_dir) -> None:
        unified_cache.put(sample_key, sample_entry)
        # File should exist in the namespace directory
        entry_dir = tmp_cache_dir / "public" / sample_key.composite_hash()
        assert entry_dir.exists()
        assert (entry_dir / "entry.json").exists()

    def test_reload_from_disk(self, tmp_cache_dir, sample_key) -> None:
        from isonome.core.unified_calibration_cache import (
            UnifiedCacheEntry,
            UnifiedCalibrationCache,
        )
        cache1 = UnifiedCalibrationCache(root_dir=tmp_cache_dir)
        cache1.put(sample_key, UnifiedCacheEntry(kernel_path="/k.pt", certified=True))

        # Create a new cache instance pointing to same dir
        cache2 = UnifiedCalibrationCache(root_dir=tmp_cache_dir)
        result = cache2.get(sample_key)
        assert result is not None
        assert result.kernel_path == "/k.pt"
        assert result.certified is True

    def test_in_memory_and_disk_consistent(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        # Both in-memory and disk should return same entry
        mem_result = unified_cache.get(sample_key)
        assert mem_result is not None

        # Force a reload by creating a new instance
        from isonome.core.unified_calibration_cache import UnifiedCalibrationCache
        cache2 = UnifiedCalibrationCache(root_dir=unified_cache.root)
        disk_result = cache2.get(sample_key)
        assert disk_result is not None
        assert disk_result.kernel_path == mem_result.kernel_path
        assert disk_result.certified == mem_result.certified


# ===========================================================================
# UnifiedCalibrationCache — Namespaces
# ===========================================================================


class TestUnifiedCacheNamespaces:
    def test_default_namespace(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        result = unified_cache.get(sample_key, namespace="public")
        assert result is not None

    def test_namespace_isolation(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        public_entry = UnifiedCacheEntry(kernel_path="/public.pt", metadata={"ns": "public"}, certified=True)
        private_entry = UnifiedCacheEntry(kernel_path="/private.pt", metadata={"ns": "private"}, certified=True)
        unified_cache.put(sample_key, public_entry, namespace="public")
        unified_cache.put(sample_key, private_entry, namespace="private")

        pub_result = unified_cache.get(sample_key, namespace="public")
        priv_result = unified_cache.get(sample_key, namespace="private")
        assert pub_result is not None
        assert priv_result is not None
        assert pub_result.kernel_path == "/public.pt"
        assert priv_result.kernel_path == "/private.pt"

    def test_list_keys_per_namespace(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        k1 = UnifiedCacheKey("h1", "reach", "v1")
        k2 = UnifiedCacheKey("h2", "grasp", "v1")
        unified_cache.put(k1, UnifiedCacheEntry(kernel_path="/a.pt"))
        unified_cache.put(k2, UnifiedCacheEntry(kernel_path="/b.pt"), namespace="private")

        public_keys = unified_cache.list_keys(namespace="public")
        private_keys = unified_cache.list_keys(namespace="private")
        assert len(public_keys) == 1
        assert len(private_keys) == 1

    def test_clear_namespace(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        k1 = UnifiedCacheKey("h1", "reach", "v1")
        k2 = UnifiedCacheKey("h2", "grasp", "v1")
        unified_cache.put(k1, UnifiedCacheEntry(kernel_path="/a.pt"))
        unified_cache.put(k2, UnifiedCacheEntry(kernel_path="/b.pt"), namespace="private")

        removed = unified_cache.clear(namespace="public")
        assert removed == 1
        assert not unified_cache.has(k1)
        assert unified_cache.has(k2, namespace="private")


# ===========================================================================
# UnifiedCalibrationCache — Near-Match Search
# ===========================================================================


class TestUnifiedCacheNearMatch:
    def test_near_match_finds_similar_topology(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        vec_a = torch.randn(32)
        vec_b = vec_a + 0.01 * torch.randn(32)  # very close
        vec_c = vec_a + 10.0 * torch.randn(32)  # far away

        k_a = UnifiedCacheKey("hash_a", "reach", "v1")
        k_b = UnifiedCacheKey("hash_b", "reach", "v1")
        k_c = UnifiedCacheKey("hash_c", "reach", "v1")

        unified_cache.put(k_a, UnifiedCacheEntry(kernel_path="/a.pt", certified=True), topology_vector=vec_a)
        unified_cache.put(k_b, UnifiedCacheEntry(kernel_path="/b.pt", certified=True), topology_vector=vec_b)
        unified_cache.put(k_c, UnifiedCacheEntry(kernel_path="/c.pt", certified=True), topology_vector=vec_c)

        query_key = UnifiedCacheKey("query", "reach", "v1")
        matches = unified_cache.find_near_matches(query_key, vec_a, epsilon=0.5)
        ids = {e.metadata.get("id", f"i{i}") for i, (_, e) in enumerate(matches)}
        paths = {e.kernel_path for _, e in matches}
        assert "/a.pt" in paths
        assert "/b.pt" in paths
        assert "/c.pt" not in paths

    def test_near_match_filters_task_and_vla(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        vec = torch.randn(32)

        unified_cache.put(
            UnifiedCacheKey("h1", "reach", "v1"),
            UnifiedCacheEntry(kernel_path="/reach_v1.pt", certified=True),
            topology_vector=vec,
        )
        unified_cache.put(
            UnifiedCacheKey("h2", "grasp", "v1"),
            UnifiedCacheEntry(kernel_path="/grasp_v1.pt", certified=True),
            topology_vector=vec,
        )
        unified_cache.put(
            UnifiedCacheKey("h3", "reach", "v2"),
            UnifiedCacheEntry(kernel_path="/reach_v2.pt", certified=True),
            topology_vector=vec,
        )

        matches = unified_cache.find_near_matches(
            UnifiedCacheKey("query", "reach", "v1"), vec, epsilon=0.1,
        )
        paths = {e.kernel_path for _, e in matches}
        assert paths == {"/reach_v1.pt"}

    def test_near_match_certified_only(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        vec = torch.randn(32)

        unified_cache.put(
            UnifiedCacheKey("h1", "reach", "v1"),
            UnifiedCacheEntry(kernel_path="/cert.pt", certified=True),
            topology_vector=vec,
        )
        unified_cache.put(
            UnifiedCacheKey("h2", "reach", "v1"),
            UnifiedCacheEntry(kernel_path="/uncert.pt", certified=False),
            topology_vector=vec + 0.01 * torch.randn(32),
        )

        # Without certified_only, both should be found
        matches_all = unified_cache.find_near_matches(
            UnifiedCacheKey("query", "reach", "v1"), vec, epsilon=1.0,
        )
        paths_all = {e.kernel_path for _, e in matches_all}
        assert "/cert.pt" in paths_all
        assert "/uncert.pt" in paths_all

        # With certified_only, only certified
        matches_cert = unified_cache.find_near_matches(
            UnifiedCacheKey("query", "reach", "v1"), vec, epsilon=1.0,
            certified_only=True,
        )
        paths_cert = {e.kernel_path for _, e in matches_cert}
        assert "/cert.pt" in paths_cert
        assert "/uncert.pt" not in paths_cert

    def test_near_match_without_stored_vector_skipped(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        vec = torch.randn(32)
        key = UnifiedCacheKey("h1", "reach", "v1")
        unified_cache.put(key, UnifiedCacheEntry(kernel_path="/no_vec.pt", certified=True))
        matches = unified_cache.find_near_matches(key, vec, epsilon=10.0)
        assert matches == []

    def test_near_matches_sorted_by_distance(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        vec_base = torch.randn(32)
        vec_a = vec_base + 0.01 * torch.randn(32)  # very close
        vec_b = vec_base + 0.3 * torch.randn(32)   # somewhat close

        unified_cache.put(
            UnifiedCacheKey("ha", "reach", "v1"),
            UnifiedCacheEntry(kernel_path="/a.pt", certified=True),
            topology_vector=vec_a,
        )
        unified_cache.put(
            UnifiedCacheKey("hb", "reach", "v1"),
            UnifiedCacheEntry(kernel_path="/b.pt", certified=True),
            topology_vector=vec_b,
        )

        matches = unified_cache.find_near_matches(
            UnifiedCacheKey("q", "reach", "v1"), vec_base, epsilon=1.0,
        )
        distances = [d for d, _ in matches]
        assert distances == sorted(distances)


# ===========================================================================
# UnifiedCalibrationCache — Convenience Lookups
# ===========================================================================


class TestUnifiedCacheConvenience:
    def test_get_by_topology_vector(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        tv = _make_topology_vector(seed=5)
        key = UnifiedCacheKey.from_topology_vector(tv, "reach", "v1")
        unified_cache.put(key, UnifiedCacheEntry(kernel_path="/k.pt", certified=True))
        result = unified_cache.get_by_topology(tv, "reach", "v1")
        assert result is not None
        assert result.kernel_path == "/k.pt"

    def test_get_by_robot_hash(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        robot_hash = "a1b2c3d4e5f6a7b8"
        result = unified_cache.get_by_robot_hash(robot_hash, "reach", "v1")
        assert result is None

        from isonome.core.unified_calibration_cache import UnifiedCacheKey
        key = UnifiedCacheKey(robot_hash, "reach", "v1")
        unified_cache.put(key, UnifiedCacheEntry(kernel_path="/k.pt", certified=True))
        result = unified_cache.get_by_robot_hash(robot_hash, "reach", "v1")
        assert result is not None

    def test_has_by_topology(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        tv = _make_topology_vector(seed=10)
        key = UnifiedCacheKey.from_topology_vector(tv, "pick", "v2")
        unified_cache.put(key, UnifiedCacheEntry(kernel_path="/k.pt"))
        assert unified_cache.has_by_topology(tv, "pick", "v2")


# ===========================================================================
# UnifiedCalibrationCache — Serialization (in-memory roundtrip)
# ===========================================================================


class TestUnifiedCacheSerialization:
    def test_to_dict(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        d = unified_cache.to_dict()
        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_from_dict_roundtrip(self, unified_cache, sample_key) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry
        unified_cache.put(sample_key, UnifiedCacheEntry(
            kernel_path="/k.pt", metadata={"episodes": 500}, certified=True,
        ))
        d = unified_cache.to_dict()
        # from_dict creates an in-memory-only cache from serialized data
        from isonome.core.unified_calibration_cache import UnifiedCalibrationCache
        restored = UnifiedCalibrationCache.from_dict(d)
        assert len(restored) == 1
        result = restored.get(sample_key)
        assert result is not None
        assert result.kernel_path == "/k.pt"

    def test_empty_cache_serialization(self) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCalibrationCache
        cache = UnifiedCalibrationCache()
        d = cache.to_dict()
        restored = UnifiedCalibrationCache.from_dict(d)
        assert len(restored) == 0


# ===========================================================================
# UnifiedCalibrationCache — Cross-feature Integration
# ===========================================================================


class TestUnifiedCacheIntegration:
    def test_different_robots_independent(self, unified_cache) -> None:
        from isonome.core.unified_calibration_cache import UnifiedCacheEntry, UnifiedCacheKey
        tv_arm = _make_topology_vector(seed=1)
        tv_leg = _make_topology_vector(seed=2)
        key_arm = UnifiedCacheKey.from_topology_vector(tv_arm, "reach", "v1")
        key_leg = UnifiedCacheKey.from_topology_vector(tv_leg, "reach", "v1")
        unified_cache.put(key_arm, UnifiedCacheEntry(kernel_path="/arm.pt", metadata={"joints": 7}))
        unified_cache.put(key_leg, UnifiedCacheEntry(kernel_path="/leg.pt", metadata={"joints": 12}))
        assert unified_cache.get(key_arm).metadata["joints"] == 7  # type: ignore
        assert unified_cache.get(key_leg).metadata["joints"] == 12  # type: ignore

    def test_certified_near_match_after_disk_reload(self, tmp_cache_dir) -> None:
        from isonome.core.unified_calibration_cache import (
            UnifiedCacheEntry,
            UnifiedCacheKey,
            UnifiedCalibrationCache,
        )
        vec = torch.randn(32)

        cache1 = UnifiedCalibrationCache(root_dir=tmp_cache_dir)
        k1 = UnifiedCacheKey("h1", "reach", "v1")
        k2 = UnifiedCacheKey("h2", "reach", "v1")
        cache1.put(k1, UnifiedCacheEntry(kernel_path="/cert.pt", certified=True), topology_vector=vec)
        cache1.put(k2, UnifiedCacheEntry(kernel_path="/uncert.pt", certified=False), topology_vector=vec)

        # Reload from disk
        cache2 = UnifiedCalibrationCache(root_dir=tmp_cache_dir)
        matches = cache2.find_near_matches(
            UnifiedCacheKey("q", "reach", "v1"), vec, epsilon=1.0, certified_only=True,
        )
        paths = {e.kernel_path for _, e in matches}
        assert "/cert.pt" in paths
        assert "/uncert.pt" not in paths

    def test_stats_survive_in_memory_roundtrip(self, unified_cache, sample_key, sample_entry) -> None:
        unified_cache.put(sample_key, sample_entry)
        unified_cache.get(sample_key)
        assert unified_cache.stats.hits == 1
        assert unified_cache.stats.puts == 1
