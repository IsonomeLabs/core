"""Tests for CalibrationCache — architecture gap #6.

Covers both implementations:
- ``isonome.core.calibration_cache`` — in-memory cache with stats and certification
- ``isonome.praxis.calibration_cache`` — on-disk cache with namespaces, near-match, and CLI
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from typer.testing import CliRunner

from isonome.cli import app
from isonome.core.calibration_cache import (
    CalibrationCache,
    CalibrationCacheEntry,
    CalibrationCacheKey,
    CalibrationCacheStats,
)
from isonome.praxis.calibration_cache import (
    CacheKey,
    CalibrationCache as PolicyPackageCache,
    CertifiedPolicyPackage,
)
from isonome.utils.morphology import TopologyVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_topology_vector(seed: int = 0) -> TopologyVector:
    """Create a deterministic 32-D TopologyVector for testing."""
    torch.manual_seed(seed)
    return TopologyVector(features=torch.randn(32))


# ===========================================================================
# Core: CalibrationCacheKey
# ===========================================================================


class TestCalibrationCacheKey:
    """Tests for the composite cache key model."""

    def test_key_attributes(self) -> None:
        key = CalibrationCacheKey("abc", "reach", "v1")
        assert key.topology_hash == "abc"
        assert key.task_type == "reach"
        assert key.vla_version == "v1"

    def test_key_deterministic_hash(self) -> None:
        k1 = CalibrationCacheKey("aaa", "pick", "v2")
        k2 = CalibrationCacheKey("aaa", "pick", "v2")
        assert k1.composite_hash() == k2.composite_hash()

    def test_key_different_inputs_different_hash(self) -> None:
        k1 = CalibrationCacheKey("aaa", "reach", "v1")
        k2 = CalibrationCacheKey("bbb", "reach", "v1")
        k3 = CalibrationCacheKey("aaa", "grasp", "v1")
        k4 = CalibrationCacheKey("aaa", "reach", "v2")
        assert k1.composite_hash() != k2.composite_hash()
        assert k1.composite_hash() != k3.composite_hash()
        assert k1.composite_hash() != k4.composite_hash()

    def test_key_composite_hash_format(self) -> None:
        h = CalibrationCacheKey("x", "y", "z").composite_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_key_composite_hash_uses_sha256(self) -> None:
        key = CalibrationCacheKey("aaa", "pick", "v2")
        expected = hashlib.sha256("aaa:pick:v2".encode()).hexdigest()
        assert key.composite_hash() == expected

    def test_key_equality(self) -> None:
        k1 = CalibrationCacheKey("x", "y", "z")
        k2 = CalibrationCacheKey("x", "y", "z")
        assert k1 == k2

    def test_key_inequality(self) -> None:
        k1 = CalibrationCacheKey("x", "y", "z")
        k2 = CalibrationCacheKey("a", "y", "z")
        assert k1 != k2

    def test_key_hashable(self) -> None:
        k1 = CalibrationCacheKey("x", "y", "z")
        k2 = CalibrationCacheKey("x", "y", "z")
        assert len({k1, k2}) == 1

    def test_key_from_topology_vector(self) -> None:
        tv = _make_topology_vector(seed=42)
        key = CalibrationCacheKey.from_topology_vector(tv, "reach", "openvla-v1")
        assert key.topology_hash == tv.topology_hash
        assert key.task_type == "reach"
        assert key.vla_version == "openvla-v1"


# ===========================================================================
# Core: CalibrationCacheEntry
# ===========================================================================


class TestCalibrationCacheEntry:
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
        time.sleep(0.01)
        assert entry.is_expired

    def test_entry_not_expired(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=3600.0)
        assert not entry.is_expired

    def test_entry_no_ttl_never_expires(self) -> None:
        entry = CalibrationCacheEntry(kernel_path="/k.pt", ttl=None)
        assert not entry.is_expired


# ===========================================================================
# Core: CalibrationCache — Core Operations
# ===========================================================================


class TestCalibrationCacheCore:
    def test_put_and_get(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        entry = CalibrationCacheEntry(kernel_path="/k.pt", certified=True)
        cache.put(key, entry)
        result = cache.get(key)
        assert result is not None
        assert result.kernel_path == "/k.pt"
        assert result.certified is True

    def test_get_missing_returns_none(self) -> None:
        cache = CalibrationCache()
        assert cache.get(CalibrationCacheKey("x", "y", "z")) is None

    def test_has_key(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        assert not cache.has(key)
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.has(key)

    def test_remove(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        removed = cache.remove(key)
        assert removed is not None
        assert not cache.has(key)

    def test_remove_missing_returns_none(self) -> None:
        cache = CalibrationCache()
        assert cache.remove(CalibrationCacheKey("x", "y", "z")) is None

    def test_put_overwrites(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/v1.pt"))
        cache.put(key, CalibrationCacheEntry(kernel_path="/v2.pt", certified=True))
        result = cache.get(key)
        assert result is not None
        assert result.kernel_path == "/v2.pt"
        assert result.certified is True

    def test_len(self) -> None:
        cache = CalibrationCache()
        assert len(cache) == 0
        cache.put(CalibrationCacheKey("a", "reach", "v1"), CalibrationCacheEntry(kernel_path="/a.pt"))
        assert len(cache) == 1

    def test_clear(self) -> None:
        cache = CalibrationCache()
        cache.put(CalibrationCacheKey("a", "reach", "v1"), CalibrationCacheEntry(kernel_path="/a.pt"))
        cache.clear()
        assert len(cache) == 0


# ===========================================================================
# Core: CalibrationCache — TTL & Eviction
# ===========================================================================


class TestCalibrationCacheEviction:
    def test_expired_entry_not_returned(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", ttl=0.0))
        time.sleep(0.01)
        assert cache.get(key) is None

    def test_expired_entry_removed_on_get(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", ttl=0.0))
        time.sleep(0.01)
        cache.get(key)
        assert len(cache) == 0

    def test_max_size_evicts_oldest(self) -> None:
        cache = CalibrationCache(max_size=2)
        k1 = CalibrationCacheKey("a", "reach", "v1")
        k2 = CalibrationCacheKey("b", "reach", "v1")
        k3 = CalibrationCacheKey("c", "reach", "v1")
        cache.put(k1, CalibrationCacheEntry(kernel_path="/1.pt"))
        cache.put(k2, CalibrationCacheEntry(kernel_path="/2.pt"))
        cache.put(k3, CalibrationCacheEntry(kernel_path="/3.pt"))
        assert not cache.has(k1)
        assert cache.has(k2)
        assert cache.has(k3)

    def test_evict_expired_removes_only_expired(self) -> None:
        cache = CalibrationCache()
        k1 = CalibrationCacheKey("exp", "reach", "v1")
        k2 = CalibrationCacheKey("fresh", "reach", "v1")
        cache.put(k1, CalibrationCacheEntry(kernel_path="/e.pt", ttl=0.0))
        cache.put(k2, CalibrationCacheEntry(kernel_path="/f.pt", ttl=3600.0))
        time.sleep(0.01)
        removed = cache.evict_expired()
        assert removed == 1
        assert not cache.has(k1)
        assert cache.has(k2)

    def test_no_ttl_entries_not_evicted(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", ttl=None))
        assert cache.evict_expired() == 0
        assert cache.has(key)


# ===========================================================================
# Core: CalibrationCache — Statistics
# ===========================================================================


class TestCalibrationCacheStats:
    def test_stats_initial(self) -> None:
        cache = CalibrationCache()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    def test_stats_tracks_puts(self) -> None:
        cache = CalibrationCache()
        cache.put(CalibrationCacheKey("abc", "reach", "v1"), CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.stats.puts == 1

    def test_stats_tracks_hits(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        cache.get(key)
        assert cache.stats.hits == 1

    def test_stats_tracks_misses(self) -> None:
        cache = CalibrationCache()
        cache.get(CalibrationCacheKey("x", "y", "z"))
        assert cache.stats.misses == 1

    def test_stats_hit_rate(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        cache.get(key)
        cache.get(CalibrationCacheKey("x", "y", "z"))
        assert cache.stats.hit_rate == 0.5


# ===========================================================================
# Core: CalibrationCache — Certification Filtering
# ===========================================================================


class TestCalibrationCacheCertification:
    def test_get_certified_only_true(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))
        assert cache.get(key, certified_only=True) is not None

    def test_get_certified_only_filters_uncertified(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=False))
        assert cache.get(key, certified_only=True) is None

    def test_get_by_topology_certified_only(self) -> None:
        cache = CalibrationCache()
        tv = _make_topology_vector(seed=5)
        key = CalibrationCacheKey.from_topology_vector(tv, "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))
        assert cache.get_by_topology(tv, "reach", "v1", certified_only=True) is not None


# ===========================================================================
# Core: CalibrationCache — Integration
# ===========================================================================


class TestCalibrationCacheIntegration:
    def test_kernel_path_uses_composite_hash(self) -> None:
        key = CalibrationCacheKey("abc", "reach", "v1")
        filename = f"{key.composite_hash()[:16]}.pt"
        assert filename.endswith(".pt")
        assert len(filename) == 19

    def test_entries_for_different_robots_independent(self) -> None:
        cache = CalibrationCache()
        tv_arm = _make_topology_vector(seed=1)
        tv_leg = _make_topology_vector(seed=2)
        key_arm = CalibrationCacheKey.from_topology_vector(tv_arm, "reach", "v1")
        key_leg = CalibrationCacheKey.from_topology_vector(tv_leg, "reach", "v1")
        cache.put(key_arm, CalibrationCacheEntry(kernel_path="/arm.pt", metadata={"joints": 7}))
        cache.put(key_leg, CalibrationCacheEntry(kernel_path="/leg.pt", metadata={"joints": 12}))
        assert cache.get(key_arm).metadata["joints"] == 7
        assert cache.get(key_leg).metadata["joints"] == 12

    def test_backward_compatible_with_robot_hash(self) -> None:
        old_hash = "a1b2c3d4e5f6a7b8"
        key = CalibrationCacheKey(old_hash, "reach", "v1")
        cache = CalibrationCache()
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt"))
        assert cache.has(key)

    def test_lookup_by_robot_hash_shortcut(self) -> None:
        cache = CalibrationCache()
        robot_hash = "a1b2c3d4e5f6a7b8"
        key = CalibrationCacheKey(robot_hash, "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))
        assert cache.get_by_robot_hash(robot_hash, "reach", "v1").kernel_path == "/k.pt"


# ===========================================================================
# Core: CalibrationCache — Serialization
# ===========================================================================


class TestCalibrationCacheSerialization:
    def test_to_dict(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", certified=True))
        d = cache.to_dict()
        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_from_dict_roundtrip(self) -> None:
        cache = CalibrationCache()
        key = CalibrationCacheKey("abc", "reach", "v1")
        cache.put(key, CalibrationCacheEntry(kernel_path="/k.pt", metadata={"episodes": 500}, certified=True))
        restored = CalibrationCache.from_dict(cache.to_dict())
        assert len(restored) == 1
        result = restored.get(key)
        assert result.kernel_path == "/k.pt"
        assert result.certified is True

    def test_empty_cache_serialization(self) -> None:
        cache = CalibrationCache()
        restored = CalibrationCache.from_dict(cache.to_dict())
        assert len(restored) == 0


# ---------------------------------------------------------------------------
# Praxis: CacheKey
# ---------------------------------------------------------------------------


def test_praxis_cache_key_sha256_is_stable_and_unique() -> None:
    k1 = CacheKey("abc", "reach", "v1")
    k2 = CacheKey("abc", "reach", "v1")
    k3 = CacheKey("abc", "grasp", "v1")
    assert k1.sha256() == k2.sha256()
    assert k1.sha256() != k3.sha256()
    assert len(k1.sha256()) == 64


def test_praxis_cache_key_to_dict_roundtrip() -> None:
    key = CacheKey("topo", "task", "vla")
    d = key.to_dict()
    restored = CacheKey.from_dict(d)
    assert restored == key


# ---------------------------------------------------------------------------
# Praxis: Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_cache(tmp_path: Path) -> PolicyPackageCache:
    return PolicyPackageCache(root_dir=tmp_path / "cache")


@pytest.fixture
def sample_key() -> CacheKey:
    return CacheKey(
        topology_hash="a1b2c3d4e5f6",
        task_type="reach",
        vla_version="openvla-7b",
    )


@pytest.fixture
def sample_package() -> CertifiedPolicyPackage:
    return CertifiedPolicyPackage(
        manifest={"task": "reach red cube"},
        agent_configs={"arm": {"dof": 7}},
        coordinator_config={"strategy": "priority"},
        reflex_gains={"kp": 1.0},
        sim_metrics={"success_rate": 0.99},
        policy_package_path="/tmp/pkg.zip",
    )


# ---------------------------------------------------------------------------
# Praxis: Basic put / get / exists
# ---------------------------------------------------------------------------


def test_praxis_put_and_get(tmp_cache: PolicyPackageCache, sample_key: CacheKey, sample_package: CertifiedPolicyPackage) -> None:
    entry_dir = tmp_cache.put(sample_key, sample_package)
    assert entry_dir.exists()
    assert (entry_dir / PolicyPackageCache.META_FILE).exists()
    assert (entry_dir / PolicyPackageCache.PACKAGE_FILE).exists()

    retrieved = tmp_cache.get(sample_key)
    assert retrieved is not None
    assert retrieved.manifest == sample_package.manifest
    assert retrieved.agent_configs == sample_package.agent_configs
    assert retrieved.policy_package_path == sample_package.policy_package_path


def test_praxis_get_missing_returns_none(tmp_cache: PolicyPackageCache) -> None:
    assert tmp_cache.get(CacheKey("missing", "task", "v1")) is None


def test_praxis_exists(tmp_cache: PolicyPackageCache, sample_key: CacheKey, sample_package: CertifiedPolicyPackage) -> None:
    assert not tmp_cache.exists(sample_key)
    tmp_cache.put(sample_key, sample_package)
    assert tmp_cache.exists(sample_key)


# ---------------------------------------------------------------------------
# Praxis: Namespaces
# ---------------------------------------------------------------------------


def test_praxis_namespace_isolation(tmp_cache: PolicyPackageCache, sample_key: CacheKey, sample_package: CertifiedPolicyPackage) -> None:
    public_pkg = CertifiedPolicyPackage(manifest={"ns": "public"})
    private_pkg = CertifiedPolicyPackage(manifest={"ns": "private"})

    tmp_cache.put(sample_key, public_pkg, namespace="public")
    tmp_cache.put(sample_key, private_pkg, namespace="private")

    assert tmp_cache.get(sample_key, namespace="public") == public_pkg
    assert tmp_cache.get(sample_key, namespace="private") == private_pkg


def test_praxis_default_namespace(tmp_cache: PolicyPackageCache, sample_key: CacheKey, sample_package: CertifiedPolicyPackage) -> None:
    tmp_cache.put(sample_key, sample_package)
    assert tmp_cache.get(sample_key, namespace="public") is not None


# ---------------------------------------------------------------------------
# Praxis: Topology vector and near-match search
# ---------------------------------------------------------------------------


def test_praxis_near_match_finds_similar_topology(tmp_cache: PolicyPackageCache) -> None:
    vec_a = torch.randn(32)
    vec_b = vec_a + 0.01 * torch.randn(32)
    vec_c = vec_a + 10.0 * torch.randn(32)

    key_a = CacheKey("hash_a", "reach", "v1")
    key_b = CacheKey("hash_b", "reach", "v1")
    key_c = CacheKey("hash_c", "reach", "v1")

    tmp_cache.put(key_a, CertifiedPolicyPackage(manifest={"id": "a"}), topology_vector=vec_a)
    tmp_cache.put(key_b, CertifiedPolicyPackage(manifest={"id": "b"}), topology_vector=vec_b)
    tmp_cache.put(key_c, CertifiedPolicyPackage(manifest={"id": "c"}), topology_vector=vec_c)

    query_key = CacheKey("query", "reach", "v1")
    matches = tmp_cache.find_near_matches(query_key, vec_a, epsilon=0.5)
    distances = [d for d, _ in matches]
    ids = [pkg.manifest["id"] for _, pkg in matches]

    assert "a" in ids
    assert "b" in ids
    assert "c" not in ids
    assert distances == sorted(distances)


def test_praxis_near_match_filters_task_and_vla_version(tmp_cache: PolicyPackageCache) -> None:
    vec = torch.randn(32)

    tmp_cache.put(
        CacheKey("h1", "reach", "v1"),
        CertifiedPolicyPackage(manifest={"id": "reach_v1"}),
        topology_vector=vec,
    )
    tmp_cache.put(
        CacheKey("h2", "grasp", "v1"),
        CertifiedPolicyPackage(manifest={"id": "grasp_v1"}),
        topology_vector=vec,
    )
    tmp_cache.put(
        CacheKey("h3", "reach", "v2"),
        CertifiedPolicyPackage(manifest={"id": "reach_v2"}),
        topology_vector=vec,
    )

    matches = tmp_cache.find_near_matches(
        CacheKey("query", "reach", "v1"), vec, epsilon=0.1
    )
    ids = {pkg.manifest["id"] for _, pkg in matches}
    assert ids == {"reach_v1"}


def test_praxis_near_match_without_stored_vector_is_ignored(tmp_cache: PolicyPackageCache) -> None:
    vec = torch.randn(32)
    key = CacheKey("h1", "reach", "v1")
    tmp_cache.put(key, CertifiedPolicyPackage(manifest={"id": "no_vec"}))
    matches = tmp_cache.find_near_matches(key, vec, epsilon=10.0)
    assert matches == []


# ---------------------------------------------------------------------------
# Praxis: Listing and clearing
# ---------------------------------------------------------------------------


def test_praxis_list_keys(tmp_cache: PolicyPackageCache) -> None:
    k1 = CacheKey("h1", "reach", "v1")
    k2 = CacheKey("h2", "grasp", "v1")
    tmp_cache.put(k1, CertifiedPolicyPackage())
    tmp_cache.put(k2, CertifiedPolicyPackage(), namespace="private")

    public_keys = tmp_cache.list_keys(namespace="public")
    assert len(public_keys) == 1
    assert public_keys[0] == k1

    private_keys = tmp_cache.list_keys(namespace="private")
    assert len(private_keys) == 1
    assert private_keys[0] == k2


def test_praxis_clear(tmp_cache: PolicyPackageCache) -> None:
    k1 = CacheKey("h1", "reach", "v1")
    k2 = CacheKey("h2", "grasp", "v1")
    tmp_cache.put(k1, CertifiedPolicyPackage())
    tmp_cache.put(k2, CertifiedPolicyPackage(), namespace="private")

    assert tmp_cache.clear(namespace="public") == 1
    assert tmp_cache.list_keys(namespace="public") == []
    assert len(tmp_cache.list_keys(namespace="private")) == 1


# ---------------------------------------------------------------------------
# Praxis: Persistence
# ---------------------------------------------------------------------------


def test_praxis_persistence_across_instances(tmp_path: Path, sample_key: CacheKey, sample_package: CertifiedPolicyPackage) -> None:
    root = tmp_path / "cache"
    cache1 = PolicyPackageCache(root_dir=root)
    cache1.put(sample_key, sample_package)

    cache2 = PolicyPackageCache(root_dir=root)
    retrieved = cache2.get(sample_key)
    assert retrieved is not None
    assert retrieved.manifest == sample_package.manifest


# ---------------------------------------------------------------------------
# Praxis: CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_cache_put_and_lookup(cli_runner: CliRunner, tmp_path: Path, sample_package: CertifiedPolicyPackage) -> None:
    cache_root = tmp_path / "cache"
    pkg_file = tmp_path / "pkg.json"
    pkg_file.write_text(json.dumps(sample_package.to_dict()))

    result = cli_runner.invoke(
        app,
        [
            "cache", "put",
            "--cache-root", str(cache_root),
            "abc123", "reach", "openvla-7b",
            str(pkg_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Cached at" in result.output

    result = cli_runner.invoke(
        app,
        [
            "cache", "lookup",
            "--cache-root", str(cache_root),
            "abc123", "reach", "openvla-7b",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["exact_match"] is not None
    assert data["exact_match"]["manifest"]["task"] == "reach red cube"


def test_cli_cache_lookup_missing(cli_runner: CliRunner, tmp_path: Path) -> None:
    result = cli_runner.invoke(
        app,
        [
            "cache", "lookup",
            "--cache-root", str(tmp_path / "cache"),
            "missing", "reach", "v1",
        ],
    )
    assert result.exit_code == 1
    assert "No matching cache entry found" in result.output


def test_cli_cache_list(cli_runner: CliRunner, tmp_path: Path, sample_package: CertifiedPolicyPackage) -> None:
    cache_root = tmp_path / "cache"
    pkg_file = tmp_path / "pkg.json"
    pkg_file.write_text(json.dumps(sample_package.to_dict()))

    cli_runner.invoke(
        app,
        [
            "cache", "put",
            "--cache-root", str(cache_root),
            "h1", "reach", "v1",
            str(pkg_file),
        ],
    )

    result = cli_runner.invoke(
        app,
        ["cache", "list", "--cache-root", str(cache_root)],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["topology_hash"] == "h1"
