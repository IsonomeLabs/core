"""UnifiedCalibrationCache — single cache combining in-memory + on-disk.

Architecture gap #6 unification: the architecture specifies a Calibration
Cache keyed by ``SHA256(topology + task_type + vla_version)`` that stores
certified policy packages.  Previously two separate implementations existed:

- ``isonome.core.calibration_cache`` — in-memory with stats + certification
  filtering, but no on-disk persistence or near-match search.
- ``isonome.praxis.calibration_cache`` — on-disk with namespaces and
  near-match search, but no stats or certification filtering.

This module unifies both into a single ``UnifiedCalibrationCache`` that
provides:

1. **In-memory hot path** with hit/miss/put/eviction stats (from core).
2. **Certification filtering** on both exact-match get and near-match
   search (from core, extended to near-match).
3. **On-disk persistence** so entries survive process restarts (from praxis).
4. **Namespaces** (public/private) with directory-level isolation (from praxis).
5. **Near-match search** by L2 topology-vector distance (from praxis).
6. **TTL support** with lazy eviction on read (from core).
7. **Max-size FIFO eviction** for bounded caches (from core).
8. **Serializable** via ``to_dict()`` / ``from_dict()`` for in-memory
   snapshots (from core).

Design principles
-----------------
* Write-through: every ``put()`` writes to both in-memory store and disk.
* Read-through: ``get()`` checks in-memory first; on miss, consults disk.
* Namespace-aware: the same composite hash in different namespaces stores
  separate entries.  Internally we key by ``"<namespace>:<composite_hash>"``.
* Backward compatible: ``UnifiedCacheKey`` is API-compatible with both
  ``CalibrationCacheKey`` (core) and ``CacheKey`` (praxis).
* Zero breaking changes: the existing ``CalibrationCache`` (core) and
  ``CalibrationCache`` (praxis) remain untouched; this is a new module.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from isonome.utils.logging import get_layer_logger


# ---------------------------------------------------------------------------
# UnifiedCacheKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedCacheKey:
    """Composite cache key: topology_hash + task_type + vla_version.

    The composite hash is ``SHA256(f"{topology_hash}:{task_type}:{vla_version}")``,
    matching the architecture spec's ``SHA256(topology + task_type + vla_version)``.
    """

    topology_hash: str
    task_type: str
    vla_version: str

    def composite_hash(self) -> str:
        """SHA-256 hex digest of the composite key components."""
        key_str = f"{self.topology_hash}:{self.task_type}:{self.vla_version}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.composite_hash())

    @classmethod
    def from_topology_vector(
        cls,
        topology_vector: Any,
        task_type: str,
        vla_version: str,
    ) -> UnifiedCacheKey:
        """Build a key from a ``TopologyVector`` instance."""
        return cls(
            topology_hash=topology_vector.topology_hash,
            task_type=task_type,
            vla_version=vla_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_hash": self.topology_hash,
            "task_type": self.task_type,
            "vla_version": self.vla_version,
            "composite_hash": self.composite_hash(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnifiedCacheKey:
        return cls(
            topology_hash=data["topology_hash"],
            task_type=data["task_type"],
            vla_version=data["vla_version"],
        )


# ---------------------------------------------------------------------------
# UnifiedCacheEntry
# ---------------------------------------------------------------------------


@dataclass
class UnifiedCacheEntry:
    """A cached calibration result / certified policy package.

    Attributes
    ----------
    kernel_path:
        Path to the kernel/policy file on disk.
    metadata:
        Arbitrary metadata (training episodes, success rate, etc.).
    certified:
        Whether this entry passed composition validation.
    cached_at:
        Monotonic timestamp when the entry was cached.
    ttl:
        Time-to-live in seconds. ``None`` means the entry never expires.
    """

    kernel_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    certified: bool = False
    cached_at: float = field(default_factory=time.monotonic)
    ttl: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """True if the entry has exceeded its TTL."""
        if self.ttl is None:
            return False
        return (time.monotonic() - self.cached_at) > self.ttl

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_path": self.kernel_path,
            "metadata": self.metadata,
            "certified": self.certified,
            "cached_at": self.cached_at,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnifiedCacheEntry:
        return cls(
            kernel_path=d["kernel_path"],
            metadata=d.get("metadata", {}),
            certified=d.get("certified", False),
            cached_at=d.get("cached_at", time.monotonic()),
            ttl=d.get("ttl"),
        )


# ---------------------------------------------------------------------------
# CertifiedPolicyPackage
# ---------------------------------------------------------------------------


@dataclass
class CertifiedPolicyPackage:
    """A certified policy package as stored in the calibration cache.

    Mirrors the artifact list from PRD FR-4.9 / FR-5.3.
    """

    manifest: dict[str, Any] = field(default_factory=dict)
    agent_configs: dict[str, Any] = field(default_factory=dict)
    coordinator_config: dict[str, Any] = field(default_factory=dict)
    reflex_gains: dict[str, Any] = field(default_factory=dict)
    sim_metrics: dict[str, Any] = field(default_factory=dict)
    policy_package_path: Optional[str] = None
    certification_video_path: Optional[str] = None
    launcher_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CertifiedPolicyPackage:
        return cls(**data)


# ---------------------------------------------------------------------------
# UnifiedCacheStats
# ---------------------------------------------------------------------------


@dataclass
class UnifiedCacheStats:
    """Hit/miss/eviction/put counters for cache observability."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    puts: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.puts = 0


# ---------------------------------------------------------------------------
# UnifiedCalibrationCache
# ---------------------------------------------------------------------------


class UnifiedCalibrationCache:
    """Unified calibration cache combining in-memory hot path with on-disk persistence.

    Keyed by ``SHA256(topology_hash + task_type + vla_version)`` as specified
    in architecture Diagram 1.

    Internally, entries are stored per-namespace using the key
    ``"<namespace>:<composite_hash>"``  so the same topology+task+VLA
    combination can have different entries in different namespaces.

    Parameters
    ----------
    root_dir:
        Directory for on-disk persistence.  Entries are stored as
        ``<root_dir>/<namespace>/<composite_hash>/entry.json``.
        If ``None``, the cache operates in memory only (no persistence).
    max_size:
        Maximum number of in-memory entries (across all namespaces).
        When exceeded, the oldest entry is evicted (FIFO).  Use ``0``
        for unlimited.
    default_namespace:
        Default namespace for put/get operations.
    """

    DEFAULT_NAMESPACE = "public"
    PRIVATE_NAMESPACE = "private"
    ENTRY_FILE = "entry.json"

    def __init__(
        self,
        root_dir: str | Path | None = None,
        max_size: int = 0,
        default_namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        # Internal keys: "<namespace>:<composite_hash>"
        self._store: Dict[str, UnifiedCacheEntry] = {}
        self._keys: Dict[str, UnifiedCacheKey] = {}
        self._namespaces: Dict[str, str] = {}  # internal_key → namespace
        self._insertion_order: List[str] = []
        self._topology_vectors: Dict[str, Optional[torch.Tensor]] = {}
        self._max_size = max_size
        self._stats = UnifiedCacheStats()
        self.default_namespace = default_namespace
        self._logger = get_layer_logger("unified_calibration_cache")

        if root_dir is not None:
            self.root = Path(root_dir).expanduser()
            self._persist = True
            self._ensure_namespace(default_namespace)
        else:
            self.root = Path("/tmp/_isonome_no_persist")
            self._persist = False

    # --- Internal key helpers ---

    @staticmethod
    def _ik(namespace: str, ch: str) -> str:
        """Build an internal key from namespace and composite hash."""
        return f"{namespace}:{ch}"

    # --- Public API ---

    def put(
        self,
        key: UnifiedCacheKey,
        entry: UnifiedCacheEntry | CertifiedPolicyPackage,
        *,
        namespace: Optional[str] = None,
        topology_vector: Optional[torch.Tensor] = None,
    ) -> None:
        """Store an entry under the given composite key.

        Accepts either a ``UnifiedCacheEntry`` or a ``CertifiedPolicyPackage``.
        When a ``CertifiedPolicyPackage`` is passed, it is converted to a
        ``UnifiedCacheEntry`` whose ``kernel_path`` is ``pkg.policy_package_path``
        and whose metadata carries the full package dict.

        If ``root_dir`` was provided at construction, the entry is also written
        to disk under the specified namespace.
        """
        namespace = namespace or self.default_namespace

        # Accept CertifiedPolicyPackage by wrapping it
        if isinstance(entry, CertifiedPolicyPackage):
            entry = UnifiedCacheEntry(
                kernel_path=entry.policy_package_path or "",
                metadata=entry.to_dict(),
                certified=bool(entry.sim_metrics.get("success_rate", 0) >= 0.99)
                if isinstance(entry.sim_metrics, dict) else False,
            )

        ch = key.composite_hash()
        ik = self._ik(namespace, ch)

        # If key already exists, remove from insertion order for re-insert
        if ik in self._store:
            self._insertion_order.remove(ik)

        # Enforce max_size before inserting
        if (
            self._max_size > 0
            and len(self._store) >= self._max_size
            and ik not in self._store
        ):
            self._evict_oldest()

        self._store[ik] = entry
        self._keys[ik] = key
        self._namespaces[ik] = namespace
        self._topology_vectors[ik] = topology_vector
        self._insertion_order.append(ik)
        self._stats.puts += 1

        # Persist to disk
        if self._persist:
            self._ensure_namespace(namespace)
            entry_dir = self._entry_dir(key, namespace)
            entry_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "key": key.to_dict(),
                "namespace": namespace,
                "topology_vector": (
                    self._tensor_to_list(topology_vector)
                    if topology_vector is not None
                    else None
                ),
            }

            payload = {
                "meta": meta,
                "entry": entry.to_dict(),
            }
            (entry_dir / self.ENTRY_FILE).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._logger.info(
                "cache_put",
                extra={
                    "namespace": namespace,
                    "key": ch[:16],
                    "task_type": key.task_type,
                    "vla_version": key.vla_version,
                },
            )

    def get(
        self,
        key: UnifiedCacheKey,
        *,
        certified_only: bool = False,
        namespace: Optional[str] = None,
    ) -> UnifiedCacheEntry | None:
        """Retrieve an entry by key.

        Checks in-memory first.  On miss, attempts to load from disk
        (if persistence is enabled).
        """
        namespace = namespace or self.default_namespace
        ch = key.composite_hash()
        ik = self._ik(namespace, ch)

        # In-memory lookup
        entry = self._store.get(ik)

        if entry is None and self._persist:
            entry = self._load_from_disk(key, namespace)
            if entry is not None:
                # Populate in-memory cache
                self._store[ik] = entry
                self._keys[ik] = key
                self._namespaces[ik] = namespace
                self._insertion_order.append(ik)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            self._remove_entry(ik)
            self._stats.misses += 1
            return None

        if certified_only and not entry.certified:
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return entry

    def has(
        self,
        key: UnifiedCacheKey,
        *,
        namespace: Optional[str] = None,
    ) -> bool:
        """Check if a non-expired entry exists for the key."""
        namespace = namespace or self.default_namespace
        ch = key.composite_hash()
        ik = self._ik(namespace, ch)

        entry = self._store.get(ik)

        if entry is None and self._persist:
            entry = self._load_from_disk(key, namespace)

        if entry is None:
            return False
        if entry.is_expired:
            self._remove_entry(ik)
            return False
        return True

    def remove(self, key: UnifiedCacheKey) -> UnifiedCacheEntry | None:
        """Remove and return an entry from the default namespace, or ``None`` if absent.

        Also removes the on-disk entry if persistence is enabled, so that
        subsequent ``has()`` / ``get()`` calls no longer find it.
        """
        namespace = self.default_namespace
        ch = key.composite_hash()
        ik = self._ik(namespace, ch)

        entry = self._store.get(ik)
        if entry is not None:
            self._remove_entry(ik)
        # Also remove from disk so has()/get() will not reload it
        if self._persist:
            entry_dir = self._entry_dir(key, namespace)
            entry_file = entry_dir / self.ENTRY_FILE
            if entry_file.exists():
                entry_file.unlink()
                try:
                    entry_dir.rmdir()
                except OSError:
                    pass
        return entry

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns the count removed."""
        expired_keys = [
            ik for ik, entry in self._store.items() if entry.is_expired
        ]
        for ik in expired_keys:
            self._remove_entry(ik)
        self._stats.evictions += len(expired_keys)
        return len(expired_keys)

    def clear(self, namespace: Optional[str] = None) -> int:
        """Remove all entries for the given namespace (in-memory + on-disk).

        Returns number of in-memory entries removed.
        """
        namespace = namespace or self.default_namespace
        removed = 0

        # Clear in-memory entries belonging to this namespace
        iks_to_remove = [
            ik for ik, ns in self._namespaces.items() if ns == namespace
        ]
        for ik in iks_to_remove:
            self._remove_entry(ik)
            removed += 1

        # Clear on-disk for the specific namespace
        if self._persist:
            self._clear_namespace_dir(namespace)

        return removed

    def __len__(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> UnifiedCacheStats:
        return self._stats

    # --- Namespace operations ---

    def list_keys(self, namespace: Optional[str] = None) -> list[UnifiedCacheKey]:
        """Return all cache keys stored in a namespace (reads from disk if persisting)."""
        if not self._persist:
            namespace = namespace or self.default_namespace
            return [
                self._keys[ik]
                for ik in self._insertion_order
                if self._namespaces.get(ik) == namespace
            ]

        namespace = namespace or self.default_namespace
        keys: list[UnifiedCacheKey] = []
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return keys

        for entry_dir in ns_dir.iterdir():
            entry_path = entry_dir / self.ENTRY_FILE
            if not entry_path.exists():
                continue
            try:
                payload = json.loads(entry_path.read_text(encoding="utf-8"))
                key_data = payload["meta"]["key"]
                keys.append(UnifiedCacheKey.from_dict(key_data))
            except (json.JSONDecodeError, KeyError):
                self._logger.warning(
                    "cache_corrupt_entry", extra={"entry": str(entry_dir)}
                )
        return keys

    def find_near_matches(
        self,
        key: UnifiedCacheKey,
        topology_vector: torch.Tensor,
        *,
        epsilon: float = 0.1,
        namespace: Optional[str] = None,
        certified_only: bool = False,
    ) -> list[tuple[float, UnifiedCacheEntry]]:
        """Near-match search by topology vector L2 distance.

        Returns a list of ``(distance, entry)`` tuples sorted by distance,
        including only entries where ``distance <= epsilon``.

        Parameters
        ----------
        key:
            Supplies ``task_type`` and ``vla_version`` filters.
        topology_vector:
            32-D topology feature vector.
        epsilon:
            Maximum L2 distance for a match.
        namespace:
            Cache namespace to search.
        certified_only:
            If True, only return certified entries.
        """
        namespace = namespace or self.default_namespace
        query = torch.as_tensor(topology_vector, dtype=torch.float32).flatten()
        matches: list[tuple[float, UnifiedCacheEntry]] = []

        # Search in-memory entries for this namespace
        for ik in list(self._store.keys()):
            if self._namespaces.get(ik) != namespace:
                continue
            stored_key = self._keys.get(ik)
            if stored_key is None:
                continue
            if stored_key.task_type != key.task_type or stored_key.vla_version != key.vla_version:
                continue
            stored_vec = self._topology_vectors.get(ik)
            if stored_vec is None:
                continue
            entry = self._store.get(ik)
            if entry is None or entry.is_expired:
                continue
            if certified_only and not entry.certified:
                continue

            stored_vec_flat = torch.as_tensor(stored_vec, dtype=torch.float32).flatten()
            if stored_vec_flat.shape != query.shape:
                continue
            distance = float(torch.norm(query - stored_vec_flat))
            if distance <= epsilon:
                matches.append((distance, entry))

        # Also search on-disk entries not in memory
        if self._persist:
            ns_dir = self.root / namespace
            if ns_dir.exists():
                for entry_dir in ns_dir.iterdir():
                    entry_path = entry_dir / self.ENTRY_FILE
                    if not entry_path.exists():
                        continue
                    try:
                        payload = json.loads(
                            entry_path.read_text(encoding="utf-8")
                        )
                        meta_key = payload["meta"]["key"]
                        if (
                            meta_key["task_type"] != key.task_type
                            or meta_key["vla_version"] != key.vla_version
                        ):
                            continue

                        # Check if already in memory
                        ch = meta_key.get("composite_hash")
                        ik_disk = self._ik(namespace, ch) if ch else None
                        if ik_disk and ik_disk in self._store:
                            continue  # Already searched in-memory

                        tv_list = payload["meta"].get("topology_vector")
                        if tv_list is None:
                            continue
                        entry_vec = torch.tensor(tv_list, dtype=torch.float32)
                        if entry_vec.shape != query.shape:
                            continue
                        distance = float(torch.norm(query - entry_vec))
                        if distance <= epsilon:
                            entry = UnifiedCacheEntry.from_dict(payload["entry"])
                            if certified_only and not entry.certified:
                                continue
                            matches.append((distance, entry))
                    except (json.JSONDecodeError, KeyError):
                        self._logger.warning(
                            "cache_corrupt_entry",
                            extra={"entry": str(entry_dir)},
                        )

        # Sort by distance
        matches.sort(key=lambda x: x[0])
        return matches

    # --- Convenience lookups ---

    def get_by_topology(
        self,
        topology_vector: Any,
        task_type: str,
        vla_version: str,
        certified_only: bool = False,
        namespace: Optional[str] = None,
    ) -> UnifiedCacheEntry | None:
        """Look up by a ``TopologyVector`` directly."""
        key = UnifiedCacheKey.from_topology_vector(
            topology_vector, task_type, vla_version
        )
        return self.get(key, certified_only=certified_only, namespace=namespace)

    def has_by_topology(
        self,
        topology_vector: Any,
        task_type: str,
        vla_version: str,
    ) -> bool:
        """Check existence by a ``TopologyVector`` directly."""
        key = UnifiedCacheKey.from_topology_vector(
            topology_vector, task_type, vla_version
        )
        return self.has(key)

    def get_by_robot_hash(
        self,
        robot_hash: str,
        task_type: str,
        vla_version: str,
        certified_only: bool = False,
        namespace: Optional[str] = None,
    ) -> UnifiedCacheEntry | None:
        """Look up by a raw robot hash string (backward compatible)."""
        key = UnifiedCacheKey(
            topology_hash=robot_hash,
            task_type=task_type,
            vla_version=vla_version,
        )
        return self.get(key, certified_only=certified_only, namespace=namespace)

    # --- Serialization (in-memory snapshot) ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize in-memory entries to a JSON-compatible dict."""
        entries = []
        for ik in self._insertion_order:
            key = self._keys[ik]
            entry = self._store[ik]
            ns = self._namespaces.get(ik, self.default_namespace)
            entries.append({
                "namespace": ns,
                "key": {
                    "topology_hash": key.topology_hash,
                    "task_type": key.task_type,
                    "vla_version": key.vla_version,
                },
                "entry": entry.to_dict(),
            })
        return {"entries": entries}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnifiedCalibrationCache:
        """Restore a cache from a serialized dict (in-memory only)."""
        cache = cls(root_dir=None)
        for item in d.get("entries", []):
            key = UnifiedCacheKey(
                topology_hash=item["key"]["topology_hash"],
                task_type=item["key"]["task_type"],
                vla_version=item["key"]["vla_version"],
            )
            entry = UnifiedCacheEntry.from_dict(item["entry"])
            namespace = item.get("namespace", cls.DEFAULT_NAMESPACE)
            cache.put(key, entry, namespace=namespace)
        return cache

    # --- Internal ---

    def _remove_entry(self, ik: str) -> None:
        """Remove an entry from in-memory store by internal key."""
        self._store.pop(ik, None)
        self._keys.pop(ik, None)
        self._namespaces.pop(ik, None)
        self._topology_vectors.pop(ik, None)
        if ik in self._insertion_order:
            self._insertion_order.remove(ik)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (FIFO) to make room.

        Also removes the on-disk entry if persistence is enabled, so that
        subsequent ``has()`` / ``get()`` calls no longer find it.
        """
        if not self._insertion_order:
            return
        oldest_ik = self._insertion_order.pop(0)
        # Capture namespace and key before deleting, for on-disk cleanup
        namespace = self._namespaces.pop(oldest_ik, None)
        oldest_key = self._keys.pop(oldest_ik, None)
        del self._store[oldest_ik]
        self._topology_vectors.pop(oldest_ik, None)
        self._stats.evictions += 1
        # Also remove from disk so has()/get() will not reload it
        if self._persist and namespace and oldest_key:
            entry_dir = self._entry_dir(oldest_key, namespace)
            entry_file = entry_dir / self.ENTRY_FILE
            if entry_file.exists():
                entry_file.unlink()
                try:
                    entry_dir.rmdir()
                except OSError:
                    pass

    def _ensure_namespace(self, namespace: str) -> None:
        if self._persist:
            (self.root / namespace).mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, key: UnifiedCacheKey, namespace: str) -> Path:
        return self.root / namespace / key.composite_hash()

    def _load_from_disk(
        self, key: UnifiedCacheKey, namespace: str
    ) -> UnifiedCacheEntry | None:
        """Load an entry from disk if persistence is enabled."""
        if not self._persist:
            return None
        entry_path = self._entry_dir(key, namespace) / self.ENTRY_FILE
        if not entry_path.exists():
            return None
        try:
            payload = json.loads(entry_path.read_text(encoding="utf-8"))
            entry = UnifiedCacheEntry.from_dict(payload["entry"])
            # Also recover topology vector if available
            ik = self._ik(namespace, key.composite_hash())
            tv_list = payload.get("meta", {}).get("topology_vector")
            if tv_list is not None:
                self._topology_vectors[ik] = torch.tensor(
                    tv_list, dtype=torch.float32
                )
            return entry
        except (json.JSONDecodeError, KeyError):
            self._logger.warning("cache_corrupt_entry", extra={"path": str(entry_path)})
            return None

    def _clear_namespace_dir(self, namespace: str) -> int:
        """Remove all on-disk entries in a namespace."""
        if not self._persist:
            return 0
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return 0
        removed = 0
        for entry_dir in ns_dir.iterdir():
            if entry_dir.is_dir():
                self._rm_tree(entry_dir)
                removed += 1
        return removed

    @staticmethod
    def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
        return torch.as_tensor(tensor).flatten().tolist()

    @staticmethod
    def _rm_tree(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                UnifiedCalibrationCache._rm_tree(child)
            else:
                child.unlink()
        path.rmdir()
