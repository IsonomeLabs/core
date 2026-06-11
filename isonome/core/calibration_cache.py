"""CalibrationCache — morphology-aware kernel/policy package cache.

Addresses architecture gap #6: the architecture specifies a Calibration Cache
keyed by ``SHA256(topology + task_type + vla_version)`` that stores certified
policy packages.  The existing ``SemanticCache`` in ``isonome/llm/cache.py`` is
a generic string TTL cache with no topology awareness.

Design principles
-----------------
1. **Topology-aware keys** — composes the TopologyVector.topology_hash with
   task_type and vla_version to produce a deterministic composite key, exactly
   matching the architecture spec: ``SHA256(topology + task_type + vla_version)``.
2. **Certification tracking** — entries carry a ``certified`` flag that indicates
   whether the policy package passed composition validation (1000 episodes,
   >99% success). Uncertified entries can still be cached for development.
3. **TTL-optional** — calibration entries typically persist indefinitely (no
   TTL), unlike the short-lived LLM advice cache. But TTL is supported for
   development entries or time-limited policy packages.
4. **Backward compatible** — the old ``SomaLayer._robot_hash()`` (16-char hex)
   can be used directly as the ``topology_hash`` field, enabling a smooth
   migration path from the old per-robot cache to the new per-topology+task+VLA
   cache.
5. **Statistics** — hit/miss/eviction/put counters for observability.
6. **Serializable** — ``to_dict()`` / ``from_dict()`` for persistence across
   sessions (e.g., saving to ``~/.isonome/calibration_cache.json``).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from isonome.utils.logging import get_layer_logger


# ---------------------------------------------------------------------------
# CalibrationCacheKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCacheKey:
    """Composite cache key: topology_hash + task_type + vla_version.

    The composite hash is ``SHA256(f"{topology_hash}:{task_type}:{vla_version}")``,
    matching the architecture spec's ``SHA256(topology + task_type + vla_version)``.
    """

    topology_hash: str
    task_type: str
    vla_version: str

    def composite_hash(self) -> str:
        """SHA-256 hex digest of the composite key components.

        Uses ``:`` as separator to prevent collision between components
        (e.g., hash "ab:reach:v1" vs. "abr:each:v1").
        """
        key_str = f"{self.topology_hash}:{self.task_type}:{self.vla_version}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.composite_hash())

    @classmethod
    def from_topology_vector(
        cls,
        topology_vector: Any,  # TopologyVector — late import to avoid circular dep
        task_type: str,
        vla_version: str,
    ) -> CalibrationCacheKey:
        """Build a key from a ``TopologyVector`` instance.

        Parameters
        ----------
        topology_vector:
            An ``isonome.utils.morphology.TopologyVector`` — we access only
            its ``topology_hash`` property.
        task_type:
            Task identifier (e.g., ``"reach"``, ``"pick"``, ``"push"``).
        vla_version:
            VLA model version string (e.g., ``"openvla-7b-v1"``).
        """
        return cls(
            topology_hash=topology_vector.topology_hash,
            task_type=task_type,
            vla_version=vla_version,
        )


# ---------------------------------------------------------------------------
# CalibrationCacheEntry
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCacheEntry:
    """A single cached calibration result / certified policy package.

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
        Time-to-live in seconds. ``None`` means the entry never expires
        (the default for calibration entries).
    """

    kernel_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    certified: bool = False
    cached_at: float = field(default_factory=time.monotonic)
    ttl: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        """True if the entry has exceeded its TTL.

        Entries with ``ttl=None`` never expire.
        """
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
    def from_dict(cls, d: dict[str, Any]) -> CalibrationCacheEntry:
        return cls(
            kernel_path=d["kernel_path"],
            metadata=d.get("metadata", {}),
            certified=d.get("certified", False),
            cached_at=d.get("cached_at", time.monotonic()),
            ttl=d.get("ttl"),
        )


# ---------------------------------------------------------------------------
# CalibrationCacheStats
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCacheStats:
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
# CalibrationCache
# ---------------------------------------------------------------------------


class CalibrationCache:
    """Morphology-aware cache for calibrated kernels and policy packages.

    Keyed by ``SHA256(topology_hash + task_type + vla_version)`` as specified
    in architecture Diagram 1.

    Parameters
    ----------
    max_size:
        Maximum number of entries. When exceeded, the oldest entry is
        evicted (FIFO). Use ``0`` for unlimited.
    """

    def __init__(self, max_size: int = 0) -> None:
        self._store: Dict[str, CalibrationCacheEntry] = {}
        self._keys: Dict[str, CalibrationCacheKey] = {}  # composite_hash → key
        self._insertion_order: List[str] = []  # for FIFO eviction
        self._max_size = max_size
        self._stats = CalibrationCacheStats()
        self._logger = get_layer_logger("calibration_cache")

    # -- Public API --

    def put(self, key: CalibrationCacheKey, entry: CalibrationCacheEntry) -> None:
        """Store an entry under the given composite key.

        If the key already exists, the entry is overwritten (re-inserted
        at the end of the insertion order).
        """
        ch = key.composite_hash()

        # If key already exists, remove from insertion order for re-insert
        if ch in self._store:
            self._insertion_order.remove(ch)

        # Enforce max_size before inserting
        if self._max_size > 0 and len(self._store) >= self._max_size and ch not in self._store:
            self._evict_oldest()

        self._store[ch] = entry
        self._keys[ch] = key
        self._insertion_order.append(ch)
        self._stats.puts += 1

    def get(
        self,
        key: CalibrationCacheKey,
        certified_only: bool = False,
    ) -> CalibrationCacheEntry | None:
        """Retrieve an entry by key.

        Parameters
        ----------
        key:
            The composite cache key.
        certified_only:
            If True, only return entries where ``certified=True``.
            Uncertified entries are treated as misses.

        Returns
        -------
        The cached entry, or ``None`` if absent, expired, or filtered.
        """
        ch = key.composite_hash()
        entry = self._store.get(ch)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            del self._store[ch]
            del self._keys[ch]
            self._insertion_order.remove(ch)
            self._stats.misses += 1
            return None

        if certified_only and not entry.certified:
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return entry

    def has(self, key: CalibrationCacheKey) -> bool:
        """Check if a non-expired entry exists for the key."""
        ch = key.composite_hash()
        entry = self._store.get(ch)
        if entry is None:
            return False
        if entry.is_expired:
            del self._store[ch]
            del self._keys[ch]
            self._insertion_order.remove(ch)
            return False
        return True

    def remove(self, key: CalibrationCacheKey) -> CalibrationCacheEntry | None:
        """Remove and return an entry, or ``None`` if absent."""
        ch = key.composite_hash()
        entry = self._store.pop(ch, None)
        if entry is not None:
            del self._keys[ch]
            self._insertion_order.remove(ch)
        return entry

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns the count removed."""
        expired_keys = [
            ch for ch, entry in self._store.items() if entry.is_expired
        ]
        for ch in expired_keys:
            del self._store[ch]
            del self._keys[ch]
            self._insertion_order.remove(ch)
        self._stats.evictions += len(expired_keys)
        return len(expired_keys)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()
        self._keys.clear()
        self._insertion_order.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def stats(self) -> CalibrationCacheStats:
        return self._stats

    # -- Convenience lookups --

    def get_by_topology(
        self,
        topology_vector: Any,
        task_type: str,
        vla_version: str,
        certified_only: bool = False,
    ) -> CalibrationCacheEntry | None:
        """Look up by a ``TopologyVector`` directly.

        Parameters
        ----------
        topology_vector:
            An ``isonome.utils.morphology.TopologyVector``.
        task_type:
            Task identifier.
        vla_version:
            VLA model version string.
        certified_only:
            If True, only return certified entries.
        """
        key = CalibrationCacheKey.from_topology_vector(
            topology_vector, task_type, vla_version
        )
        return self.get(key, certified_only=certified_only)

    def has_by_topology(
        self,
        topology_vector: Any,
        task_type: str,
        vla_version: str,
    ) -> bool:
        """Check existence by a ``TopologyVector`` directly."""
        key = CalibrationCacheKey.from_topology_vector(
            topology_vector, task_type, vla_version
        )
        return self.has(key)

    def get_by_robot_hash(
        self,
        robot_hash: str,
        task_type: str,
        vla_version: str,
        certified_only: bool = False,
    ) -> CalibrationCacheEntry | None:
        """Look up by a raw robot hash string (backward compatible).

        This is the 16-char hex string from ``SomaLayer._robot_hash()``.
        """
        key = CalibrationCacheKey(
            topology_hash=robot_hash,
            task_type=task_type,
            vla_version=vla_version,
        )
        return self.get(key, certified_only=certified_only)

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize the cache to a JSON-compatible dict.

        The keys are stored as composite hashes with their components so
        they can be reconstructed on deserialization.
        """
        entries = []
        for ch in self._insertion_order:
            key = self._keys[ch]
            entry = self._store[ch]
            entries.append({
                "key": {
                    "topology_hash": key.topology_hash,
                    "task_type": key.task_type,
                    "vla_version": key.vla_version,
                },
                "entry": entry.to_dict(),
            })
        return {"entries": entries}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationCache:
        """Restore a cache from a serialized dict."""
        cache = cls()
        for item in d.get("entries", []):
            key = CalibrationCacheKey(
                topology_hash=item["key"]["topology_hash"],
                task_type=item["key"]["task_type"],
                vla_version=item["key"]["vla_version"],
            )
            entry = CalibrationCacheEntry.from_dict(item["entry"])
            cache.put(key, entry)
        return cache

    # -- Internal --

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (FIFO) to make room."""
        if not self._insertion_order:
            return
        oldest_ch = self._insertion_order.pop(0)
        del self._store[oldest_ch]
        del self._keys[oldest_ch]
        self._stats.evictions += 1
