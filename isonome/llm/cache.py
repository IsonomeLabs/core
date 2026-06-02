from __future__ import annotations
import hashlib
import time
import logging
from typing import Any


class CacheEntry:
    """A single cached value with a TTL expiry."""

    def __init__(self, value: Any, ttl: float = 300.0) -> None:
        self.value = value
        self.created_at = time.monotonic()
        self.ttl = ttl

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl


class SemanticCache:
    """Hash-based dedup of Cortex advice. Simple dict-based with TTL eviction."""

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._logger = logging.getLogger("isonome.llm.cache")

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key, returning None if absent or expired."""
        h = self._hash(key)
        entry = self._store.get(h)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[h]
            return None
        return entry.value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with optional TTL override."""
        if len(self._store) >= self._max_size:
            self._evict()
        h = self._hash(key)
        self._store[h] = CacheEntry(value, ttl=ttl or self._default_ttl)

    def _evict(self) -> None:
        """Evict expired entries, or the oldest entry if none are expired."""
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]
        if not expired and self._store:
            oldest = min(self._store.items(), key=lambda x: x[1].created_at)
            del self._store[oldest[0]]

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()
