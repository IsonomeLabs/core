"""Immutable, hashable dictionary wrapper.

Used across pillars (mneme metadata, praxis action params) as a
read-only dict that can participate in sets and as dict keys.
"""
from __future__ import annotations

from typing import Any, Iterator


class frozendict:
    """Immutable dictionary - usable in frozenset and as dict keys.

    Supports all dict read operations (get, items, keys, values,
    __contains__, __len__, __iter__, __eq__) but blocks all mutating
    methods. Construct from any mapping or iterable of pairs.
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, mapping_or_iterable: Any = (), **kwargs: Any) -> None:
        if kwargs:
            if mapping_or_iterable:
                mapping_or_iterable = dict(mapping_or_iterable)
                mapping_or_iterable.update(kwargs)
            else:
                mapping_or_iterable = kwargs

        if isinstance(mapping_or_iterable, dict):
            self._data: dict[str, Any] = dict(mapping_or_iterable)
        elif hasattr(mapping_or_iterable, "items"):
            self._data = dict(mapping_or_iterable.items())
        else:
            self._data = dict(mapping_or_iterable)

        self._hash: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self):  # noqa: ANN201
        return self._data.keys()

    def values(self):  # noqa: ANN201
        return self._data.values()

    def items(self):  # noqa: ANN201
        return self._data.items()

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, frozendict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("frozendict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self._data.items())
        return f"frozendict({{{items}}})"

    def __bool__(self) -> bool:
        return bool(self._data)

    def copy(self) -> dict[str, Any]:
        """Return a mutable shallow copy."""
        return dict(self._data)
