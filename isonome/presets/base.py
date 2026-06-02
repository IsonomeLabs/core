from __future__ import annotations
from abc import ABC, abstractmethod
from isonome.core.config import AppConfig


class Preset(ABC):
    """Base class for intelligence presets."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable preset identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable one-line description."""
        ...

    @abstractmethod
    def default_config(self) -> AppConfig:
        """Return the default AppConfig for this preset."""
        ...
