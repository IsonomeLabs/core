from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for all Isonome components."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge structured extra fields
        for key in (
            "layer",
            "agent_id",
            "event",
            "data",
            "tick",
            "error",
            "channel",
            "frequency_hz",
            "model_path",
            "provider",
            "model",
            "gui",
            "path",
            "joints",
            "latency_ms",
            "count",
            "patch_id",
            "type",
            "target",
            "confidence",
            "state",
            "signal",
            "agent",
            "summary",
            "member",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        # Pydantic extra dict if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            entry.update(record.extra)
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, default=str)


def get_layer_logger(name: str) -> logging.Logger:
    """Get or create a structured logger for a layer or subsystem."""
    logger = logging.getLogger(f"isonome.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure the root isonome logger with JSON output."""
    root = logging.getLogger("isonome")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        root.addHandler(handler)
