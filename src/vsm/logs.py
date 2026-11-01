"""structlog setup: console by default, JSON lines on the recorder host."""

import json
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog


class JsonlTee:
    """structlog processor that appends each event to a JSON-lines file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, logger: Any, method: str, event: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), default=str) + "\n")
        return event


def configure_logging(level: str = "INFO", *, as_json: bool, jsonl: Path | None = None) -> None:
    """Configure structlog for this process; ``jsonl`` also tees events to a file."""
    processors: list[Any] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if jsonl is not None:
        processors.append(JsonlTee(jsonl))
    renderer = (
        structlog.processors.JSONRenderer()
        if as_json
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
