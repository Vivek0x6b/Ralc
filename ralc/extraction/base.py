"""The extraction interface: a Message in, structured Signals out.

This is the contract a Gemma-backed extractor will implement in Phase 6. The
heuristic extractor in this phase implements the same interface, so the linker
is agnostic to how signals are produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Message:
    """A single conversation message fed to the linker.

    ``timestamp`` is optional epoch seconds; when omitted the linker stamps the
    message with the wall clock at ingest time.
    """

    role: str
    content: str
    timestamp: float | None = None


@dataclass
class Signals:
    """Structured signals extracted from one message."""

    entities: list[str] = field(default_factory=list)
    is_decision: bool = False


@runtime_checkable
class Extractor(Protocol):
    """Turns a message into structured signals."""

    def extract(self, message: Message) -> Signals:
        ...
