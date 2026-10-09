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
    """Structured signals extracted from one message.

    ``topic`` is an optional short phrase naming what a decision is about; it is
    only set by extractors that produce it (Gemma) and only for decisions. The
    heuristic extractor leaves it None, so the linker stays agnostic.
    """

    entities: list[str] = field(default_factory=list)
    is_decision: bool = False
    topic: str | None = None


@runtime_checkable
class Extractor(Protocol):
    """Turns a message into structured signals."""

    def extract(self, message: Message) -> Signals:
        ...


def normalize_entities(raw: list[str]) -> list[str]:
    """Normalize an ordered list of entity strings.

    Strips surrounding whitespace, drops empties, and deduplicates
    case-insensitively while keeping the first-seen spelling and order. Shared
    by every Extractor so the linker deduplicates identically regardless of how
    the entities were produced.
    """
    seen: dict[str, str] = {}
    ordered: list[str] = []
    for text in raw:
        norm = text.strip()
        if not norm:
            continue
        key = norm.lower()
        if key not in seen:
            seen[key] = norm
            ordered.append(norm)
    return ordered
