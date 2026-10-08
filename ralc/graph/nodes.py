"""The Node: a single piece of context in the relational graph."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Node:
    """A context node.

    Fields follow section 9 of the project instructions. ``timestamp`` is epoch
    seconds. ``embedding_ref`` is an optional reference to an embedding stored
    elsewhere (the embedding itself is out of scope for Phase 1).
    """

    id: str
    content: str
    type: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    embedding_ref: str | None = None
