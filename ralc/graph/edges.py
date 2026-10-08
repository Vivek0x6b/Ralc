"""The Edge: a typed, weighted relationship between two nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Edge:
    """A directed relationship from ``source`` to ``target``.

    Fields follow section 10 of the project instructions. ``created_at`` is
    epoch seconds. The pair (``source``, ``target``, ``type``) is unique within
    a graph: at most one edge of a given type per ordered pair.
    """

    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
