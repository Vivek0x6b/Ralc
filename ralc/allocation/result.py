"""The structured output of context selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from ralc.graph.edges import Edge
from ralc.graph.nodes import Node


@dataclass
class ContextResult:
    """Selected context, ready for an application to build its LLM request.

    ``nodes`` are the selected nodes in conversation order. ``relationships``
    are the graph edges whose endpoints are both selected. ``metadata`` holds
    real, measured numbers about the selection. The query is never stored here.
    """

    nodes: list[Node]
    token_count: int
    relationships: list[Edge]
    metadata: dict = field(default_factory=dict)
