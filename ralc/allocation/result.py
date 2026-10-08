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

    def to_text(self) -> str:
        """Format the selected nodes as plain text for a prompt.

        Each node is rendered as ``"{role}: {content}"`` (or just the content
        when it has no role) in conversation order. The query is never included.
        """
        lines = []
        for node in self.nodes:
            role = node.metadata.get("role")
            lines.append(f"{role}: {node.content}" if role else node.content)
        return "\n\n".join(lines)
