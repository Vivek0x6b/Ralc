"""Context extraction: turning raw context into structured signals."""

from ralc.extraction.base import Extractor, Message, Signals
from ralc.extraction.heuristic import HeuristicExtractor
from ralc.extraction.linker import DEFAULT_EDGE_WEIGHTS, MessageLinker

__all__ = [
    "DEFAULT_EDGE_WEIGHTS",
    "Extractor",
    "HeuristicExtractor",
    "Message",
    "MessageLinker",
    "Signals",
]
