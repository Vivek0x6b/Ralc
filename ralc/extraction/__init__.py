"""Context extraction: turning raw context into structured signals."""

from ralc.extraction.base import Extractor, Message, Signals, normalize_entities
from ralc.extraction.gemma import GemmaExtractor, hybrid_signals
from ralc.extraction.heuristic import HeuristicExtractor
from ralc.extraction.linker import DEFAULT_EDGE_WEIGHTS, MessageLinker

__all__ = [
    "DEFAULT_EDGE_WEIGHTS",
    "Extractor",
    "GemmaExtractor",
    "HeuristicExtractor",
    "Message",
    "MessageLinker",
    "Signals",
    "hybrid_signals",
    "normalize_entities",
]
