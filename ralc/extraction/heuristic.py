"""A regex-only Extractor: entities and decision cues, no model required."""

from __future__ import annotations

import re

from ralc.extraction.base import Extractor, Message, Signals, normalize_entities

# Backticked terms: `redis`, `get_user`.
_BACKTICK = re.compile(r"`([^`]+)`")

# Slash paths: ralc/graph/nodes.py, src/app.
_PATH = re.compile(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+")

# Filenames: app.py, config.json. The extension must be alphabetic, so purely
# numeric tokens like 3.10 or 2.14.1 are ignored. The lookbehind stops a
# filename inside a path (nodes.py within ralc/graph/nodes.py) matching twice.
_FILENAME = re.compile(r"(?<![\w./\-])[A-Za-z0-9_\-]+\.[A-Za-z]{2,4}\b")

# snake_case and UPPER_SNAKE: user_id, TEMPORALLY_FOLLOWS.
_SNAKE = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")

# camelCase / PascalCase with an internal uppercase: getUser, ContextGraph.
# Requires a lowercase run before an uppercase, so plain "Hello" and acronyms
# like "API" are not captured.
_CAMEL = re.compile(r"\b[A-Za-z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")

_DECISION_CUES = ("we decided", "let's go with", "switch to", "instead of")


class HeuristicExtractor:
    """Extracts entities and a decision flag using regex rules only.

    ``vocabulary`` is an optional list of domain terms matched case-insensitively
    as whole words; a match is recorded with its supplied (canonical) spelling.
    """

    def __init__(self, vocabulary: list[str] | None = None):
        self.vocabulary = list(vocabulary) if vocabulary else []
        self._vocab_patterns = [
            (re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE), term)
            for term in self.vocabulary
        ]

    def extract(self, message: Message) -> Signals:
        return Signals(
            entities=self._entities(message.content),
            is_decision=self._is_decision(message.content),
        )

    def _entities(self, content: str) -> list[str]:
        # Collect (start, end, text) spans from every rule.
        spans: list[tuple[int, int, str]] = []
        for match in _BACKTICK.finditer(content):
            spans.append((match.start(), match.end(), match.group(1)))
        for pattern in (_PATH, _FILENAME, _SNAKE, _CAMEL):
            for match in pattern.finditer(content):
                spans.append((match.start(), match.end(), match.group(0)))
        for pattern, term in self._vocab_patterns:
            for match in pattern.finditer(content):
                spans.append((match.start(), match.end(), term))

        kept = self._drop_contained(spans)
        ordered_texts = [text for _, _, text in sorted(kept, key=lambda s: s[0])]
        return normalize_entities(ordered_texts)

    @staticmethod
    def _drop_contained(
        spans: list[tuple[int, int, str]]
    ) -> list[tuple[int, int, str]]:
        """Drop any span fully contained within a different span.

        This removes a filename or identifier that sits inside a longer path
        match, so each piece of text yields one entity.
        """
        ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[tuple[int, int, str]] = []
        for start, end, text in ordered:
            contained = any(
                k_start <= start and end <= k_end and (k_start, k_end) != (start, end)
                for k_start, k_end, _ in kept
            )
            if not contained:
                kept.append((start, end, text))
        return kept

    @staticmethod
    def _is_decision(content: str) -> bool:
        lowered = content.lower()
        return any(cue in lowered for cue in _DECISION_CUES)


# Static assurance that the heuristic matches the Extractor protocol.
_: Extractor = HeuristicExtractor()
