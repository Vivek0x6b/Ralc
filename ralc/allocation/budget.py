"""Token counting for budget allocation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a piece of text.

    ``approximate`` is True when the count is a heuristic rather than a real
    tokenizer's output, so results can be labeled accordingly.
    """

    approximate: bool

    def count(self, text: str) -> int:
        ...


def _approximate_tokens(text: str) -> int:
    # Roughly four characters per token; ceil so any non-empty text costs >= 1.
    return -(-len(text) // 4)


class ApproximateTokenCounter:
    """Character-based fallback counter (chars / 4), cached per text."""

    approximate = True

    def __init__(self):
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if text not in self._cache:
            self._cache[text] = _approximate_tokens(text)
        return self._cache[text]


class DefaultTokenCounter:
    """Uses tiktoken (cl100k_base) when installed, else an approximate count.

    tiktoken is imported lazily so the core package works without the optional
    ``tokens`` extra. Counts are cached per text.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        self._cache: dict[str, int] = {}
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding)
            self.approximate = False
        except Exception:
            self._encoding = None
            self.approximate = True

    def count(self, text: str) -> int:
        if text not in self._cache:
            if self._encoding is not None:
                self._cache[text] = len(self._encoding.encode(text))
            else:
                self._cache[text] = _approximate_tokens(text)
        return self._cache[text]
