"""A Gemma-backed Extractor: entity and decision extraction via the Gemini API.

Gemma produces the same Signals as HeuristicExtractor, so a benchmark can swap
one for the other. The API key is read only from the GEMINI_API_KEY environment
variable and is never logged or written anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from ralc.extraction.base import Extractor, Message, Signals, normalize_entities
from ralc.extraction.heuristic import HeuristicExtractor

# Bump when the prompt changes so cached results from an old prompt are not reused.
PROMPT_VERSION = "1"

_RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "RATE_LIMIT", "TOO MANY REQUESTS")

_PROMPT = """You extract structured signals from a single chat message.

Return ONLY a JSON object with exactly these keys:
- "entities": a list of strings naming the specific technical entities, identifiers, files, components, or domain terms the message mentions. Use the exact spelling from the message. Empty list if none.
- "is_decision": a boolean that is true only if the message states or commits to a decision.

Do not add any other keys, text, or explanation.

Message:
{content}
"""


class GemmaExtractor:
    def __init__(
        self,
        model: str = "gemma-4-26b-a4b-it",
        client=None,
        heuristic: Extractor | None = None,
        hybrid: bool = False,
        strict: bool = False,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        cache: dict | None = None,
    ):
        self.model = model
        self.client = client
        self.heuristic = heuristic or HeuristicExtractor()
        self.hybrid = hybrid
        self.strict = strict
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._cache = cache if cache is not None else {}
        self.stats = {"calls": 0, "fallbacks": 0, "cache_hits": 0}

    def extract(self, message: Message) -> Signals:
        key = self._cache_key(message.content)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]

        gemma_signals, success = self._gemma_signals(message)
        if self.hybrid:
            heuristic_signals = self.heuristic.extract(message)
            result = Signals(
                entities=normalize_entities(
                    list(heuristic_signals.entities) + list(gemma_signals.entities)
                ),
                is_decision=heuristic_signals.is_decision or gemma_signals.is_decision,
            )
        else:
            result = gemma_signals

        if success:   # only successful Gemma output is cached
            self._cache[key] = result
        return result

    # -- internals ---------------------------------------------------------

    def _gemma_signals(self, message: Message) -> tuple[Signals, bool]:
        self.stats["calls"] += 1
        try:
            text = self._call_api(message.content)
            entities, is_decision = self._parse(text)
            return Signals(entities=entities, is_decision=is_decision), True
        except Exception:
            if self.strict:
                raise
            self.stats["fallbacks"] += 1
            return self.heuristic.extract(message), False

    def _cache_key(self, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"{self.model}|{int(self.hybrid)}|{PROMPT_VERSION}|{digest}"

    def _call_api(self, content: str) -> str:
        prompt = _PROMPT.format(content=content)
        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=prompt, config={"temperature": 0},
                )
                return response.text
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limit(exc) and attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
                    continue
                raise
        raise last_exc   # pragma: no cover - loop always returns or raises above

    def _get_client(self):
        if self.client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            self.client = genai.Client(api_key=api_key)
        return self.client

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        text = str(exc).upper()
        return any(marker in text for marker in _RATE_LIMIT_MARKERS)

    @staticmethod
    def _parse(text: str) -> tuple[list[str], bool]:
        data = json.loads(_strip_fences(text))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        entities = data.get("entities")
        is_decision = data.get("is_decision")
        if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
            raise ValueError("'entities' must be a list of strings")
        if not isinstance(is_decision, bool):
            raise ValueError("'is_decision' must be a boolean")
        return normalize_entities(entities), is_decision


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]   # drop the opening ``` or ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
