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
PROMPT_VERSION = "2"

# Only genuine server or rate-limit conditions are worth retrying, including 504
# deadline timeouts from slow models. An AttributeError, a parse error, or an
# empty/blocked response is a real failure and is never retried.
_RETRYABLE_CODES = (429, 500, 503, 504)
_RETRYABLE_MARKERS = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED",
                      "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")

_PROMPT = """You extract structured signals from a single chat message.

Return ONLY a JSON object with exactly these keys:
- "entities": a list of strings naming the specific technical entities, identifiers, files, components, or domain terms the message mentions. Use the exact spelling from the message. Empty list if none.
- "is_decision": a boolean that is true only if the message states or commits to a decision.
- "topic": if is_decision is true, a short phrase (a few words) naming what the decision is about, for example the subject or component it concerns. If is_decision is false, use null.

Do not add any other keys, text, or explanation.

Message:
{content}
"""

_BATCH_PROMPT = """You extract structured signals from several chat messages.

For EACH message below, produce one JSON object with exactly these keys:
- "id": the message id exactly as given in brackets.
- "entities": a list of strings naming the specific technical entities, identifiers, files, components, or domain terms the message mentions. Use the exact spelling from the message. Empty list if none.
- "is_decision": a boolean that is true only if the message states or commits to a decision.
- "topic": if is_decision is true, a short phrase (a few words) naming what the decision is about. If is_decision is false, use null.

Return ONLY a JSON array of these objects, one per input message, in the same order. Do not add any other text or explanation.

Messages:
{block}
"""


def hybrid_signals(heuristic: Signals, gemma: Signals) -> Signals:
    """Combine heuristic and plain-Gemma signals exactly as hybrid mode does.

    The entity union (heuristic first, then Gemma, normalized) and the OR of the
    decision flags. Shared so a hybrid result can be derived from cached plain
    Gemma output with no extra API call.
    """
    return Signals(
        entities=normalize_entities(list(heuristic.entities) + list(gemma.entities)),
        is_decision=heuristic.is_decision or gemma.is_decision,
        topic=gemma.topic,
    )


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
        timeout: float = 120.0,
        cache: dict | None = None,
    ):
        self.model = model
        self.client = client
        self.heuristic = heuristic or HeuristicExtractor()
        self.hybrid = hybrid
        self.strict = strict
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout
        self._cache = cache if cache is not None else {}
        self.stats = {"calls": 0, "fallbacks": 0, "cache_hits": 0,
                      "prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    def extract(self, message: Message) -> Signals:
        key = self._cache_key(message.content)
        if key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]

        gemma_signals, success = self._gemma_signals(message)
        if self.hybrid:
            result = hybrid_signals(self.heuristic.extract(message), gemma_signals)
        else:
            result = gemma_signals

        if success:   # only successful Gemma output is cached
            self._cache[key] = result
        return result

    def extract_batch(self, messages: list[Message]) -> list[Signals]:
        """Extract several messages in one call, falling back to single calls.

        Cached messages are served from the cache; the rest are sent as one
        batch. If the batch response cannot be parsed, each uncached message is
        retried with a single-message call (which enforces strict mode). This
        path is for plain extraction; it does not apply the hybrid union.
        """
        results: list[Signals | None] = [None] * len(messages)
        pending: list[tuple[int, Message]] = []
        for i, message in enumerate(messages):
            key = self._cache_key(message.content)
            if key in self._cache:
                self.stats["cache_hits"] += 1
                results[i] = self._cache[key]
            else:
                pending.append((i, message))

        if pending:
            try:
                parsed = self._call_api_batch([m for _, m in pending])
                self.stats["calls"] += 1
                for (i, message), signals in zip(pending, parsed):
                    self._cache[self._cache_key(message.content)] = signals
                    results[i] = signals
            except Exception:
                for i, message in pending:
                    results[i] = self.extract(message)

        return [r if r is not None else Signals() for r in results]

    # -- internals ---------------------------------------------------------

    def _gemma_signals(self, message: Message) -> tuple[Signals, bool]:
        self.stats["calls"] += 1
        try:
            text = self._call_api(message.content)
            entities, is_decision, topic = self._parse(text)
            return Signals(entities=entities, is_decision=is_decision, topic=topic), True
        except Exception:
            if self.strict:
                raise
            self.stats["fallbacks"] += 1
            return self.heuristic.extract(message), False

    def _cache_key(self, content: str) -> str:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"{self.model}|{int(self.hybrid)}|{PROMPT_VERSION}|{digest}"

    def cache_key(self, content: str) -> str:
        """Public accessor for the cache key of a message content."""
        return self._cache_key(content)

    def _call_api(self, content: str) -> str:
        return self._generate(_PROMPT.format(content=content))

    def _call_api_batch(self, messages: list[Message]) -> list[Signals]:
        block = "\n".join(f"[id={i}] {m.content}" for i, m in enumerate(messages))
        text = self._generate(_BATCH_PROMPT.format(block=block))
        return self._parse_batch(text, len(messages))

    def _generate(self, prompt: str) -> str:
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=prompt, config={"temperature": 0},
                )
                self._record_usage(response)
                text = getattr(response, "text", None)
                if not text or not text.strip():
                    # A blocked or empty response has no usable text. Treat it as
                    # a real failure (not transient) so it is never retried.
                    raise ValueError("empty or blocked response from the model")
                return text
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.max_retries - 1:
                    time.sleep(self.retry_base_delay * (2 ** attempt))
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.stats["prompt_tokens"] += getattr(usage, "prompt_token_count", 0) or 0
        self.stats["candidates_tokens"] += getattr(usage, "candidates_token_count", 0) or 0
        self.stats["total_tokens"] += getattr(usage, "total_token_count", 0) or 0

    def _get_client(self):
        if self.client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set")
            self.client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(self.timeout * 1000)},
            )
        return self.client

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        if code is None:
            code = getattr(exc, "status_code", None)
        if code in _RETRYABLE_CODES:
            return True
        text = str(exc).upper()
        return any(marker in text for marker in _RETRYABLE_MARKERS)

    @classmethod
    def _parse(cls, text: str) -> tuple[list[str], bool, str | None]:
        return cls._parse_obj(json.loads(_strip_fences(text)))

    @staticmethod
    def _parse_obj(data) -> tuple[list[str], bool, str | None]:
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        entities = data.get("entities")
        is_decision = data.get("is_decision")
        if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
            raise ValueError("'entities' must be a list of strings")
        if not isinstance(is_decision, bool):
            raise ValueError("'is_decision' must be a boolean")
        topic = data.get("topic")
        if topic is not None and not isinstance(topic, str):
            raise ValueError("'topic' must be a string or null")
        # A topic only means anything for a decision; clear it otherwise, and
        # drop empty strings to None so the linker has a clean signal.
        if not is_decision or not (topic and topic.strip()):
            topic = None
        else:
            topic = topic.strip()
        return normalize_entities(entities), is_decision, topic

    @classmethod
    def _parse_batch(cls, text: str, n: int) -> list[Signals]:
        data = json.loads(_strip_fences(text))
        if not isinstance(data, list) or len(data) != n:
            raise ValueError(f"expected a JSON array of {n} objects")
        by_id: dict[str, Signals] = {}
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("each batch item must be an object with an 'id'")
            entities, is_decision, topic = cls._parse_obj(item)
            by_id[str(item["id"])] = Signals(entities=entities, is_decision=is_decision, topic=topic)
        try:
            return [by_id[str(i)] for i in range(n)]
        except KeyError as exc:
            raise ValueError(f"batch response is missing id {exc}") from exc


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]   # drop the opening ``` or ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped
