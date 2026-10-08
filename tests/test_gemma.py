"""Tests for the Gemma extractor (fake client, no network)."""

import os

import numpy as np
import pytest

from ralc import ContextManager
from ralc.extraction import GemmaExtractor, HeuristicExtractor, Message, hybrid_signals
from ralc.extraction.base import Extractor, normalize_entities


# --------------------------------------------------------------------------
# Fake google-genai client
# --------------------------------------------------------------------------

class _Resp:
    def __init__(self, text):
        self.text = text


class _Models:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model=None, contents=None, config=None):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome)


class FakeClient:
    def __init__(self, *outcomes):
        self.models = _Models(outcomes)


def _ex(*outcomes, **kwargs):
    kwargs.setdefault("retry_base_delay", 0.0)
    return GemmaExtractor(client=FakeClient(*outcomes), **kwargs)


# small helpers for the ContextManager test
class DictEmbedder:
    def __init__(self, mapping, dim):
        self.mapping = mapping
        self.dim = dim

    def embed(self, texts):
        rows = []
        for t in texts:
            v = np.zeros(self.dim)
            if t in self.mapping:
                v[self.mapping[t]] = 1.0
            rows.append(v)
        return np.vstack(rows) if rows else np.zeros((0, self.dim))


class WordCounter:
    approximate = False

    def count(self, text):
        return len(text.split())


# --------------------------------------------------------------------------
# Protocol and parsing
# --------------------------------------------------------------------------

def test_is_an_extractor():
    assert isinstance(_ex('{"entities": [], "is_decision": false}'), Extractor)


def test_parse_clean_json():
    s = _ex('{"entities": ["redis"], "is_decision": true}').extract(Message("user", "x"))
    assert s.entities == ["redis"]
    assert s.is_decision is True


def test_parse_fenced_json():
    s = _ex('```json\n{"entities": ["Cache_Key"], "is_decision": false}\n```').extract(
        Message("user", "x")
    )
    assert s.entities == ["Cache_Key"]
    assert s.is_decision is False


def test_parse_malformed_falls_back_to_heuristic():
    ex = _ex("not json at all")
    message = Message("user", "please use `redis`")
    s = ex.extract(message)
    assert s == HeuristicExtractor().extract(message)
    assert ex.stats["fallbacks"] == 1


# --------------------------------------------------------------------------
# strict mode
# --------------------------------------------------------------------------

def test_strict_raises_on_parse_error():
    ex = _ex("not json", strict=True)
    with pytest.raises(Exception):
        ex.extract(Message("user", "x"))


def test_strict_raises_on_api_error():
    ex = _ex(RuntimeError("boom"), strict=True)
    with pytest.raises(Exception):
        ex.extract(Message("user", "x"))


def test_non_strict_falls_back_on_api_error():
    ex = _ex(RuntimeError("boom"))
    message = Message("user", "use `redis`")
    s = ex.extract(message)
    assert s == HeuristicExtractor().extract(message)
    assert ex.stats["fallbacks"] == 1


def test_fallback_counting_multiple():
    ex = _ex(RuntimeError("boom"))   # every call raises
    ex.extract(Message("user", "use `redis`"))
    ex.extract(Message("user", "and `postgres`"))
    assert ex.stats["fallbacks"] == 2


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------

def test_success_is_cached_second_call_skips_client():
    ex = _ex('{"entities": ["redis"], "is_decision": true}')
    ex.extract(Message("user", "same content"))
    ex.extract(Message("user", "same content"))
    assert ex.client.models.calls == 1
    assert ex.stats["cache_hits"] == 1


def test_fallback_is_not_cached():
    # call 1 raises -> fallback (not cached); call 2 succeeds -> cached; call 3 cache hit
    ex = _ex(
        RuntimeError("boom"),
        '{"entities": ["redis"], "is_decision": true}',
        '{"entities": ["redis"], "is_decision": true}',
    )
    content = "use `redis`"
    ex.extract(Message("user", content))
    second = ex.extract(Message("user", content))
    third = ex.extract(Message("user", content))
    assert ex.client.models.calls == 2            # fallback was not cached, so call 2 hit the client
    assert second.is_decision is True             # the successful Gemma result
    assert third == second                        # served from cache
    assert ex.stats["fallbacks"] == 1
    assert ex.stats["cache_hits"] == 1


def test_cache_key_includes_model_and_hybrid():
    shared: dict = {}
    content = "same text"
    a = GemmaExtractor(model="m1", client=FakeClient('{"entities": ["from_m1"], "is_decision": false}'),
                       cache=shared, retry_base_delay=0.0)
    b = GemmaExtractor(model="m2", client=FakeClient('{"entities": ["from_m2"], "is_decision": false}'),
                       cache=shared, retry_base_delay=0.0)

    ra = a.extract(Message("user", content))
    rb = b.extract(Message("user", content))
    assert ra.entities == ["from_m1"]
    assert rb.entities == ["from_m2"]              # not served from a's cache entry
    assert b.client.models.calls == 1

    # the hybrid flag also separates cache entries
    c = GemmaExtractor(model="m1", hybrid=True,
                       client=FakeClient('{"entities": ["from_hybrid"], "is_decision": false}'),
                       cache=shared, retry_base_delay=0.0)
    c.extract(Message("user", content))
    assert c.client.models.calls == 1             # different key than a, so it called its client


# --------------------------------------------------------------------------
# Hybrid, normalization, immutability, retry
# --------------------------------------------------------------------------

def test_derived_hybrid_equals_real_hybrid():
    # Deriving hybrid from cached plain-Gemma output plus the heuristic must
    # match what GemmaExtractor(hybrid=True) produces for the same Gemma output.
    message = Message("user", "we decided to use cache_key for the lookup")
    gemma_json = '{"entities": ["gamma_id", "cache_key"], "is_decision": false}'

    real = GemmaExtractor(hybrid=True, client=FakeClient(gemma_json),
                          retry_base_delay=0.0).extract(message)
    plain = GemmaExtractor(hybrid=False, client=FakeClient(gemma_json),
                           retry_base_delay=0.0).extract(message)
    derived = hybrid_signals(HeuristicExtractor().extract(message), plain)

    assert derived == real


def test_hybrid_unions_entities_and_or_decision():
    ex = _ex('{"entities": ["gamma_id"], "is_decision": false}', hybrid=True)
    s = ex.extract(Message("user", "we decided to use cache_key"))
    assert set(s.entities) == {"cache_key", "gamma_id"}
    assert s.is_decision is True                   # heuristic cue "we decided"


def test_entity_normalization_matches_heuristic_convention():
    raw = ["Cache_Key", "cache_key", " Redis "]
    ex = _ex('{"entities": ["Cache_Key", "cache_key", " Redis "], "is_decision": false}')
    s = ex.extract(Message("user", "x"))
    assert s.entities == normalize_entities(raw)
    assert s.entities == ["Cache_Key", "Redis"]    # strip + case-insensitive dedup, first spelling


def test_message_content_not_mutated():
    content = "we decided to use `redis` for cache_key"
    message = Message("user", content)
    _ex('{"entities": ["redis"], "is_decision": true}').extract(message)
    assert message.content == content


def test_retry_then_success_on_rate_limit():
    ex = _ex(RuntimeError("429 RESOURCE_EXHAUSTED"),
             '{"entities": ["redis"], "is_decision": false}',
             max_retries=3)
    s = ex.extract(Message("user", "x"))
    assert s.entities == ["redis"]
    assert ex.stats["fallbacks"] == 0
    assert ex.client.models.calls == 2             # one retry after the rate-limit error


def test_empty_response_is_a_failure_not_retried():
    message = Message("user", "use `redis`")
    ex = _ex(None, max_retries=5)                  # response.text is None (blocked/empty)
    s = ex.extract(message)
    assert s == HeuristicExtractor().extract(message)   # fell back, did not crash
    assert ex.stats["fallbacks"] == 1
    assert ex.client.models.calls == 1             # not retried

    strict = _ex(None, strict=True, max_retries=5)
    with pytest.raises(Exception):
        strict.extract(Message("user", "x"))
    assert strict.client.models.calls == 1


def test_non_retryable_error_is_not_retried():
    ex = _ex(RuntimeError("boom"), max_retries=5)
    ex.extract(Message("user", "use `redis`"))
    assert ex.client.models.calls == 1             # a plain error is not a transient one


# --------------------------------------------------------------------------
# ContextManager integration and live
# --------------------------------------------------------------------------

def test_context_manager_with_gemma_injected():
    embedder = DictEmbedder({"q": 0, "alpha": 0, "beta": 1}, dim=2)
    extractor = GemmaExtractor(
        client=FakeClient('{"entities": ["postgres"], "is_decision": true}'),
        retry_base_delay=0.0,
    )
    cm = ContextManager(embedder=embedder, extractor=extractor, token_counter=WordCounter())
    cm.add_message("user", "alpha")
    cm.add_message("assistant", "beta")
    result = cm.retrieve("q", token_budget=100, seed_k=1)
    assert len(result.nodes) >= 1
    assert result.token_count <= 100


_LIVE = os.environ.get("GEMINI_API_KEY") and os.environ.get("RALC_LIVE_TESTS") == "1"


@pytest.mark.skipif(not _LIVE, reason="requires GEMINI_API_KEY and RALC_LIVE_TESTS=1")
def test_live_gemma_extraction():
    # strict=True raises on any API or parse error, so this can never pass via
    # the heuristic fallback: it only passes on a genuine Gemma response.
    ex = GemmaExtractor(strict=True)
    s = ex.extract(Message("user", "We decided to use PostgreSQL for storage."))
    assert isinstance(s.entities, list)
    assert isinstance(s.is_decision, bool)
    assert ex.stats["fallbacks"] == 0
