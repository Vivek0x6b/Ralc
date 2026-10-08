"""Warm the Gemma extraction cache without loading any embedding stack.

This runs only the Gemma extractions (both the plain and the hybrid extractor)
over every conversation message and every question string, saving successes to
the shared disk cache as it goes so the run is resumable. It never imports torch
or sentence-transformers, so it is light on memory.

Retry policy lives in GemmaExtractor and covers only genuine server/rate-limit
errors (429, 500, 503). Anything else (a blocked or empty response, a parse
error) is a real failure: the string is recorded in a failures file and skipped
rather than retried forever or crashing the run.

    python benchmarks/warm_gemma_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ralc.extraction import GemmaExtractor, Message

from benchmarks.gemma_cache import (
    GEMMA_MODEL,
    extraction_texts,
    load_gemma_cache,
    save_failures,
    save_gemma_cache,
)

# Guard the memory promise: extraction must not pull in the embedding stack.
assert "torch" not in sys.modules, "torch must not be imported by the warm-up"
assert "sentence_transformers" not in sys.modules, "sentence-transformers must not be imported"

DATA = Path(__file__).resolve().parent / "data"
PAUSE_SECONDS = 1.0


def _warm_one(label, extractor, texts, cache, failures):
    for i, text in enumerate(texts):
        calls_before = extractor.stats["calls"]
        size_before = len(cache)
        status = "ok"
        try:
            extractor.extract(Message("user", text))
        except Exception as exc:
            status = "FAIL"
            failures.append({
                "index": i,
                "mode": label,
                "error": str(exc)[:200],
                "preview": text[:80],
            })
            save_failures(failures)
        made_api_call = extractor.stats["calls"] > calls_before
        if len(cache) > size_before:
            save_gemma_cache(cache)
        mode_fails = sum(1 for f in failures if f["mode"] == label)
        print(f"[{label}] {i + 1}/{len(texts)} {status} "
              f"calls={extractor.stats['calls']} hits={extractor.stats['cache_hits']} "
              f"fails={mode_fails}", flush=True)
        if made_api_call:
            time.sleep(PAUSE_SECONDS)
    save_gemma_cache(cache)


def main():
    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    texts = extraction_texts(messages, questions)

    cache = load_gemma_cache()
    failures: list[dict] = []
    print(f"Loaded {len(cache)} cached entries. Warming over {len(texts)} strings "
          f"per extractor.", flush=True)

    gemma = GemmaExtractor(model=GEMMA_MODEL, strict=True, cache=cache,
                           max_retries=5, retry_base_delay=2.0, timeout=60.0)
    _warm_one("gemma", gemma, texts, cache, failures)

    hybrid = GemmaExtractor(model=GEMMA_MODEL, hybrid=True, strict=True, cache=cache,
                            max_retries=5, retry_base_delay=2.0, timeout=60.0)
    _warm_one("hybrid", hybrid, texts, cache, failures)

    save_failures(failures)
    print(f"\nWarm-up complete. failures: {len(failures)} "
          f"(gemma={sum(1 for f in failures if f['mode'] == 'gemma')}, "
          f"hybrid={sum(1 for f in failures if f['mode'] == 'hybrid')})", flush=True)
    if failures:
        print(f"Recorded in benchmarks/results/gemma_warm_failures.json; "
              f"run.py will use the heuristic for those strings.", flush=True)


if __name__ == "__main__":
    main()
