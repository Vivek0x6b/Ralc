"""Warm the Gemma extraction cache without loading any embedding stack.

This runs only the Gemma extractions (both the plain and the hybrid extractor)
over every conversation message and every question string, saving results to
the shared disk cache incrementally so the run is resumable. It never imports
torch or sentence-transformers, so it is light on memory and can run while the
full benchmark cannot.

    python benchmarks/warm_gemma_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ralc.extraction import GemmaExtractor, Message

from benchmarks.gemma_cache import GEMMA_MODEL, load_gemma_cache, save_gemma_cache

# Guard the memory promise: extraction must not pull in the embedding stack.
assert "torch" not in sys.modules, "torch must not be imported by the warm-up"
assert "sentence_transformers" not in sys.modules, "sentence-transformers must not be imported"

DATA = Path(__file__).resolve().parent / "data"
SAVE_EVERY = 20


def _texts() -> list[str]:
    messages = json.loads((DATA / "conversation.json").read_text(encoding="utf-8"))
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    # Extraction is keyed by content only, so the role here does not matter.
    return [m["content"] for m in messages] + [q["question"] for q in questions]


def _warm_one(label, extractor, texts, cache):
    done = 0
    for text in texts:
        for attempt in range(6):
            try:
                extractor.extract(Message("user", text))
                break
            except Exception as exc:
                if attempt == 5:
                    raise
                print(f"    transient {type(exc).__name__} on {label}; "
                      f"retry {attempt + 1}/6", flush=True)
                time.sleep(2 * (attempt + 1))
        done += 1
        if done % SAVE_EVERY == 0:
            save_gemma_cache(cache)
            print(f"  {label}: {done}/{len(texts)} "
                  f"(calls={extractor.stats['calls']}, "
                  f"cache_hits={extractor.stats['cache_hits']})", flush=True)
    save_gemma_cache(cache)
    print(f"  {label}: done {done}/{len(texts)} stats={extractor.stats}", flush=True)


def main():
    texts = _texts()
    cache = load_gemma_cache()
    print(f"Loaded {len(cache)} cached entries. Warming over {len(texts)} strings "
          f"per extractor.", flush=True)

    gemma = GemmaExtractor(model=GEMMA_MODEL, strict=True, cache=cache,
                           max_retries=6, retry_base_delay=2.0)
    _warm_one("gemma", gemma, texts, cache)

    hybrid = GemmaExtractor(model=GEMMA_MODEL, hybrid=True, strict=True, cache=cache,
                            max_retries=6, retry_base_delay=2.0)
    _warm_one("hybrid", hybrid, texts, cache)

    print("Warm-up complete.", flush=True)


if __name__ == "__main__":
    main()
