"""Strict Gemma re-extraction with topics (see
benchmarks/PREREGISTRATION_topic_linking.md and its deviation note).

Re-extracts every unique conversation message and question string across v1 and
v2 under the new topic prompt (PROMPT_VERSION 2), strict mode, 45s timeout, 3
retries, warming benchmarks/results/gemma_cache.json (shared, so v1 and v2
dedupe). It is resumable: already-cached strings are skipped and the cache is
saved as it goes.

Deviation from the pre-registration: batching was abandoned. On this endpoint a
batched generation deadlines at 45s (504 DEADLINE_EXCEEDED) even at batch 5,
while a single-message call succeeds in about 16s. Per the pre-registered fixed
45s timeout, extraction uses single calls. The batch-leakage check is therefore
not applicable and is not run.

Run from the repo root with GEMINI_API_KEY set:

    python -m benchmarks.experiments.reextract_gemma_topics
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ralc.extraction import Message
from ralc.extraction.gemma import GemmaExtractor

from benchmarks.gemma_cache import GEMMA_MODEL, load_gemma_cache, save_gemma_cache

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
REPORT_PATH = HERE / "reextract_gemma_topics_report.json"
SAVE_EVERY = 10


def unique_strings():
    seen, out = set(), []
    for d in (DATA, DATA / "v2"):
        messages = json.loads((d / "conversation.json").read_text(encoding="utf-8"))
        questions = json.loads((d / "questions.json").read_text(encoding="utf-8"))
        for s in [m["content"] for m in messages] + [q["question"] for q in questions]:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is not set; re-extraction needs it.")

    strings = unique_strings()
    print(f"unique strings to extract: {len(strings)} (single calls, 45s timeout)", flush=True)

    cache = load_gemma_cache()
    extractor = GemmaExtractor(model=GEMMA_MODEL, strict=True, hybrid=False, cache=cache,
                               max_retries=3, retry_base_delay=2.0, timeout=45.0)
    failures = []
    for i, s in enumerate(strings):
        if extractor.cache_key(s) in cache:
            continue
        try:
            extractor.extract(Message("user", s))
        except Exception as exc:
            failures.append({"preview": s[:70], "error": str(exc)[:160]})
        if (i + 1) % SAVE_EVERY == 0:
            save_gemma_cache(cache)
            print(f"  {i + 1}/{len(strings)} calls={extractor.stats['calls']} "
                  f"total_tokens={extractor.stats['total_tokens']} failures={len(failures)}", flush=True)
    save_gemma_cache(cache)

    missing = sum(1 for s in strings if extractor.cache_key(s) not in cache)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": GEMMA_MODEL,
        "mode": "single_calls",
        "note": "batching abandoned: batched generations deadline at 45s (504); single calls at 45s",
        "batch_leakage_check": "not applicable: batching was dropped, so there is no batched path to compare against",
        "unique_strings": len(strings),
        "reextraction_stats": dict(extractor.stats),
        "uncached_after_run": missing,
        "failures": failures,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.name}")
    print(f"stats: {report['reextraction_stats']}")
    print(f"uncached after run: {missing}; failures: {len(failures)}")


if __name__ == "__main__":
    main()
