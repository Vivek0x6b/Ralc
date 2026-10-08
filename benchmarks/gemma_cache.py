"""Shared on-disk Gemma extraction cache for the benchmark.

The cache lives in benchmarks/results/gemma_cache.json as a plain JSON map from
a GemmaExtractor cache key to {"entities": [...], "is_decision": bool}. Keeping
it here (rather than in GemmaExtractor) lets the warm-up script and the runner
share it while the extractor itself stays unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from ralc.extraction import Signals

GEMMA_MODEL = "gemma-4-26b-a4b-it"
RESULTS = Path(__file__).resolve().parent / "results"
GEMMA_CACHE = RESULTS / "gemma_cache.json"
GEMMA_FAILURES = RESULTS / "gemma_warm_failures.json"


def extraction_texts(messages: list[dict], questions: list[dict]) -> list[str]:
    """The exact ordered list of strings extracted by the warm-up and the run.

    Every conversation message content, then every question string. The warm-up
    records failures by index into this list; the runner maps the same indices
    back to content, so both must build it identically.
    """
    return [m["content"] for m in messages] + [q["question"] for q in questions]


def load_failures() -> list[dict]:
    if not GEMMA_FAILURES.exists():
        return []
    return json.loads(GEMMA_FAILURES.read_text(encoding="utf-8"))


def save_failures(failures: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    GEMMA_FAILURES.write_text(json.dumps(failures, indent=2), encoding="utf-8")


def load_gemma_cache() -> dict[str, Signals]:
    if not GEMMA_CACHE.exists():
        return {}
    raw = json.loads(GEMMA_CACHE.read_text(encoding="utf-8"))
    return {k: Signals(entities=v["entities"], is_decision=v["is_decision"]) for k, v in raw.items()}


def save_gemma_cache(cache: dict[str, Signals]) -> None:
    RESULTS.mkdir(exist_ok=True)
    raw = {k: {"entities": list(s.entities), "is_decision": bool(s.is_decision)}
           for k, s in cache.items()}
    GEMMA_CACHE.write_text(json.dumps(raw, indent=2), encoding="utf-8")
