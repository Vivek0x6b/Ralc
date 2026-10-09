"""Recompute the relational ablation with the corrected metric and tags.

Offline, no API and no retrieval: it reads the stored per-cell selected ids from
relational_ablation_results.json and the hand-reviewed labels in
question_labels.json, then reports complete-story both ways side by side:

- OLD: complete = full gold set selected (recall == 1 on gold_message_ids),
  question tag from the gold count (>= 2 gold = relationship).
- NEW: complete = every required_message_id selected, question tag from the
  required count (>= 2 required = relationship).

Recall is unchanged (always on gold_message_ids), so it is not re-reported here.
Run with `python -m benchmarks.experiments.recompute_corrected_metric`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SRC = HERE / "relational_ablation_results.json"
MD_PATH = HERE / "relational_ablation_corrected.md"
JSON_PATH = HERE / "relational_ablation_corrected.json"


def load_labels(name):
    d = DATA / "v2" if name == "v2" else DATA
    labels = json.loads((d / "question_labels.json").read_text(encoding="utf-8"))
    return {x["question"]: x for x in labels}


def rates(dataset, labels):
    """Per budget, per tag, per method: old and new complete-story rate."""
    methods = dataset["method_order"]
    budgets = dataset["budgets"]
    pq = dataset["per_question"]
    out = {}
    for b in budgets:
        out[str(b)] = {}
        for tagkind in ("relationship", "lookup"):
            # Row sets differ between old and new tagging, so compute each on its
            # own membership.
            old_rows = [q for q in pq if q["type"] == tagkind]
            new_rows = [q for q in pq if labels[q["question"]]["tag"] == tagkind]
            out[str(b)][tagkind] = {}
            for m in methods:
                old_vals = [q["results"][str(b)][m]["complete"] for q in old_rows]
                new_vals = []
                for q in new_rows:
                    required = set(labels[q["question"]]["required_message_ids"])
                    selected = set(q["results"][str(b)][m]["selected"])
                    new_vals.append(1 if required <= selected else 0)
                out[str(b)][tagkind][m] = {
                    "old": sum(old_vals) / len(old_vals) if old_vals else 0.0,
                    "new": sum(new_vals) / len(new_vals) if new_vals else 0.0,
                    "old_n": len(old_rows), "new_n": len(new_rows),
                }
    return out


def render_md(result):
    lines = ["# Relational ablation with corrected complete-story metric and tags", "",
             f"- generated: {result['generated_at']}",
             "- OLD = full gold selected, tag from gold count. NEW = required subset "
             "selected, tag from required count.",
             ""]
    for name, d in result["datasets"].items():
        counts = d["counts"]
        lines.append(f"## {name} (old tags: rel {counts['old']['relationship']} / lookup "
                     f"{counts['old']['lookup']}; new tags: rel {counts['new']['relationship']} / "
                     f"lookup {counts['new']['lookup']})")
        lines.append("")
        methods = d["methods"]
        for b in d["budgets"]:
            lines.append(f"### Budget {b} (complete-story: old -> new)")
            lines.append("")
            lines.append("| method | relationship old | relationship new | lookup old | lookup new |")
            lines.append("| --- | --- | --- | --- | --- |")
            r = d["rates"][str(b)]
            for m in methods:
                rel = r["relationship"][m]
                lk = r["lookup"][m]
                lines.append(f"| {m} | {rel['old']:.2f} | {rel['new']:.2f} | "
                             f"{lk['old']:.2f} | {lk['new']:.2f} |")
            lines.append("")
    return "\n".join(lines)


def main():
    src = json.loads(SRC.read_text(encoding="utf-8"))
    datasets = {}
    for name, dataset in src["datasets"].items():
        labels = load_labels(name)
        pq = dataset["per_question"]
        old_counts = {t: sum(1 for q in pq if q["type"] == t) for t in ("relationship", "lookup")}
        new_counts = {t: sum(1 for q in pq if labels[q["question"]]["tag"] == t)
                      for t in ("relationship", "lookup")}
        datasets[name] = {
            "budgets": dataset["budgets"],
            "methods": dataset["method_order"],
            "counts": {"old": old_counts, "new": new_counts},
            "rates": rates(dataset, labels),
        }
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "source": SRC.name, "datasets": datasets}
    JSON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = render_md(result)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {JSON_PATH.name} and {MD_PATH.name}\n")
    print(md)


if __name__ == "__main__":
    main()
