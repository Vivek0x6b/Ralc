"""Build hand-reviewed question labels alongside questions.json (not changing it).

Adds, per question:
- required_message_ids: the minimal set of gold messages that together state
  every gold fact (from the hand review recorded in the pre-registration
  amendment of 2026-10-09),
- tag: relationship if >= 2 required messages, else lookup.

gold_message_ids stays the supporting set used for recall. Only the questions
listed in OVERRIDES have required != gold; all others keep the full gold set.
Run with `python -m benchmarks.data.build_labels`.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per dataset, per question index: which gold indices to keep as required.
# Everything not listed keeps its full gold set.
OVERRIDES = {
    "v1": {3: [1], 6: [1], 11: [0, 2]},
    "v2": {3: [1], 6: [1], 11: [0, 2], 14: [0, 2, 3]},
}


def build(name: str, data_dir: Path) -> list[dict]:
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    overrides = OVERRIDES[name]
    labels = []
    for i, q in enumerate(questions):
        gold = q["gold_message_ids"]
        keep = overrides.get(i)
        required = [gold[j] for j in keep] if keep is not None else list(gold)
        labels.append({
            "question": q["question"],
            "gold_message_ids": gold,
            "required_message_ids": required,
            "tag": "relationship" if len(required) >= 2 else "lookup",
        })
    (data_dir / "question_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def main():
    for name, data_dir in (("v1", HERE), ("v2", HERE / "v2")):
        labels = build(name, data_dir)
        rel = sum(1 for x in labels if x["tag"] == "relationship")
        changed = [x["question"][:50] for x in labels
                   if x["required_message_ids"] != x["gold_message_ids"]]
        print(f"{name}: {len(labels)} questions, relationship {rel}, lookup {len(labels) - rel}")
        print(f"   required != gold for {len(changed)}: {changed}")


if __name__ == "__main__":
    main()
