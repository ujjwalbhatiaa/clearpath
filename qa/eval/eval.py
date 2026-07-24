#!/usr/bin/env python3
"""
eval.py -- real, honestly-computed retrieval accuracy for the Phase 1
extractive Q&A core, against eval/labeled_questions.json.

Reports:
- Top-1 accuracy on questions that have an expected chunk id.
- Top-3 accuracy (is the correct chunk anywhere in the top 3?) on the same.
- Refusal-path correctness on questions labeled expected_chunk_id: null
  (i.e. does the retriever actually refuse instead of guessing?).
- Every disagreement, printed individually for review -- not hidden.

Run: python eval/eval.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval import RetrievalIndex  # noqa: E402

LABELED_PATH = os.path.join(os.path.dirname(__file__), "labeled_questions.json")


def load_labeled(path: str = LABELED_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def main() -> int:
    questions = load_labeled()
    index = RetrievalIndex().build()

    answerable = [q for q in questions if q["expected_chunk_id"] is not None]
    refusal_cases = [q for q in questions if q["expected_chunk_id"] is None]

    top1_hits = 0
    top3_hits = 0
    misses = []

    for q in answerable:
        results = index.retrieve(q["question"], top_k=3)
        top1 = results[0]
        got_id = top1.chunk.id if not top1.refused else None
        top3_ids = [r.chunk.id for r in results if not r.refused]

        if got_id == q["expected_chunk_id"]:
            top1_hits += 1
        else:
            misses.append(
                {
                    "question": q["question"],
                    "expected": q["expected_chunk_id"],
                    "got_top1": got_id,
                    "top1_score": round(top1.score, 3),
                    "kind": "top1_miss",
                }
            )

        if q["expected_chunk_id"] in top3_ids:
            top3_hits += 1
        elif q["expected_chunk_id"] != got_id:
            # already recorded as a top1 miss above; note top-3 status too
            misses[-1]["in_top3"] = False

    refusal_correct = 0
    refusal_misses = []
    for q in refusal_cases:
        r = index.top1(q["question"])
        if r.refused:
            refusal_correct += 1
        else:
            refusal_misses.append(
                {
                    "question": q["question"],
                    "expected": "REFUSAL",
                    "got_top1": r.chunk.id,
                    "top1_score": round(r.score, 3),
                    "kind": "should_have_refused",
                }
            )

    n_answerable = len(answerable)
    n_refusal = len(refusal_cases)

    top1_acc = top1_hits / n_answerable if n_answerable else 0.0
    top3_acc = top3_hits / n_answerable if n_answerable else 0.0
    refusal_acc = refusal_correct / n_refusal if n_refusal else 0.0

    print("=" * 70)
    print("ClearPath Phase 1 Q&A -- retrieval eval")
    print(f"Corpus snapshot date: {index.snapshot_date}")
    print(f"Eval set: {len(questions)} questions "
          f"({n_answerable} answerable, {n_refusal} expected-refusal)")
    print("=" * 70)
    print(f"Top-1 accuracy (answerable questions):  {top1_hits}/{n_answerable}  ({top1_acc:.1%})")
    print(f"Top-3 accuracy (answerable questions):  {top3_hits}/{n_answerable}  ({top3_acc:.1%})")
    print(f"Refusal-path correctness (expected-refusal questions): "
          f"{refusal_correct}/{n_refusal}  ({refusal_acc:.1%})")

    all_misses = misses + refusal_misses
    if all_misses:
        print()
        print("-" * 70)
        print(f"Disagreements ({len(all_misses)}), printed for review:")
        for m in all_misses:
            print(f"  [{m['kind']}] Q: {m['question']!r}")
            print(f"      expected={m['expected']!r}  got_top1={m['got_top1']!r}  "
                  f"score={m['top1_score']}")
    else:
        print()
        print("No disagreements -- every question matched its expected label.")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
