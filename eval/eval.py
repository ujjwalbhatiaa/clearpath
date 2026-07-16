#!/usr/bin/env python3
"""
Eval harness for the ClearPath sponsorship/eligibility classifier.

Runs the classifier against a hand-labeled set of job postings
(labeled_postings.json) and reports real, honestly-computed metrics:
  - 3-class accuracy for sponsorship_likelihood (likely/unlikely/unclear)
  - per-class precision/recall/F1 for sponsorship_likelihood
  - per-category precision/recall/F1 for eligible_categories (multi-label)
  - every disagreement, printed for manual review

Usage:
    python eval/eval.py
    python eval/eval.py --labels eval/labeled_postings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Allow running as `python eval/eval.py` from the project root without
# installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import HeuristicBackend  # noqa: E402

SPONSORSHIP_CLASSES = ["likely", "unlikely", "unclear"]
ALL_CATEGORIES = [
    "on_campus_only",
    "co_op_exempt_eligible",
    "pgwp_track",
    "unrestricted",
]


def load_labeled_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def run_eval(labeled_set: list[dict], backend) -> dict:
    n = len(labeled_set)
    correct_sponsorship = 0

    # confusion counts per class for precision/recall
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    # multi-label category counts
    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)

    disagreements = []

    for item in labeled_set:
        text = item["text"]
        gold = item["label"]
        gold_sponsorship = gold["sponsorship_likelihood"]
        gold_categories = set(gold.get("eligible_categories", []))

        result = backend.classify(text)
        pred_sponsorship = result.sponsorship_likelihood
        pred_categories = {tag.category for tag in result.eligible_categories}

        # sponsorship_likelihood scoring
        if pred_sponsorship == gold_sponsorship:
            correct_sponsorship += 1
            class_tp[gold_sponsorship] += 1
        else:
            class_fp[pred_sponsorship] += 1
            class_fn[gold_sponsorship] += 1

        # category scoring (multi-label set comparison)
        for cat in ALL_CATEGORIES:
            in_pred = cat in pred_categories
            in_gold = cat in gold_categories
            if in_pred and in_gold:
                cat_tp[cat] += 1
            elif in_pred and not in_gold:
                cat_fp[cat] += 1
            elif not in_pred and in_gold:
                cat_fn[cat] += 1

        if pred_sponsorship != gold_sponsorship or pred_categories != gold_categories:
            disagreements.append({
                "id": item["id"],
                "text_preview": text[:90] + ("..." if len(text) > 90 else ""),
                "gold_sponsorship": gold_sponsorship,
                "pred_sponsorship": pred_sponsorship,
                "gold_categories": sorted(gold_categories),
                "pred_categories": sorted(pred_categories),
                "confidence": round(result.confidence, 2),
                "evidence": result.evidence_spans,
            })

    accuracy = correct_sponsorship / n if n else 0.0

    per_class = {}
    for cls in SPONSORSHIP_CLASSES:
        p, r, f1 = precision_recall_f1(class_tp[cls], class_fp[cls], class_fn[cls])
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": class_tp[cls] + class_fn[cls]}

    per_category = {}
    for cat in ALL_CATEGORIES:
        p, r, f1 = precision_recall_f1(cat_tp[cat], cat_fp[cat], cat_fn[cat])
        per_category[cat] = {"precision": p, "recall": r, "f1": f1, "support": cat_tp[cat] + cat_fn[cat]}

    return {
        "n": n,
        "accuracy": accuracy,
        "correct": correct_sponsorship,
        "per_class": per_class,
        "per_category": per_category,
        "disagreements": disagreements,
    }


def print_report(report: dict) -> None:
    print("=" * 72)
    print("ClearPath sponsorship/eligibility classifier -- eval report")
    print("=" * 72)
    print(f"Labeled examples: {report['n']}")
    print(f"Sponsorship-likelihood accuracy: {report['correct']}/{report['n']} = {report['accuracy']:.2%}")
    print()
    print("Per-class (sponsorship_likelihood) precision / recall / F1:")
    for cls, m in report["per_class"].items():
        print(f"  {cls:10s} support={m['support']:2d}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
    print()
    print("Per-category (eligible_categories, multi-label) precision / recall / F1:")
    for cat, m in report["per_category"].items():
        print(f"  {cat:24s} support={m['support']:2d}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")
    print()
    if report["disagreements"]:
        print(f"Disagreements ({len(report['disagreements'])} of {report['n']}):")
        for d in report["disagreements"]:
            print(f"  [{d['id']}] \"{d['text_preview']}\"")
            print(f"      gold: sponsorship={d['gold_sponsorship']}  categories={d['gold_categories']}")
            print(f"      pred: sponsorship={d['pred_sponsorship']} (conf={d['confidence']})  categories={d['pred_categories']}")
            print(f"      evidence: {d['evidence']}")
    else:
        print("No disagreements -- classifier matched every gold label.")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the ClearPath classifier against a labeled set.")
    parser.add_argument(
        "--labels",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_postings.json"),
        help="Path to the labeled postings JSON file.",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to also write the full report as JSON.",
    )
    args = parser.parse_args(argv)

    labeled_set = load_labeled_set(args.labels)
    backend = HeuristicBackend()
    report = run_eval(labeled_set, backend)
    print_report(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report written to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
