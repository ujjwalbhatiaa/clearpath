#!/usr/bin/env python3
"""
ask.py -- CLI for ClearPath's Phase 1 IRCC compliance Q&A (extractive v0).

Given a question about Canadian study-permit work rules, returns the exact
quoted passage from the grounding corpus (corpus/chunks.json) that best
matches -- never a generated or paraphrased answer. If nothing in the
corpus is a confident match, refuses rather than guessing.

Usage:
    python ask.py --question "How many hours can I work during summer break?"
    python ask.py -q "Do I need a co-op work permit?" --top-k 3
    echo "What is the capital of Canada?" | python ask.py
"""

from __future__ import annotations

import argparse
import json
import sys

from retrieval import REFUSAL_THRESHOLD, RetrievalIndex, RetrievalResult


def format_result(rank: int, result: RetrievalResult) -> str:
    if result.refused:
        return (
            f"[{rank}] No grounded answer found for this question "
            f"(best match similarity {result.score:.3f} is below the "
            f"{REFUSAL_THRESHOLD} refusal threshold) -- consult canada.ca "
            f"directly rather than relying on this tool for this question."
        )
    c = result.chunk
    return (
        f"[{rank}] (similarity {result.score:.3f})\n"
        f'    "{c.text}"\n'
        f"    Source: {c.source_title}\n"
        f"    URL: {c.source_url}\n"
        f"    Last modified: {c.date_modified}"
    )


def result_to_dict(result: RetrievalResult) -> dict:
    if result.refused:
        return {"refused": True, "score": round(result.score, 4), "chunk": None}
    c = result.chunk
    return {
        "refused": False,
        "score": round(result.score, 4),
        "chunk": {
            "id": c.id,
            "text": c.text,
            "source_title": c.source_title,
            "source_url": c.source_url,
            "date_modified": c.date_modified,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extractive, citation-only Q&A over a static snapshot of IRCC "
            "study-permit work-rule pages. Never generates an answer -- "
            "always returns an exact quote + source, or refuses."
        )
    )
    parser.add_argument("--question", "-q", type=str, default=None, help="The question to ask.")
    parser.add_argument(
        "--top-k", "-k", type=int, default=1, help="Number of ranked results to return (default 1)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of formatted text."
    )
    args = parser.parse_args(argv)

    question = args.question
    if question is None:
        if not sys.stdin.isatty():
            question = sys.stdin.read().strip()
        if not question:
            parser.error("no --question given and nothing piped on stdin")

    index = RetrievalIndex().build()
    results = index.retrieve(question, top_k=args.top_k)

    if args.json:
        print(json.dumps({"question": question, "results": [result_to_dict(r) for r in results]}, indent=2))
    else:
        print(f'Question: "{question}"')
        print(f"Corpus snapshot date: {index.snapshot_date}")
        print("-" * 60)
        for i, r in enumerate(results, start=1):
            print(format_result(i, r))
            print()
        print(
            "Note: this is a citation finder over a static snapshot of "
            "canada.ca/ircc.canada.ca text, not legal or immigration "
            "advice. Always verify against the live source before acting."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
