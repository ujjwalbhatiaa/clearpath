#!/usr/bin/env python3
"""
ClearPath sponsorship/eligibility classifier — CLI entry point.

Usage:
    python classify.py --input posting.txt
    python classify.py --text "We do not sponsor employment visas..."
    echo "some posting text" | python classify.py

Prints structured JSON to stdout:
{
  "sponsorship_likelihood": "likely" | "unlikely" | "unclear",
  "confidence": 0.0-1.0,
  "evidence_spans": ["exact quote from the posting", ...],
  "eligible_categories": [{"category": "...", "reason": "..."}, ...]
}
"""

from __future__ import annotations

import argparse
import json
import sys

from classifier import HeuristicBackend


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a job posting's sponsorship likelihood and IRCC eligibility categories."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input", "-i", help="Path to a text file containing the job posting.")
    group.add_argument("--text", "-t", help="Job posting text passed directly on the command line.")
    parser.add_argument(
        "--backend",
        default="heuristic",
        choices=["heuristic"],
        help="Which classifier backend to use (only 'heuristic' is implemented in this increment).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: on).",
    )
    return parser


def load_posting_text(args: argparse.Namespace) -> str:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            return f.read()
    if args.text:
        return args.text
    # fall back to stdin
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise SystemExit(
        "No posting text provided. Use --input <file>, --text \"...\", or pipe text via stdin."
    )


def get_backend(name: str):
    if name == "heuristic":
        return HeuristicBackend()
    raise SystemExit(f"Unknown backend: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    posting_text = load_posting_text(args)
    backend = get_backend(args.backend)
    result = backend.classify(posting_text)

    print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
