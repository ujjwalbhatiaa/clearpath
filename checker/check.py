#!/usr/bin/env python3
"""
check.py -- CLI for ClearPath's Phase 1 personal compliance & PGWP
eligibility checker.

Given a structured eligibility profile (JSON), runs the rule engine in
eligibility.py and prints a structured, cited report of work-hour
compliance and PGWP eligibility. Pure deterministic rule logic -- no API
key, no model, no network call.

Usage:
    python check.py --profile examples/profile_ujjwal.json
    python check.py --profile examples/profile_college_polytechnic.json --json
    python check.py --profile examples/profile_ujjwal.json --as-of-date 2026-09-01
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from corpus_loader import CHUNKS_BY_ID
from eligibility import AssessmentReport, Profile, RuleResult, assess


def format_report(report: AssessmentReport) -> str:
    lines = ["ClearPath compliance & PGWP eligibility report"]
    if report.profile_name:
        lines.append(f"Profile: {report.profile_name}")
    lines.append(f"As-of date: {report.as_of_date.isoformat()}")
    lines.append("-" * 70)
    for r in report.results:
        lines.append(f"[{r.status.value.upper()}] {r.rule}")
        lines.append(f"    {r.explanation}")
        for cid in r.citations:
            chunk = CHUNKS_BY_ID.get(cid)
            if chunk:
                lines.append(f"    Source: {chunk.source_title} ({chunk.source_url}, modified {chunk.date_modified})")
        lines.append("")
    counts = report.summary()
    lines.append("Summary: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    lines.append(
        "Note: this is an informational compliance checker, not legal or immigration advice. "
        "Always verify against the live canada.ca/ircc.canada.ca source before acting."
    )
    return "\n".join(lines)


def _rule_result_to_dict(r: RuleResult) -> dict:
    return {
        "rule": r.rule,
        "status": r.status.value,
        "explanation": r.explanation,
        "citations": [
            {
                "id": cid,
                "source_title": CHUNKS_BY_ID[cid].source_title,
                "source_url": CHUNKS_BY_ID[cid].source_url,
                "date_modified": CHUNKS_BY_ID[cid].date_modified,
            }
            for cid in r.citations
            if cid in CHUNKS_BY_ID
        ],
    }


def report_to_dict(report: AssessmentReport) -> dict:
    return {
        "profile_name": report.profile_name,
        "as_of_date": report.as_of_date.isoformat(),
        "results": [_rule_result_to_dict(r) for r in report.results],
        "summary": report.summary(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ClearPath personal compliance & PGWP eligibility checker (pure rule logic, no API key)."
    )
    parser.add_argument("--profile", "-p", required=True, help="Path to a profile JSON file.")
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Override the as-of date (YYYY-MM-DD) used for the scheduled-break budget check.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of formatted text."
    )
    args = parser.parse_args(argv)

    with open(args.profile, "r", encoding="utf-8") as f:
        data = json.load(f)
    profile = Profile.from_dict(data)

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    report = assess(profile, as_of_date=as_of)

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(format_report(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
