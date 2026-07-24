#!/usr/bin/env python3
"""
draft_facts.py -- ClearPath Phase 3a: compliance-fact grounding layer.

For one (resume profile, checker eligibility profile, matched posting)
triple -- i.e. one `RankedResult` out of `matcher.rank.rank()`'s output --
assembles a `FactPacket`: the complete, structured set of true statements
that could legitimately appear in a future application/cover-letter draft
for this specific person and this specific posting, each one traced to its
source (a checker `RuleResult`, the classifier's evidence spans/sponsorship
signal, or the matcher's own compatibility verdict). No prose is generated
here -- the output is structured data, not sentences.

Why this exists as its own increment, not folded into generation: see
`matcher/DRAFTING.md` for the full "Attribute First, then Generate"
rationale (this run's research finding, NEXT-BUILD-SPEC.md 2026-07-22).
Short version: build and test the fact-retrieval/attribution layer
completely before any generation touches it, so a future drafting step
(Phase 3b -- NOT built in this increment, needs a free-tier LLM key with
Ujjwal's sign-off) is structurally constrained to only the facts this layer
assembles, rather than bolting attribution on after the fact.

No LLM call, no prose generation, no API key, no network call -- pure
assembly/validation logic over three already-shipped, already-tested
modules (classifier, checker, matcher.rank), imported and reused, never
re-derived. Same free-tier-by-construction discipline as every prior
ClearPath module.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Sibling-module imports -- same convention rank.py already uses: add
# classifier/, checker/, qa/ to sys.path explicitly (they're not Python-path
# siblings of matcher/ by default), then import rank.py itself (same-
# directory import) so we reuse its already-fused RankedResult /
# CategoryCompatibility / Compatibility types and its `rank()` /
# `_load_postings()` helpers rather than re-deriving any of them.
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _sub in ("classifier", "checker", "qa"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eligibility import AssessmentReport, Profile, RuleResult, Status, assess  # noqa: E402

from rank import (  # noqa: E402  (same-directory import)
    CategoryCompatibility,
    Compatibility,
    RankedResult,
    _load_postings,
    rank as rank_postings,
)
from resume_profile import ResumeProfile  # noqa: E402  (same-directory import)


# ---------------------------------------------------------------------------
# FactPacket data model
# ---------------------------------------------------------------------------

SourceType = Literal["checker", "classifier", "matcher"]


@dataclass
class Claim:
    """One statement a future drafting step is (or is not) allowed to make.

    `citation` and `rule_result_status` are deliberately both optional
    individually -- `packet_is_well_formed` requires at least one of them
    to be non-null on every `allowed_claims` entry, so nothing is ever
    asserted with no traceable source. `citation` is a corpus/rule chunk id
    (e.g. "on_campus_unlimited") when the source is `checker`, or a real,
    literal substring of the posting text (an evidence span) when the
    source is `classifier` -- never a paraphrase.
    """

    text: str
    source_type: SourceType
    citation: Optional[str] = None
    rule_result_status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source_type": self.source_type,
            "citation": self.citation,
            "rule_result_status": self.rule_result_status,
        }


@dataclass
class FactPacket:
    posting_id: str
    resume_profile_id: str
    allowed_claims: List[Claim] = field(default_factory=list)
    forbidden_claims: List[Claim] = field(default_factory=list)
    skill_overlap: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "posting_id": self.posting_id,
            "resume_profile_id": self.resume_profile_id,
            "allowed_claims": [c.to_dict() for c in self.allowed_claims],
            "forbidden_claims": [c.to_dict() for c in self.forbidden_claims],
            "skill_overlap": self.skill_overlap,
        }


# ---------------------------------------------------------------------------
# Category -> checker RuleResult attribution
# ---------------------------------------------------------------------------

# Which checker RuleResult "rule" names are relevant evidence for each
# classifier-tagged eligible_category. Only `pgwp_track` maps to real,
# profile-dependent checker RuleResults (pgwp_general_eligibility,
# pgwp_field_of_study). The other three categories' compatibility is, by
# rank.py's own design (`_on_campus_compatibility`, `_co_op_compatibility`,
# `_unrestricted_compatibility`), checker-profile-independent -- it doesn't
# come from evaluating any of this candidate's specific RuleResults, so
# there is nothing to attribute a claim to there beyond
# `CategoryCompatibility`'s own reason/citations. Documented judgment call
# (per NEXT-BUILD-SPEC.md's "flag rather than guess silently" instruction):
# an alternative would be to attribute on_campus_only to `weekly_hours`
# RuleResults when the profile happens to have any logged, but those are
# per-week entries about a specific week's actual hours, not a general
# "on-campus work is uncapped" fact -- conflating the two would misattribute
# a general rule to an incidental personal data point. Kept separate.
_CATEGORY_TO_RULE_NAMES: Dict[str, List[str]] = {
    "pgwp_track": ["pgwp_general_eligibility", "pgwp_field_of_study"],
}


def _relevant_rule_results(category: str, checker_report: AssessmentReport) -> List[RuleResult]:
    names = _CATEGORY_TO_RULE_NAMES.get(category, [])
    if not names:
        return []
    return [r for r in checker_report.results if r.rule in names]


def _claims_for_category(
    cc: CategoryCompatibility, checker_report: AssessmentReport
) -> Tuple[List[Claim], List[Claim]]:
    """Returns (allowed, forbidden) claims for one CategoryCompatibility."""
    allowed: List[Claim] = []
    forbidden: List[Claim] = []
    rule_results = _relevant_rule_results(cc.category, checker_report)

    if cc.verdict == Compatibility.COMPATIBLE:
        if rule_results:
            for r in rule_results:
                # By construction (see rank.py's `_pgwp_compatibility`), a
                # COMPATIBLE verdict only happens when every relevant rule
                # result is COMPLIANT. Assert the invariant rather than
                # silently trusting it -- if this ever fires it means
                # rank.py's mapping changed underneath this module.
                assert r.status == Status.COMPLIANT, (
                    f"COMPATIBLE verdict for category {cc.category!r} but rule {r.rule!r} is "
                    f"{r.status.value!r}, not compliant -- rank.py's compatibility mapping "
                    "invariant is broken; draft_facts.py's attribution assumes it holds."
                )
                allowed.append(
                    Claim(
                        text=(
                            f"You may state you meet the '{r.rule}' requirement for the "
                            f"'{cc.category}' pathway: {r.explanation}"
                        ),
                        source_type="checker",
                        citation=", ".join(r.citations) if r.citations else None,
                        rule_result_status=r.status.value,
                    )
                )
        else:
            citation = ", ".join(cc.citations) if cc.citations else None
            allowed.append(
                Claim(
                    text=(
                        f"You may state you are eligible to pursue this posting via the "
                        f"'{cc.category}' pathway: {cc.reason}"
                    ),
                    source_type="matcher",
                    citation=citation,
                    rule_result_status=cc.verdict.value,
                )
            )
        return allowed, forbidden

    # UNCLEAR or INCOMPATIBLE: never emit an allowed claim for this
    # category -- this is the whole safety point of the layer.
    non_compliant = [r for r in rule_results if r.status != Status.COMPLIANT]
    if non_compliant:
        for r in non_compliant:
            forbidden.append(
                Claim(
                    text=(
                        f"Do not claim eligibility via the '{cc.category}' pathway on the basis of "
                        f"'{r.rule}' -- checker status is {r.status.value}: {r.explanation}"
                    ),
                    source_type="checker",
                    citation=", ".join(r.citations) if r.citations else None,
                    rule_result_status=r.status.value,
                )
            )
    else:
        citation = ", ".join(cc.citations) if cc.citations else None
        forbidden.append(
            Claim(
                text=(
                    f"Do not claim confirmed eligibility via the '{cc.category}' pathway -- {cc.reason}"
                ),
                source_type="matcher",
                citation=citation,
                rule_result_status=cc.verdict.value,
            )
        )
    return allowed, forbidden


def _claims_for_sponsorship(ranked_result: RankedResult) -> Tuple[List[Claim], List[Claim]]:
    """Claims derived from the Phase 0 classifier's sponsorship signal.

    Always emits at least one forbidden claim guarding against overclaiming
    a *guaranteed* sponsorship -- the classifier is an honest heuristic
    (see `classifier/backend.py`'s own docstring: "not a calibrated
    probability"), never a verified employer confirmation, regardless of
    which way the signal leans.
    """
    allowed: List[Claim] = []
    forbidden: List[Claim] = []
    likelihood = ranked_result.sponsorship_likelihood
    confidence = ranked_result.classifier_confidence
    evidence = ", ".join(ranked_result.evidence_spans) if ranked_result.evidence_spans else None
    status_note = f"sponsorship_likelihood={likelihood} (confidence {confidence:.2f})"

    if likelihood == "unlikely":
        forbidden.append(
            Claim(
                text=(
                    "Do not imply or ask about employer-sponsored work-permit support for this "
                    "role -- the posting signals it will not sponsor."
                ),
                source_type="classifier",
                citation=evidence,
                rule_result_status=status_note,
            )
        )
        return allowed, forbidden

    if likelihood == "likely" and evidence:
        allowed.append(
            Claim(
                text=(
                    "You may note the posting signals openness to sponsoring/supporting a work "
                    "permit for the right candidate."
                ),
                source_type="classifier",
                citation=evidence,
                rule_result_status=status_note,
            )
        )

    # Regardless of direction (likely/unclear), a heuristic confidence
    # score is never a guarantee -- forbid overclaiming certainty.
    forbidden.append(
        Claim(
            text=(
                "Do not claim guaranteed or confirmed sponsorship for this role -- the sponsorship "
                f"signal is a heuristic classification ({likelihood}, confidence {confidence:.2f}), "
                "not a verified employer confirmation."
            ),
            source_type="classifier",
            citation=None,
            rule_result_status=status_note,
        )
    )
    return allowed, forbidden


# ---------------------------------------------------------------------------
# Skill overlap -- new helper (no existing overlap signal to reuse)
# ---------------------------------------------------------------------------


def compute_skill_overlap(resume: ResumeProfile, posting_text: str) -> List[str]:
    """Returns the subset of `resume.skills` that literally appear in
    `posting_text`, matched case-insensitively on word boundaries.

    No skill-overlap signal exists anywhere else in the codebase to reuse --
    `rank.py`'s `compute_fit_scores` is whole-resume-text-vs-whole-posting-
    text embedding cosine similarity; it never identifies *which* specific
    skills overlap. Per NEXT-BUILD-SPEC.md Scope-in section 1 ("reuse ... or
    add a small, separately-testable keyword-overlap helper if none exists
    yet -- state which"): none existed, so this is that helper. Deliberately
    simple, literal, word-boundary-matched -- not fuzzy, not ML-based -- so
    every entry stays 100% traceable to a real substring of the posting
    text, the same "no invented substrings" discipline the classifier's
    `evidence_spans` already follow (see `tests/test_classify.py`).
    """
    overlap: List[str] = []
    for skill in resume.skills:
        s = skill.strip()
        if not s:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)", re.IGNORECASE)
        if pattern.search(posting_text):
            overlap.append(s)
    return overlap


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_fact_packet(
    ranked_result: RankedResult,
    resume_profile: ResumeProfile,
    checker_report: AssessmentReport,
) -> FactPacket:
    """Assembles the complete, traceable fact packet a future drafting step
    (Phase 3b -- not built in this increment) would be allowed to build a
    prose application/cover-letter draft from. See `matcher/DRAFTING.md`.

    Reuses `ranked_result`'s already-computed `category_compatibilities`
    (from `matcher.rank.assess_compatibility`) and the classifier's
    sponsorship signal/evidence directly -- nothing here re-derives fit
    scores, compatibility verdicts, or classification, it only attributes
    and validates what those modules already produced.
    """
    allowed: List[Claim] = []
    forbidden: List[Claim] = []

    for cc in ranked_result.category_compatibilities:
        a, f = _claims_for_category(cc, checker_report)
        allowed.extend(a)
        forbidden.extend(f)

    a, f = _claims_for_sponsorship(ranked_result)
    allowed.extend(a)
    forbidden.extend(f)

    skill_overlap = compute_skill_overlap(resume_profile, ranked_result.posting_text)

    return FactPacket(
        posting_id=ranked_result.posting_id,
        resume_profile_id=resume_profile.name,
        allowed_claims=allowed,
        forbidden_claims=forbidden,
        skill_overlap=skill_overlap,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def packet_is_well_formed(packet: FactPacket, postings: Optional[List[dict]] = None) -> List[str]:
    """Returns a list of integrity problems (empty list == well-formed).

    - Every `allowed_claims` entry must have a non-null `citation` or
      `rule_result_status` -- nothing ever asserted with no traceable
      source. (Not required of `forbidden_claims` -- a forbidden claim is
      allowed to be a blanket caution even without a specific citation,
      though in practice every one this module emits has one.)
    - No claim text may appear in both `allowed_claims` and
      `forbidden_claims`.
    - `posting_id` must be non-empty, and if `postings` is supplied, must
      resolve to a real entry in it.
    - `resume_profile_id` must be non-empty.
    """
    problems: List[str] = []

    for c in packet.allowed_claims:
        if not c.citation and not c.rule_result_status:
            problems.append(f"allowed claim has no citation or rule_result_status: {c.text!r}")

    allowed_texts = {c.text for c in packet.allowed_claims}
    forbidden_texts = {c.text for c in packet.forbidden_claims}
    both = allowed_texts & forbidden_texts
    if both:
        problems.append(
            f"claim(s) appear in both allowed_claims and forbidden_claims: {sorted(both)}"
        )

    if not packet.posting_id:
        problems.append("posting_id is empty")
    elif postings is not None:
        ids = {p["id"] for p in postings}
        if packet.posting_id not in ids:
            problems.append(
                f"posting_id {packet.posting_id!r} does not resolve to any posting in the given corpus"
            )

    if not packet.resume_profile_id:
        problems.append("resume_profile_id is empty")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def format_packet(packet: FactPacket) -> str:
    lines = [f"ClearPath Phase 3a -- fact packet for posting {packet.posting_id}"]
    lines.append(f"Resume profile: {packet.resume_profile_id}")
    lines.append("=" * 78)
    lines.append(
        f"Skill overlap: {', '.join(packet.skill_overlap) if packet.skill_overlap else '(none found)'}"
    )
    lines.append("")
    lines.append(f"ALLOWED CLAIMS ({len(packet.allowed_claims)}) -- safe to state in a draft:")
    for c in packet.allowed_claims:
        lines.append(f"  [{c.source_type}] {c.text}")
        lines.append(f"      citation={c.citation!r}  status={c.rule_result_status!r}")
    lines.append("")
    lines.append(f"FORBIDDEN CLAIMS ({len(packet.forbidden_claims)}) -- must NOT be stated:")
    for c in packet.forbidden_claims:
        lines.append(f"  [{c.source_type}] {c.text}")
        lines.append(f"      citation={c.citation!r}  status={c.rule_result_status!r}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ClearPath Phase 3a: compliance-fact grounding layer. Assembles the structured, "
            "cited fact packet a future drafting step is allowed to build from. No LLM call, "
            "no prose generation -- see matcher/DRAFTING.md."
        )
    )
    parser.add_argument("--resume-profile", required=True, help="Path to a resume profile JSON.")
    parser.add_argument("--eligibility-profile", required=True, help="Path to a checker Profile JSON.")
    parser.add_argument("--postings", required=True, help="Path to a JSON list of {id, text, ...} postings.")
    parser.add_argument("--posting-id", required=True, help="Which posting id to build a fact packet for.")
    parser.add_argument("--as-of-date", default=None, help="Override as-of date (YYYY-MM-DD) for the checker.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of formatted text.")
    args = parser.parse_args(argv)

    resume = ResumeProfile.load(args.resume_profile)
    with open(args.eligibility_profile, "r", encoding="utf-8") as f:
        elig_profile = Profile.from_dict(json.load(f))
    postings = _load_postings(args.postings)
    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None

    ranked = rank_postings(resume, elig_profile, postings, as_of_date=as_of)
    match = next((r for r in ranked if r.posting_id == args.posting_id), None)
    if match is None:
        print(f"No posting with id {args.posting_id!r} found in {args.postings}", file=sys.stderr)
        return 1

    checker_report = assess(elig_profile, as_of_date=as_of)
    packet = assemble_fact_packet(match, resume, checker_report)
    problems = packet_is_well_formed(packet, postings=postings)

    if args.json:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        print(format_packet(packet))

    if problems:
        print("\nIntegrity problems found:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
