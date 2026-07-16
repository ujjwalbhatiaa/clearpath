"""
Sponsorship / eligibility classification backends for ClearPath.

Design: `ClassifierBackend` is an abstract interface so the v0 rule-based
`HeuristicBackend` can later be swapped for a free-tier LLM backend
(e.g. a Groq or Gemini free-tier call) without changing any calling code.
No backend other than HeuristicBackend is implemented in this increment —
per STARTUP.md's "ask before using any API key" operating constraint, no
LLM API has been activated. See BUILD-STATUS.md for the pending-decision
log entry.

IRCC rule grounding for eligible_categories (do not invent thresholds —
these are the facts logged in STARTUP.md's research log, 2026-07-16):
- on_campus_only: on-campus work by a study-permit holder is NOT subject
  to the off-campus 24-hr/week cap; that cap only applies to off-campus
  work. Source: https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/work-off-campus.html
- co_op_exempt_eligible: as of April 1, 2026, eligible co-op/internship
  placements that are <=50% of the program length no longer require a
  separate co-op work permit. Source: CIC News, 2026-04,
  https://www.cicnews.com/2026/04/canada-moves-to-expand-work-authorization-for-international-students-and-graduates-0473917.html
- pgwp_track: full-time/permanent career-track roles are the kind of
  experience relevant to a Post-Graduation Work Permit holder or
  soon-to-be-graduate; PGWP field-of-study list was frozen for 2026.
  Source: CIC News, 2026-01,
  https://www.cicnews.com/2026/01/ircc-freezes-list-of-pgwp-eligible-fields-of-study-for-2026-0167305.html
- unrestricted: posting explicitly states it accepts any legal work
  status / open work permit holders, i.e. not gated by a specific permit
  category.

This is a heuristic, evidence-quoting engine, not a legal-advice tool.
Every category tag carries a one-line, source-grounded reason so a human
can audit the call.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

SponsorshipLikelihood = Literal["likely", "unlikely", "unclear"]
EligibleCategory = Literal[
    "on_campus_only",
    "co_op_exempt_eligible",
    "pgwp_track",
    "unrestricted",
]


@dataclass
class CategoryTag:
    category: EligibleCategory
    reason: str


@dataclass
class ClassificationResult:
    sponsorship_likelihood: SponsorshipLikelihood
    confidence: float
    evidence_spans: list[str] = field(default_factory=list)
    eligible_categories: list[CategoryTag] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sponsorship_likelihood": self.sponsorship_likelihood,
            "confidence": round(self.confidence, 2),
            "evidence_spans": self.evidence_spans,
            "eligible_categories": [
                {"category": t.category, "reason": t.reason}
                for t in self.eligible_categories
            ],
        }


class ClassifierBackend(ABC):
    """Interface every classification backend must implement."""

    @abstractmethod
    def classify(self, posting_text: str) -> ClassificationResult:
        ...


# ---------------------------------------------------------------------------
# Heuristic backend
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, weight). Weight is added to a running score;
# positive weight pushes toward "likely" (will sponsor / consider
# non-PR/non-citizen candidates), negative weight pushes toward "unlikely".
# Patterns are intentionally conservative (real phrasing seen in postings)
# rather than single keywords, to cut down on false triggers.
_NO_SPONSOR_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bwe do not sponsor\b", re.I), -3.0),
    (re.compile(r"\bwill not sponsor\b", re.I), -3.0),
    (re.compile(r"\bunable to (?:offer|provide) (?:visa )?sponsorship\b", re.I), -3.0),
    (re.compile(r"\bnot able to sponsor\b", re.I), -3.0),
    (re.compile(r"\bno (?:visa )?sponsorship (?:is )?(?:available|offered|provided)\b", re.I), -3.0),
    (re.compile(r"\bsponsorship is not available\b", re.I), -3.0),
    (re.compile(r"\bmust be (?:a )?(?:canadian citizen|permanent resident)\b", re.I), -2.5),
    (re.compile(r"\bcanadian citizens? (?:and|or) permanent residents? only\b", re.I), -3.0),
    (re.compile(r"\bmust have (?:valid |unrestricted )?(?:permanent )?work authorization\b", re.I), -1.5),
    (re.compile(r"\blegally (?:entitled|authorized) to work in canada without (?:employer )?sponsorship\b", re.I), -3.0),
    (re.compile(r"\bwithout the need for (?:current or future )?(?:visa )?sponsorship\b", re.I), -2.5),
    (re.compile(r"\bdoes not (?:currently )?provide (?:immigration|visa) sponsorship\b", re.I), -3.0),
]

_SPONSOR_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bwe (?:sponsor|offer sponsorship)\b", re.I), 3.0),
    (re.compile(r"\bsponsorship (?:is )?available\b", re.I), 3.0),
    (re.compile(r"\bopen to sponsoring\b", re.I), 2.5),
    (re.compile(r"\bwill sponsor (?:eligible )?candidates\b", re.I), 3.0),
    (re.compile(r"\blmia support (?:is )?available\b", re.I), 2.5),
    (re.compile(r"\binternational students? (?:are )?(?:welcome|encouraged to apply)\b", re.I), 1.5),
    (re.compile(r"\bstudy permit holders? (?:are )?(?:welcome|encouraged to apply)\b", re.I), 2.0),
    (re.compile(r"\bopen work permit(?:s)? (?:holders? )?(?:are )?(?:welcome|accepted)\b", re.I), 2.0),
    (re.compile(r"\bwe welcome applicants? (?:of )?(?:any|all) (?:legal )?work status\b", re.I), 2.0),
]

_ON_CAMPUS_PATTERNS = [
    re.compile(r"\bon[- ]campus\b", re.I),
    re.compile(r"\bstudent (?:employee|assistant) position\b", re.I),
    re.compile(r"\bcampus job\b", re.I),
    re.compile(r"\bwork[- ]study\b", re.I),
]

_CO_OP_PATTERNS = [
    re.compile(r"\bco[- ]op\b", re.I),
    re.compile(r"\binternship\b", re.I),
    re.compile(r"\bpracticum\b", re.I),
    re.compile(r"\bfield placement\b", re.I),
    re.compile(r"\bwork[- ]integrated learning\b", re.I),
]

_PGWP_PATTERNS = [
    re.compile(r"\bfull[- ]time,? permanent\b", re.I),
    re.compile(r"\bpermanent,? full[- ]time\b", re.I),
    re.compile(r"\bnew grad(?:uate)? program\b", re.I),
    re.compile(r"\bcareer opportunity\b", re.I),
    re.compile(r"\bpost[- ]graduat(?:e|ion)\b", re.I),
    re.compile(r"\bentry[- ]level full[- ]time\b", re.I),
]

_UNRESTRICTED_PATTERNS = [
    re.compile(r"\bany legal work status\b", re.I),
    re.compile(r"\bopen work permit(?:s)? accepted\b", re.I),
    re.compile(r"\bregardless of (?:immigration|visa) status\b", re.I),
]


def _find_matches(text: str, patterns: list[re.Pattern]) -> list[str]:
    """Return the exact matched substrings (real spans of `text`) for the
    first match of each pattern that fires, deduplicated, in order found."""
    spans: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        m = pat.search(text)
        if m:
            span = m.group(0)
            key = span.lower()
            if key not in seen:
                seen.add(key)
                spans.append(span)
    return spans


class HeuristicBackend(ClassifierBackend):
    """
    Rule-based v0 classifier. No network calls, no API key required.

    Scoring: each weighted pattern that matches contributes to a running
    score. Score > 0 -> likely, score < 0 -> unlikely, score == 0 (no
    signal at all) -> unclear. Confidence is derived from the magnitude
    of the score and the number of independent signals found, clamped to
    [0, 1] — this is an honest heuristic confidence, not a calibrated
    probability.
    """

    def classify(self, posting_text: str) -> ClassificationResult:
        text = posting_text

        no_sponsor_hits = _find_matches(text, [p for p, _ in _NO_SPONSOR_PATTERNS])
        sponsor_hits = _find_matches(text, [p for p, _ in _SPONSOR_PATTERNS])

        score = 0.0
        evidence: list[str] = []
        seen_evidence: set[str] = set()

        for pat, weight in _NO_SPONSOR_PATTERNS:
            m = pat.search(text)
            if m:
                score += weight
                span = m.group(0)
                if span.lower() not in seen_evidence:
                    seen_evidence.add(span.lower())
                    evidence.append(span)

        for pat, weight in _SPONSOR_PATTERNS:
            m = pat.search(text)
            if m:
                score += weight
                span = m.group(0)
                if span.lower() not in seen_evidence:
                    seen_evidence.add(span.lower())
                    evidence.append(span)

        n_signals = len(no_sponsor_hits) + len(sponsor_hits)

        if n_signals == 0:
            likelihood: SponsorshipLikelihood = "unclear"
            confidence = 0.2  # low confidence: no explicit signal found
        elif score > 0:
            likelihood = "likely"
            confidence = min(1.0, 0.5 + 0.15 * n_signals + 0.05 * score)
        elif score < 0:
            likelihood = "unlikely"
            confidence = min(1.0, 0.5 + 0.15 * n_signals + 0.05 * abs(score))
        else:
            # equal and opposite signals fired -> genuinely mixed/ambiguous
            likelihood = "unclear"
            confidence = 0.35

        categories: list[CategoryTag] = []

        on_campus_matches = _find_matches(text, _ON_CAMPUS_PATTERNS)
        if on_campus_matches:
            categories.append(CategoryTag(
                category="on_campus_only",
                reason=(
                    "Posting describes an on-campus role; on-campus work by "
                    "study-permit holders is exempt from the off-campus "
                    "24-hr/week cap (canada.ca/work-off-campus)."
                ),
            ))
            for s in on_campus_matches:
                if s.lower() not in seen_evidence:
                    seen_evidence.add(s.lower())
                    evidence.append(s)

        co_op_matches = _find_matches(text, _CO_OP_PATTERNS)
        if co_op_matches:
            categories.append(CategoryTag(
                category="co_op_exempt_eligible",
                reason=(
                    "Posting is a co-op/internship/placement role; as of "
                    "Apr 1, 2026, eligible placements <=50% of program "
                    "length no longer require a separate co-op work permit "
                    "(CIC News, 2026-04)."
                ),
            ))
            for s in co_op_matches:
                if s.lower() not in seen_evidence:
                    seen_evidence.add(s.lower())
                    evidence.append(s)

        pgwp_matches = _find_matches(text, _PGWP_PATTERNS)
        if pgwp_matches:
            categories.append(CategoryTag(
                category="pgwp_track",
                reason=(
                    "Posting describes a full-time/permanent career-track "
                    "role consistent with PGWP-relevant post-grad "
                    "employment; PGWP field-of-study list was frozen for "
                    "2026 (CIC News, 2026-01)."
                ),
            ))
            for s in pgwp_matches:
                if s.lower() not in seen_evidence:
                    seen_evidence.add(s.lower())
                    evidence.append(s)

        unrestricted_matches = _find_matches(text, _UNRESTRICTED_PATTERNS)
        if unrestricted_matches:
            categories.append(CategoryTag(
                category="unrestricted",
                reason=(
                    "Posting explicitly states it accepts candidates with "
                    "any legal work status / open work permit, i.e. not "
                    "gated to a specific permit category."
                ),
            ))
            for s in unrestricted_matches:
                if s.lower() not in seen_evidence:
                    seen_evidence.add(s.lower())
                    evidence.append(s)

        return ClassificationResult(
            sponsorship_likelihood=likelihood,
            confidence=confidence,
            evidence_spans=evidence,
            eligible_categories=categories,
        )
