#!/usr/bin/env python3
"""
rank.py -- ClearPath Phase 2 (v1): eligibility-aware ranked job list.

Fuses three already-shipped, independently-tested ClearPath modules --
imported and reused, never re-derived:
  - classifier/backend.py's HeuristicBackend -- sponsorship_likelihood +
    eligible_categories per job posting (Phase 0).
  - checker/eligibility.py's Profile/assess() -- the candidate's *current*
    personal compliance / PGWP-eligibility state (Phase 1).
  - qa/retrieval.py's embedding model choice (all-MiniLM-L6-v2, local,
    no API key) -- we import the model name constant to guarantee we embed
    with the exact same model as the Q&A tool, not re-declare it. We do
    NOT reuse `RetrievalIndex` itself: that class is coupled to the
    chunk-corpus / refusal-threshold retrieval flow, which doesn't apply to
    free-text resume<->posting similarity -- forcing that fit would be a
    worse abstraction than a small, separate embedding call here.

Why this isn't "just another resume matcher": see matcher/README.md's
"Not a generic resume matcher" section. The differentiator is
`assess_compatibility()` below -- fusing semantic fit with the Phase 0
classifier's sponsorship/eligibility tags and the Phase 1 checker's
personal compliance state as an explained, three-state compatibility gate.
No plain cosine-similarity resume matcher does that combination.

No API key, no LLM call, no persistence -- pure local embeddings + rule
logic, same free-tier-by-construction discipline as every prior module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sibling-module imports (checker/, classifier/, qa/ are not Python-path
# siblings of matcher/ by default -- add them explicitly, same convention
# checker/corpus_loader.py uses for file paths, applied here to imports).
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _sub in ("classifier", "checker", "qa"):
    _p = os.path.join(_REPO_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import CategoryTag, ClassificationResult, ClassifierBackend, HeuristicBackend  # noqa: E402
from eligibility import AssessmentReport, Profile, RuleResult, Status, assess  # noqa: E402
from retrieval import MODEL_NAME as EMBEDDING_MODEL_NAME  # noqa: E402

from resume_profile import ResumeProfile  # noqa: E402  (same-directory import)


# ---------------------------------------------------------------------------
# Compatibility model
# ---------------------------------------------------------------------------


class Compatibility(str, Enum):
    COMPATIBLE = "compatible"
    UNCLEAR = "unclear"
    INCOMPATIBLE = "incompatible"


# Sort priority within the eligibility-compatibility gate: compatible first,
# then unclear, then incompatible surfaced last -- see README "Ranking logic".
_VERDICT_PRIORITY: Dict[Compatibility, int] = {
    Compatibility.COMPATIBLE: 0,
    Compatibility.UNCLEAR: 1,
    Compatibility.INCOMPATIBLE: 2,
}

# Adjusted-score penalty applied within a compatibility bucket when the
# classifier's sponsorship_likelihood is "unlikely" -- keeps a
# compatible-but-unlikely-sponsorship posting visible (never dropped) while
# still ranking it below otherwise-similar compatible postings. See
# README "Ranking logic" for the exact rationale and worked example.
SPONSORSHIP_UNLIKELY_PENALTY = 0.15


@dataclass
class CategoryCompatibility:
    category: str
    verdict: Compatibility
    reason: str
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "citations": self.citations,
        }


def _pgwp_compatibility(checker_report: AssessmentReport) -> CategoryCompatibility:
    """Maps the checker's pgwp_general_eligibility + pgwp_field_of_study
    results onto a single compatible/unclear/incompatible verdict for the
    `pgwp_track` posting category.

    Collapsing two RuleResults (each 4-state: COMPLIANT / VIOLATION /
    NOT_ENOUGH_INFO / NOT_RESOLVED) into one 3-state compatibility verdict,
    documented reasoning:
      - Either result VIOLATION -> INCOMPATIBLE. A checker-confirmed
        eligibility violation (an exclusion trigger, a failed length/
        window/permit-validity check) means this profile is not currently
        PGWP-eligible, regardless of the other result's status.
      - Both results COMPLIANT -> COMPATIBLE. Only when *both* the general
        checklist and the field-of-study branch clear does the checker
        actually confirm PGWP eligibility.
      - Otherwise (any NOT_ENOUGH_INFO / NOT_RESOLVED, no VIOLATION) ->
        UNCLEAR. This is the common real case for a currently-enrolled
        student (e.g. Ujjwal's own real profile: no PGWP application date
        yet because he hasn't graduated) -- the checker cannot yet confirm
        or deny PGWP eligibility, so the honest answer is "unclear," not a
        guess in either direction. Mirrors the checker's own
        NOT_ENOUGH_INFO discipline (see checker/README.md).
    """
    by_rule = {r.rule: r for r in checker_report.results}
    general = by_rule.get("pgwp_general_eligibility")
    fos = by_rule.get("pgwp_field_of_study")
    relevant = [r for r in (general, fos) if r is not None]

    citations = sorted({cid for r in relevant for cid in r.citations})

    violations = [r for r in relevant if r.status == Status.VIOLATION]
    if violations:
        reason = "Not currently PGWP-eligible per the compliance checker: " + " ".join(
            r.explanation for r in violations
        )
        return CategoryCompatibility("pgwp_track", Compatibility.INCOMPATIBLE, reason, citations)

    if relevant and all(r.status == Status.COMPLIANT for r in relevant):
        reason = (
            "Checker assesses this profile as meeting both general PGWP eligibility "
            "(program length, full-time status, application window, permit validity, no "
            "exclusions) and the field-of-study requirement."
        )
        return CategoryCompatibility("pgwp_track", Compatibility.COMPATIBLE, reason, citations)

    unresolved = [r for r in relevant if r.status in (Status.NOT_ENOUGH_INFO, Status.NOT_RESOLVED)]
    if unresolved:
        reason = "PGWP eligibility isn't fully resolved by the compliance checker yet: " + " ".join(
            r.explanation for r in unresolved
        )
    else:
        reason = "PGWP eligibility could not be determined from the checker profile."
    return CategoryCompatibility("pgwp_track", Compatibility.UNCLEAR, reason, citations)


def _on_campus_compatibility() -> CategoryCompatibility:
    """on_campus_only is always COMPATIBLE: on-campus work has no hour
    restriction and no other gating condition for a study-permit holder in
    good standing (on_campus_unlimited) -- it doesn't depend on the
    checker's weekly-hours or PGWP assessments at all."""
    return CategoryCompatibility(
        "on_campus_only",
        Compatibility.COMPATIBLE,
        "On-campus work has no hour restriction and is available to any study-permit "
        "holder in good standing, independent of weekly off-campus hours or PGWP status.",
        ["on_campus_unlimited"],
    )


def _co_op_compatibility() -> CategoryCompatibility:
    """co_op_exempt_eligible is always UNCLEAR: whether a specific
    placement qualifies for the April 2026 co-op work-permit exemption
    depends on program structure (a DLI-required placement totaling <=50%
    of the program) that this checker's Profile does not capture at all --
    there is no field for it. Rather than silently assume yes/no, this is
    treated the same way the checker treats a structurally out-of-scope
    case (its own NOT_RESOLVED status) -- an honest "can't determine from
    this profile," not a guess."""
    return CategoryCompatibility(
        "co_op_exempt_eligible",
        Compatibility.UNCLEAR,
        "Whether a specific placement qualifies for the co-op work-permit exemption depends "
        "on program structure (a DLI-required placement totaling <=50% of the program length) "
        "that this compliance profile doesn't capture -- confirm with your co-op/experiential "
        "learning office.",
        ["co_op_exemption_2026"],
    )


def _unrestricted_compatibility() -> CategoryCompatibility:
    """unrestricted is always COMPATIBLE: the posting itself states it
    accepts any legal work status, so by construction it isn't gated by a
    specific student work-authorization category the checker would need to
    verify."""
    return CategoryCompatibility(
        "unrestricted",
        Compatibility.COMPATIBLE,
        "Posting explicitly states it accepts any legal work status, so it isn't gated by a "
        "specific student work-authorization category.",
        [],
    )


_CATEGORY_HANDLERS = {
    "on_campus_only": lambda report: _on_campus_compatibility(),
    "co_op_exempt_eligible": lambda report: _co_op_compatibility(),
    "pgwp_track": _pgwp_compatibility,
    "unrestricted": lambda report: _unrestricted_compatibility(),
}


def assess_compatibility(
    cls: ClassificationResult, checker_report: AssessmentReport
) -> Tuple[Compatibility, str, List[CategoryCompatibility]]:
    """Maps a posting's classifier output + the candidate's checker report
    onto one overall compatibility verdict, an explanatory reason, and the
    per-category breakdown that produced it.

    Default case -- posting has NO eligible_categories tagged at all (the
    classifier found no on-campus / co-op / PGWP-track / unrestricted
    signal in the posting text):
      - sponsorship_likelihood == "unlikely" -> INCOMPATIBLE. There is no
        recognized student work-authorization category *and* the posting
        signals it won't sponsor/support work-permit authorization for
        this role -- no currently visible legal work-authorization route
        for a study-permit holder. (Satisfies NEXT-BUILD-SPEC.md's
        acceptance criterion: an unlikely-sponsorship posting with no
        compatible category is marked incompatible, not silently
        omitted.)
      - sponsorship_likelihood == "likely" -> UNCLEAR, not COMPATIBLE.
        A "likely to sponsor" signal on a posting with no matched category
        plausibly points at a separate LMIA-backed employer-sponsored work
        permit pathway -- real, but not one of the four categories this
        checker models (it only covers study-permit-based work
        authorization). Claiming COMPATIBLE here would overclaim what the
        checker actually verified; UNCLEAR says so honestly while still
        surfacing the positive signal in the reason text.
      - sponsorship_likelihood == "unclear" -> UNCLEAR. No signal in
        either direction.

    Otherwise: run every tagged category through `_CATEGORY_HANDLERS`, take
    the best (lowest-priority) verdict across tagged categories as the
    overall verdict -- if ANY tagged category is a real compatible pathway,
    the posting is realistically pursuable via that category, even if
    another tagged category on the same posting is unclear or
    incompatible. The reason string reports every category compatibility
    at the winning verdict level (there can be more than one), never just
    the first.
    """
    if not cls.eligible_categories:
        if cls.sponsorship_likelihood == "unlikely":
            reason = (
                "Posting doesn't match any of the recognized student work-authorization "
                "categories (on-campus, co-op-exempt, PGWP-track, or explicitly unrestricted), "
                "and it signals it won't sponsor/support work-permit authorization for this "
                "role -- no currently visible legal work-authorization route for a study-permit "
                "holder."
            )
            cc = CategoryCompatibility("(none)", Compatibility.INCOMPATIBLE, reason, [])
            return Compatibility.INCOMPATIBLE, reason, [cc]

        reason = (
            "Posting doesn't clearly state which student work-authorization category it falls "
            "under, so eligibility can't be determined from the compliance profile alone."
        )
        if cls.sponsorship_likelihood == "likely":
            reason += (
                " The posting does signal openness to sponsorship, but that pathway (e.g. an "
                "employer-sponsored LMIA-backed work permit) isn't modeled by this checker, "
                "which only covers study-permit-based work-authorization categories -- treat "
                "this as a lead worth investigating directly, not a confirmed compatibility."
            )
        cc = CategoryCompatibility("(none)", Compatibility.UNCLEAR, reason, [])
        return Compatibility.UNCLEAR, reason, [cc]

    per_category = [_CATEGORY_HANDLERS[t.category](checker_report) for t in cls.eligible_categories]
    best_priority = min(_VERDICT_PRIORITY[c.verdict] for c in per_category)
    overall = next(v for v, p in _VERDICT_PRIORITY.items() if p == best_priority)
    winning = [c for c in per_category if c.verdict == overall]
    reason = " / ".join(f"{c.category}: {c.reason}" for c in winning)
    return overall, reason, per_category


# ---------------------------------------------------------------------------
# Semantic similarity layer (reuses the Q&A tool's embedding model choice)
# ---------------------------------------------------------------------------

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        # Imported lazily so `python rank.py --help` doesn't pay the
        # torch/sentence-transformers import cost -- same discipline
        # qa/retrieval.py already uses for the same reason.
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _MODEL


def compute_fit_scores(resume: ResumeProfile, postings: List[dict]) -> Dict[str, float]:
    """Embeds the resume's raw_text once and every posting's text once
    (single batched `encode` call), returns {posting_id: cosine_similarity}.

    Cosine similarity via dot product of L2-normalized embeddings, same
    pattern as qa/retrieval.py's RetrievalIndex.retrieve()."""
    model = _get_model()
    texts = [resume.raw_text] + [p["text"] for p in postings]
    embs = np.asarray(model.encode(texts, normalize_embeddings=True))
    resume_emb = embs[0]
    posting_embs = embs[1:]
    scores = posting_embs @ resume_emb
    return {p["id"]: float(s) for p, s in zip(postings, scores)}


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------


@dataclass
class RankedResult:
    posting_id: str
    rank: int
    fit_score: float
    adjusted_score: float
    sponsorship_likelihood: str
    classifier_confidence: float
    evidence_spans: List[str]
    eligible_categories: List[dict]
    category_compatibilities: List[CategoryCompatibility]
    overall_compatibility: Compatibility
    overall_reason: str
    posting_text: str

    def to_dict(self) -> dict:
        return {
            "posting_id": self.posting_id,
            "rank": self.rank,
            "fit_score": round(self.fit_score, 4),
            "adjusted_score": round(self.adjusted_score, 4),
            "sponsorship_likelihood": self.sponsorship_likelihood,
            "classifier_confidence": round(self.classifier_confidence, 2),
            "evidence_spans": self.evidence_spans,
            "eligible_categories": self.eligible_categories,
            "category_compatibilities": [c.to_dict() for c in self.category_compatibilities],
            "overall_compatibility": self.overall_compatibility.value,
            "overall_reason": self.overall_reason,
        }


def rank(
    resume: ResumeProfile,
    eligibility_profile: Profile,
    postings: List[dict],
    classifier: Optional[ClassifierBackend] = None,
    as_of_date: Optional[date] = None,
) -> List[RankedResult]:
    """resume/eligibility profiles fused against a list of {"id","text"}
    postings -> a ranked List[RankedResult].

    Ranking logic (see README "Ranking logic" for the full worked
    rationale): primary key is the eligibility-compatibility bucket
    (compatible, then unclear, then incompatible -- never dropped);
    secondary key within a bucket is `adjusted_score` descending, where
    adjusted_score = semantic fit_score minus a SPONSORSHIP_UNLIKELY_PENALTY
    (0.15) if the classifier's sponsorship_likelihood is "unlikely" -- this
    keeps a compatible-but-unlikely-sponsorship posting visible in its
    bucket while still ranking it below otherwise-similar compatible
    postings whose sponsorship signal is neutral or positive.
    """
    classifier = classifier or HeuristicBackend()
    checker_report = assess(eligibility_profile, as_of_date=as_of_date)
    fit_scores = compute_fit_scores(resume, postings)

    results: List[RankedResult] = []
    for p in postings:
        cls = classifier.classify(p["text"])
        overall, reason, per_cat = assess_compatibility(cls, checker_report)
        fit = fit_scores[p["id"]]
        penalty = SPONSORSHIP_UNLIKELY_PENALTY if cls.sponsorship_likelihood == "unlikely" else 0.0
        adjusted = fit - penalty
        results.append(
            RankedResult(
                posting_id=p["id"],
                rank=0,
                fit_score=fit,
                adjusted_score=adjusted,
                sponsorship_likelihood=cls.sponsorship_likelihood,
                classifier_confidence=cls.confidence,
                evidence_spans=cls.evidence_spans,
                eligible_categories=[{"category": t.category, "reason": t.reason} for t in cls.eligible_categories],
                category_compatibilities=per_cat,
                overall_compatibility=overall,
                overall_reason=reason,
                posting_text=p["text"],
            )
        )

    results.sort(key=lambda r: (_VERDICT_PRIORITY[r.overall_compatibility], -r.adjusted_score))
    for i, r in enumerate(results, start=1):
        r.rank = i
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_postings(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": p["id"], "text": p["text"]} for p in data]


def format_results(results: List[RankedResult], top: Optional[int] = None) -> str:
    shown = results[:top] if top else results
    lines = [f"ClearPath Phase 2 -- eligibility-aware ranked job list ({len(shown)}/{len(results)} shown)"]
    lines.append("=" * 78)
    for r in shown:
        lines.append(
            f"#{r.rank}  [{r.overall_compatibility.value.upper()}]  posting {r.posting_id}  "
            f"fit={r.fit_score:.3f} adjusted={r.adjusted_score:.3f} "
            f"sponsorship={r.sponsorship_likelihood} (conf {r.classifier_confidence:.2f})"
        )
        lines.append(f"    Why: {r.overall_reason}")
        if r.evidence_spans:
            lines.append(f"    Evidence: {', '.join(r.evidence_spans)}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ClearPath Phase 2 (v1): eligibility-aware ranked job list. "
        "Local embeddings + rule logic only -- no API key, no LLM call."
    )
    parser.add_argument("--resume-profile", required=True, help="Path to a resume profile JSON.")
    parser.add_argument("--eligibility-profile", required=True, help="Path to a checker Profile JSON.")
    parser.add_argument("--postings", required=True, help="Path to a JSON list of {id, text, ...} postings.")
    parser.add_argument("--as-of-date", default=None, help="Override as-of date (YYYY-MM-DD) for the checker.")
    parser.add_argument("--top", type=int, default=None, help="Only show the top N results.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of formatted text.")
    args = parser.parse_args(argv)

    resume = ResumeProfile.load(args.resume_profile)
    with open(args.eligibility_profile, "r", encoding="utf-8") as f:
        elig_profile = Profile.from_dict(json.load(f))
    postings = _load_postings(args.postings)
    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None

    results = rank(resume, elig_profile, postings, as_of_date=as_of)

    if args.json:
        shown = results[: args.top] if args.top else results
        print(json.dumps([r.to_dict() for r in shown], indent=2))
    else:
        print(format_results(results, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
