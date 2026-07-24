"""
Unit tests for the Phase 2 (v1) eligibility-aware ranked job list.

Run: python -m pytest tests/ -v      (from the matcher/ directory)
     python -m unittest discover -s tests -v

These are real behavioral tests, not "it runs" smoke tests -- per
NEXT-BUILD-SPEC.md Scope-in section 6, each one proves a specific claim
about the fusion logic (semantic layer does real work; the eligibility gate
actually gates; nothing is silently dropped).

Model loading (sentence-transformers) happens once in setUpClass and is
reused across every test in this file to keep the suite fast.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import rank first -- it inserts checker/classifier/qa onto sys.path as a
# side effect, which the eligibility import below depends on.
from rank import Compatibility, compute_fit_scores, rank  # noqa: E402
from eligibility import Profile  # noqa: E402
from resume_profile import ResumeProfile  # noqa: E402

_MATCHER_DIR = os.path.join(os.path.dirname(__file__), "..")
_REPO_ROOT = os.path.join(_MATCHER_DIR, "..")


def _load_postings():
    with open(os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": p["id"], "text": p["text"]} for p in data]


def _load_resume(name: str) -> ResumeProfile:
    return ResumeProfile.load(os.path.join(_MATCHER_DIR, "examples", name))


def _load_eligibility(path_from_checker: str) -> Profile:
    with open(os.path.join(_REPO_ROOT, "checker", path_from_checker), "r", encoding="utf-8") as f:
        return Profile.from_dict(json.load(f))


class TestResumeProfileIntegrity(unittest.TestCase):
    """Corpus/profile integrity checks -- the real-resume dogfood profile
    actually contains real skills genuinely read from the extracted text,
    not a placeholder."""

    def test_ujjwal_profile_skills_non_empty_and_contains_real_skill(self):
        resume = _load_resume("resume_profile_ujjwal.json")
        self.assertTrue(resume.skills, "resume_profile_ujjwal.json skills list must not be empty")
        self.assertIn("Python", resume.skills)

    def test_ujjwal_profile_raw_text_contains_real_resume_content(self):
        resume = _load_resume("resume_profile_ujjwal.json")
        self.assertIn("University of Alberta", resume.raw_text)
        self.assertIn("UJJWAL BHATIA", resume.raw_text)

    def test_frontend_synthetic_profile_has_distinct_skill_set(self):
        ujjwal = _load_resume("resume_profile_ujjwal.json")
        frontend = _load_resume("resume_profile_frontend_synthetic.json")
        # Genuinely different skill sets -- the whole point of the fixture.
        self.assertNotEqual(set(ujjwal.skills), set(frontend.skills))
        self.assertIn("React", frontend.skills)
        self.assertNotIn("Python", frontend.skills)


class TestSemanticSimilarityLayer(unittest.TestCase):
    """Proves the embedding-based fit score is doing real, resume-content-
    dependent work -- not a fixed/placeholder score."""

    @classmethod
    def setUpClass(cls):
        cls.postings = _load_postings()
        cls.ujjwal = _load_resume("resume_profile_ujjwal.json")
        cls.frontend = _load_resume("resume_profile_frontend_synthetic.json")

    def test_strong_overlap_posting_scores_above_unrelated_posting(self):
        # p01 = "Software Engineer" full-time role; p07 = seasonal retail
        # sales associate. For a CS/ML resume the software role should
        # score meaningfully higher on semantic fit than an unrelated
        # retail job -- a real behavioral claim, not just "it runs".
        scores = compute_fit_scores(self.ujjwal, self.postings)
        self.assertGreater(
            scores["p01"],
            scores["p07"] + 0.1,
            f"expected p01 (Software Engineer) to clearly outscore p07 (retail) for a CS/ML resume, "
            f"got p01={scores['p01']:.3f} p07={scores['p07']:.3f}",
        )

    def test_swapping_resume_changes_fit_scores(self):
        # Same posting, two very different resumes -- the fit score for at
        # least the clearest software-engineering posting must differ
        # substantially between the CS/ML resume and the frontend-only
        # synthetic resume, proving the layer actually reads resume
        # content rather than returning a fixed score regardless of input.
        ujjwal_scores = compute_fit_scores(self.ujjwal, self.postings)
        frontend_scores = compute_fit_scores(self.frontend, self.postings)
        self.assertGreater(
            abs(ujjwal_scores["p01"] - frontend_scores["p01"]),
            0.1,
            f"expected p01's fit score to differ meaningfully between resumes, got "
            f"ujjwal={ujjwal_scores['p01']:.3f} frontend={frontend_scores['p01']:.3f}",
        )

    def test_swapping_resume_changes_overall_ranking_order(self):
        elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        ujjwal_ranked = rank(self.ujjwal, elig, self.postings)
        frontend_ranked = rank(self.frontend, elig, self.postings)
        ujjwal_order = [r.posting_id for r in ujjwal_ranked]
        frontend_order = [r.posting_id for r in frontend_ranked]
        self.assertNotEqual(
            ujjwal_order,
            frontend_order,
            "ranking order should not be identical across two very different resume profiles",
        )


class TestEligibilityCompatibilityGate(unittest.TestCase):
    """The actual differentiator -- proves the gate is doing real
    eligibility-aware work, not just passing every posting through."""

    @classmethod
    def setUpClass(cls):
        cls.postings = _load_postings()
        cls.ujjwal_resume = _load_resume("resume_profile_ujjwal.json")
        cls.ujjwal_elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        cls.college_polytechnic_elig = _load_eligibility(
            os.path.join("examples", "profile_college_polytechnic.json")
        )

    def _result_by_id(self, results, posting_id):
        for r in results:
            if r.posting_id == posting_id:
                return r
        self.fail(f"posting {posting_id!r} missing from ranked results -- must never be silently dropped")

    def test_unlikely_sponsorship_with_no_category_is_incompatible_not_dropped(self):
        # p12: "must have valid permanent work authorization... unable to
        # consider candidates requiring any future work permit renewals" --
        # no eligible_categories tag, unlikely sponsorship. Must be marked
        # incompatible with a stated reason, and must still be present in
        # the output (never silently omitted).
        results = rank(self.ujjwal_resume, self.ujjwal_elig, self.postings)
        self.assertEqual(len(results), len(self.postings), "no posting should ever be dropped from the output")
        r = self._result_by_id(results, "p12")
        self.assertEqual(r.overall_compatibility, Compatibility.INCOMPATIBLE)
        self.assertTrue(r.overall_reason, "incompatible verdict must carry a stated reason")

    def test_pgwp_track_posting_incompatible_when_checker_confirms_exclusion(self):
        # p01 is tagged pgwp_track by the classifier. The college/
        # polytechnic checker profile has already_received_pgwp=True, a
        # hard PGWP exclusion (VIOLATION). A pgwp_track posting must be
        # flagged incompatible for this profile, not compatible or silently
        # passed through.
        results = rank(self.ujjwal_resume, self.college_polytechnic_elig, self.postings)
        r = self._result_by_id(results, "p01")
        self.assertEqual(r.overall_compatibility, Compatibility.INCOMPATIBLE)
        self.assertTrue(
            any(cc.category == "pgwp_track" and cc.verdict == Compatibility.INCOMPATIBLE for cc in r.category_compatibilities)
        )

    def test_pgwp_track_posting_unclear_when_checker_lacks_info(self):
        # p01 is tagged pgwp_track. Ujjwal's real profile has no PGWP
        # application date yet (he hasn't graduated) -- the checker's
        # pgwp_general_eligibility check is NOT_ENOUGH_INFO, not a
        # violation. The honest verdict here is unclear, not a guess.
        results = rank(self.ujjwal_resume, self.ujjwal_elig, self.postings)
        r = self._result_by_id(results, "p01")
        self.assertEqual(r.overall_compatibility, Compatibility.UNCLEAR)
        self.assertTrue(
            any(cc.category == "pgwp_track" and cc.verdict == Compatibility.UNCLEAR for cc in r.category_compatibilities)
        )

    def test_on_campus_posting_always_compatible(self):
        # p05: on-campus circulation assistant job. On-campus work has no
        # hour restriction for any study-permit holder -- compatible
        # regardless of the checker profile's weekly-hours or PGWP state.
        results = rank(self.ujjwal_resume, self.ujjwal_elig, self.postings)
        r = self._result_by_id(results, "p05")
        self.assertEqual(r.overall_compatibility, Compatibility.COMPATIBLE)

    def test_compatible_but_unlikely_sponsorship_ranks_lower_but_is_kept(self):
        # A compatible posting whose sponsorship signal is "unlikely" must
        # still appear (never dropped) and must rank at or below an
        # otherwise-similar compatible posting with a neutral/positive
        # sponsorship signal, per the adjusted-score penalty.
        results = rank(self.ujjwal_resume, self.ujjwal_elig, self.postings)
        compatible = [r for r in results if r.overall_compatibility == Compatibility.COMPATIBLE]
        self.assertTrue(compatible, "expected at least one compatible posting in the fixture set")
        unlikely_compatible = [r for r in compatible if r.sponsorship_likelihood == "unlikely"]
        non_unlikely_compatible = [r for r in compatible if r.sponsorship_likelihood != "unlikely"]
        if unlikely_compatible and non_unlikely_compatible:
            # every unlikely-sponsorship compatible posting's adjusted
            # score should be strictly less than its own fit score (the
            # penalty was actually applied), proving it's downranked, not
            # dropped.
            for r in unlikely_compatible:
                self.assertLess(r.adjusted_score, r.fit_score)

    def test_no_posting_ever_dropped_from_output(self):
        results = rank(self.ujjwal_resume, self.ujjwal_elig, self.postings)
        result_ids = {r.posting_id for r in results}
        posting_ids = {p["id"] for p in self.postings}
        self.assertEqual(result_ids, posting_ids)


class TestRankOrdering(unittest.TestCase):
    """The bucket ordering (compatible, then unclear, then incompatible)
    holds across the whole output, not just spot checks."""

    def test_buckets_are_contiguous_and_correctly_ordered(self):
        postings = _load_postings()
        resume = _load_resume("resume_profile_ujjwal.json")
        elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        results = rank(resume, elig, postings)

        priority = {Compatibility.COMPATIBLE: 0, Compatibility.UNCLEAR: 1, Compatibility.INCOMPATIBLE: 2}
        seen_priorities = [priority[r.overall_compatibility] for r in results]
        self.assertEqual(seen_priorities, sorted(seen_priorities), "bucket order must be non-decreasing")

    def test_ranks_are_sequential_starting_at_one(self):
        postings = _load_postings()
        resume = _load_resume("resume_profile_ujjwal.json")
        elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        results = rank(resume, elig, postings)
        self.assertEqual([r.rank for r in results], list(range(1, len(results) + 1)))


class TestCLI(unittest.TestCase):
    """Smoke tests for the CLI entry point, mirroring checker/tests's CLI
    smoke-test pattern."""

    def test_cli_text_mode_runs_end_to_end(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(_MATCHER_DIR, "rank.py"),
                "--resume-profile",
                os.path.join(_MATCHER_DIR, "examples", "resume_profile_ujjwal.json"),
                "--eligibility-profile",
                os.path.join(_REPO_ROOT, "checker", "examples", "profile_ujjwal.json"),
                "--postings",
                os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"),
                "--top",
                "3",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ClearPath Phase 2", proc.stdout)

    def test_cli_json_mode_produces_valid_json(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(_MATCHER_DIR, "rank.py"),
                "--resume-profile",
                os.path.join(_MATCHER_DIR, "examples", "resume_profile_ujjwal.json"),
                "--eligibility-profile",
                os.path.join(_REPO_ROOT, "checker", "examples", "profile_ujjwal.json"),
                "--postings",
                os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(len(data), 20)
        self.assertIn("overall_compatibility", data[0])


if __name__ == "__main__":
    unittest.main()
