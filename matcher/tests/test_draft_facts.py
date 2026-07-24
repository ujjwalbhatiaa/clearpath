"""
Unit tests for the Phase 3a compliance-fact grounding layer.

Run: python -m pytest tests/ -v      (from the matcher/ directory)
     python -m unittest discover -s tests -v

Real behavioral tests, not "it runs" smoke tests -- per NEXT-BUILD-SPEC.md
Scope-in section 5, each proves a specific claim about attribution
correctness, most importantly that a NOT-yet-confirmed eligibility state
never leaks into `allowed_claims`.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import rank first -- it inserts checker/classifier/qa onto sys.path as a
# side effect, which the eligibility import below (transitively, via
# draft_facts) depends on.
from rank import Compatibility, rank  # noqa: E402
from draft_facts import (  # noqa: E402
    Claim,
    FactPacket,
    assemble_fact_packet,
    compute_skill_overlap,
    packet_is_well_formed,
)
from eligibility import Profile, assess  # noqa: E402
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


def _packet_for(resume, elig_profile, postings, posting_id, as_of_date=None):
    ranked = rank(resume, elig_profile, postings, as_of_date=as_of_date)
    match = next(r for r in ranked if r.posting_id == posting_id)
    checker_report = assess(elig_profile, as_of_date=as_of_date)
    return assemble_fact_packet(match, resume, checker_report)


class TestSkillOverlapHelper(unittest.TestCase):
    """Direct unit tests of the new keyword-overlap helper -- no existing
    signal for "which specific skills overlap" existed before this
    increment (rank.py's compute_fit_scores is whole-text cosine
    similarity only)."""

    def test_matches_real_substring_case_insensitively(self):
        resume = ResumeProfile(name="t", raw_text="", skills=["Python", "React"])
        overlap = compute_skill_overlap(resume, "Looking for a python developer.")
        self.assertIn("Python", overlap)
        self.assertNotIn("React", overlap)

    def test_no_partial_word_false_positive(self):
        # "Java" must not match inside "JavaScript" -- word-boundary
        # matching, not naive substring search.
        resume = ResumeProfile(name="t", raw_text="", skills=["Java"])
        overlap = compute_skill_overlap(resume, "We use JavaScript extensively.")
        self.assertEqual(overlap, [])

    def test_empty_skill_entries_ignored(self):
        resume = ResumeProfile(name="t", raw_text="", skills=["", "  ", "Python"])
        overlap = compute_skill_overlap(resume, "Python required.")
        self.assertEqual(overlap, ["Python"])


class TestSkillOverlapDiffersAcrossResumeProfiles(unittest.TestCase):
    """Proves skill_overlap reads real resume content -- not a hardcoded
    or fixed value regardless of which resume profile is passed in. The
    shared 20-posting eval corpus (Phase 0's hand-labeled sponsorship-
    language set) doesn't happen to mention specific tech skills by name,
    so this uses a purpose-built posting text, same fixture-construction
    approach test_rank.py used for the frontend-synthetic resume."""

    @classmethod
    def setUpClass(cls):
        cls.ujjwal_resume = _load_resume("resume_profile_ujjwal.json")
        cls.frontend_resume = _load_resume("resume_profile_frontend_synthetic.json")
        cls.elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        cls.posting = {
            "id": "skilltest01",
            "text": (
                "We are hiring a developer skilled in Python, pandas, and React to join our "
                "small product team. Sponsorship is available for the right candidate."
            ),
        }

    def _packet(self, resume):
        ranked = rank(resume, self.elig, [self.posting])
        checker_report = assess(self.elig)
        return assemble_fact_packet(ranked[0], resume, checker_report)

    def test_skill_overlap_differs_between_ujjwal_and_frontend_profile_same_posting(self):
        ujjwal_packet = self._packet(self.ujjwal_resume)
        frontend_packet = self._packet(self.frontend_resume)

        self.assertIn("Python", ujjwal_packet.skill_overlap)
        self.assertIn("pandas", ujjwal_packet.skill_overlap)
        self.assertNotIn("React", ujjwal_packet.skill_overlap)

        self.assertIn("React", frontend_packet.skill_overlap)
        self.assertNotIn("Python", frontend_packet.skill_overlap)

        self.assertNotEqual(set(ujjwal_packet.skill_overlap), set(frontend_packet.skill_overlap))


class TestPGWPForbiddenClaims(unittest.TestCase):
    """The single most important behavior in this increment (per
    NEXT-BUILD-SPEC.md's acceptance criteria): a not-yet-confirmed or
    excluded PGWP eligibility state must never leak into allowed_claims."""

    @classmethod
    def setUpClass(cls):
        cls.postings = _load_postings()
        cls.ujjwal_resume = _load_resume("resume_profile_ujjwal.json")
        cls.ujjwal_elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        cls.college_polytechnic_elig = _load_eligibility(
            os.path.join("examples", "profile_college_polytechnic.json")
        )

    def test_pgwp_not_enough_info_lands_in_forbidden_not_allowed(self):
        # p01 is tagged pgwp_track. Ujjwal's real profile has no PGWP
        # application date yet (NOT_ENOUGH_INFO, not a violation) -- the
        # honest packet must forbid the PGWP claim, and must NOT allow it.
        packet = _packet_for(self.ujjwal_resume, self.ujjwal_elig, self.postings, "p01")

        allowed_mentions_pgwp = any("pgwp_track" in c.text for c in packet.allowed_claims)
        forbidden_mentions_pgwp = any("pgwp_track" in c.text for c in packet.forbidden_claims)

        self.assertFalse(allowed_mentions_pgwp, "PGWP claim must not appear in allowed_claims when checker status is NOT_ENOUGH_INFO")
        self.assertTrue(forbidden_mentions_pgwp, "PGWP claim must appear in forbidden_claims when checker status is NOT_ENOUGH_INFO")

        pgwp_forbidden = [c for c in packet.forbidden_claims if "pgwp_track" in c.text]
        self.assertTrue(any(c.rule_result_status == "not_enough_info" for c in pgwp_forbidden))

    def test_pgwp_exclusion_violation_produces_explicit_forbidden_claim_citing_exclusion(self):
        # p01 against the college/polytechnic profile: already_received_pgwp
        # = True, a confirmed hard exclusion (VIOLATION). Must produce an
        # explicit forbidden claim citing pgwp_not_eligible_exclusions.
        packet = _packet_for(
            self.ujjwal_resume, self.college_polytechnic_elig, self.postings, "p01"
        )

        pgwp_forbidden = [c for c in packet.forbidden_claims if "pgwp_track" in c.text]
        self.assertTrue(pgwp_forbidden, "expected at least one forbidden claim about the pgwp_track pathway")
        self.assertTrue(
            any(
                c.citation and "pgwp_not_eligible_exclusions" in c.citation and c.rule_result_status == "violation"
                for c in pgwp_forbidden
            ),
            f"expected a forbidden claim citing pgwp_not_eligible_exclusions with violation status, got: {pgwp_forbidden}",
        )
        allowed_mentions_pgwp = any("pgwp_track" in c.text for c in packet.allowed_claims)
        self.assertFalse(allowed_mentions_pgwp)


class TestOnCampusAllowedClaim(unittest.TestCase):
    """on_campus_only is always COMPATIBLE (checker-independent) -- proves
    the allowed claim carries the checker's *actual* citation string, not a
    paraphrase."""

    def test_on_campus_allowed_claim_has_real_checker_citation(self):
        postings = _load_postings()
        resume = _load_resume("resume_profile_ujjwal.json")
        elig = _load_eligibility(os.path.join("examples", "profile_ujjwal.json"))
        packet = _packet_for(resume, elig, postings, "p05")  # on_campus_only posting

        on_campus_allowed = [c for c in packet.allowed_claims if "on_campus_only" in c.text]
        self.assertTrue(on_campus_allowed, "expected an allowed claim for the on_campus_only pathway")
        self.assertTrue(
            any(c.citation == "on_campus_unlimited" for c in on_campus_allowed),
            f"expected the exact checker citation string 'on_campus_unlimited', got: "
            f"{[c.citation for c in on_campus_allowed]}",
        )
        # Must not also be forbidden.
        on_campus_forbidden = [c for c in packet.forbidden_claims if "on_campus_only" in c.text]
        self.assertEqual(on_campus_forbidden, [])


class TestFullCorpusWellFormedness(unittest.TestCase):
    """The non-negotiable correctness property for this increment: every
    fact packet produced across the full labeled corpus, against both
    example resume profiles paired with both example checker profiles (4
    pairings x 20 postings = 80 packets), passes packet_is_well_formed with
    zero integrity problems."""

    def test_all_packets_well_formed_across_full_corpus_and_all_profile_pairings(self):
        postings = _load_postings()
        resumes = {
            "ujjwal": _load_resume("resume_profile_ujjwal.json"),
            "frontend": _load_resume("resume_profile_frontend_synthetic.json"),
        }
        eligs = {
            "ujjwal": _load_eligibility(os.path.join("examples", "profile_ujjwal.json")),
            "college_polytechnic": _load_eligibility(
                os.path.join("examples", "profile_college_polytechnic.json")
            ),
        }

        total_checked = 0
        for resume_key, resume in resumes.items():
            for elig_key, elig in eligs.items():
                ranked = rank(resume, elig, postings)
                checker_report = assess(elig)
                self.assertEqual(len(ranked), len(postings))
                for ranked_result in ranked:
                    packet = assemble_fact_packet(ranked_result, resume, checker_report)
                    problems = packet_is_well_formed(packet, postings=postings)
                    self.assertEqual(
                        problems,
                        [],
                        f"integrity problems for resume={resume_key!r} elig={elig_key!r} "
                        f"posting={ranked_result.posting_id!r}: {problems}",
                    )
                    total_checked += 1

        self.assertEqual(total_checked, len(postings) * len(resumes) * len(eligs))


class TestPacketIsWellFormedValidator(unittest.TestCase):
    """Direct unit tests of the validator itself -- proves it actually
    catches each class of problem it claims to catch, not just that it
    returns [] on well-formed input."""

    def test_flags_allowed_claim_with_no_citation_or_status(self):
        packet = FactPacket(
            posting_id="p01",
            resume_profile_id="r1",
            allowed_claims=[Claim(text="unsupported claim", source_type="matcher")],
        )
        problems = packet_is_well_formed(packet)
        self.assertTrue(any("no citation or rule_result_status" in p for p in problems))

    def test_flags_claim_in_both_allowed_and_forbidden(self):
        dup_text = "duplicate claim text"
        packet = FactPacket(
            posting_id="p01",
            resume_profile_id="r1",
            allowed_claims=[Claim(text=dup_text, source_type="matcher", rule_result_status="compatible")],
            forbidden_claims=[Claim(text=dup_text, source_type="matcher", rule_result_status="incompatible")],
        )
        problems = packet_is_well_formed(packet)
        self.assertTrue(any("both allowed_claims and forbidden_claims" in p for p in problems))

    def test_flags_posting_id_not_in_corpus(self):
        packet = FactPacket(posting_id="does-not-exist", resume_profile_id="r1")
        problems = packet_is_well_formed(packet, postings=[{"id": "p01", "text": "x"}])
        self.assertTrue(any("does not resolve" in p for p in problems))

    def test_flags_empty_posting_id_and_empty_resume_profile_id(self):
        packet = FactPacket(posting_id="", resume_profile_id="")
        problems = packet_is_well_formed(packet)
        self.assertTrue(any("posting_id is empty" in p for p in problems))
        self.assertTrue(any("resume_profile_id is empty" in p for p in problems))

    def test_well_formed_packet_returns_no_problems(self):
        packet = FactPacket(
            posting_id="p01",
            resume_profile_id="r1",
            allowed_claims=[Claim(text="ok claim", source_type="checker", citation="some_rule")],
            forbidden_claims=[Claim(text="do not claim x", source_type="checker", citation="some_rule")],
        )
        problems = packet_is_well_formed(packet, postings=[{"id": "p01", "text": "x"}])
        self.assertEqual(problems, [])


class TestCLI(unittest.TestCase):
    """Smoke tests for the CLI entry point, mirroring rank.py's / checker's
    CLI smoke-test pattern."""

    def test_cli_text_mode_runs_end_to_end(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(_MATCHER_DIR, "draft_facts.py"),
                "--resume-profile",
                os.path.join(_MATCHER_DIR, "examples", "resume_profile_ujjwal.json"),
                "--eligibility-profile",
                os.path.join(_REPO_ROOT, "checker", "examples", "profile_ujjwal.json"),
                "--postings",
                os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"),
                "--posting-id",
                "p01",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ClearPath Phase 3a", proc.stdout)
        self.assertIn("FORBIDDEN CLAIMS", proc.stdout)

    def test_cli_json_mode_produces_valid_well_formed_json(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(_MATCHER_DIR, "draft_facts.py"),
                "--resume-profile",
                os.path.join(_MATCHER_DIR, "examples", "resume_profile_ujjwal.json"),
                "--eligibility-profile",
                os.path.join(_REPO_ROOT, "checker", "examples", "profile_ujjwal.json"),
                "--postings",
                os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"),
                "--posting-id",
                "p05",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["posting_id"], "p05")
        self.assertIn("allowed_claims", data)
        self.assertIn("forbidden_claims", data)

    def test_cli_unknown_posting_id_fails_cleanly(self):
        import subprocess

        proc = subprocess.run(
            [
                sys.executable,
                os.path.join(_MATCHER_DIR, "draft_facts.py"),
                "--resume-profile",
                os.path.join(_MATCHER_DIR, "examples", "resume_profile_ujjwal.json"),
                "--eligibility-profile",
                os.path.join(_REPO_ROOT, "checker", "examples", "profile_ujjwal.json"),
                "--postings",
                os.path.join(_REPO_ROOT, "eval", "labeled_postings.json"),
                "--posting-id",
                "does-not-exist",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
