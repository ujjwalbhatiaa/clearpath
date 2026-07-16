"""
Unit tests for the ClearPath heuristic classifier backend.

Run with: python -m pytest tests/ -v
(or python -m unittest discover -s tests -v if pytest isn't installed)
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import HeuristicBackend, ClassificationResult  # noqa: E402


class TestHeuristicBackendBasics(unittest.TestCase):
    def setUp(self):
        self.backend = HeuristicBackend()

    def test_returns_classification_result(self):
        result = self.backend.classify("A plain job posting with no signal.")
        self.assertIsInstance(result, ClassificationResult)

    def test_confidence_in_valid_range(self):
        cases = [
            "We do not sponsor employment visas.",
            "Sponsorship is available for the right candidate.",
            "A totally neutral job description with no relevant language at all.",
        ]
        for text in cases:
            result = self.backend.classify(text)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_sponsorship_likelihood_is_valid_label(self):
        result = self.backend.classify("Some posting text.")
        self.assertIn(result.sponsorship_likelihood, {"likely", "unlikely", "unclear"})


class TestEvidenceSpansAreRealSubstrings(unittest.TestCase):
    """The spec requires evidence_spans to never be fabricated -- every span
    must be an exact substring of the input text. This is the single most
    important correctness property of the classifier."""

    def setUp(self):
        self.backend = HeuristicBackend()

    def test_evidence_spans_are_substrings_no_sponsor(self):
        text = "We do not sponsor employment visas. Must be a Canadian citizen or permanent resident."
        result = self.backend.classify(text)
        self.assertTrue(len(result.evidence_spans) > 0)
        for span in result.evidence_spans:
            self.assertIn(span, text, f"Evidence span {span!r} is not a substring of the input text")

    def test_evidence_spans_are_substrings_sponsor(self):
        text = "Sponsorship is available and we welcome international students are encouraged to apply."
        result = self.backend.classify(text)
        for span in result.evidence_spans:
            self.assertIn(span, text)

    def test_evidence_spans_are_substrings_on_all_labeled_examples(self):
        """Run the substring guarantee across the entire hand-labeled eval
        set as a regression check, not just hand-picked examples."""
        labels_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "eval",
            "labeled_postings.json",
        )
        with open(labels_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            result = self.backend.classify(item["text"])
            for span in result.evidence_spans:
                self.assertIn(
                    span,
                    item["text"],
                    f"[{item['id']}] evidence span {span!r} not found verbatim in posting text",
                )


class TestClearSponsorshipCases(unittest.TestCase):
    def setUp(self):
        self.backend = HeuristicBackend()

    def test_explicit_no_sponsor_statement(self):
        result = self.backend.classify(
            "We do not sponsor employment visas or work permits for this role."
        )
        self.assertEqual(result.sponsorship_likelihood, "unlikely")

    def test_citizens_or_pr_only(self):
        result = self.backend.classify(
            "Applicants must be a Canadian citizen or permanent resident to be considered."
        )
        self.assertEqual(result.sponsorship_likelihood, "unlikely")

    def test_explicit_sponsor_statement(self):
        result = self.backend.classify(
            "Sponsorship is available for the right candidate. We have sponsored work permits before."
        )
        self.assertEqual(result.sponsorship_likelihood, "likely")

    def test_no_signal_is_unclear(self):
        result = self.backend.classify(
            "We are looking for a friendly, organized office coordinator."
        )
        self.assertEqual(result.sponsorship_likelihood, "unclear")
        self.assertLess(result.confidence, 0.5)


class TestEligibleCategoryTagging(unittest.TestCase):
    def setUp(self):
        self.backend = HeuristicBackend()

    def test_on_campus_tag(self):
        result = self.backend.classify(
            "The University Library is hiring a part-time on-campus Circulation Assistant."
        )
        categories = {t.category for t in result.eligible_categories}
        self.assertIn("on_campus_only", categories)

    def test_co_op_tag(self):
        result = self.backend.classify("Summer Co-op Student, Data Analytics team.")
        categories = {t.category for t in result.eligible_categories}
        self.assertIn("co_op_exempt_eligible", categories)

    def test_pgwp_tag(self):
        result = self.backend.classify(
            "This is a full-time, permanent Business Analyst role in our downtown office."
        )
        categories = {t.category for t in result.eligible_categories}
        self.assertIn("pgwp_track", categories)

    def test_unrestricted_tag(self):
        result = self.backend.classify(
            "We welcome applicants of any legal work status, including open work permit holders."
        )
        categories = {t.category for t in result.eligible_categories}
        self.assertIn("unrestricted", categories)

    def test_every_category_has_a_reason(self):
        result = self.backend.classify(
            "On-campus Research Assistant Co-op position, full-time permanent after graduation."
        )
        for tag in result.eligible_categories:
            self.assertTrue(tag.reason.strip(), f"Category {tag.category} has an empty reason")

    def test_no_category_signal_produces_empty_list(self):
        result = self.backend.classify(
            "Seasonal Sales Associate needed for the holiday rush."
        )
        self.assertEqual(result.eligible_categories, [])


class TestCLIEndToEnd(unittest.TestCase):
    """Smoke test that classify.py's CLI plumbing (arg parsing, backend
    selection, JSON serialization) works, not just the library code."""

    def test_cli_produces_valid_json_via_subprocess(self):
        import subprocess

        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classify.py"
        )
        proc = subprocess.run(
            [sys.executable, script, "--text", "We do not sponsor employment visas."],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        data = json.loads(proc.stdout)
        self.assertIn("sponsorship_likelihood", data)
        self.assertIn("confidence", data)
        self.assertIn("evidence_spans", data)
        self.assertIn("eligible_categories", data)
        self.assertEqual(data["sponsorship_likelihood"], "unlikely")


if __name__ == "__main__":
    unittest.main()
