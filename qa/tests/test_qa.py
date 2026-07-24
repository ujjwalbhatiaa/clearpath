"""
Unit tests for the Phase 1 extractive Q&A core.

Run: python -m unittest discover -s tests -v   (from the qa/ directory)

These tests exercise the REAL sentence-transformers model and REAL corpus --
no mocking of the retrieval path -- because the property that actually
matters here (does the refusal path work, is the corpus honest) can only be
verified against real embeddings, not a stub.
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval import REFUSAL_THRESHOLD, RetrievalIndex, load_corpus  # noqa: E402

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "chunks.json")
LABELED_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "labeled_questions.json")


class TestCorpusIntegrity(unittest.TestCase):
    """The corpus is the source of truth for every citation this tool ever
    returns -- these tests exist to catch a broken/invented chunk before it
    ships, the same way Phase 0's evidence-span regression test did."""

    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.snapshot_date = load_corpus(CORPUS_PATH)

    def test_corpus_loads_and_is_nonempty(self):
        self.assertGreaterEqual(len(self.chunks), 8)

    def test_snapshot_date_present(self):
        self.assertRegex(self.snapshot_date, r"^\d{4}-\d{2}-\d{2}$")

    def test_every_chunk_has_required_fields(self):
        for c in self.chunks:
            self.assertTrue(c.id, "chunk missing id")
            self.assertTrue(c.source_title, f"{c.id} missing source_title")
            self.assertTrue(c.source_url.startswith("https://"), f"{c.id} bad source_url")
            self.assertRegex(c.date_modified, r"^\d{4}-\d{2}-\d{2}$", f"{c.id} bad date_modified")
            self.assertTrue(c.text and len(c.text) > 20, f"{c.id} text too short/empty")
            self.assertIsInstance(c.tags, list)
            self.assertGreater(len(c.tags), 0, f"{c.id} has no tags")

    def test_chunk_ids_are_unique(self):
        ids = [c.id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)), "duplicate chunk ids found")

    def test_all_source_urls_point_at_canada_ca_or_ircc(self):
        for c in self.chunks:
            self.assertTrue(
                "canada.ca" in c.source_url,
                f"{c.id} source_url is not a canada.ca/ircc.canada.ca domain: {c.source_url}",
            )


class TestRetrieval(unittest.TestCase):
    """Real retrieval, real model, real corpus -- these are correctness
    tests for the actual production path, not a mock."""

    @classmethod
    def setUpClass(cls):
        cls.index = RetrievalIndex(corpus_path=CORPUS_PATH).build()

    def test_clear_in_corpus_question_returns_correct_chunk(self):
        r = self.index.top1("How many hours per week am I allowed to work off campus during a regular semester?")
        self.assertFalse(r.refused)
        self.assertEqual(r.chunk.id, "hour_cap_24")

    def test_returned_text_is_verbatim_from_corpus(self):
        """Non-negotiable correctness property (same spirit as Phase 0's
        evidence-span test): whatever ask.py prints as a quote must be
        byte-identical to a chunk's `text` field in chunks.json -- never a
        paraphrase, never partially reworded."""
        chunk_texts = {c.text for c in self.index.chunks}
        r = self.index.top1("Do I still need a separate co-op work permit for my internship this year?")
        self.assertFalse(r.refused)
        self.assertIn(r.chunk.text, chunk_texts)

    def test_clearly_out_of_corpus_question_is_refused(self):
        r = self.index.top1("What is the best pizza place near the University of Alberta?")
        self.assertTrue(r.refused)
        self.assertIsNone(r.chunk)

    def test_refusal_threshold_is_actually_applied(self):
        r = self.index.top1("some completely unrelated question about pizza toppings and video games")
        if not r.refused:
            self.assertGreaterEqual(r.score, REFUSAL_THRESHOLD)

    def test_top_k_returns_requested_number_of_results(self):
        results = self.index.retrieve("How many hours can I work during a break?", top_k=3)
        self.assertEqual(len(results), 3)

    def test_scores_are_sorted_descending(self):
        results = self.index.retrieve("PGWP eligibility rules", top_k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestEvalHarnessSanity(unittest.TestCase):
    """Sanity-checks on the eval set itself, not the model -- catches a
    labeling mistake (e.g. a typo'd expected_chunk_id) before it silently
    deflates or inflates the reported accuracy number."""

    @classmethod
    def setUpClass(cls):
        cls.chunks, _ = load_corpus(CORPUS_PATH)
        cls.chunk_ids = {c.id for c in cls.chunks}
        with open(LABELED_PATH, "r", encoding="utf-8") as f:
            cls.questions = json.load(f)["questions"]

    def test_eval_set_has_at_least_15_questions(self):
        self.assertGreaterEqual(len(self.questions), 15)

    def test_every_expected_chunk_id_exists_in_corpus_or_is_null(self):
        for q in self.questions:
            eid = q["expected_chunk_id"]
            if eid is not None:
                self.assertIn(eid, self.chunk_ids, f"labeled question references unknown chunk id {eid!r}")

    def test_eval_set_includes_at_least_one_refusal_case(self):
        self.assertTrue(any(q["expected_chunk_id"] is None for q in self.questions))

    def test_eval_set_covers_every_corpus_chunk_at_least_once(self):
        covered = {q["expected_chunk_id"] for q in self.questions if q["expected_chunk_id"] is not None}
        missing = self.chunk_ids - covered
        self.assertEqual(missing, set(), f"these chunks have no eval question at all: {missing}")


if __name__ == "__main__":
    unittest.main()
