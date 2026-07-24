"""
corpus_loader.py -- merges the Phase 1 Q&A grounding corpus
(../qa/corpus/chunks.json) with this checker's own additional chunks
(rules_corpus.json) into one id-keyed lookup, so the eligibility rule
engine (eligibility.py) can cite a chunk id from either file without
duplicating any government source text.

Why two files instead of one: NEXT-BUILD-SPEC.md left the choice open but
required "both files must derive from one source of truth or clearly
cross-reference each other." Extending qa/corpus/chunks.json in place was
considered and rejected -- the Q&A tool's existing unit tests
(qa/tests/test_qa.py) hard-code expected top-1 retrieval results for
specific questions, and adding chunks with overlapping vocabulary
(on-campus vs. off-campus, "20 hours" vs "24 hours") risks silently
shifting embedding rankings and breaking those already-shipped,
already-verified tests. Keeping the checker's new chunks in a separate
file avoids touching the Q&A tool's corpus/tests at all, while still
reusing (not duplicating) the two existing chunks the checker also needs
(pgwp_general_eligibility, pgwp_field_of_study_exempt) directly by id.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

QA_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "qa", "corpus", "chunks.json")
CHECKER_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "rules_corpus.json")


@dataclass
class Chunk:
    id: str
    tags: List[str]
    source_title: str
    source_url: str
    date_modified: str
    text: str


def _load_one(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**c) for c in data["chunks"]]


def load_merged_corpus(
    qa_path: str = QA_CORPUS_PATH, checker_path: str = CHECKER_CORPUS_PATH
) -> Dict[str, Chunk]:
    """Loads both corpus files and returns a single {chunk_id: Chunk} dict.
    Raises if a chunk id in the checker's own file collides with one
    already defined in the Q&A corpus -- that would mean a fact got
    silently forked instead of reused, which this module exists to
    prevent.
    """
    chunks: Dict[str, Chunk] = {}
    for c in _load_one(qa_path):
        chunks[c.id] = c
    for c in _load_one(checker_path):
        if c.id in chunks:
            raise ValueError(
                f"chunk id collision between qa corpus and checker corpus: {c.id!r} "
                "-- reuse the existing chunk by id instead of redefining it"
            )
        chunks[c.id] = c
    return chunks


# Loaded once at import time -- at ~11 total chunks this is effectively free,
# and every caller in this package (eligibility.py, check.py, tests) wants
# the same merged view.
CHUNKS_BY_ID: Dict[str, Chunk] = load_merged_corpus()
