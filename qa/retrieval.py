"""
retrieval.py -- local, no-API-key extractive retrieval core for ClearPath's
Phase 1 IRCC compliance Q&A.

Design (see NEXT-BUILD-SPEC.md / README.md for the full rationale):
- Extractive only: given a question, embed it, find the nearest grounded
  corpus chunk by cosine similarity, and return that chunk's EXACT quoted
  text plus its source URL and modified-date. Never paraphrase, never
  generate a new sentence.
- Local embeddings only: sentence-transformers/all-MiniLM-L6-v2 runs on
  CPU, needs no API key, no signup, and (after the one-time model
  download) no network access at query time.
- Below-threshold queries are refused rather than answered with a
  low-confidence guess -- see REFUSAL_THRESHOLD below.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus", "chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"

# Cosine-similarity threshold below which we refuse to answer rather than
# return a low-confidence match. Tuned against the full eval set in
# eval/labeled_questions.json after the retrieval representation below was
# finalized (see README.md "Retrieval threshold" for the full table):
# every correctly-matched answerable question in the eval set scored
# >= 0.487, while 3 of 4 deliberately out-of-corpus/refusal-expected
# questions scored <= 0.395. 0.40 sits directly between those two clusters
# -- it doesn't cost a single true-positive hit in the eval set, and it
# correctly refuses the "how do I file my taxes" case that a lower
# threshold (0.35) let through. The 4th refusal case (a deliberately
# adversarial multi-hop question, see README.md) scores 0.588 -- inside
# the legitimate-hit range -- and is NOT reliably catchable by a
# similarity threshold alone; documented as an honest limitation rather
# than tuned away.
REFUSAL_THRESHOLD = 0.40


@dataclass
class Chunk:
    id: str
    tags: List[str]
    source_title: str
    source_url: str
    date_modified: str
    text: str


@dataclass
class RetrievalResult:
    chunk: Optional[Chunk]
    score: float
    refused: bool


def load_corpus(path: str = CORPUS_PATH) -> Tuple[List[Chunk], str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = [Chunk(**c) for c in data["chunks"]]
    snapshot_date = data.get("snapshot_date", "unknown")
    return chunks, snapshot_date


class RetrievalIndex:
    """Embeds the corpus once, then answers queries by cosine similarity.

    Embeddings are recomputed at startup (no persisted vector DB) -- at
    8-15 chunks this takes well under a second on CPU and keeps the
    implementation simple, per NEXT-BUILD-SPEC.md's explicit "no need for a
    real vector DB yet" note.
    """

    def __init__(self, corpus_path: str = CORPUS_PATH, model_name: str = MODEL_NAME):
        self.chunks, self.snapshot_date = load_corpus(corpus_path)
        self._model = None
        self._model_name = model_name
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so that things like `python ask.py --help`
            # don't pay the torch/sentence-transformers import cost.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    @staticmethod
    def _retrieval_representation(chunk: Chunk) -> str:
        """Text actually embedded for similarity search.

        This is deliberately richer than `chunk.text` alone -- it prepends
        the source title and tags so the embedding captures which rule
        area a chunk belongs to, not just its raw sentence content. Several
        chunks share a lot of surface vocabulary (hours, PGWP, work permit)
        so bare-quote embeddings confused adjacent-but-different rules in
        early testing (documented in README.md eval notes). What actually
        gets *returned and cited* to the user is always `chunk.text` --
        this representation only affects retrieval ranking, never the
        quoted output, so it can't introduce a fabricated citation.
        """
        tags = ", ".join(chunk.tags)
        return f"{chunk.source_title}. Topics: {tags}. {chunk.text}"

    def build(self):
        self._ensure_model()
        texts = [self._retrieval_representation(c) for c in self.chunks]
        embs = self._model.encode(texts, normalize_embeddings=True)
        self._embeddings = np.asarray(embs)
        return self

    def retrieve(self, question: str, top_k: int = 1) -> List[RetrievalResult]:
        if self._embeddings is None:
            self.build()
        self._ensure_model()
        q_emb = self._model.encode([question], normalize_embeddings=True)[0]
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        scores = self._embeddings @ q_emb
        order = np.argsort(-scores)[:top_k]

        results: List[RetrievalResult] = []
        for idx in order:
            score = float(scores[idx])
            if score < REFUSAL_THRESHOLD:
                results.append(RetrievalResult(chunk=None, score=score, refused=True))
            else:
                results.append(
                    RetrievalResult(chunk=self.chunks[idx], score=score, refused=False)
                )
        return results

    def top1(self, question: str) -> RetrievalResult:
        return self.retrieve(question, top_k=1)[0]
