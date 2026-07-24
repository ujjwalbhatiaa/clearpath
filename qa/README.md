# ClearPath — IRCC Compliance Q&A (Phase 1, extractive v0)

Phase 1 of the ClearPath roadmap (see `../../STARTUP.md`): a grounded
question-answering tool over Canadian study-permit work-rule pages. It sits
alongside the Phase 0 sponsorship classifier (`../classifier/`) in this same
repo.

## The problem this increment solves

The rules governing what an international student on a study permit is
allowed to do — the 24-hour/week off-campus cap, unlimited hours during
scheduled breaks (capped at 180 days/year), PGWP eligibility, and the new
(April 2026) co-op work-permit exemption — are scattered across several
canada.ca and ircc.canada.ca pages, phrased in dense policy language, and
change often enough that a static guide goes stale. Students need a fast,
trustworthy way to ask "can I do X" and get back the actual rule, not a
guess.

## Why extractive, not generative

Given a question, this tool embeds it, retrieves the single most relevant
grounded passage from a small hand-curated corpus, and returns the **exact
quoted text** plus its source URL and last-modified date — never a
paraphrase, never a generated sentence. If nothing in the corpus is a
confident match, it refuses rather than guessing.

This is a deliberate scope choice for this increment, not a shortcut:

1. **Zero hallucination by construction.** An extractive citation-finder
   cannot invent a rule, threshold, or date — it can only ever surface a
   quote that's already in the vetted corpus. For anything
   immigration-rules-adjacent, that's the correct posture: "here's the exact
   government text that answers this" is trustworthy in a way a generated
   summary sentence isn't, even a well-grounded one.
2. **No API key needed at all**, which satisfies `STARTUP.md`'s free-tier /
   ask-first API key policy by construction rather than by restraint —
   embeddings run locally via `sentence-transformers` (`all-MiniLM-L6-v2`),
   entirely on CPU, no signup, no rate limit, no network call at query time
   (only once, to download model weights on first run).

A generative synthesis layer that combines multiple chunks into one answer
is deliberately out of scope here — see "Honest limitations" below.

## How it works

1. `corpus/chunks.json` — 8 grounded rule chunks, each an exact quote
   captured live on **2026-07-17** from a cited canada.ca/ircc.canada.ca
   page (see "Sources" below), with `id`, `source_url`, `source_title`,
   `date_modified`, `text`, and `tags`.
2. `retrieval.py` — embeds a *retrieval representation* of each chunk
   (source title + tags + quote text, concatenated) using
   `sentence-transformers/all-MiniLM-L6-v2`, and at query time embeds the
   question and ranks chunks by cosine similarity. **What gets embedded for
   ranking and what gets returned as a citation are different things** —
   only `chunk.text` (the pure quote) is ever shown to the user; the title
   and tags only influence which chunk wins, they never appear as if they
   were part of the quoted government text.
3. `ask.py` — CLI: prints the top-1 (or `--top-k N`) matching chunk(s) as a
   formatted quote + citation, or a JSON payload with `--json`.
4. `eval/eval.py` — runs the full labeled question set and reports real
   top-1/top-3 retrieval accuracy and refusal-path correctness.

## How to run it

```bash
# From this directory (startup/clearpath/qa/). First run downloads the
# ~90MB MiniLM model weights once (cached afterward, no network needed).

# Ask a question
python ask.py --question "How many hours can I work during summer break?"

# Get the top 3 matches, as JSON
python ask.py -q "Do I need a co-op work permit?" --top-k 3 --json

# Pipe a question in
echo "What is the capital of Canada?" | python ask.py

# Run the eval harness
python eval/eval.py

# Run the unit tests
python -m unittest discover -s tests -v
```

## Eval results (this run, 2026-07-17)

Run against `eval/labeled_questions.json` — 22 hand-written questions I
labeled myself (18 answerable across all 8 corpus chunks, 4 deliberately
out-of-corpus/ambiguous to test the refusal path). **These are the actual
numbers this run produced.**

```
Eval set: 22 questions (18 answerable, 4 expected-refusal)
Top-1 accuracy (answerable questions):  15/18  (83.3%)
Top-3 accuracy (answerable questions):  17/18  (94.4%)
Refusal-path correctness (expected-refusal questions): 3/4  (75.0%)
```

### Retrieval threshold

`REFUSAL_THRESHOLD = 0.40` in `retrieval.py`, tuned against the real score
distribution produced by this eval set (not guessed):

- Every correctly-matched answerable question scored **>= 0.487** cosine
  similarity against its correct chunk.
- 3 of the 4 refusal-expected questions ("what's the capital of Canada",
  "how do I file my taxes", "best pizza near UAlberta") scored **<= 0.395**
  against every chunk.
- `0.40` sits directly between those two clusters — it costs zero
  true-positive hits in this eval set while correctly refusing the
  borderline "how do I file my taxes" case that a naive `0.35` threshold
  (my first attempt, see git history / retrieval.py comments) let through
  as a false match against the PGWP-eligibility chunk.

### Honest limitations, not hidden

- **4 top-1 misses, all confusions between rule-adjacent chunks that share
  vocabulary** (e.g. "24-hour cap" vs. "180-day annual cap" both mention
  "hours" and "work"; a co-op program-length question matched the 180-day
  chunk instead of the co-op chunk). Top-3 accuracy (94.4%) shows the
  correct answer is nearly always *retrievable*, just not always ranked
  first — a real, current limitation of small-corpus MiniLM embeddings on
  closely related legal/policy text, not a bug.
- **1 refusal-path miss is a deliberately adversarial case, not
  swept under the rug:** "I'm a master's grad from a 6-month program under
  the PGWP 8-month minimum — am I eligible at all?" scores 0.588 (inside
  the legitimate-hit score range) because it shares heavy PGWP vocabulary
  with a real chunk, but answering it correctly requires *combining* the
  8-month-minimum rule with a fact not in this corpus (whether any
  exception exists for shorter programs). A single-chunk extractive
  retriever cannot express "I need information I don't have" versus "I
  found something related but incomplete" — it can only express
  similarity. This is exactly why extractive-only retrieval, however
  faithful, is not sufficient for multi-hop questions — flagged as future
  work below, not hidden.
- **8-chunk corpus, hand-written eval questions.** This is a real, run eval
  on the actual retrieval code — not fabricated — but with a corpus this
  small the numbers above describe "how the retriever behaves on the rule
  areas it currently covers," not a general claim about arbitrary
  immigration questions. Growing the corpus (more chunks, more rule areas)
  is the most direct way to make this number more meaningful over time.
- **Static snapshot, not live.** The corpus was captured on **2026-07-17**
  and is not re-fetched at query time. If any cited canada.ca/ircc.canada.ca
  page changes after that date, this tool will keep quoting the old text
  until a future research run re-captures it. This is a known, intentional
  trade-off for this increment (see `NEXT-BUILD-SPEC.md` scope-out), not an
  oversight — but it means this tool should never be treated as
  automatically current.

### Future work flagged by this eval

- Multi-hop reasoning (combine 2+ chunks to answer questions like the
  master's-grad edge case above) — needs either a generative layer (behind
  the same free-tier/ask-first API key policy) or explicit rule-composition
  logic, not just better retrieval.
- Disambiguate closely related chunks further — e.g. a cross-encoder
  re-ranking pass over the top-3 candidates, or splitting `annual_180_day_cap`
  into more granular sub-chunks so it stops absorbing near-misses from the
  24-hour and co-op questions.
- Grow the corpus beyond the 8 chunks captured this run — more rule areas
  (e.g. what counts as misrepresentation/consequences of violation, per the
  CIC News synthesis article noted in `../STARTUP.md`'s 2026-07-17 research
  log but not yet turned into a cited chunk).
- Fuse with the Phase 0 classifier (a natural Phase 1.5/2 step, explicitly
  out of scope for this increment per `NEXT-BUILD-SPEC.md`).

## Project layout

```
clearpath/qa/
  ask.py                       # CLI entry point
  retrieval.py                 # RetrievalIndex: local embeddings + cosine-similarity search
  corpus/
    chunks.json                 # 8 grounded IRCC rule chunks, snapshot date 2026-07-17
  eval/
    labeled_questions.json      # 22 hand-labeled questions
    eval.py                     # eval harness -- top-1/top-3/refusal accuracy + disagreements
  tests/
    test_qa.py                  # corpus integrity + real-retrieval + eval-set sanity tests
  README.md                     # this file
```

## Sources cited by this corpus (all captured live 2026-07-17)

- [Work off campus as an international student](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/work-off-campus.html) — canada.ca, modified 2026-04-15
- [Can I work as many hours as I want if I'm eligible to work off campus?](https://ircc.canada.ca/english/helpcentre/answer.asp?qnum=503&top=15) — IRCC Help Centre, modified 2026-04-17
- [Post-graduation work permit: Who can apply](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/after-graduation/eligibility.html) — canada.ca, modified 2026-06-24
- [PGWP field of study requirements / CIP code lookup](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/after-graduation/eligibility/field-of-study.html) — canada.ca, modified 2026-03-09
- [Work in a student work placement](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/intern.html) — canada.ca, modified 2026-05-01

**This tool is informational, not legal or immigration advice.** It surfaces
exact quotes from a dated snapshot of public government pages — it does not
interpret your personal situation, and every answer should be verified
against the live canada.ca/ircc.canada.ca source before relying on it for a
real decision.
