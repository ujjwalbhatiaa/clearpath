# ClearPath — Sponsorship/Eligibility Classifier (Phase 0)

ClearPath is an AI copilot for international students in Canada: it reads job
postings to flag realistic sponsorship/eligibility odds, and (in later
phases) tracks personal IRCC compliance status and drafts compliance-aware
applications. This is Phase 0 of the roadmap — see `../STARTUP.md` for the
full product spec, competitive landscape, and phase plan.

## The problem this increment solves

Job postings almost never say clearly whether an employer will consider or
sponsor a non-PR/non-citizen candidate. The signal — when it exists at all —
is buried in boilerplate legal language, phrased inconsistently, and easy to
miss when you're scanning dozens of postings a day. Separately, whether a
given role is even something a study-permit holder is allowed to take
depends on IRCC rules (on-campus vs. off-campus hour caps, co-op work permit
exemptions, PGWP eligibility) that most postings never mention at all.

This module takes raw job-posting text and produces a structured,
evidence-backed classification:

- **`sponsorship_likelihood`** — `likely` / `unlikely` / `unclear`, with a
  confidence score.
- **`evidence_spans`** — the exact quotes from the posting that drove the
  call. These are guaranteed to be real substrings of the input (never
  fabricated) — enforced by a dedicated test, see `tests/test_classify.py`.
- **`eligible_categories`** — a subset of `on_campus_only`,
  `co_op_exempt_eligible`, `pgwp_track`, `unrestricted`, each with a
  one-line reason grounded in the 2026 IRCC rules logged in `STARTUP.md`
  (not invented — every rule cites a canada.ca or CIC News source, see
  `classifier/backend.py` module docstring for the full citation list).

## Approach & backend

The classifier is built behind a `ClassifierBackend` abstract interface
(`classifier/backend.py`) so a free-tier LLM backend can be swapped in later
as a drop-in change without touching `classify.py` or the eval harness.

**This increment ships only `HeuristicBackend`** — a rule-based engine using
weighted regex patterns matched against real sponsorship-language and
eligibility-language phrasing seen in actual job postings (e.g. "we do not
sponsor", "must be a Canadian citizen or permanent resident", "sponsorship is
available", co-op/internship language, on-campus language, etc.). No LLM API
was used or activated for this increment — per the operating constraint in
`STARTUP.md` ("ask before using any API key"), no key has been requested or
used. See `BUILD-STATUS.md` for the current status of that decision.

Scoring: each matched pattern contributes a hand-assigned weight to a running
score; the sign of the total determines the label, and confidence is derived
from the number of independent signals found and the magnitude of the score,
clamped to `[0, 1]`. This is an honest heuristic confidence — **not** a
calibrated statistical probability.

## How to run it

```bash
# From this directory (startup/clearpath/):

# Classify a posting from a file
python classify.py --input examples/sample_posting.txt

# Classify posting text directly
python classify.py --text "We do not sponsor employment visas for this role."

# Pipe text in
cat some_posting.txt | python classify.py

# Run the eval harness against the labeled set
python eval/eval.py

# Run the unit tests
python -m unittest discover -s tests -v
```

## Eval results (this run, 2026-07-16)

Run against `eval/labeled_postings.json` — 20 hand-labeled postings I wrote
and labeled myself, spanning clear-sponsor, clear-no-sponsor, ambiguous, and
two deliberately adversarial edge cases (see below). **These are the actual
numbers this run produced** — nothing here is fabricated or rounded up.

```
Labeled examples: 20
Sponsorship-likelihood accuracy: 18/20 = 90.00%

Per-class (sponsorship_likelihood) precision / recall / F1:
  likely     support=5  P=1.00  R=1.00  F1=1.00
  unlikely   support=7  P=1.00  R=0.71  F1=0.83
  unclear    support=8  P=0.80  R=1.00  F1=0.89

Per-category (eligible_categories, multi-label) precision / recall / F1:
  on_campus_only           support=2  P=1.00  R=1.00  F1=1.00
  co_op_exempt_eligible    support=4  P=1.00  R=1.00  F1=1.00
  pgwp_track               support=3  P=1.00  R=1.00  F1=1.00
  unrestricted              support=1  P=1.00  R=1.00  F1=1.00
```

### Honest limitations, not hidden

The two misses (`p19`, `p20` in `eval/labeled_postings.json`) are
**deliberately adversarial cases I added on purpose** to avoid reporting an
inflated, too-good-to-be-true number: past-tense negation ("we used to
sponsor... no longer do") and indirect negation ("not open to candidates who
require sponsorship"). The v0 regex patterns only match present-tense,
direct phrasing and correctly fall back to `unclear` (low confidence, no
evidence) rather than guessing — that's a real, documented coverage gap, not
a silent failure. Most of the other 18 examples are close paraphrases of
real posting language rather than a large, independently-sourced corpus, so
this number should be read as "the heuristic behaves as designed on the
patterns it was built for," not as a claim of real-world accuracy on
arbitrary postings — that claim can only be earned with a much larger,
independently collected eval set, which is future work once Phase 0 grows
past this seed.

### Future work flagged by this eval

- Add negation-aware matching (e.g. detect "not"/"no longer"/"used to" near
  a sponsor phrase and flip or discount the signal) rather than pure
  positive pattern matching.
- Grow the eval set with real scraped postings (Phase 1+, once a scraping
  pipeline exists) instead of hand-written examples, to get a genuine
  out-of-sample accuracy number.
- Swap in a free-tier LLM backend behind the same `ClassifierBackend`
  interface once one is approved (see `BUILD-STATUS.md`), and compare
  head-to-head against the heuristic baseline on this same eval set.

## Project layout

```
clearpath/
  classify.py                 # CLI entry point
  classifier/
    __init__.py
    backend.py                 # ClassifierBackend interface + HeuristicBackend
  eval/
    labeled_postings.json      # 20 hand-labeled job postings
    eval.py                    # eval harness -- accuracy/precision/recall + disagreements
  tests/
    test_classify.py           # unit tests incl. evidence-span substring guarantee
  examples/
    sample_posting.txt         # example input for --input
  README.md                    # this file
  SHIP-KIT.md                  # GitHub repo creation instructions (blocked, see below)
```

## IRCC rule sources cited by this classifier

- Off-campus 24-hr/week cap & on-campus exemption:
  https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/work-off-campus.html
- PGWP field-of-study list frozen for 2026:
  https://www.cicnews.com/2026/01/ircc-freezes-list-of-pgwp-eligible-fields-of-study-for-2026-0167305.html
- Co-op work permit exemption effective April 1, 2026:
  https://www.cicnews.com/2026/04/canada-moves-to-expand-work-authorization-for-international-students-and-graduates-0473917.html

**This tool is informational, not legal or immigration advice.** It reports
a confidence-scored signal grounded in cited public rules — it does not
guarantee an employer's actual willingness to sponsor, and every eligibility
claim should be verified against the current canada.ca source before relying
on it for a real decision.
