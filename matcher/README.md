# ClearPath — Eligibility-Aware Ranked Job List (Phase 2, v1)

Phase 2 of ClearPath's roadmap (see `../STARTUP.md` for the full product
spec): given a resume and a personal compliance profile, rank a list of job
postings by fit — but only among the postings this specific person can
currently, legally pursue, with every eligibility claim traced back to a
cited government rule. This is the first increment of Phase 2; see "Scope
and honest limitations" below for what's deliberately not built yet.

**Phase 3a (the compliance-fact grounding layer that a future application-
drafting step will be built on top of) now also lives in this directory —
see [`DRAFTING.md`](DRAFTING.md).** This document below covers Phase 2 only.

## Not a generic resume matcher

Plain resume↔job semantic matching (embed both, cosine-similarity, sort) is
a **saturated, generic portfolio pattern** — a quick search while planning
this increment turned up dozens of near-identical "AI resume matcher"
repos, all using the same recipe (see `../STARTUP.md`'s 2026-07-20 research
log entry for the specific finding and links). Building Phase 2 as a
standalone matcher would blend into that crowd, not stand out.

**This module is deliberately not that.** The differentiator is
`assess_compatibility()` in `rank.py`: it fuses the semantic fit score with
two already-shipped ClearPath modules —

- the Phase 0 sponsorship/eligibility **classifier** (`classifier/backend.py`),
  which tags each posting's `sponsorship_likelihood` and `eligible_categories`
  (`on_campus_only`, `co_op_exempt_eligible`, `pgwp_track`, `unrestricted`);
- the Phase 1 personal compliance/PGWP **checker** (`checker/eligibility.py`),
  which knows this specific candidate's *current* eligibility state (are they
  currently PGWP-eligible? still just a study-permit holder? unresolved
  pending more info?);

into a three-state eligibility-compatibility gate (`compatible` /
`unclear` / `incompatible`) with a plain-language, cited reason attached to
every verdict. **Rank by fit, but only among postings this specific person
can actually pursue right now, with every eligibility claim traced to a
cited government rule where one applies** — that combination is the wedge
`STARTUP.md`'s 2026-07-16 competitive-landscape research identified, and no
generic resume-matcher repo does it. This is a resume-narrative requirement
as much as a technical one, so it's stated here explicitly, not left
implicit.

## What it fuses (imported, never re-derived)

| Signal | Source | Reused via |
|---|---|---|
| Sponsorship likelihood + eligible categories per posting | Phase 0 classifier | `from backend import HeuristicBackend` |
| Candidate's current compliance/PGWP-eligibility state | Phase 1 checker | `from eligibility import Profile, assess` |
| Resume↔posting semantic fit | Phase 1 Q&A tool's embedding model choice | `from retrieval import MODEL_NAME` (same `all-MiniLM-L6-v2`, not re-declared) |

We deliberately do **not** reuse `qa/retrieval.py`'s `RetrievalIndex` class
directly — that class is coupled to the chunk-corpus / refusal-threshold
retrieval flow (built for "find the one grounded canada.ca chunk that
answers this question"), which doesn't fit free-text resume↔posting
similarity at all. Importing just the model-name constant guarantees we
embed with the exact same model as the Q&A tool (no config drift) without
forcing a bad abstraction fit.

## The eligibility-compatibility gate, in detail

For each posting, every `eligible_categories` tag the classifier fired is
mapped to a `compatible` / `unclear` / `incompatible` verdict:

- **`on_campus_only` → always `compatible`.** On-campus work has no hour
  restriction for any study-permit holder in good standing
  (`on_campus_unlimited`) — it doesn't depend on the checker's weekly-hours
  or PGWP assessments at all.
- **`unrestricted` → always `compatible`.** The posting itself states it
  accepts any legal work status, so by construction it isn't gated by a
  category the checker needs to verify.
- **`co_op_exempt_eligible` → always `unclear`.** Whether a specific
  placement qualifies for the April 2026 co-op work-permit exemption
  depends on program structure (a DLI-required placement ≤50% of the
  program) that the checker's `Profile` has no field for at all — this is a
  genuine, documented gap in what the checker can verify, not a guess in
  either direction. **Judgment call, flagged explicitly (per
  `NEXT-BUILD-SPEC.md`'s instruction to flag ambiguous mappings rather than
  guess silently):** a plausible alternative design would infer
  co-op-eligibility from `program_end_date` being in the future (i.e. "is
  this person still an active student at all"), but that would only rule
  out the clearly-graduated case and wouldn't verify the far more specific
  DLI/50%-of-program condition — it would look more confident than it
  actually is. `unclear` is the more honest choice.
- **`pgwp_track` → depends on the checker's `pgwp_general_eligibility` and
  `pgwp_field_of_study` results** (each already 4-state: `COMPLIANT` /
  `VIOLATION` / `NOT_ENOUGH_INFO` / `NOT_RESOLVED`), collapsed to 3 states:
  - either result is `VIOLATION` → **`incompatible`** (a confirmed
    exclusion or a failed eligibility check overrides everything else);
  - both results `COMPLIANT` → **`compatible`**;
  - otherwise (any `NOT_ENOUGH_INFO` / `NOT_RESOLVED`, no violation) →
    **`unclear`**. This is the real, common case for a currently-enrolled
    student — see the dogfood result below.
- **No `eligible_categories` tagged at all** (the classifier found no
  category signal in the posting text):
  - `sponsorship_likelihood: unlikely` → **`incompatible`** — no recognized
    student category *and* the posting signals it won't sponsor/support
    work-permit authorization, so there's no currently visible legal route.
  - `sponsorship_likelihood: likely` → **`unclear`**, not `compatible` — a
    "likely to sponsor" signal here plausibly points at a separate
    LMIA-backed employer-sponsored work-permit pathway that this checker
    doesn't model at all (it only covers study-permit-based work
    authorization). Calling this `compatible` would overclaim what was
    actually verified; `unclear` says so honestly while still surfacing the
    positive signal in the reason text.
  - `sponsorship_likelihood: unclear` → **`unclear`**.

A posting can be tagged with multiple categories at once (e.g. `p14`: both
`on_campus_only` and `co_op_exempt_eligible`). The overall verdict is the
**best** (lowest-priority: compatible < unclear < incompatible) verdict
across its tagged categories — if any one recognized pathway is compatible,
the posting is realistically pursuable via that pathway, even if another
tag on the same posting is less certain.

## Ranking logic

1. **Primary key: the compatibility bucket** — `compatible` postings first,
   then `unclear`, then `incompatible` **surfaced last, never hidden**. A
   posting is never dropped from the output regardless of verdict (tested,
   see below).
2. **Secondary key within a bucket: `adjusted_score` descending**, where
   `adjusted_score = fit_score - 0.15` if the classifier's
   `sponsorship_likelihood` is `"unlikely"`, else `adjusted_score =
   fit_score`. This is how "compatible-but-unlikely-sponsorship postings
   should rank lower but not be silently dropped" (`NEXT-BUILD-SPEC.md`'s
   explicit requirement) is implemented: the penalty moves a posting down
   *within* its bucket without ever changing its bucket, and the reason
   text (`sponsorship_likelihood`, `evidence_spans`) is always attached so
   the "why" is visible, not just the reordering. `0.15` is a documented,
   hand-chosen constant (`SPONSORSHIP_UNLIKELY_PENALTY` in `rank.py`) sized
   to be a meaningful tie-breaker without being able to flip a
   strong-fit/unlikely-sponsorship posting below a weak-fit/neutral one —
   it is explicitly **not** a tuned or learned weight (see Scope out: no
   ML-trained ranking model).

## How to run it

```bash
# From this directory (startup/clearpath/matcher/):

python rank.py \
  --resume-profile examples/resume_profile_ujjwal.json \
  --eligibility-profile ../checker/examples/profile_ujjwal.json \
  --postings ../eval/labeled_postings.json

# Machine-readable JSON, top 5 only
python rank.py \
  --resume-profile examples/resume_profile_ujjwal.json \
  --eligibility-profile ../checker/examples/profile_ujjwal.json \
  --postings ../eval/labeled_postings.json \
  --json --top 5

# Run the unit tests
python -m pytest tests/ -v
```

## Real dogfood run (this run, 2026-07-20)

Ujjwal's real resume (`examples/resume_profile_ujjwal.json`, extracted from
`portfolio/Ujjwal_Bhatia_Resume.docx` via `python-docx`, skills manually
curated by reading the extracted text) ranked against his real Phase 1
checker profile (`../checker/examples/profile_ujjwal.json`) and Phase 0's
20-posting labeled set (`../eval/labeled_postings.json`) — these are the
**actual numbers this run produced**, not illustrative:

- `p01` ("Software Engineer", full-time/permanent, explicit no-sponsor
  language) scored a semantic fit of **0.472** against Ujjwal's real
  CS/ML resume — clearly higher than `p07` (seasonal retail sales
  associate) at **0.168**, proving the embedding layer is reading real
  resume content, not returning a placeholder score.
- `p01` is tagged `pgwp_track` by the classifier. Against Ujjwal's real
  checker profile, the compatibility verdict is **`unclear`** — his profile
  has no PGWP application date yet (he hasn't graduated), so
  `pgwp_general_eligibility` comes back `NOT_ENOUGH_INFO`, not a guess in
  either direction. This is the honest, expected answer for a
  currently-enrolled student, and it's a real, un-cherry-picked outcome of
  running the tool against his actual profile.
- Swapping in the synthetic frontend-focused resume
  (`examples/resume_profile_frontend_synthetic.json`, different skill set:
  React/Next.js/UI-UX vs. Python/ML/data) changes both individual fit
  scores (`p01`'s score drops from 0.472 to 0.268) and the overall ranking
  order — proving the semantic layer does real, resume-dependent work, not
  a fixed order regardless of input.
- `p12` (weak no-sponsor signal, no eligible-category tag) is correctly
  flagged **`incompatible`** with a stated reason and still appears in the
  output — nothing is silently dropped.
- Feeding the *college/polytechnic* checker example profile
  (`../checker/examples/profile_college_polytechnic.json`, which has a
  confirmed PGWP exclusion — `already_received_pgwp: true`) against the
  same postings correctly flags every `pgwp_track`-tagged posting (`p01`,
  `p10`, `p16`) as **`incompatible`**, demonstrating the gate actually
  changes behavior based on the checker profile, not just the resume.

## Test results

**16/16 passing** (`python -m pytest tests/ -v`, from this directory):

- Resume/profile integrity (2 tests): the real resume profile's `skills`
  list is non-empty and contains a real, spot-checkable skill (`Python`);
  `raw_text` genuinely contains real resume content; the synthetic
  contrasting profile has a genuinely distinct skill set.
- Semantic similarity layer (3 tests): a strong-overlap posting
  (`p01`) scores meaningfully above an unrelated one (`p07`); the fit score
  for the same posting changes meaningfully when the resume is swapped;
  the overall ranking order changes when the resume is swapped.
- Eligibility-compatibility gate (6 tests): unlikely-sponsorship-with-no-
  category is `incompatible` and not dropped; a `pgwp_track` posting is
  `incompatible` when the checker confirms a PGWP exclusion (asserted by
  posting id `p01` against the college/polytechnic profile); the same
  posting id is `unclear` against Ujjwal's real profile (the
  `NOT_ENOUGH_INFO`-driven honest case); an `on_campus_only` posting is
  always `compatible`; a compatible-but-unlikely-sponsorship posting's
  `adjusted_score` is strictly below its `fit_score` (the penalty was
  actually applied, not just documented); no posting is ever dropped from
  the output regardless of verdict.
- Rank ordering (2 tests): the compatibility buckets are contiguous and
  correctly ordered across the whole output; ranks are sequential starting
  at 1.
- CLI (2 tests): text-mode and `--json` mode both run end-to-end via
  `subprocess` against the real fixtures.

## Scope and honest limitations

**Deliberately not built in this increment** (see `NEXT-BUILD-SPEC.md`
"Scope out" for the full list and reasoning):

- **No automated NLP resume parsing.** The `skills` list is manually
  curated by reading the extracted resume text — no `spaCy`/`pyresparser`
  NER. Real future work, not attempted here.
- **No live job-board ingestion.** The job-side corpus is Phase 0's
  existing 20-posting hand-labeled set (`../eval/labeled_postings.json`),
  not a live scrape from `opportunities-hunt` or elsewhere.
- **No trained ranking model.** The combined score is a documented,
  explainable rule (semantic score + a hand-chosen sponsorship penalty +
  the compatibility gate), not a learned model — keeps this free-tier-by-
  construction and matches the explainable/cited ethos of every other
  ClearPath module.
- **No web/UI, no persistence.** Still CLI + JSON files, same as every
  prior module.
- **No generative synthesis, no LLM calls.** Pure embedding-similarity +
  rule logic — zero new API key needed for this increment (the only new
  dependency is `python-docx`, free/local).
- **The `co_op_exempt_eligible` compatibility verdict is always
  `unclear`, never `compatible`**, because the checker's `Profile` has no
  field for program co-op structure at all (see "The eligibility-
  compatibility gate, in detail" above for the specific judgment call and
  the alternative that was considered and rejected).
- **The sponsorship-unlikely ranking penalty (0.15) is a hand-chosen
  constant, not tuned or learned** — it's sized to be a meaningful
  tie-breaker, not a calibrated weight.
- **The heuristic classifier's known coverage gaps carry through
  unchanged** — see `../classifier/README` equivalent docs in the
  top-level `README.md` (adversarial negation phrasing `p19`/`p20` are
  still misclassified by Phase 0's heuristic backend; this increment
  inherits that, it doesn't fix it).

## Project layout

```
matcher/
  rank.py                              # core fusion logic + CLI entry point
  resume_profile.py                    # ResumeProfile dataclass
  resume_extract.py                    # python-docx text extraction helper
  examples/
    resume_profile_ujjwal.json         # real resume profile (dogfood case)
    resume_profile_frontend_synthetic.json  # synthetic contrasting profile
  tests/
    test_rank.py                       # 16 behavioral tests
  README.md                            # this file
```

**This tool is informational, not legal or immigration advice** — same
disclaimer as every other ClearPath module. Every eligibility claim traces
to a cited canada.ca/IRCC source where the checker provides one; sponsorship
likelihood is a heuristic signal, not a guarantee of an employer's actual
willingness to sponsor.
