# SHIP-KIT — ClearPath

Status: **Phase 0 SHIPPED and live at https://github.com/ujjwalbhatiaa/clearpath
(public). Phase 1, Phase 2 (v1), and Phase 3a are code-complete and committed
locally across 4 commits (`381b11d` the qa/ build, 2026-07-17; `c856627` the
checker/ build, 2026-07-18; `c24961c` the matcher/ build, 2026-07-20; `cb09a9d`
the matcher/draft_facts.py (Phase 3a) build, 2026-07-22/23) but NOT yet
pushed** — per `NEXT-BUILD-SPEC.md`'s explicit instruction, pushing to the
existing repo needs the same exact-spoken-confirmation-sentence policy as a
new repo creation, and all four were unattended runs with no one present to
say it.

## To push the pending commits, say exactly:

> "Create a public GitHub repo named `clearpath` under ujjwalbhatiaa and push the code"

(Same sentence as before — this is an existing repo, so in practice this
just needs the equivalent green light to push local commits `381b11d`,
`c856627`, `c24961c`, and `cb09a9d`:
- `381b11d` [10 files: `qa/README.md`, `qa/ask.py`, `qa/corpus/chunks.json`,
  `qa/eval/eval.py`, `qa/eval/labeled_questions.json`, `qa/requirements.txt`,
  `qa/retrieval.py`, `qa/tests/test_qa.py`, plus updated `README.md` and
  `SHIP-KIT.md`]
- `c856627` [10 files: `checker/README.md`, `checker/__init__.py`,
  `checker/check.py`, `checker/corpus_loader.py`, `checker/eligibility.py`,
  `checker/examples/profile_college_polytechnic.json`,
  `checker/examples/profile_ujjwal.json`, `checker/rules_corpus.json`,
  `checker/tests/test_eligibility.py`, plus updated top-level `README.md`]
- `c24961c` [9 files: `matcher/README.md`, `matcher/__init__.py`,
  `matcher/examples/resume_profile_frontend_synthetic.json`,
  `matcher/examples/resume_profile_ujjwal.json`, `matcher/rank.py`,
  `matcher/resume_extract.py`, `matcher/resume_profile.py`,
  `matcher/tests/test_rank.py`, plus updated top-level `README.md`]
- `cb09a9d` [5 files: `matcher/DRAFTING.md`, `matcher/draft_facts.py`,
  `matcher/tests/test_draft_facts.py`, plus updated `README.md` and
  `matcher/README.md`]

via GitHub's web UI, verified byte-identical via the REST API the same way
Phase 0 was.)

Ujjwal gave the exact confirmation sentence on 2026-07-16 ("Create a public
GitHub repo named `clearpath` under ujjwalbhatiaa and push the code"). The
repo was created and all 10 tracked files were pushed via GitHub's web UI
(no git CLI credentials available in the sandbox). Every file was verified
byte-identical to the local, tested source via the GitHub REST API
(`api.github.com/repos/.../contents/<path>`, base64-decoded and diffed) —
not just a UI screenshot. Two files needed a second pass before verification
passed: `classifier/__init__.py` had a stray 2-space indentation bug from an
early push attempt (fixed by re-editing via GitHub's web editor), and
`eval/labeled_postings.json` silently failed to commit on the first attempt
despite the UI showing a completed commit flow (fixed by redoing the commit
and re-verifying via the API immediately after).

## What ships in this push

Everything under `Remote job asap/startup/clearpath/`:
- `classify.py` — CLI entry point
- `classifier/` — `ClassifierBackend` interface + `HeuristicBackend`
- `eval/` — labeled eval set + eval harness
- `tests/` — unit tests (17 passing, incl. evidence-span substring guarantee)
- `examples/sample_posting.txt`
- `README.md` — problem, approach, actual eval numbers, honest limitations

## What's in the pending Phase 1 push

Everything under `Remote job asap/startup/clearpath/qa/`:
- `ask.py` — CLI entry point (extractive Q&A)
- `retrieval.py` — local sentence-transformers embeddings + cosine-similarity retrieval
- `corpus/chunks.json` — 8 grounded IRCC rule chunks, snapshot date 2026-07-17
- `eval/` — 22 hand-labeled questions + eval harness (83.3% top-1, 94.4% top-3, 75% refusal accuracy)
- `tests/test_qa.py` — 15 unit tests, all passing
- `README.md` — problem, approach, actual eval numbers, honest limitations
- Plus updated top-level `README.md` (pointer to `qa/`) and this file

Everything under `Remote job asap/startup/clearpath/checker/` (2026-07-18):
- `check.py` — CLI entry point (personal compliance & PGWP eligibility checker)
- `eligibility.py` — rule engine: weekly-hours cap, scheduled-break day-count budget
  (150/180-day rules), PGWP general eligibility, PGWP field-of-study branching
- `corpus_loader.py` + `rules_corpus.json` — 3 new grounded chunks, merged with
  `qa/corpus/chunks.json` by id (no forked facts)
- `examples/` — Ujjwal's real bachelor's-level profile + a synthetic
  college/polytechnic profile exercising a different branch
- `tests/test_eligibility.py` — 53 unit tests, all passing (every boundary
  condition in NEXT-BUILD-SPEC.md's Scope-in section 5, incl. the legacy
  20-hour-permit regression test)
- `README.md` — problem, rule-engine design, real test counts, honest limitations
- Plus updated top-level `README.md` (pointer to `checker/`)

Everything new under `Remote job asap/startup/clearpath/matcher/` (Phase 3a, 2026-07-22/23):
- `draft_facts.py` — `assemble_fact_packet()`: fuses one `matcher.rank`
  `RankedResult` + the checker's `AssessmentReport` + a `ResumeProfile` into
  a `FactPacket` of `allowed_claims`/`forbidden_claims`, each traced to a
  real checker rule citation, classifier evidence span, or matcher verdict.
  New `compute_skill_overlap()` helper. `packet_is_well_formed()` validator.
- `DRAFTING.md` — problem, the "Attribute First, then Generate" rationale,
  the `FactPacket` schema, real CLI output, real test counts, and an
  explicit "Phase 3b — not built yet" section (needs a free-tier LLM key +
  Ujjwal's sign-off, deliberately out of scope).
- `tests/test_draft_facts.py` — 16 unit tests, all passing, most importantly
  the PGWP `NOT_ENOUGH_INFO`/confirmed-exclusion cases correctly landing in
  `forbidden_claims`, never `allowed_claims`, plus an 80-fact-packet
  full-corpus well-formedness sweep (zero integrity problems).
- Plus updated top-level `README.md` and `matcher/README.md` (pointers to
  `DRAFTING.md`).
- **No LLM call, no API key, no prose generation** — pure assembly/
  validation logic. Zero regressions: classifier 17/17, qa 15/15, checker
  53/53, matcher/rank 16/16, all still green. **117/117 passing across the
  whole repo.**

Everything under `Remote job asap/startup/clearpath/matcher/` (Phase 2, 2026-07-20):
- `rank.py` — CLI entry point + fusion logic: eligibility-aware ranked job
  list (semantic fit via reused Q&A embedding model, sponsorship/category
  tags via the Phase 0 classifier, eligibility-compatibility gate via the
  Phase 1 checker)
- `resume_profile.py` + `resume_extract.py` — `ResumeProfile` data shape +
  `python-docx`-based resume text extraction (new dependency, free/local,
  no API key)
- `examples/` — Ujjwal's real resume profile (extracted from
  `portfolio/Ujjwal_Bhatia_Resume.docx`) + a synthetic frontend-focused
  contrasting profile
- `tests/test_rank.py` — 16 unit tests, all passing (semantic-layer
  behavioral proofs, eligibility-gate behavioral proofs asserted by posting
  id, rank-ordering invariants, CLI smoke tests)
- `README.md` — problem, "not a generic resume matcher" differentiation
  rationale, ranking logic worked example, real dogfood numbers, honest
  limitations
- Plus updated top-level `README.md` (pointer to `matcher/`)

## Suggested first-commit steps (once the repo exists)

```bash
cd "startup/clearpath"
git init
git add .
git commit -m "Phase 0: sponsorship/eligibility classifier (heuristic backend) + eval harness"
git branch -M main
git remote add origin https://github.com/ujjwalbhatiaa/clearpath.git
git push -u origin main
```

## Suggested repo metadata

- **Description:** "AI copilot for international students in Canada — classifies job postings for sponsorship likelihood and IRCC work-eligibility signals, grounded in cited canada.ca rules."
- **Topics:** `nlp`, `python`, `canada`, `international-students`, `job-search`, `classifier`
- Add the README's eval numbers directly to the repo description or pinned-repo blurb once live — they're real numbers, safe to lead with.

## Not yet ready to ship (do not claim otherwise)

- No UI, no web app (Phase 0 scope explicitly excludes this — see `NEXT-BUILD-SPEC.md`).
- No LLM backend (heuristic only this increment; see README "Approach & backend").
- Eval set is 20 hand-written examples, not an independent/scraped corpus — say so if asked, don't oversell the 90% number.
- **Phase 1 (`qa/`):** extractive-only, 8-chunk static snapshot dated 2026-07-17, not re-fetched live. Cannot answer multi-hop questions that need combining two rules (documented honestly in `qa/README.md`). 22 hand-written eval questions, not independently sourced. Don't claim it's a general IRCC chatbot — it's a citation-finder over a small, dated corpus.
- **Phase 1 (`checker/`):** pure rule engine, no CIP-code field-of-study resolution (reports `not_resolved` honestly instead), no distance-learning/curriculum-licensing/flight-school special-case modeling, no web/UI, no persistence, no fusion with Phase 0 or the Q&A tool yet. Static corpus, same re-fetch caveat as `qa/`. See `checker/README.md` "Honest limitations" for the full list.
- **Phase 2 v1 (`matcher/`):** no automated NLP resume parsing (skills manually curated from extracted text), no live job-board ingestion (job-side corpus is Phase 0's 20-posting hand-labeled set, not a live scrape), no trained ranking model (documented rule: semantic score + a hand-chosen 0.15 sponsorship-unlikely penalty + the compatibility gate), no web/UI, no persistence. `co_op_exempt_eligible` compatibility is always `unclear` by design (the checker has no field for program co-op structure) — a documented judgment call, not an oversight. Inherits Phase 0's known classifier coverage gaps (adversarial negation phrasing) unchanged. See `matcher/README.md` "Scope and honest limitations" for the full list.
- **Phase 3a (`matcher/draft_facts.py`):** fact-grounding/attribution only — **no prose generation, no LLM call, no API key of any kind.** Not an application drafter yet; Phase 3b (the actual generation step) is explicitly not built and needs a free-tier LLM key with Ujjwal's sign-off first. Only `pgwp_track` maps to real per-profile checker `RuleResult`s; the other three categories' claims come from `CategoryCompatibility`'s own reason/citations. `skill_overlap` is literal word-boundary string matching, not semantic. See `matcher/DRAFTING.md` "Scope and honest limitations" for the full list.
