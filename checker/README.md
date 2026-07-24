# ClearPath — Personal Compliance & PGWP Eligibility Checker (Phase 1, second half)

This is the second half of Phase 1 named in `../STARTUP.md`'s roadmap
("RAG-grounded IRCC compliance Q&A — **work-hour tracker, PGWP eligibility
checker**"). The extractive Q&A tool (`../qa/`) shipped the first half on
2026-07-17; this module is the personalized rule engine that turns those
same static facts (plus deeper detail found on the same government pages)
into a dated, cited assessment of *your* specific situation.

## The problem this increment solves

`../qa/` answers "what does the rule say" with an exact quote. It cannot
answer "given my actual hours, my actual scheduled breaks, my actual
program dates — am I compliant, and am I PGWP-eligible?" That requires
combining several grounded rules with a user's own facts and doing real
day-count arithmetic (is this break long enough to qualify, has the
150-day or 180-day budget been used up, is a PGWP application inside its
180-day window). This module is that: a deterministic rule engine over a
structured profile, never a generative guess.

## Rule engine design

- **Pure functions, one per rule area** (`eligibility.py`):
  `check_weekly_hours`, `check_scheduled_break_budget`,
  `check_pgwp_general_eligibility`, `check_pgwp_field_of_study`, composed
  by `assess()` into one report.
- **Four-state status**, not a binary pass/fail: `compliant` / `violation`
  / `not_enough_info` / `not_resolved`. An unset profile field degrades to
  `not_enough_info` rather than assuming a default. `not_resolved` is a
  4th state for cases this checker recognizes but deliberately does not
  attempt to resolve (CIP-code field-of-study lookup, the flight-school
  special case) — see "Honest limitations" below. Every result also
  carries an `explanation` string and a list of `citations` (chunk ids
  resolvable in the merged corpus — see below).
- **The 24-hour cap is structurally always 24.** `effective_off_campus_weekly_cap()`
  never reads `study_permit_printed_hours` — it exists specifically so a
  legacy permit printed with a stale "20 hours" condition can never leak
  into the enforced cap. See `TestLegacyPermitTextTwentyHours` in the test
  suite for the dedicated regression tests.
- **Day-count math is done at day granularity**, using only stdlib
  `datetime`/`calendar` (`add_months`, `compute_break_day_status`). A
  7-day-minimum-length check, back-to-back break chaining (via interval
  merging), the 150-consecutive-day cap, and the 180-day/calendar-year
  budget are all computed by walking real dates, not approximated — this
  is the part of the spec most likely to hide an off-by-one bug, so it's
  covered by the most boundary-heavy tests in the suite.

## Corpus design — why two files, not one

`checker/rules_corpus.json` holds 3 new chunks captured 2026-07-18
(`on_campus_unlimited`, `legacy_permit_text_20_hours`,
`pgwp_not_eligible_exclusions`). It does **not** duplicate the two
existing chunks this checker also needs (`pgwp_general_eligibility`,
`pgwp_field_of_study_exempt`) — those are reused directly by id from
`../qa/corpus/chunks.json`. `corpus_loader.py` merges both files into one
`{id: Chunk}` lookup at import time and raises if a chunk id ever
collides between the two files, so the two corpora can never silently
fork the same fact.

Extending `../qa/corpus/chunks.json` in place was considered and
rejected: that file's existing unit tests (`../qa/tests/test_qa.py`)
hard-code expected top-1 retrieval results for specific questions against
specific chunks, and the 3 new chunks share enough vocabulary with
existing ones (on-campus vs. off-campus hours, "20 hours" vs "24 hours")
that adding them risked silently shifting embedding rankings and breaking
those already-shipped, already-verified tests. Keeping the new chunks in
a separate file avoided touching the Q&A tool's corpus or tests at all —
confirmed by re-running `../tests/` and `../checker/tests/` together
after this build (see "Test results" below); the Q&A/classifier test
files were not modified.

## How to run it

```bash
# From this directory (startup/clearpath/checker/):

# Run the checker against Ujjwal's own real (bachelor's-level) profile
python check.py --profile examples/profile_ujjwal.json

# Run against a synthetic college/polytechnic profile that exercises a
# different field-of-study branch, a weekly-hours violation, and a PGWP
# exclusion violation
python check.py --profile examples/profile_college_polytechnic.json --json

# Override the as-of date used for the scheduled-break budget check
python check.py --profile examples/profile_ujjwal.json --as-of-date 2026-09-01

# Run the test suite
python -m pytest tests/ -v
# or: python -m unittest discover -s tests -v
```

## Test results (this run, 2026-07-18)

```
53/53 tests passing (checker/tests/test_eligibility.py)
17/17 tests passing (../tests/test_classify.py, unaffected, re-run to confirm no regression)
```

**53/53, not a subset** — every boundary condition `NEXT-BUILD-SPEC.md`'s
Scope-in section 5 called out has a real, named, passing test:

- Exactly 24 / 24.01 / 25 off-campus hours during term (compliant / violation / violation).
- On-campus hours at any number during term (always compliant).
- A 5-day break (doesn't qualify, cap still applies) vs. an exactly-7-day break (qualifies, unlimited).
- Chained breaks: day 150 of a chain (still unlimited) vs. day 151 (reverts to capped).
- Cumulative annual budget: day 180 (still authorized) vs. day 181 (reverts to capped), plus a
  calendar-year-boundary reset test.
- The legacy "20 hours printed, 24 applies" case — 3 dedicated tests, not implied by another test,
  including one that would fail under a naive who-trusts-the-printed-number implementation.
- PGWP program length: exactly 8 months (eligible) vs. 7 months 29 days (ineligible), computed via
  real calendar-month arithmetic (`add_months`), not an approximated day count.
- PGWP application window: exactly day 180 (eligible) vs. day 181 (ineligible).
- PGWP field of study: bachelor's/master's/doctoral (no requirement, incl. Ujjwal's real profile),
  college/polytechnic before vs. on/after Nov 1, 2024 (exempt vs. `not_resolved`), flight school
  (`not_resolved`).
- At least one `not_enough_info` case for a missing required field, in both the weekly-hours check and
  the PGWP general-eligibility check.
- Corpus integrity: every new chunk is well-formed and canada.ca-sourced, the two reused chunks are
  confirmed present-but-not-duplicated, and every citation id every rule function can emit resolves in
  the merged corpus.
- CLI end-to-end smoke tests against both example profiles, in both text and `--json` modes, plus the
  `--as-of-date` override.

## Project layout

```
clearpath/checker/
  eligibility.py               # Profile/WeekEntry/ScheduledBreak dataclasses + the rule engine
  corpus_loader.py              # merges ../qa/corpus/chunks.json + rules_corpus.json by id
  rules_corpus.json             # 3 new grounded chunks, captured 2026-07-18
  check.py                      # CLI entry point
  examples/
    profile_ujjwal.json          # Ujjwal's real bachelor's-level case (no field-of-study requirement)
    profile_college_polytechnic.json  # synthetic profile: different branch + violations
  tests/
    test_eligibility.py          # 53 tests -- boundary conditions, corpus integrity, CLI smoke tests
  README.md                      # this file
```

## Honest limitations, not hidden

- **No CIP-code-level field-of-study resolution.** `check_pgwp_field_of_study` correctly determines
  *whether* a field-of-study requirement applies (by credential level and application date) but never
  attempts to resolve a specific program against the 6-digit CIP code list — it reports `not_resolved`
  and points to the Q&A tool / canada.ca instead. Deliberately out of scope for this increment (see
  `NEXT-BUILD-SPEC.md`).
- **No distance-learning lock-in-date logic, curriculum-licensing (P3) grandfather-date logic, or
  flight-school special case.** These are real, grounded rules noted in `STARTUP.md`'s 2026-07-18
  research log, but are narrow special cases left unmodeled here rather than half-implemented.
  Flight-school graduates get an explicit `not_resolved` result rather than a wrong answer.
- **`check_scheduled_break_budget` is informational, not a violation judgment.** It reports whether
  unlimited-hours authorization is active on a given date — it does not itself know how many hours were
  actually worked that day (that's `check_weekly_hours`'s job, at week granularity). Its `status` field
  is `compliant` (authorization active or correctly not claimed) or `violation` (authorization has
  lapsed for that specific day due to the 150-day or 180-day cap) — but "violation" here means
  "unlimited-hours status has lapsed," not "hours were necessarily exceeded."
- **No web/UI.** Still CLI, same as the classifier and the Q&A tool.
- **No fusion with the Phase 0 classifier or the Phase 1 Q&A retrieval tool yet.** Flagged as future
  work (Phase 1.5), not built here.
- **No persistence/database for profiles.** A JSON file passed via `--profile` is enough for this
  increment.
- **No generative synthesis, no LLM calls.** Pure rule logic over grounded facts — free-tier-by-
  construction, same rationale as `../qa/`.
- **Static, dated corpus, not re-fetched live.** Same caveat as `../qa/README.md`: if any cited
  canada.ca/ircc.canada.ca page changes after the dates recorded in each chunk's `date_modified`, this
  tool will keep citing the old text until a future research run re-verifies it.

## Sources cited by this checker

All 3 new chunks captured live 2026-07-18 (see `rules_corpus.json`); reused chunks captured 2026-07-16/17
(see `../qa/README.md`):

- [Work on campus as an international student](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/work-on-campus.html) — canada.ca, modified 2026-05-01
- [Work off campus as an international student](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/work-off-campus.html) — canada.ca, modified 2026-04-15
- [Can I work as many hours as I want if I'm eligible to work off campus?](https://ircc.canada.ca/english/helpcentre/answer.asp?qnum=503&top=15) — IRCC Help Centre, modified 2026-04-17
- [Post-graduation work permit: Who can apply](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/after-graduation/eligibility.html) — canada.ca, modified 2026-06-24
- [PGWP field of study requirements / CIP code lookup](https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/work/after-graduation/eligibility/field-of-study.html) — canada.ca, modified 2026-03-09

**This tool is informational, not legal or immigration advice.** It reports a rule-based assessment
grounded in cited public rules and your own stated facts — it does not replace professional
immigration advice, and every result should be verified against the live canada.ca/ircc.canada.ca
source before relying on it for a real decision.
