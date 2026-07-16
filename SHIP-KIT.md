# SHIP-KIT — ClearPath Phase 0

Status: **code-complete locally, not yet pushed to GitHub.**

Per standing repo-creation policy, new GitHub repo creation requires the
exact spoken confirmation sentence from Ujjwal, given in the moment — a
blanket/generic approval does not clear it, and this was an unattended
scheduled run with no one present to say it. Nothing was pushed.

## To ship this, say exactly:

> "Create a public GitHub repo named `clearpath` under ujjwalbhatiaa and push the code"

(Swap `clearpath` for a different repo name first if you've settled on a
better product name than the STARTUP.md placeholder — "ClearPath" is
explicitly marked as not-yet-locked branding.)

## What ships in this push

Everything under `Remote job asap/startup/clearpath/`:
- `classify.py` — CLI entry point
- `classifier/` — `ClassifierBackend` interface + `HeuristicBackend`
- `eval/` — labeled eval set + eval harness
- `tests/` — unit tests (17 passing, incl. evidence-span substring guarantee)
- `examples/sample_posting.txt`
- `README.md` — problem, approach, actual eval numbers, honest limitations

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
