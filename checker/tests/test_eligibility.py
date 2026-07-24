"""
Unit tests for the Phase 1 personal compliance & PGWP eligibility checker.

Run: python -m unittest discover -s tests -v   (from the checker/ directory)
     python -m pytest tests/ -v                 (from the checker/ directory)

These tests exist to nail down the exact boundary conditions
NEXT-BUILD-SPEC.md calls out as the increment's real bar -- off-by-one bugs
in day-count math are the most likely failure mode here, so almost every
test below is a boundary case (exactly-at-the-limit vs. one-past-it), not a
comfortably-inside-the-range happy path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from corpus_loader import CHUNKS_BY_ID  # noqa: E402
from eligibility import (  # noqa: E402
    FIELD_OF_STUDY_CUTOFF_DATE,
    Profile,
    ScheduledBreak,
    Status,
    WeekEntry,
    add_months,
    assess,
    check_pgwp_field_of_study,
    check_pgwp_general_eligibility,
    check_scheduled_break_budget,
    check_weekly_hours,
    compute_break_day_status,
    effective_off_campus_weekly_cap,
)

CHECKER_DIR = os.path.join(os.path.dirname(__file__), "..")


# ---------------------------------------------------------------------------
# 1. Weekly off-campus / on-campus hours
# ---------------------------------------------------------------------------


class TestWeeklyHoursBoundaries(unittest.TestCase):
    def _term_time_profile(self) -> Profile:
        # No scheduled breaks at all -> every week is term-time.
        return Profile(scheduled_breaks=[])

    def test_exactly_24_off_campus_hours_is_compliant(self):
        profile = self._term_time_profile()
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=24.0)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_24_01_off_campus_hours_is_violation(self):
        profile = self._term_time_profile()
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=24.01)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.VIOLATION)

    def test_25_off_campus_hours_is_violation(self):
        profile = self._term_time_profile()
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=25)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.VIOLATION)

    def test_on_campus_hours_at_any_number_during_term_is_always_compliant(self):
        profile = self._term_time_profile()
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=0, on_campus_hours=40)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.COMPLIANT)
        self.assertIn("40", r.explanation)

    def test_missing_off_campus_hours_is_not_enough_info(self):
        profile = self._term_time_profile()
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=None)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.NOT_ENOUGH_INFO)


class TestLegacyPermitTextTwentyHours(unittest.TestCase):
    """Dedicated, explicitly-named test for the one rule a naive
    implementation would silently get wrong: trusting the number printed
    on the study permit instead of the current 24-hour rule."""

    def test_effective_cap_is_always_24_regardless_of_printed_hours(self):
        profile = Profile(study_permit_printed_hours=20)
        self.assertEqual(effective_off_campus_weekly_cap(profile), 24)

    def test_legacy_20_hour_permit_profile_with_22_hours_is_compliant_not_violation(self):
        """A naive implementation that capped hours at the PRINTED number
        (20) would wrongly flag 22 off-campus hours as a violation. The
        real rule is 24 applies regardless -- this must be compliant."""
        profile = Profile(study_permit_printed_hours=20, scheduled_breaks=[])
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=22)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.COMPLIANT)
        self.assertIn("24", r.explanation)

    def test_legacy_20_hour_permit_profile_with_23_hours_is_still_compliant(self):
        """23 hours is ineligible under a naive 20-hour cap but must be
        compliant under the real 24-hour cap -- the sharpest possible
        regression check for this rule."""
        profile = Profile(study_permit_printed_hours=20, scheduled_breaks=[])
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=23)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_legacy_20_hour_permit_profile_with_25_hours_is_still_a_violation(self):
        """Confirms the real 24-hour cap is still enforced -- this isn't
        an unlimited-hours pass, just not-the-printed-20 number."""
        profile = Profile(study_permit_printed_hours=20, scheduled_breaks=[])
        week = WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=25)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.VIOLATION)


# ---------------------------------------------------------------------------
# 2. Scheduled-break day-count math
# ---------------------------------------------------------------------------


class TestScheduledBreakDayCount(unittest.TestCase):
    def test_single_5_day_break_does_not_qualify(self):
        """A break shorter than 7 consecutive days does not count as a
        scheduled break at all -- the 24-hour cap still applies."""
        breaks = [ScheduledBreak(start_date=date(2026, 3, 2), end_date=date(2026, 3, 6))]  # 5 days
        status = compute_break_day_status(breaks, date(2026, 3, 4))
        self.assertFalse(status.is_break_day)
        self.assertFalse(status.authorized_unlimited)

    def test_5_day_break_week_still_enforces_24_hour_cap(self):
        breaks = [ScheduledBreak(start_date=date(2026, 3, 2), end_date=date(2026, 3, 6))]
        profile = Profile(scheduled_breaks=breaks)
        week = WeekEntry(week_start=date(2026, 3, 2), off_campus_hours=25)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.VIOLATION)

    def test_exactly_7_day_break_qualifies(self):
        breaks = [ScheduledBreak(start_date=date(2026, 3, 2), end_date=date(2026, 3, 8))]  # 7 days
        status = compute_break_day_status(breaks, date(2026, 3, 5))
        self.assertTrue(status.is_break_day)
        self.assertTrue(status.authorized_unlimited)

    def test_7_day_break_week_allows_unlimited_hours(self):
        breaks = [ScheduledBreak(start_date=date(2026, 3, 2), end_date=date(2026, 3, 8))]
        profile = Profile(scheduled_breaks=breaks)
        week = WeekEntry(week_start=date(2026, 3, 2), off_campus_hours=100)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_chained_breaks_150th_consecutive_day_still_unlimited(self):
        """Two adjacent (back-to-back, no gap) breaks chain into one
        continuous period. Day 150 of that chain must still be
        unlimited."""
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 4, 10)),  # 100 days
            ScheduledBreak(start_date=date(2026, 4, 11), end_date=date(2026, 7, 19)),  # next 100 days, adjacent
        ]
        day150 = date(2026, 1, 1) + __import__("datetime").timedelta(days=149)
        status = compute_break_day_status(breaks, day150)
        self.assertEqual(status.chain_position, 150)
        self.assertTrue(status.authorized_unlimited)

    def test_chained_breaks_151st_consecutive_day_reverts_to_capped(self):
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 4, 10)),
            ScheduledBreak(start_date=date(2026, 4, 11), end_date=date(2026, 7, 19)),
        ]
        day151 = date(2026, 1, 1) + __import__("datetime").timedelta(days=150)
        status = compute_break_day_status(breaks, day151)
        self.assertEqual(status.chain_position, 151)
        self.assertFalse(status.authorized_unlimited)

    def test_151st_day_week_enforces_24_hour_cap(self):
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 4, 10)),
            ScheduledBreak(start_date=date(2026, 4, 11), end_date=date(2026, 7, 19)),
        ]
        profile = Profile(scheduled_breaks=breaks)
        # Week starting on the 151st consecutive day of the chain.
        week_start = date(2026, 1, 1) + __import__("datetime").timedelta(days=150)
        week = WeekEntry(week_start=week_start, off_campus_hours=30)
        r = check_weekly_hours(profile, week)
        self.assertEqual(r.status, Status.VIOLATION)

    def test_cumulative_180th_day_in_year_still_authorized(self):
        """Multiple separate (non-adjacent, each < 150 consecutive days)
        breaks in the same calendar year whose cumulative qualifying days
        reach exactly 180 -- day 180 must still be authorized."""
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 5, 30)),  # 150 days
            ScheduledBreak(start_date=date(2026, 6, 15), end_date=date(2026, 7, 14)),  # 30 days, gap before it
        ]
        # 150 (first break) + 30 (second break) = 180 total qualifying days;
        # the 180th falls on the last day of the second break.
        status = compute_break_day_status(breaks, date(2026, 7, 14))
        self.assertEqual(status.cumulative_year_days_used, 180)
        self.assertTrue(status.authorized_unlimited)

    def test_cumulative_181st_day_in_year_reverts_to_capped(self):
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 5, 30)),  # 150 days
            ScheduledBreak(start_date=date(2026, 6, 15), end_date=date(2026, 7, 20)),  # 36 days
        ]
        # Day 181 of cumulative usage: 150 + 31 = day 2026-07-15 is the 181st.
        status = compute_break_day_status(breaks, date(2026, 7, 15))
        self.assertEqual(status.cumulative_year_days_used, 180)  # doesn't advance past the cap
        self.assertFalse(status.authorized_unlimited)

    def test_180_day_cap_resets_across_calendar_years(self):
        """The 180-day cap is per calendar year -- a break spanning into
        January of the next year should reset the running count."""
        breaks = [ScheduledBreak(start_date=date(2025, 12, 20), end_date=date(2026, 2, 15))]  # 58 days
        # Jan 1 2026 is day 13 of this break overall, but only the 13th day
        # *within 2026* -- confirm cumulative_year_days_used counts from
        # Jan 1 2026, not from the break's true start in 2025.
        status = compute_break_day_status(breaks, date(2026, 1, 1))
        self.assertEqual(status.cumulative_year_days_used, 1)


# ---------------------------------------------------------------------------
# 3. check_scheduled_break_budget (RuleResult-level reporting)
# ---------------------------------------------------------------------------


class TestScheduledBreakBudgetRule(unittest.TestCase):
    def test_no_breaks_at_all_is_compliant_informational(self):
        profile = Profile(scheduled_breaks=[])
        r = check_scheduled_break_budget(profile, date(2026, 7, 18))
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_within_budget_break_day_is_compliant(self):
        breaks = [ScheduledBreak(start_date=date(2026, 5, 1), end_date=date(2026, 8, 31))]
        profile = Profile(scheduled_breaks=breaks)
        r = check_scheduled_break_budget(profile, date(2026, 5, 10))
        self.assertEqual(r.status, Status.COMPLIANT)
        self.assertIn("authorized", r.explanation)

    def test_past_150_day_chain_is_violation(self):
        breaks = [
            ScheduledBreak(start_date=date(2026, 1, 1), end_date=date(2026, 4, 10)),
            ScheduledBreak(start_date=date(2026, 4, 11), end_date=date(2026, 7, 19)),
        ]
        profile = Profile(scheduled_breaks=breaks)
        day151 = date(2026, 1, 1) + __import__("datetime").timedelta(days=150)
        r = check_scheduled_break_budget(profile, day151)
        self.assertEqual(r.status, Status.VIOLATION)


# ---------------------------------------------------------------------------
# 4. PGWP general eligibility boundaries
# ---------------------------------------------------------------------------


class TestPgwpGeneralEligibilityBoundaries(unittest.TestCase):
    def _base_profile(self, **overrides) -> Profile:
        defaults = dict(
            program_start_date=date(2024, 1, 1),
            program_end_date=add_months(date(2024, 1, 1), 8),  # exactly 8 months
            full_time_each_semester=True,
            program_completion_confirmation_date=date(2024, 9, 1),
            pgwp_application_date=date(2025, 2, 27),  # will be recomputed per-test
            study_permit_valid_during_window=True,
        )
        defaults.update(overrides)
        return Profile(**defaults)

    def test_exactly_8_month_program_is_length_eligible(self):
        start = date(2024, 1, 1)
        end = add_months(start, 8)
        profile = self._base_profile(program_start_date=start, program_end_date=end)
        r = check_pgwp_general_eligibility(profile)
        self.assertNotIn("program_length_at_least_8_months", r.explanation) if r.status == Status.COMPLIANT else None
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_7_months_29_days_program_is_length_ineligible(self):
        start = date(2024, 1, 1)
        end = add_months(start, 8) - __import__("datetime").timedelta(days=1)  # one day short
        profile = self._base_profile(program_start_date=start, program_end_date=end)
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.VIOLATION)
        self.assertIn("program_length_at_least_8_months", r.explanation)

    def test_application_exactly_day_180_is_eligible(self):
        completion = date(2025, 1, 1)
        application = completion + __import__("datetime").timedelta(days=180)
        profile = self._base_profile(
            program_completion_confirmation_date=completion, pgwp_application_date=application
        )
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_application_day_181_is_ineligible(self):
        completion = date(2025, 1, 1)
        application = completion + __import__("datetime").timedelta(days=181)
        profile = self._base_profile(
            program_completion_confirmation_date=completion, pgwp_application_date=application
        )
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.VIOLATION)
        self.assertIn("applied_within_180_day_window", r.explanation)

    def test_missing_required_field_is_not_enough_info(self):
        profile = self._base_profile(full_time_each_semester=None)
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.NOT_ENOUGH_INFO)
        self.assertIn("full_time_each_semester", r.explanation)

    def test_exclusion_flag_triggers_violation_even_if_everything_else_passes(self):
        profile = self._base_profile(already_received_pgwp=True)
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.VIOLATION)
        self.assertIn("already received a PGWP", r.explanation)

    def test_all_criteria_met_and_no_exclusions_is_compliant(self):
        start = date(2024, 1, 1)
        end = add_months(start, 9)
        completion = date(2024, 9, 1)
        application = completion + __import__("datetime").timedelta(days=30)
        profile = self._base_profile(
            program_start_date=start,
            program_end_date=end,
            program_completion_confirmation_date=completion,
            pgwp_application_date=application,
        )
        r = check_pgwp_general_eligibility(profile)
        self.assertEqual(r.status, Status.COMPLIANT)


# ---------------------------------------------------------------------------
# 5. PGWP field-of-study branching
# ---------------------------------------------------------------------------


class TestPgwpFieldOfStudy(unittest.TestCase):
    def test_bachelors_has_no_requirement(self):
        """Ujjwal's own real case: a bachelor's-degree graduate has no
        field-of-study requirement at all."""
        profile = Profile(credential_level="bachelors", study_permit_application_date=date(2025, 1, 1))
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_masters_has_no_requirement(self):
        profile = Profile(credential_level="masters")
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_doctoral_has_no_requirement(self):
        profile = Profile(credential_level="doctoral")
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_college_polytechnic_before_cutoff_is_exempt(self):
        before = FIELD_OF_STUDY_CUTOFF_DATE - __import__("datetime").timedelta(days=1)
        profile = Profile(credential_level="college_polytechnic", study_permit_application_date=before)
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_college_polytechnic_on_cutoff_date_requires_field_of_study(self):
        profile = Profile(
            credential_level="college_polytechnic", study_permit_application_date=FIELD_OF_STUDY_CUTOFF_DATE
        )
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.NOT_RESOLVED)

    def test_college_polytechnic_after_cutoff_requires_field_of_study(self):
        after = FIELD_OF_STUDY_CUTOFF_DATE + __import__("datetime").timedelta(days=1)
        profile = Profile(credential_level="college_polytechnic", study_permit_application_date=after)
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.NOT_RESOLVED)

    def test_other_university_program_same_branching_as_college_polytechnic(self):
        before = FIELD_OF_STUDY_CUTOFF_DATE - __import__("datetime").timedelta(days=1)
        profile = Profile(credential_level="other_university", study_permit_application_date=before)
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.COMPLIANT)

    def test_flight_school_is_not_resolved(self):
        profile = Profile(credential_level="flight_school")
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.NOT_RESOLVED)

    def test_date_gated_level_missing_application_date_is_not_enough_info(self):
        profile = Profile(credential_level="college_polytechnic", study_permit_application_date=None)
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.NOT_ENOUGH_INFO)

    def test_missing_credential_level_is_not_enough_info(self):
        profile = Profile(credential_level=None)
        r = check_pgwp_field_of_study(profile)
        self.assertEqual(r.status, Status.NOT_ENOUGH_INFO)


# ---------------------------------------------------------------------------
# 6. add_months helper
# ---------------------------------------------------------------------------


class TestAddMonths(unittest.TestCase):
    def test_simple_case(self):
        self.assertEqual(add_months(date(2024, 1, 1), 8), date(2024, 9, 1))

    def test_clamps_short_month(self):
        self.assertEqual(add_months(date(2024, 1, 31), 1), date(2024, 2, 29))  # 2024 is a leap year

    def test_crosses_year_boundary(self):
        self.assertEqual(add_months(date(2024, 6, 15), 8), date(2025, 2, 15))


# ---------------------------------------------------------------------------
# 7. assess() end-to-end against the example profiles
# ---------------------------------------------------------------------------


class TestAssessEndToEnd(unittest.TestCase):
    def _load(self, filename: str) -> Profile:
        import json

        path = os.path.join(CHECKER_DIR, "examples", filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Profile.from_dict(data)

    def test_ujjwal_profile_field_of_study_is_compliant(self):
        profile = self._load("profile_ujjwal.json")
        report = assess(profile)
        fos = next(r for r in report.results if r.rule == "pgwp_field_of_study")
        self.assertEqual(fos.status, Status.COMPLIANT)

    def test_ujjwal_profile_break_week_is_compliant(self):
        profile = self._load("profile_ujjwal.json")
        report = assess(profile)
        weekly = [r for r in report.results if r.rule == "weekly_hours"]
        self.assertEqual(len(weekly), 2)
        self.assertTrue(all(r.status == Status.COMPLIANT for r in weekly))

    def test_college_polytechnic_profile_has_violation_and_not_resolved(self):
        profile = self._load("profile_college_polytechnic.json")
        report = assess(profile)
        statuses = {r.rule: r.status for r in report.results}
        self.assertEqual(statuses["weekly_hours"], Status.VIOLATION)
        self.assertEqual(statuses["pgwp_general_eligibility"], Status.VIOLATION)
        self.assertEqual(statuses["pgwp_field_of_study"], Status.NOT_RESOLVED)

    def test_assess_uses_profile_as_of_date_when_not_overridden(self):
        profile = self._load("profile_ujjwal.json")
        report = assess(profile)
        self.assertEqual(report.as_of_date, date(2026, 7, 18))

    def test_assess_as_of_date_override_takes_precedence(self):
        profile = self._load("profile_ujjwal.json")
        report = assess(profile, as_of_date=date(2026, 1, 1))
        self.assertEqual(report.as_of_date, date(2026, 1, 1))


# ---------------------------------------------------------------------------
# 8. Corpus integrity -- every citation the rule engine emits must resolve
# ---------------------------------------------------------------------------


class TestCorpusIntegrity(unittest.TestCase):
    """Mirrors qa/tests/test_qa.py's corpus-integrity discipline: a broken
    or invented citation should never ship silently."""

    NEW_CHUNK_IDS = {"on_campus_unlimited", "legacy_permit_text_20_hours", "pgwp_not_eligible_exclusions"}
    REUSED_CHUNK_IDS = {
        "hour_cap_24",
        "scheduled_break_unlimited",
        "annual_180_day_cap",
        "pgwp_general_eligibility",
        "pgwp_field_of_study_exempt",
        "pgwp_field_of_study_freeze_2026",
    }

    def test_new_chunks_exist_and_are_well_formed(self):
        for cid in self.NEW_CHUNK_IDS:
            self.assertIn(cid, CHUNKS_BY_ID, f"new chunk {cid!r} missing from merged corpus")
            chunk = CHUNKS_BY_ID[cid]
            self.assertTrue(chunk.source_url.startswith("https://"))
            self.assertIn("canada.ca", chunk.source_url)
            self.assertRegex(chunk.date_modified, r"^\d{4}-\d{2}-\d{2}$")
            self.assertGreater(len(chunk.text), 20)

    def test_reused_qa_chunks_are_present_not_duplicated(self):
        """These chunks must come from qa/corpus/chunks.json, not be
        redefined in rules_corpus.json -- confirms the no-fork guarantee."""
        import json

        checker_corpus_path = os.path.join(CHECKER_DIR, "rules_corpus.json")
        with open(checker_corpus_path, "r", encoding="utf-8") as f:
            checker_ids = {c["id"] for c in json.load(f)["chunks"]}
        for cid in self.REUSED_CHUNK_IDS:
            self.assertIn(cid, CHUNKS_BY_ID, f"expected reused chunk {cid!r} not found in merged corpus")
            self.assertNotIn(cid, checker_ids, f"chunk {cid!r} should not be redefined in rules_corpus.json")

    def test_every_citation_emitted_by_every_rule_resolves(self):
        """Runs every rule function across a battery of profiles and checks
        every citation id it emits actually resolves in CHUNKS_BY_ID."""
        profiles_and_weeks = [
            (Profile(scheduled_breaks=[]), WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=24)),
            (Profile(scheduled_breaks=[]), WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=None)),
            (
                Profile(scheduled_breaks=[ScheduledBreak(date(2026, 5, 1), date(2026, 8, 31))]),
                WeekEntry(week_start=date(2026, 5, 4), off_campus_hours=50),
            ),
            (
                Profile(study_permit_printed_hours=20, scheduled_breaks=[]),
                WeekEntry(week_start=date(2026, 9, 14), off_campus_hours=22),
            ),
        ]
        for profile, week in profiles_and_weeks:
            r = check_weekly_hours(profile, week)
            for cid in r.citations:
                self.assertIn(cid, CHUNKS_BY_ID, f"weekly_hours emitted unresolved citation {cid!r}")

        for level in ["bachelors", "masters", "doctoral", "college_polytechnic", "other_university", "flight_school", None]:
            r = check_pgwp_field_of_study(Profile(credential_level=level, study_permit_application_date=date(2025, 1, 1)))
            for cid in r.citations:
                self.assertIn(cid, CHUNKS_BY_ID, f"pgwp_field_of_study emitted unresolved citation {cid!r}")

        r = check_pgwp_general_eligibility(Profile(already_received_pgwp=True))
        for cid in r.citations:
            self.assertIn(cid, CHUNKS_BY_ID, f"pgwp_general_eligibility emitted unresolved citation {cid!r}")

        r = check_scheduled_break_budget(Profile(scheduled_breaks=[]), date(2026, 7, 18))
        for cid in r.citations:
            self.assertIn(cid, CHUNKS_BY_ID, f"scheduled_break_budget emitted unresolved citation {cid!r}")


# ---------------------------------------------------------------------------
# 9. CLI end-to-end smoke tests
# ---------------------------------------------------------------------------


class TestCLIEndToEnd(unittest.TestCase):
    def _run(self, *args):
        script = os.path.join(CHECKER_DIR, "check.py")
        return subprocess.run(
            [sys.executable, script, *args], capture_output=True, text=True, timeout=15, cwd=CHECKER_DIR
        )

    def test_cli_runs_end_to_end_on_ujjwal_profile(self):
        proc = self._run("--profile", "examples/profile_ujjwal.json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ClearPath compliance", proc.stdout)
        self.assertIn("pgwp_field_of_study", proc.stdout)

    def test_cli_runs_end_to_end_on_college_polytechnic_profile_json_mode(self):
        proc = self._run("--profile", "examples/profile_college_polytechnic.json", "--json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        import json

        data = json.loads(proc.stdout)
        self.assertIn("results", data)
        self.assertIn("summary", data)
        rule_statuses = {r["rule"]: r["status"] for r in data["results"]}
        self.assertEqual(rule_statuses["weekly_hours"], "violation")
        self.assertEqual(rule_statuses["pgwp_field_of_study"], "not_resolved")

    def test_cli_as_of_date_override(self):
        proc = self._run("--profile", "examples/profile_ujjwal.json", "--as-of-date", "2026-01-15", "--json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        import json

        data = json.loads(proc.stdout)
        self.assertEqual(data["as_of_date"], "2026-01-15")


if __name__ == "__main__":
    unittest.main()
