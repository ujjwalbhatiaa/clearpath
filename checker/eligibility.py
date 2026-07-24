"""
eligibility.py -- personal compliance & PGWP eligibility rule engine for
ClearPath's Phase 1 checker.

Pure deterministic rule logic over a structured Profile -- no API key, no
model, no network call, no generative synthesis. Every rule traces to a
cited chunk (id) resolvable in corpus_loader.CHUNKS_BY_ID (a merge of
../qa/corpus/chunks.json and rules_corpus.json).

Design notes (see checker/README.md for the full rationale):
- Every rule function returns a RuleResult with an explicit Status --
  COMPLIANT / VIOLATION / NOT_ENOUGH_INFO / NOT_RESOLVED. An unset profile
  field degrades to NOT_ENOUGH_INFO rather than being guessed at a default.
  NOT_RESOLVED is a 4th, deliberately out-of-scope status for cases this
  checker recognizes but does not attempt to resolve (CIP-code field-of-
  study lookup, flight-school special case) -- see NEXT-BUILD-SPEC.md
  "Scope out".
- The 24-hour off-campus cap is ALWAYS 24, regardless of what a legacy
  study permit prints (see effective_off_campus_weekly_cap and the
  legacy_permit_text_20_hours chunk) -- a naive implementation might read
  the printed number directly; this module makes that mistake structurally
  impossible by never reading study_permit_printed_hours as the cap.
- Scheduled-break/day-count math (7-day minimum, 150-consecutive-day
  chaining, 180-day/calendar-year budget) is implemented at day
  granularity using only stdlib `datetime` -- see compute_break_day_status.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (every number here traces to a cited chunk -- see corpus_loader)
# ---------------------------------------------------------------------------

OFF_CAMPUS_WEEKLY_CAP_HOURS = 24  # hour_cap_24 / legacy_permit_text_20_hours
SCHEDULED_BREAK_MIN_DAYS = 7  # annual_180_day_cap
SCHEDULED_BREAK_CHAIN_MAX_DAYS = 150  # annual_180_day_cap
SCHEDULED_BREAK_ANNUAL_MAX_DAYS = 180  # annual_180_day_cap
PGWP_MIN_PROGRAM_MONTHS = 8  # pgwp_general_eligibility
PGWP_APPLICATION_WINDOW_DAYS = 180  # pgwp_general_eligibility
FIELD_OF_STUDY_CUTOFF_DATE = date(2024, 11, 1)  # pgwp_field_of_study_exempt

NO_FIELD_OF_STUDY_REQUIREMENT_LEVELS = {"bachelors", "masters", "doctoral"}
DATE_GATED_FIELD_OF_STUDY_LEVELS = {"other_university", "college_polytechnic"}
CREDENTIAL_LEVELS = NO_FIELD_OF_STUDY_REQUIREMENT_LEVELS | DATE_GATED_FIELD_OF_STUDY_LEVELS | {
    "flight_school"
}


class Status(str, Enum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    NOT_ENOUGH_INFO = "not_enough_info"
    NOT_RESOLVED = "not_resolved"  # recognized, deliberately out of scope -- see README limitations


# ---------------------------------------------------------------------------
# Profile data shape
# ---------------------------------------------------------------------------


@dataclass
class ScheduledBreak:
    start_date: date
    end_date: date  # inclusive

    @staticmethod
    def from_dict(d: dict) -> "ScheduledBreak":
        return ScheduledBreak(
            start_date=date.fromisoformat(d["start_date"]),
            end_date=date.fromisoformat(d["end_date"]),
        )


@dataclass
class WeekEntry:
    week_start: date  # first of 7 consecutive days this entry covers
    off_campus_hours: Optional[float] = None
    on_campus_hours: float = 0.0
    label: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "WeekEntry":
        return WeekEntry(
            week_start=date.fromisoformat(d["week_start"]),
            off_campus_hours=d.get("off_campus_hours"),
            on_campus_hours=d.get("on_campus_hours", 0.0) or 0.0,
            label=d.get("label"),
        )


# The 8 PGWP "who's not eligible" exclusion fields, paired with a
# human-readable reason for the report. Grounded in
# pgwp_not_eligible_exclusions (see rules_corpus.json).
EXCLUSION_FIELDS: List[Tuple[str, str]] = [
    ("already_received_pgwp", "already received a PGWP before"),
    ("studied_esl_or_fsl_only", "studied only English or French as a second language"),
    ("took_general_interest_courses_only", "took only general interest or self-improvement courses"),
    (
        "completed_non_credit_program",
        "completed a non-credit program of study (an exception exists for flight school programs)",
    ),
    (
        "received_gac_funding_with_return_requirement",
        "received Global Affairs Canada funding/scholarship that requires returning to their home "
        "country after graduation",
    ),
    ("over_50pct_distance_learning", "completed more than 50% of the study program through distance learning"),
    (
        "studied_at_non_canadian_institution_in_canada",
        "studied at a non-Canadian institution located in Canada",
    ),
    (
        "studied_at_non_pgwp_eligible_dli_or_p3_program",
        "studied at a designated learning institution/program that isn't PGWP-eligible, including a "
        "non-grandfathered curriculum-licensing (P3) program",
    ),
]


@dataclass
class Profile:
    name: Optional[str] = None
    as_of_date: Optional[date] = None

    # Work-hour tracking
    weekly_hours: List[WeekEntry] = field(default_factory=list)
    scheduled_breaks: List[ScheduledBreak] = field(default_factory=list)
    study_permit_printed_hours: Optional[int] = None  # literal number printed on the physical permit

    # Academic / PGWP profile
    credential_level: Optional[str] = None  # one of CREDENTIAL_LEVELS
    program_start_date: Optional[date] = None
    program_end_date: Optional[date] = None
    full_time_each_semester: Optional[bool] = None
    study_permit_application_date: Optional[date] = None
    program_completion_confirmation_date: Optional[date] = None
    pgwp_application_date: Optional[date] = None
    study_permit_valid_during_window: Optional[bool] = None

    # PGWP exclusion flags -- True/False/None (unknown)
    already_received_pgwp: Optional[bool] = None
    studied_esl_or_fsl_only: Optional[bool] = None
    took_general_interest_courses_only: Optional[bool] = None
    completed_non_credit_program: Optional[bool] = None
    received_gac_funding_with_return_requirement: Optional[bool] = None
    over_50pct_distance_learning: Optional[bool] = None
    studied_at_non_canadian_institution_in_canada: Optional[bool] = None
    studied_at_non_pgwp_eligible_dli_or_p3_program: Optional[bool] = None

    @staticmethod
    def from_dict(d: dict) -> "Profile":
        def opt_date(key: str) -> Optional[date]:
            v = d.get(key)
            return date.fromisoformat(v) if v else None

        return Profile(
            name=d.get("name"),
            as_of_date=opt_date("as_of_date"),
            weekly_hours=[WeekEntry.from_dict(w) for w in d.get("weekly_hours", [])],
            scheduled_breaks=[ScheduledBreak.from_dict(b) for b in d.get("scheduled_breaks", [])],
            study_permit_printed_hours=d.get("study_permit_printed_hours"),
            credential_level=d.get("credential_level"),
            program_start_date=opt_date("program_start_date"),
            program_end_date=opt_date("program_end_date"),
            full_time_each_semester=d.get("full_time_each_semester"),
            study_permit_application_date=opt_date("study_permit_application_date"),
            program_completion_confirmation_date=opt_date("program_completion_confirmation_date"),
            pgwp_application_date=opt_date("pgwp_application_date"),
            study_permit_valid_during_window=d.get("study_permit_valid_during_window"),
            already_received_pgwp=d.get("already_received_pgwp"),
            studied_esl_or_fsl_only=d.get("studied_esl_or_fsl_only"),
            took_general_interest_courses_only=d.get("took_general_interest_courses_only"),
            completed_non_credit_program=d.get("completed_non_credit_program"),
            received_gac_funding_with_return_requirement=d.get("received_gac_funding_with_return_requirement"),
            over_50pct_distance_learning=d.get("over_50pct_distance_learning"),
            studied_at_non_canadian_institution_in_canada=d.get("studied_at_non_canadian_institution_in_canada"),
            studied_at_non_pgwp_eligible_dli_or_p3_program=d.get(
                "studied_at_non_pgwp_eligible_dli_or_p3_program"
            ),
        )


@dataclass
class RuleResult:
    rule: str
    status: Status
    explanation: str
    citations: List[str]


@dataclass
class AssessmentReport:
    profile_name: Optional[str]
    as_of_date: date
    results: List[RuleResult]

    def summary(self) -> Dict[str, int]:
        counts = {s.value: 0 for s in Status}
        for r in self.results:
            counts[r.status.value] += 1
        return counts


# ---------------------------------------------------------------------------
# Date arithmetic helpers (stdlib only, per NEXT-BUILD-SPEC.md's note)
# ---------------------------------------------------------------------------


def add_months(d: date, months: int) -> date:
    """Adds `months` calendar months to `d`, clamping the day if the target
    month is shorter (e.g. Jan 31 + 1 month -> Feb 28/29, not Mar 3)."""
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def effective_off_campus_weekly_cap(profile: Profile) -> int:
    """Always 24, regardless of what a legacy permit prints.

    Some students still hold a study permit physically printed with a
    stale "20 hours per week" condition; the current rule (see
    legacy_permit_text_20_hours) is that 24 hours/week applies regardless,
    as long as normal eligibility is met. A naive implementation might
    read `profile.study_permit_printed_hours` and use that number as the
    cap directly -- this function exists specifically to make that mistake
    impossible to make silently: `profile` is accepted as an argument for
    API symmetry/future-proofing, but its printed-hours field is never
    read here.
    """
    return OFF_CAMPUS_WEEKLY_CAP_HOURS


def _merge_breaks(breaks: List[ScheduledBreak]) -> List[Tuple[date, date]]:
    """Merges scheduled breaks that are back-to-back (no gap day) or
    overlapping into continuous chained intervals, per the "back-to-back
    scheduled breaks" rule in annual_180_day_cap. Returns a sorted list of
    inclusive (start, end) merged intervals."""
    if not breaks:
        return []
    ordered = sorted((b.start_date, b.end_date) for b in breaks)
    merged: List[Tuple[date, date]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + timedelta(days=1):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _qualifying_intervals(breaks: List[ScheduledBreak]) -> List[Tuple[date, date]]:
    """Only merged intervals of >= SCHEDULED_BREAK_MIN_DAYS count as a
    "scheduled break" at all -- a single short break (e.g. a 5-day reading
    week, a lone stat holiday) does not, per annual_180_day_cap."""
    merged = _merge_breaks(breaks)
    return [(s, e) for (s, e) in merged if (e - s).days + 1 >= SCHEDULED_BREAK_MIN_DAYS]


@dataclass
class BreakDayStatus:
    date: date
    is_break_day: bool
    chain_position: Optional[int]  # 1-indexed day number within its qualifying interval
    cumulative_year_days_used: int  # authorized-unlimited days used this calendar year, through this date
    authorized_unlimited: bool


def compute_break_day_status(scheduled_breaks: List[ScheduledBreak], target_date: date) -> BreakDayStatus:
    """Walks day-by-day from Jan 1 of target_date's year through
    target_date, tracking (a) whether each day falls in a qualifying
    (>=7-consecutive-day) scheduled break, (b) its 1-indexed position
    within that break's chained interval (for the 150-consecutive-day
    cap), and (c) the running count of unlimited-hours days authorized so
    far this calendar year (for the 180-day/year cap). A day is only
    "authorized_unlimited" if it clears BOTH caps.
    """
    intervals = _qualifying_intervals(scheduled_breaks)
    year_start = date(target_date.year, 1, 1)
    running = 0
    result: Optional[BreakDayStatus] = None
    d = year_start
    while d <= target_date:
        interval = next(((s, e) for (s, e) in intervals if s <= d <= e), None)
        if interval is not None:
            chain_pos = (d - interval[0]).days + 1
            if chain_pos <= SCHEDULED_BREAK_CHAIN_MAX_DAYS and running < SCHEDULED_BREAK_ANNUAL_MAX_DAYS:
                authorized = True
                running += 1
            else:
                authorized = False
        else:
            chain_pos = None
            authorized = False
        if d == target_date:
            result = BreakDayStatus(
                date=d,
                is_break_day=interval is not None,
                chain_position=chain_pos,
                cumulative_year_days_used=running,
                authorized_unlimited=authorized,
            )
        d += timedelta(days=1)
    assert result is not None  # loop always reaches target_date since year_start <= target_date
    return result


# ---------------------------------------------------------------------------
# Rule functions
# ---------------------------------------------------------------------------


def check_weekly_hours(profile: Profile, week: WeekEntry) -> RuleResult:
    """Applies the 24-hour off-campus cap only to off-campus, term-time
    hours. On-campus hours are always uncapped (on_campus_unlimited).
    Hours during a fully-qualifying scheduled break week are uncapped
    off-campus too, bounded by the 150/180-day rules (annual_180_day_cap).
    """
    label = week.label or f"week of {week.week_start.isoformat()}"

    if week.off_campus_hours is None:
        return RuleResult(
            rule="weekly_hours",
            status=Status.NOT_ENOUGH_INFO,
            explanation=f"{label}: no off-campus hours were recorded for this week -- cannot assess.",
            citations=["hour_cap_24"],
        )

    days = [week.week_start + timedelta(days=i) for i in range(7)]
    day_statuses = [compute_break_day_status(profile.scheduled_breaks, d) for d in days]
    week_fully_unlimited = all(s.authorized_unlimited for s in day_statuses)
    on_campus_hours = week.on_campus_hours or 0.0

    legacy_note = ""
    citations = ["hour_cap_24"]
    if profile.study_permit_printed_hours is not None and profile.study_permit_printed_hours != OFF_CAMPUS_WEEKLY_CAP_HOURS:
        legacy_note = (
            f" Note: your study permit is printed with a '{profile.study_permit_printed_hours} hours/week' "
            f"condition, but this checker always applies the current {OFF_CAMPUS_WEEKLY_CAP_HOURS}-hour/week "
            "rule, not the number printed on the permit."
        )
        citations.append("legacy_permit_text_20_hours")

    if week_fully_unlimited:
        status = Status.COMPLIANT
        explanation = (
            f"{label}: every day this week falls within a qualifying, budget-authorized scheduled break, "
            f"so off-campus hours ({week.off_campus_hours}) are unlimited. On-campus hours "
            f"({on_campus_hours}) are always unlimited regardless."
        )
        citations = ["scheduled_break_unlimited", "annual_180_day_cap", "on_campus_unlimited"]
    else:
        cap = effective_off_campus_weekly_cap(profile)
        if week.off_campus_hours > cap:
            status = Status.VIOLATION
            explanation = (
                f"{label}: {week.off_campus_hours} off-campus hours exceeds the {cap}-hour/week cap "
                f"that applies during term time. On-campus hours ({on_campus_hours}) are unaffected -- "
                "there is no limit on on-campus hours."
            )
        else:
            status = Status.COMPLIANT
            explanation = (
                f"{label}: {week.off_campus_hours} off-campus hours is within the {cap}-hour/week "
                f"term-time cap. On-campus hours ({on_campus_hours}) are unaffected -- there is no "
                "limit on on-campus hours."
            )
        citations.append("on_campus_unlimited")

    return RuleResult(rule="weekly_hours", status=status, explanation=explanation + legacy_note, citations=citations)


def check_scheduled_break_budget(profile: Profile, as_of_date: date) -> RuleResult:
    """Reports the scheduled-break budget status as of a given date: is
    today part of a qualifying break, what position in the current chain,
    and how many unlimited-hours days have been used this calendar year.
    This is an informational status check (it does not itself know actual
    hours worked on `as_of_date`), not a pass/fail judgment of behavior --
    see check_weekly_hours for the rule that actually evaluates hours."""
    s = compute_break_day_status(profile.scheduled_breaks, as_of_date)

    if not s.is_break_day:
        return RuleResult(
            rule="scheduled_break_budget",
            status=Status.COMPLIANT,
            explanation=(
                f"{as_of_date.isoformat()} is not part of a qualifying scheduled break (a scheduled break "
                f"must be at least {SCHEDULED_BREAK_MIN_DAYS} consecutive days). The standard "
                f"{OFF_CAMPUS_WEEKLY_CAP_HOURS}-hour/week off-campus cap applies."
            ),
            citations=["annual_180_day_cap"],
        )

    if s.authorized_unlimited:
        return RuleResult(
            rule="scheduled_break_budget",
            status=Status.COMPLIANT,
            explanation=(
                f"{as_of_date.isoformat()} is day {s.chain_position} of a qualifying scheduled break "
                f"(<= {SCHEDULED_BREAK_CHAIN_MAX_DAYS}-consecutive-day limit) and unlimited-hours day "
                f"{s.cumulative_year_days_used} used so far this calendar year "
                f"(<= {SCHEDULED_BREAK_ANNUAL_MAX_DAYS}-day/year limit) -- unlimited off-campus hours "
                "are authorized today."
            ),
            citations=["annual_180_day_cap", "scheduled_break_unlimited"],
        )

    if s.chain_position is not None and s.chain_position > SCHEDULED_BREAK_CHAIN_MAX_DAYS:
        reason = (
            f"day {s.chain_position} of the current scheduled-break chain, which exceeds the "
            f"{SCHEDULED_BREAK_CHAIN_MAX_DAYS}-consecutive-day limit"
        )
    else:
        reason = (
            f"the {SCHEDULED_BREAK_ANNUAL_MAX_DAYS}-day/calendar-year unlimited-hours budget has already "
            "been used up"
        )
    return RuleResult(
        rule="scheduled_break_budget",
        status=Status.VIOLATION,
        explanation=(
            f"{as_of_date.isoformat()} falls within a scheduled break, but unlimited-hours authorization "
            f"has lapsed for today: {reason}. The standard {OFF_CAMPUS_WEEKLY_CAP_HOURS}-hour/week "
            "off-campus cap applies today, not unlimited hours."
        ),
        citations=["annual_180_day_cap"],
    )


def check_pgwp_general_eligibility(profile: Profile) -> RuleResult:
    """Evaluates the general PGWP eligibility checklist (program length,
    full-time-per-semester, 180-day application window, permit validity
    during that window) plus the 8-item exclusion list, per
    pgwp_general_eligibility and pgwp_not_eligible_exclusions."""

    # Exclusions are checked first and short-circuit: any True disqualifies
    # regardless of everything else.
    triggered = []
    for field_name, desc in EXCLUSION_FIELDS:
        if getattr(profile, field_name) is True:
            triggered.append(desc)
    if triggered:
        return RuleResult(
            rule="pgwp_general_eligibility",
            status=Status.VIOLATION,
            explanation="Not eligible for a PGWP due to exclusion: " + "; ".join(triggered) + ".",
            citations=["pgwp_not_eligible_exclusions"],
        )

    checks: Dict[str, Optional[bool]] = {}

    if profile.program_start_date and profile.program_end_date:
        min_end = add_months(profile.program_start_date, PGWP_MIN_PROGRAM_MONTHS)
        checks["program_length_at_least_8_months"] = profile.program_end_date >= min_end
    else:
        checks["program_length_at_least_8_months"] = None

    checks["full_time_each_semester"] = profile.full_time_each_semester

    if profile.pgwp_application_date and profile.program_completion_confirmation_date:
        delta_days = (profile.pgwp_application_date - profile.program_completion_confirmation_date).days
        checks["applied_within_180_day_window"] = 0 <= delta_days <= PGWP_APPLICATION_WINDOW_DAYS
    else:
        checks["applied_within_180_day_window"] = None

    checks["study_permit_valid_during_window"] = profile.study_permit_valid_during_window

    failed = [k for k, v in checks.items() if v is False]
    if failed:
        return RuleResult(
            rule="pgwp_general_eligibility",
            status=Status.VIOLATION,
            explanation="Fails PGWP general eligibility on: " + ", ".join(failed) + ".",
            citations=["pgwp_general_eligibility"],
        )

    missing = [k for k, v in checks.items() if v is None]
    if missing:
        return RuleResult(
            rule="pgwp_general_eligibility",
            status=Status.NOT_ENOUGH_INFO,
            explanation="Not enough information to assess: " + ", ".join(missing) + ".",
            citations=["pgwp_general_eligibility"],
        )

    return RuleResult(
        rule="pgwp_general_eligibility",
        status=Status.COMPLIANT,
        explanation=(
            "Meets all general PGWP eligibility criteria (program length, full-time status each semester, "
            "180-day application window, permit validity during that window), and no exclusions apply."
        ),
        citations=["pgwp_general_eligibility", "pgwp_not_eligible_exclusions"],
    )


def check_pgwp_field_of_study(profile: Profile) -> RuleResult:
    """Branches on credential level, per pgwp_field_of_study_exempt.
    Bachelor's/master's/doctoral graduates have no field-of-study
    requirement, ever. Graduates of "any other university program" or a
    college/polytechnic program need an eligible field of study only if
    their study permit application was submitted on/after Nov 1, 2024.
    Flight-school graduates have a distinct condition not modeled here.
    This function never performs a CIP-code lookup -- see README
    "Honest limitations"."""
    level = profile.credential_level

    if level is None:
        return RuleResult(
            rule="pgwp_field_of_study",
            status=Status.NOT_ENOUGH_INFO,
            explanation="credential_level was not provided -- cannot assess the field-of-study requirement.",
            citations=["pgwp_field_of_study_exempt"],
        )

    if level in NO_FIELD_OF_STUDY_REQUIREMENT_LEVELS:
        return RuleResult(
            rule="pgwp_field_of_study",
            status=Status.COMPLIANT,
            explanation=(
                f"No field-of-study requirement applies to {level} graduates, regardless of when the "
                "study permit application was submitted."
            ),
            citations=["pgwp_field_of_study_exempt"],
        )

    if level == "flight_school":
        return RuleResult(
            rule="pgwp_field_of_study",
            status=Status.NOT_RESOLVED,
            explanation=(
                "Flight school graduates have a distinct pilot-license/instructor condition instead of "
                "the standard field-of-study/language requirements -- not modeled by this checker. "
                "Consult canada.ca directly."
            ),
            citations=["pgwp_field_of_study_exempt"],
        )

    if level in DATE_GATED_FIELD_OF_STUDY_LEVELS:
        if profile.study_permit_application_date is None:
            return RuleResult(
                rule="pgwp_field_of_study",
                status=Status.NOT_ENOUGH_INFO,
                explanation=(
                    "study_permit_application_date was not provided -- cannot determine whether the "
                    f"field-of-study requirement applies to this {level} graduate."
                ),
                citations=["pgwp_field_of_study_exempt"],
            )
        if profile.study_permit_application_date < FIELD_OF_STUDY_CUTOFF_DATE:
            return RuleResult(
                rule="pgwp_field_of_study",
                status=Status.COMPLIANT,
                explanation=(
                    f"Study permit application ({profile.study_permit_application_date.isoformat()}) was "
                    f"submitted before {FIELD_OF_STUDY_CUTOFF_DATE.isoformat()} -- exempt from the "
                    f"field-of-study requirement for {level} graduates."
                ),
                citations=["pgwp_field_of_study_exempt"],
            )
        return RuleResult(
            rule="pgwp_field_of_study",
            status=Status.NOT_RESOLVED,
            explanation=(
                f"Study permit application ({profile.study_permit_application_date.isoformat()}) was "
                f"submitted on/after {FIELD_OF_STUDY_CUTOFF_DATE.isoformat()} -- {level} graduates need an "
                "eligible field of study (6-digit CIP code lookup). This checker does not resolve specific "
                "CIP codes -- see the Q&A tool or consult canada.ca's field-of-study page directly."
            ),
            citations=["pgwp_field_of_study_exempt", "pgwp_field_of_study_freeze_2026"],
        )

    return RuleResult(
        rule="pgwp_field_of_study",
        status=Status.NOT_ENOUGH_INFO,
        explanation=f"Unrecognized credential_level {level!r} -- expected one of {sorted(CREDENTIAL_LEVELS)}.",
        citations=[],
    )


def assess(profile: Profile, as_of_date: Optional[date] = None) -> AssessmentReport:
    """Runs every rule area and returns one structured report."""
    resolved_as_of = as_of_date or profile.as_of_date or date.today()

    results: List[RuleResult] = []
    for week in profile.weekly_hours:
        results.append(check_weekly_hours(profile, week))
    results.append(check_scheduled_break_budget(profile, resolved_as_of))
    results.append(check_pgwp_general_eligibility(profile))
    results.append(check_pgwp_field_of_study(profile))

    return AssessmentReport(profile_name=profile.name, as_of_date=resolved_as_of, results=results)
