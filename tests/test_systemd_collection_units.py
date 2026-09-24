"""Invariants of the collection/watchdog systemd units.

These units are not executed by CI, so every safety property they carry lives in
a file nobody runs.  This repo's recurring failure is a comment that stopped
being true — a timer claiming "never two concurrent runs" while nothing enforced
it, a watchdog documented as catching zero-row runs while its ExecStart never
passed the flag that does the catching.  Each test below pins one of those to
the unit file rather than to a comment.

Deliberately structural, not textual: they read directives, not prose, so
rewording a comment cannot break them and deleting a directive cannot pass them.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

SYSTEMD_DIR = Path(__file__).resolve().parent.parent / "deploy" / "systemd"

COLLECT_TIMER = SYSTEMD_DIR / "tirra-collect.timer"
WEEKEND_TIMER = SYSTEMD_DIR / "tirra-collect-continuous.timer"
FRESHNESS_TIMER = SYSTEMD_DIR / "tirra-freshness.timer"
FRESHNESS_SERVICE = SYSTEMD_DIR / "tirra-freshness.service"


def directives(path: Path) -> list[tuple[str, str]]:
    """Every ``Key=value`` line, comments and blanks dropped, in file order."""
    out: list[tuple[str, str]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        key, _, value = line.partition("=")
        out.append((key.strip(), value))
    return out


def values(path: Path, key: str) -> list[str]:
    return [v for k, v in directives(path) if k == key]


def value(path: Path, key: str) -> str:
    found = values(path, key)
    assert len(found) == 1, f"{path.name}: expected exactly one {key}=, got {found}"
    return found[0]


# ── concurrency: two collection processes on one WAL database ──


def test_weekend_timer_declares_no_persistent_catchup():
    """Persistent= here can queue a second daily_collection at boot.

    tirra-chain.timer already carries Persistent=true for a *different* unit.
    systemd serialises starts of the same unit, not of two units, so a boot after
    downtime spanning a weekday and a weekend slot would start tirra-chain.service
    and tirra-collect.service together — two writers, and the only write guard in
    PipelineStore is an in-process threading.RLock.  Restore this only together
    with a flock in run_scheduled.sh or Conflicts= on tirra-collect.service.
    """
    assert values(WEEKEND_TIMER, "Persistent") == []


def test_both_collection_timers_target_the_same_unit():
    """What actually makes the weekday/weekend split safe is job coalescing.

    Two timers pointing at one unit produce one job; the calendar split is a
    convenience on top of that.  Pointing this timer at a second service unit
    would silently remove the only real protection.
    """
    assert value(WEEKEND_TIMER, "Unit") == "tirra-collect.service"
    # tirra-collect.timer has no Unit= — systemd derives tirra-collect.service
    # from the timer's own name, which is the same unit.
    assert values(COLLECT_TIMER, "Unit") == []


def test_collection_timers_do_not_overlap_on_the_calendar():
    """Mon..Fri + Sat,Sun, not a widened Mon..Sun beside a weekend timer."""
    weekday = value(COLLECT_TIMER, "OnCalendar")
    weekend = value(WEEKEND_TIMER, "OnCalendar")
    assert weekday.startswith("Mon..Fri ")
    assert weekend.startswith("Sat,Sun ")
    # Same clock time: even 24h spacing for the point-in-time probes.
    assert weekday.split(None, 1)[1] == weekend.split(None, 1)[1]


# ── the watchdog actually asks the question it documents ──


def test_watchdog_runs_in_row_growth_mode_not_age_only():
    """--since-snapshot is what catches a green run that wrote nothing.

    Without it the unit grades ages, and an age threshold is satisfied by a
    single row — the partial-write failure this repo keeps shipping stays green.
    """
    assert "--since-snapshot" in value(FRESHNESS_SERVICE, "ExecStart")


def test_watchdog_tolerates_a_missing_baseline():
    """First run has no snapshot; comparing the DB to itself flags every source.

    The ExecStart must therefore carry a fallback invocation without the flag.
    """
    exec_start = value(FRESHNESS_SERVICE, "ExecStart")
    assert exec_start.count("check_freshness.py") == 2, (
        "expected a --since-snapshot invocation and a bare fallback invocation"
    )
    assert "-mmin" in exec_start, "expected a minimum-baseline-age guard"


def test_baseline_is_refreshed_even_when_the_check_fails():
    """ExecStartPost is skipped after a non-zero ExecStart.

    This unit is designed to exit non-zero, so a baseline written by
    ExecStartPost would freeze on the first red day and never move again.
    """
    assert values(FRESHNESS_SERVICE, "ExecStartPost") == []
    stop_post = value(FRESHNESS_SERVICE, "ExecStopPost")
    assert stop_post.startswith("-"), "a failed snapshot must not mask the check's exit code"


def test_watchdog_never_opens_the_pipeline_db_writable():
    """Both the check and the baseline copy read through mode=ro + query_only."""
    body = FRESHNESS_SERVICE.read_text()
    exec_lines = [v for k, v in directives(FRESHNESS_SERVICE) if k.startswith("Exec")]
    pipeline_opens = [line for line in exec_lines if "pipeline.db" in line]
    assert pipeline_opens, "no unit line touches the pipeline DB at all"
    for line in pipeline_opens:
        if "sqlite3.connect" in line:
            assert "mode=ro" in line and "query_only" in line, line
    assert "StateDirectory=tirramind" in body, "the baseline needs a writable state dir of its own"


def test_watchdog_exit_codes_are_not_masked():
    """SuccessExitStatus= turns the alarm back into the silence it exists to break."""
    assert values(FRESHNESS_SERVICE, "SuccessExitStatus") == []


# ── systemd would eat these before the shell ever saw them ──


@pytest.mark.parametrize(
    "path",
    [COLLECT_TIMER, WEEKEND_TIMER, FRESHNESS_TIMER, FRESHNESS_SERVICE],
    ids=lambda p: p.name,
)
def test_exec_lines_are_free_of_systemd_specifiers(path: Path):
    """``$VAR`` is expanded by systemd, ``%x`` is a unit specifier.

    Either one silently rewrites a command line before /bin/sh or python sees it,
    which for the baseline copy would mean writing to an empty path.
    """
    for key, val in directives(path):
        if not key.startswith("Exec"):
            continue
        assert "$" not in val, f"{path.name}: {key} would undergo systemd variable expansion"
        assert not re.search(r"%[a-zA-Z%]", val), f"{path.name}: {key} contains a systemd specifier"


@pytest.mark.parametrize(
    "path",
    [COLLECT_TIMER, WEEKEND_TIMER, FRESHNESS_TIMER, FRESHNESS_SERVICE],
    ids=lambda p: p.name,
)
def test_exec_lines_parse_as_systemd_would_split_them(path: Path):
    """systemd splits Exec= with shell-like quoting; an unbalanced quote is fatal."""
    for key, val in directives(path):
        if not key.startswith("Exec"):
            continue
        argv = shlex.split(val.lstrip("-@:+!"))
        assert argv, f"{path.name}: {key} split to nothing"
        assert argv[0].startswith("/"), f"{path.name}: {key} must use an absolute executable path"
