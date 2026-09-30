"""Recorded test counts must equal a live collection (ROADMAP Phase 8 criterion 9, PER-10).

``CLAUDE.md`` and ``README.md`` state the suite's size at four sites. For most of this
project's history they said **1705** while the suite grew past 2000 — the shape of recorded
number that has misled this project before (``08-CONTEXT.md`` D-07). This module pins the
four sites against ``pytest --collect-only`` run live in a subprocess.

**Equality, not a bound.** A ``>=`` here would pass on every future growth of the suite and
so could only confirm — the defect criterion 9 exists to correct. Growing the suite
therefore turns this module red until the four sites are updated in the same commit.

**Each site has its own pattern**, so a failure names which of the four is stale, and a
pattern that matches nothing is a failure, never a skip: a silently unmatched site is how
three of the four could drift while this stayed green.

**No recursion.** ``--collect-only`` executes nothing. The subprocess still carries an
environment sentinel, and the test that spawns it asserts the sentinel is absent — so if
collection ever starts executing tests, this fails loudly instead of recursing.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SENTINEL = "TC_DOCS_RECORDED_COUNTS_COLLECT_ONLY"

#: (file, site name, pattern). One pattern per site, each with one capture group.
_SITES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("CLAUDE.md", "layout tree", re.compile(r"pytest test suite \((\d+) tests\)")),
    ("CLAUDE.md", "current status", re.compile(r"\*\*(\d+) tests collected\*\*")),
    ("README.md", "badge URL", re.compile(r"img\.shields\.io/badge/tests-(\d+)%20passing")),
    ("README.md", "feature list", re.compile(r"(\d+) tests \(unit \+ integration\)")),
)


def recorded_counts(texts: dict[str, str]) -> dict[str, list[int]]:
    """Every value each site's pattern finds, keyed ``"<file>: <site>"``."""
    return {
        f"{fname}: {site}": [int(v) for v in pattern.findall(texts.get(fname, ""))]
        for fname, site, pattern in _SITES
    }


def stale_sites(texts: dict[str, str], live: int) -> list[str]:
    """One message per site that is missing, ambiguous, or not equal to ``live``."""
    problems = []
    for key, values in recorded_counts(texts).items():
        if len(values) != 1:
            problems.append(f"{key}: expected exactly one recorded count, found {values or 'none'}")
        elif values[0] != live:
            problems.append(f"{key}: records {values[0]}, live collection is {live}")
    return problems


def _live_collected_count() -> int:
    assert os.environ.get(_SENTINEL) is None, (
        "this test is running inside its own --collect-only subprocess: collection has started "
        "EXECUTING tests. Stopping here rather than recursing."
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests/"],
        cwd=_ROOT,
        env={**os.environ, _SENTINEL: "1"},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, f"collection failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}"
    found = re.findall(r"^(\d+) tests? collected", proc.stdout, re.M)
    assert len(found) == 1, f"no single 'N tests collected' summary line in:\n{proc.stdout[-1000:]}"
    assert not re.search(r"\berrors?\b", proc.stdout.splitlines()[-1]), proc.stdout.splitlines()[-1]
    return int(found[0])


@pytest.fixture(scope="module")
def live() -> int:
    return _live_collected_count()


@pytest.fixture(scope="module")
def docs() -> dict[str, str]:
    return {name: (_ROOT / name).read_text(encoding="utf-8") for name in ("CLAUDE.md", "README.md")}


def test_all_four_sites_are_found_exactly_once(docs):
    counts = recorded_counts(docs)
    assert len(counts) == 4
    missing = {key: values for key, values in counts.items() if len(values) != 1}
    assert not missing, f"a recorded-count site is unmatched or ambiguous (a failure, not a skip): {missing}"


@pytest.mark.parametrize("fname,site,pattern", _SITES, ids=[f"{f}:{s}" for f, s, _ in _SITES])
def test_each_site_equals_the_live_collection(docs, live, fname, site, pattern):
    values = [int(v) for v in pattern.findall(docs[fname])]
    assert values == [live], f"{fname} ({site}) records {values}; `pytest --collect-only` collects {live}"


def test_the_comparison_fails_on_a_stale_document_and_names_both_values():
    """The failure path, exercised: without it a broken pattern would present as a pass."""
    stale = {
        "CLAUDE.md": "pytest test suite (1705 tests)\n**1705 tests collected**\n",
        "README.md": "img.shields.io/badge/tests-1705%20passing\n- 1705 tests (unit + integration)\n",
    }
    problems = stale_sites(stale, live=2361)
    assert len(problems) == 4
    for problem, (fname, site, _) in zip(problems, _SITES):
        assert fname in problem and site in problem and "1705" in problem and "2361" in problem
    assert stale_sites(stale, live=1705) == [], "the comparison must also accept a matching document"


def test_an_unmatched_site_is_reported_as_a_failure():
    """A document missing its badge must fail on the badge, not pass on the other three."""
    texts = {
        "CLAUDE.md": "pytest test suite (7 tests)\n**7 tests collected**\n",
        "README.md": "- 7 tests (unit + integration)\n",
    }
    assert stale_sites(texts, live=7) == ["README.md: badge URL: expected exactly one recorded count, found none"]
