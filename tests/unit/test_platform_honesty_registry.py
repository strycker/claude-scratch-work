"""Unit tests for ``total_trial_count()`` (HON-02 / D-16, Phase 7 wave 2, Task 2).

Complements ``tests/unit/test_platform_registry.py`` (append_trial/read_trials coverage)
with a dedicated ``TestTotalTrialCount`` class. Synthetic ledgers are built directly under
``tmp_path`` (never the real registry) so every behavior — including the two-header-row
sum and the malformed-config degrade-gracefully path — is pinned by an exact expected
integer, not a shape check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_crab_lib.platform.honesty.registry import (
    DEFAULT_REGISTRY_PATH,
    PROVENANCE_RECORD_TYPE,
    append_trial,
    read_trials,
    total_trial_count,
)

# ── Helpers ──────────────────────────────────────────────────────────────────────


def _write_ledger(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _header_row(prior: int, *, discarded: int = 0) -> dict:
    return {
        "config_hash": "RESET",
        "config": {
            "trial_tag": f"SYNTHETIC-RESET-{prior}",
            "record_type": PROVENANCE_RECORD_TYPE,
            "prior_genuine_trials": prior,
            "discarded_smoke_rows": discarded,
        },
        "features": [],
        "metrics": {"prior_genuine_trials": prior, "discarded_smoke_rows": discarded},
        "git_sha": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _trial_row(tag: str) -> dict:
    return {
        "config_hash": "abc123",
        "config": {"trial_tag": tag, "model": "rf"},
        "features": ["f1"],
        "metrics": {"sharpe": 0.5},
        "git_sha": "deadbeef",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


# ── total_trial_count() ────────────────────────────────────────────────────────────


class TestTotalTrialCount:
    """D-16's denominator: header prior(s) + non-header rows, never a raw row count."""

    def test_header_only_returns_prior(self, tmp_path):
        """A ledger holding only the provenance header returns its prior, not 1."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38, discarded=4)])
        assert total_trial_count(path=path) == 38

    def test_header_plus_one_genuine_row_returns_39(self, tmp_path):
        """After one genuine tagged row is appended, the count is 39."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38, discarded=4)])
        append_trial(
            config={"trial_tag": "P7-W2-first-real-trial"},
            features=["f1"],
            metrics={"sharpe": 0.3},
            path=path,
        )
        assert total_trial_count(path=path) == 39

    def test_two_header_rows_sum_both_priors(self, tmp_path):
        """Two header rows (priors 10 and 5) plus 3 ordinary rows -> 18, not 13 or 8."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(
            path,
            [
                _header_row(10),
                _trial_row("a"),
                _header_row(5),
                _trial_row("b"),
                _trial_row("c"),
            ],
        )
        assert total_trial_count(path=path) == 18

    def test_no_header_returns_plain_row_count(self, tmp_path):
        """A ledger with no header row and 7 ordinary rows returns exactly 7."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_trial_row(str(i)) for i in range(7)])
        assert total_trial_count(path=path) == 7

    def test_missing_ledger_returns_zero_without_raising(self, tmp_path):
        """A nonexistent ledger path returns 0, does not raise."""
        assert total_trial_count(path=tmp_path / "does_not_exist.jsonl") == 0

    def test_empty_ledger_file_returns_zero(self, tmp_path):
        """An existing-but-empty ledger file returns 0."""
        path = tmp_path / "trials.jsonl"
        path.write_text("")
        assert total_trial_count(path=path) == 0

    def test_malformed_config_counts_as_a_trial_not_a_crash(self, tmp_path):
        """A row whose config is not a dict (e.g. null) degrades to 'counted as a trial'."""
        path = tmp_path / "trials.jsonl"
        rows = [
            {
                "config_hash": "weird",
                "config": None,
                "features": [],
                "metrics": {},
                "git_sha": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            _trial_row("ordinary"),
        ]
        _write_ledger(path, rows)
        # Neither row is a recognized header -> both counted -> 2, no exception.
        assert total_trial_count(path=path) == 2

    def test_header_missing_prior_key_defaults_to_zero(self, tmp_path):
        """A header row lacking prior_genuine_trials contributes 0, not a raised KeyError."""
        path = tmp_path / "trials.jsonl"
        rows = [
            {
                "config_hash": "RESET",
                "config": {"trial_tag": "BROKEN-RESET", "record_type": PROVENANCE_RECORD_TYPE},
                "features": [],
                "metrics": {},
                "git_sha": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            _trial_row("a"),
            _trial_row("b"),
        ]
        _write_ledger(path, rows)
        # Header contributes 0 (missing key), 2 ordinary rows -> 2.
        assert total_trial_count(path=path) == 2

    def test_result_never_below_headers_own_prior(self, tmp_path):
        """The floor invariant: total_trial_count is never less than the header's own prior."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38, discarded=4)])
        count = total_trial_count(path=path)
        assert count >= 38, f"total_trial_count() fell below the header's own prior (38): got {count}"

    def test_live_ledger_floor_at_38(self):
        """Regression pin: a future reset that loses the header must not undercount below 38.

        Reads the REAL registry/trials.jsonl (read-only) — skips cleanly if absent (fresh
        checkout with no registry committed yet).
        """
        if not DEFAULT_REGISTRY_PATH.exists():
            pytest.skip("registry/trials.jsonl does not exist in this checkout")
        count = total_trial_count()
        assert count >= 38, (
            f"total_trial_count() against the live ledger returned {count}, below the "
            "provenance header's own stated floor of 38 prior_genuine_trials — this means "
            "the header was not read (undercount) or was lost entirely."
        )

    def test_read_trials_unchanged_stays_header_unaware(self, tmp_path):
        """read_trials() itself must stay a bare reader — total_trial_count wraps it, not the reverse."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38, discarded=4), _trial_row("a")])
        df = read_trials(path=path)
        assert len(df) == 2  # read_trials sees the header AS a row; it does not special-case it
