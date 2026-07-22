"""Unit tests for trading_crab_lib.platform.honesty.registry (HON-02).

Follows the incumbent tests/unit/test_platform_taxonomy.py structure: property
tests over a tmp_path-scoped ledger (never the real registry/trials.jsonl).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime

import pytest

from trading_crab_lib import ROOT
from trading_crab_lib.platform.honesty.registry import (
    DEFAULT_REGISTRY_PATH,
    append_trial,
    config_hash,
    read_trials,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _trial_kwargs(tag: str = "a") -> dict:
    return {
        "config": {"model": "rf", "tag": tag},
        "features": ["feat_1", "feat_2"],
        "metrics": {"sharpe": 0.5},
    }


# ── append_trial: append-only guarantee ─────────────────────────────────────────


class TestAppendNeverTruncates:
    def test_append_never_truncates(self, tmp_path):
        """Writing a sentinel trial then 3 more never rewrites the first line."""
        path = tmp_path / "trials.jsonl"
        append_trial(**_trial_kwargs("sentinel"), path=path)
        first_line_before = path.read_text().splitlines()[0]

        for i in range(3):
            append_trial(**_trial_kwargs(f"extra_{i}"), path=path)

        lines = path.read_text().splitlines()
        assert len(lines) == 4
        assert lines[0] == first_line_before

    def test_one_line_per_call(self, tmp_path):
        """N append_trial calls yield exactly N newline-terminated valid-JSON lines."""
        path = tmp_path / "trials.jsonl"
        n = 5
        for i in range(n):
            append_trial(**_trial_kwargs(str(i)), path=path)

        content = path.read_text()
        assert content.endswith("\n")
        lines = content.splitlines()
        assert len(lines) == n
        for line in lines:
            json.loads(line)  # each line is valid JSON, does not raise


# ── Row schema ───────────────────────────────────────────────────────────────────


class TestRowSchema:
    def test_row_schema(self, tmp_path):
        """Every appended row has the complete schema with a tz-aware ISO timestamp."""
        path = tmp_path / "trials.jsonl"
        row = append_trial(**_trial_kwargs(), path=path)

        assert set(row.keys()) == {
            "config_hash",
            "config",
            "features",
            "metrics",
            "git_sha",
            "timestamp",
        }
        parsed_ts = datetime.fromisoformat(row["timestamp"])
        assert parsed_ts.tzinfo is not None


# ── config_hash ────────────────────────────────────────────────────────────────


class TestConfigHash:
    def test_config_hash_deterministic_regardless_of_key_order(self):
        """Equal configs, differently-ordered dict literals, hash identically."""
        cfg_a = {"model": "rf", "depth": 5}
        cfg_b = {"depth": 5, "model": "rf"}
        assert config_hash(cfg_a) == config_hash(cfg_b)

    def test_config_hash_differs_for_changed_value(self):
        """A changed value hashes differently."""
        cfg_a = {"model": "rf", "depth": 5}
        cfg_b = {"model": "rf", "depth": 6}
        assert config_hash(cfg_a) != config_hash(cfg_b)

    def test_config_hash_length(self):
        """Hash is a 12-char hex string."""
        h = config_hash({"model": "rf"})
        assert len(h) == 12
        int(h, 16)  # valid hex, does not raise


# ── Append-mode guard ────────────────────────────────────────────────────────────


class TestOpenModeIsAppend:
    def test_open_mode_is_append(self):
        """The module source opens the ledger with 'a' mode only, never 'w'."""
        source = (ROOT / "src" / "trading_crab_lib" / "platform" / "honesty" / "registry.py").read_text()
        assert '"a"' in source or "'a'" in source
        # No ledger write in "w"/"wb" mode anywhere in the module.
        assert '"w"' not in source
        assert "'w'" not in source


# ── read_trials ──────────────────────────────────────────────────────────────────


class TestReadTrials:
    def test_read_trials_roundtrip(self, tmp_path):
        """After K appends, read_trials returns a K-row DataFrame with a config_hash column."""
        path = tmp_path / "trials.jsonl"
        k = 4
        for i in range(k):
            append_trial(**_trial_kwargs(str(i)), path=path)

        df = read_trials(path=path)
        assert len(df) == k
        assert "config_hash" in df.columns

    def test_read_trials_missing_path_returns_empty_dataframe(self, tmp_path):
        """A nonexistent path returns an empty DataFrame, does not raise."""
        df = read_trials(path=tmp_path / "does_not_exist.jsonl")
        assert df.empty


# ── git SHA capture ──────────────────────────────────────────────────────────────


class TestGitShaPresent:
    def test_git_sha_present(self, tmp_path):
        """The appended row's git_sha is a non-empty string."""
        path = tmp_path / "trials.jsonl"
        row = append_trial(**_trial_kwargs(), path=path)
        assert isinstance(row["git_sha"], str)
        assert len(row["git_sha"]) > 0


# ── Default path location ─────────────────────────────────────────────────────────


class TestDefaultPathNotUnderData:
    def test_default_path_not_under_data(self):
        """DEFAULT_REGISTRY_PATH is under ROOT, in 'registry/', never under 'data/'."""
        assert DEFAULT_REGISTRY_PATH.is_relative_to(ROOT)
        relative_parts = DEFAULT_REGISTRY_PATH.relative_to(ROOT).parts
        assert "registry" in relative_parts
        assert "data" not in relative_parts


# ── Git-trackable path (Task 2) ────────────────────────────────────────────────────


def test_registry_path_is_git_trackable():
    """`git check-ignore registry/trials.jsonl` exits non-zero — the path is not ignored.

    Does not write registry/trials.jsonl to the working tree; asserts on the
    path string only.
    """
    try:
        result = subprocess.run(
            ["git", "check-ignore", "registry/trials.jsonl"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("git is not available")
    assert result.returncode == 1
