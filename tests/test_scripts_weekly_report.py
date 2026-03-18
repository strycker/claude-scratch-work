"""Tests for scripts/run_weekly_report.py (Phase 6)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the script as a module (script lives in scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import run_weekly_report as weekly  # noqa: E402


# ── Archive logic (timestamped copy + email_body.txt) ────────────────────────

class TestArchiveWeeklyReport:
    def test_creates_timestamped_copy_and_email_body(self, tmp_path):
        content = "# Weekly Regime Report\n\n**Current regime:** 2 — Growth Boom\n"
        (tmp_path / "weekly_report.md").write_text(content, encoding="utf-8")

        weekly.archive_weekly_report(tmp_path)

        today = date.today().isoformat()
        stamped = tmp_path / f"weekly_{today}.md"
        email_body = tmp_path / "email_body.txt"
        assert stamped.exists()
        assert email_body.exists()
        assert stamped.read_text(encoding="utf-8") == content
        assert email_body.read_text(encoding="utf-8") == content

    def test_skips_when_no_report(self, tmp_path):
        # No weekly_report.md in dir
        weekly.archive_weekly_report(tmp_path)

        today = date.today().isoformat()
        assert not (tmp_path / f"weekly_{today}.md").exists()
        assert not (tmp_path / "email_body.txt").exists()

    def test_email_body_matches_report_content(self, tmp_path):
        line = "Confidence in this regime: 75%.\n"
        (tmp_path / "weekly_report.md").write_text(line, encoding="utf-8")

        weekly.archive_weekly_report(tmp_path)

        assert (tmp_path / "email_body.txt").read_text(encoding="utf-8") == line


# ── CLI argv (subprocess args) ────────────────────────────────────────────────

class TestScriptArgv:
    def test_default_argv_includes_steps_2_through_7(self):
        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            with patch("sys.argv", ["run_weekly_report.py"]):
                result = weekly.main()
            assert result == 0
            m.assert_called_once()
            argv = m.call_args[0][0]
            assert "--steps" in argv
            idx = argv.index("--steps")
            assert argv[idx + 1] == "2,3,4,5,6,7"

    def test_full_argv_includes_steps_1_through_7(self):
        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            with patch("sys.argv", ["run_weekly_report.py", "--full"]):
                result = weekly.main()
            assert result == 0
            argv = m.call_args[0][0]
            assert "--steps" in argv
            idx = argv.index("--steps")
            assert argv[idx + 1] == "1,2,3,4,5,6,7"

    def test_plots_and_verbose_passed_through(self):
        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            with patch("sys.argv", ["run_weekly_report.py", "--full", "--plots", "--verbose"]):
                result = weekly.main()
            assert result == 0
            argv = m.call_args[0][0]
            assert "--plots" in argv
            assert "--verbose" in argv
            assert argv[argv.index("--steps") + 1] == "1,2,3,4,5,6,7"

    def test_returns_nonzero_when_pipeline_fails(self):
        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=1)
            with patch("sys.argv", ["run_weekly_report.py"]):
                result = weekly.main()
            assert result == 1


class TestScriptSendEmail:
    def test_send_email_flag_calls_helpers_when_config_valid(self, tmp_path, monkeypatch):
        # Point REPORTS_DIR to a temp dir with a weekly_report.md present
        monkeypatch.setattr(weekly, "REPORTS_DIR", tmp_path)
        (tmp_path / "weekly_report.md").write_text("# Report\n", encoding="utf-8")

        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            monkeypatch.setattr("run_weekly_report.load_email_config", lambda: {"smtp_host": "h", "smtp_port": 587, "username": "u", "password": "p", "from_address": "f", "to_address": "t", "use_tls": True, "use_ssl": False})
            monkeypatch.setattr("run_weekly_report.send_weekly_email", lambda cfg, subj, body: True)
            with patch("sys.argv", ["run_weekly_report.py", "--send-email"]):
                result = weekly.main()
            assert result == 0

    def test_send_email_skips_when_config_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(weekly, "REPORTS_DIR", tmp_path)
        (tmp_path / "weekly_report.md").write_text("# Report\n", encoding="utf-8")

        with patch("run_weekly_report.subprocess.run") as m:
            m.return_value = MagicMock(returncode=0)
            monkeypatch.setattr("run_weekly_report.load_email_config", lambda: {})
            send_mock = MagicMock()
            monkeypatch.setattr("run_weekly_report.send_weekly_email", send_mock)
            with patch("sys.argv", ["run_weekly_report.py", "--send-email"]):
                result = weekly.main()
            assert result == 0
            send_mock.assert_not_called()
