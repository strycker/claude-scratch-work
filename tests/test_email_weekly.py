"""Tests for email.py and scripts/run_weekly_report.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from market_regime.email import (
    build_weekly_email_body,
    load_email_config,
    send_weekly_email,
)


# ── load_email_config tests ──────────────────────────────────────────────────


def test_missing_file_returns_empty(tmp_path):
    result = load_email_config(tmp_path / "nonexistent.yaml")
    assert result == {}


def test_loads_valid_config(tmp_path):
    cfg_file = tmp_path / "email.yaml"
    cfg_file.write_text(
        "smtp_host: smtp.example.com\n"
        "smtp_port: 587\n"
        "username: user@example.com\n"
        "password: secret\n"
        "sender: user@example.com\n"
        "recipients:\n  - admin@example.com\n"
    )
    result = load_email_config(cfg_file)
    assert result["smtp_host"] == "smtp.example.com"
    assert result["smtp_port"] == 587
    assert result["recipients"] == ["admin@example.com"]


def test_malformed_yaml_returns_empty(tmp_path):
    cfg_file = tmp_path / "email.yaml"
    cfg_file.write_text("just a string, not a dict")
    result = load_email_config(cfg_file)
    assert result == {}


# ── build_weekly_email_body tests ────────────────────────────────────────────


def test_build_body_from_email_body_txt(tmp_path):
    (tmp_path / "email_body.txt").write_text("Hello from the pipeline!")
    subject, body = build_weekly_email_body(tmp_path)
    assert "Market Regime Weekly Report" in subject
    assert body == "Hello from the pipeline!"


def test_build_body_from_weekly_report_md(tmp_path):
    (tmp_path / "weekly_report.md").write_text("# Weekly Report\nAll good.")
    subject, body = build_weekly_email_body(tmp_path)
    assert "# Weekly Report" in body


def test_build_body_from_dashboard_csv(tmp_path):
    (tmp_path / "dashboard.csv").write_text("asset,signal\nSPY,GREEN\nGLD,YELLOW\n")
    subject, body = build_weekly_email_body(tmp_path)
    assert "Dashboard summary" in body
    assert "SPY" in body


def test_build_body_no_files(tmp_path):
    subject, body = build_weekly_email_body(tmp_path)
    assert "No report files found" in body


def test_build_body_priority_order(tmp_path):
    """email_body.txt takes priority over weekly_report.md."""
    (tmp_path / "email_body.txt").write_text("preferred body")
    (tmp_path / "weekly_report.md").write_text("# Markdown report")
    _, body = build_weekly_email_body(tmp_path)
    assert body == "preferred body"


# ── send_weekly_email tests ──────────────────────────────────────────────────


def test_send_email_missing_keys():
    result = send_weekly_email({}, "Subject", "Body")
    assert result is False


def test_send_email_no_recipients():
    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "username": "user",
        "password": "pass",
        "sender": "user@example.com",
        "recipients": [],
    }
    result = send_weekly_email(cfg, "Subject", "Body")
    assert result is False


@patch("market_regime.email.smtplib.SMTP")
def test_send_email_tls_success(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value = mock_smtp

    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "username": "user",
        "password": "pass",
        "sender": "user@example.com",
        "recipients": ["admin@example.com"],
        "use_ssl": False,
    }
    result = send_weekly_email(cfg, "Test Subject", "Test Body")
    assert result is True
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user", "pass")
    mock_smtp.sendmail.assert_called_once()
    mock_smtp.quit.assert_called_once()


@patch("market_regime.email.smtplib.SMTP_SSL")
def test_send_email_ssl_success(mock_smtp_ssl_cls):
    mock_smtp = MagicMock()
    mock_smtp_ssl_cls.return_value = mock_smtp

    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 465,
        "username": "user",
        "password": "pass",
        "sender": "user@example.com",
        "recipients": ["admin@example.com"],
        "use_ssl": True,
    }
    result = send_weekly_email(cfg, "Test Subject", "Test Body")
    assert result is True
    mock_smtp.login.assert_called_once()
    mock_smtp.sendmail.assert_called_once()


# ── run_weekly_report.py tests ───────────────────────────────────────────────


def test_archive_weekly_report_creates_archive(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import archive_weekly_report

    (tmp_path / "dashboard.csv").write_text("asset,signal\nSPY,GREEN\n")
    result = archive_weekly_report(tmp_path)
    assert result is not None
    assert result.exists()
    assert (result / "dashboard.csv").exists()


def test_archive_weekly_report_no_files(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import archive_weekly_report

    result = archive_weekly_report(tmp_path)
    assert result is None


def test_build_pipeline_command():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import build_pipeline_command

    cmd = build_pipeline_command(full=True, steps="5,6,7", verbose=True)
    assert "--refresh" in cmd
    assert "--recompute" in cmd
    assert "--steps" in cmd
    assert "5,6,7" in cmd
    assert "--verbose" in cmd
    assert "--plots" in cmd


def test_build_pipeline_command_defaults():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import build_pipeline_command

    cmd = build_pipeline_command()
    assert "--recompute" in cmd
    assert "--refresh" not in cmd
    assert "--plots" in cmd
