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


def test_archive_weekly_report_creates_dated_copy(tmp_path):
    import sys
    from datetime import date
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import archive_weekly_report

    content = "# Weekly Report\nAll good."
    (tmp_path / "weekly_report.md").write_text(content, encoding="utf-8")
    archive_weekly_report(tmp_path)

    today = date.today().isoformat()
    stamped = tmp_path / f"weekly_{today}.md"
    email_body = tmp_path / "email_body.txt"
    assert stamped.exists()
    assert email_body.exists()
    assert stamped.read_text(encoding="utf-8") == content
    assert email_body.read_text(encoding="utf-8") == content


def test_archive_weekly_report_noop_when_no_report(tmp_path):
    import sys
    from datetime import date
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    from run_weekly_report import archive_weekly_report

    archive_weekly_report(tmp_path)

    today = date.today().isoformat()
    assert not (tmp_path / f"weekly_{today}.md").exists()
    assert not (tmp_path / "email_body.txt").exists()
