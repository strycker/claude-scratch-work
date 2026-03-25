"""Tests for email.py and scripts/run_weekly_report.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trading_crab_lib.email import (
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
        "from_address: user@example.com\n"
        "to_address: admin@example.com\n"
    )
    result = load_email_config(cfg_file)
    assert result["smtp_host"] == "smtp.example.com"
    assert result["smtp_port"] == 587
    assert result["to_address"] == "admin@example.com"


def test_malformed_yaml_returns_empty(tmp_path):
    cfg_file = tmp_path / "email.yaml"
    cfg_file.write_text("just a string, not a dict")
    result = load_email_config(cfg_file)
    assert result == {}


def test_fallback_to_email_local_yaml(tmp_path, monkeypatch):
    """When email.yaml is absent, load_email_config falls back to email.local.yaml."""
    import trading_crab_lib
    monkeypatch.setattr(trading_crab_lib, "CONFIG_DIR", tmp_path)
    local_cfg = tmp_path / "email.local.yaml"
    local_cfg.write_text(
        "smtp_host: local.example.com\n"
        "smtp_port: 465\n"
        "username: user\n"
        "password: pass\n"
        "from_address: user@example.com\n"
        "to_address: admin@example.com\n"
    )
    result = load_email_config()
    assert result["smtp_host"] == "local.example.com"


def test_email_yaml_takes_priority_over_local(tmp_path, monkeypatch):
    """email.yaml is preferred when both email.yaml and email.local.yaml exist."""
    import trading_crab_lib
    monkeypatch.setattr(trading_crab_lib, "CONFIG_DIR", tmp_path)
    _write_full_config(tmp_path / "email.yaml", host="primary.example.com")
    _write_full_config(tmp_path / "email.local.yaml", host="local.example.com")
    result = load_email_config()
    assert result["smtp_host"] == "primary.example.com"


def test_strict_validation_rejects_incomplete_config(tmp_path):
    """Missing required keys → returns empty dict (fail-fast)."""
    cfg_file = tmp_path / "email.yaml"
    cfg_file.write_text("smtp_host: smtp.example.com\n")
    result = load_email_config(cfg_file)
    assert result == {}


def test_env_var_overrides(tmp_path, monkeypatch):
    """TC_* env vars override YAML values."""
    import trading_crab_lib
    monkeypatch.setattr(trading_crab_lib, "CONFIG_DIR", tmp_path)
    _write_full_config(tmp_path / "email.yaml", host="yaml.example.com")
    monkeypatch.setenv("TC_SMTP_HOST", "env.example.com")
    result = load_email_config()
    assert result["smtp_host"] == "env.example.com"


def test_env_vars_only_no_yaml(tmp_path, monkeypatch):
    """Email config works purely from env vars without any YAML file."""
    import trading_crab_lib
    monkeypatch.setattr(trading_crab_lib, "CONFIG_DIR", tmp_path)
    monkeypatch.setenv("TC_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("TC_SMTP_PORT", "587")
    monkeypatch.setenv("TC_SMTP_USER", "user@example.com")
    monkeypatch.setenv("TC_SMTP_PASSWORD", "secret")
    monkeypatch.setenv("TC_EMAIL_FROM", "user@example.com")
    monkeypatch.setenv("TC_EMAIL_TO", "admin@example.com")
    result = load_email_config()
    assert result["smtp_host"] == "smtp.example.com"
    assert result["smtp_port"] == 587
    assert result["from_address"] == "user@example.com"


def test_env_var_port_coercion(tmp_path, monkeypatch):
    """TC_SMTP_PORT is coerced to int."""
    import trading_crab_lib
    monkeypatch.setattr(trading_crab_lib, "CONFIG_DIR", tmp_path)
    _write_full_config(tmp_path / "email.yaml")
    monkeypatch.setenv("TC_SMTP_PORT", "465")
    result = load_email_config()
    assert result["smtp_port"] == 465
    assert isinstance(result["smtp_port"], int)


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
    cfg = _make_config(to_address="")
    result = send_weekly_email(cfg, "Subject", "Body")
    assert result is False


@patch("trading_crab_lib.email.smtplib.SMTP")
def test_send_email_tls_success(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value = mock_smtp

    cfg = _make_config(use_ssl=False)
    result = send_weekly_email(cfg, "Test Subject", "Test Body")
    assert result is True
    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("user", "pass")
    mock_smtp.sendmail.assert_called_once()
    mock_smtp.quit.assert_called_once()


@patch("trading_crab_lib.email.smtplib.SMTP_SSL")
def test_send_email_ssl_success(mock_smtp_ssl_cls):
    mock_smtp = MagicMock()
    mock_smtp_ssl_cls.return_value = mock_smtp

    cfg = _make_config(use_ssl=True, smtp_port=465)
    result = send_weekly_email(cfg, "Test Subject", "Test Body")
    assert result is True
    mock_smtp.login.assert_called_once()
    mock_smtp.sendmail.assert_called_once()


@patch("trading_crab_lib.email.smtplib.SMTP")
def test_send_email_comma_separated_to_address(mock_smtp_cls):
    """to_address as comma-separated string is split into multiple recipients."""
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value = mock_smtp

    cfg = _make_config(to_address="a@example.com, b@example.com")
    result = send_weekly_email(cfg, "Test", "Body")
    assert result is True
    call_args = mock_smtp.sendmail.call_args
    assert len(call_args[0][1]) == 2  # 2 recipients


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


# ── Test helpers ─────────────────────────────────────────────────────────────


def _make_config(
    *,
    smtp_host: str = "smtp.example.com",
    smtp_port: int = 587,
    username: str = "user",
    password: str = "pass",
    from_address: str = "user@example.com",
    to_address: str = "admin@example.com",
    use_ssl: bool = False,
) -> dict:
    return {
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "username": username,
        "password": password,
        "from_address": from_address,
        "to_address": to_address,
        "use_ssl": use_ssl,
    }


def _write_full_config(path: Path, *, host: str = "smtp.example.com") -> None:
    path.write_text(
        f"smtp_host: {host}\n"
        "smtp_port: 587\n"
        "username: user@example.com\n"
        "password: secret\n"
        "from_address: user@example.com\n"
        "to_address: admin@example.com\n"
    )
