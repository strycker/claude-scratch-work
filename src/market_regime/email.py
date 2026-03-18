"""
Weekly email delivery for market regime reports.

Loads SMTP configuration from config/email.yaml, composes an email body
from the most recent weekly report, and sends via SMTP (TLS or SSL).

  load_email_config()        — parse config/email.yaml
  build_weekly_email_body()  — compose subject + body from report files
  send_weekly_email()        — send via SMTP with TLS/SSL support
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_email_config(config_path: Path | None = None) -> dict:
    """
    Load email configuration from a YAML file.

    Expected keys:
        smtp_host:     SMTP server hostname
        smtp_port:     port (587 for TLS, 465 for SSL)
        username:      SMTP username
        password:      SMTP password (or app-specific password)
        sender:        sender email address
        recipients:    list of recipient email addresses
        use_ssl:       true for SSL (port 465), false for STARTTLS (port 587)

    Returns empty dict if file is missing or malformed.
    """
    if config_path is None:
        from market_regime import CONFIG_DIR
        config_path = CONFIG_DIR / "email.yaml"

    if not config_path.exists():
        log.warning("Email config not found at %s", config_path)
        return {}

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            log.warning("Email config is not a dict: %s", config_path)
            return {}
        return cfg
    except Exception as exc:
        log.warning("Failed to load email config: %s", exc)
        return {}


def build_weekly_email_body(
    report_dir: Path,
    subject_prefix: str = "Market Regime Weekly Report",
) -> tuple[str, str]:
    """
    Build email subject and body from the most recent report files.

    Looks for (in order):
      1. report_dir/email_body.txt  — pre-formatted email body
      2. report_dir/weekly_report.md — markdown report (used as-is)
      3. report_dir/dashboard.csv — fallback: summarize CSV

    Returns:
        (subject, body) tuple.
    """
    from datetime import date
    subject = f"{subject_prefix} — {date.today().isoformat()}"

    # Try pre-formatted email body
    email_body_file = report_dir / "email_body.txt"
    if email_body_file.exists():
        body = email_body_file.read_text(encoding="utf-8")
        return subject, body

    # Try markdown report
    report_file = report_dir / "weekly_report.md"
    if report_file.exists():
        body = report_file.read_text(encoding="utf-8")
        return subject, body

    # Fallback: dashboard CSV summary
    dashboard_file = report_dir / "dashboard.csv"
    if dashboard_file.exists():
        import pandas as pd
        df = pd.read_csv(dashboard_file)
        body = f"Dashboard summary ({len(df)} assets):\n\n{df.to_string(index=False)}"
        return subject, body

    return subject, "(No report files found)"


def send_weekly_email(
    config: dict,
    subject: str,
    body: str,
) -> bool:
    """
    Send an email via SMTP.

    Args:
        config:  dict from load_email_config()
        subject: email subject line
        body:    email body (plain text)

    Returns:
        True if sent successfully, False otherwise.
    """
    required = ["smtp_host", "smtp_port", "username", "password", "sender", "recipients"]
    missing = [k for k in required if k not in config or not config[k]]
    if missing:
        log.error("Email config missing required keys: %s", missing)
        return False

    recipients = config["recipients"]
    if isinstance(recipients, str):
        recipients = [recipients]

    if not recipients:
        log.error("No recipients configured")
        return False

    msg = MIMEMultipart()
    msg["From"] = config["sender"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    use_ssl = config.get("use_ssl", False)
    host = config["smtp_host"]
    port = int(config["smtp_port"])

    try:
        if use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(host, port, context=context)
        else:
            server = smtplib.SMTP(host, port)
            server.starttls()

        server.login(config["username"], config["password"])
        server.sendmail(config["sender"], recipients, msg.as_string())
        server.quit()
        log.info("Weekly email sent to %s", recipients)
        return True
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        return False
