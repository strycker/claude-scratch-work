"""
Weekly email delivery for market regime reports.

Loads SMTP configuration from config/email.yaml (with fallback to
config/email.local.yaml, then environment variables), composes an email
body from the most recent weekly report, and sends via SMTP (TLS or SSL).

  load_email_config()        — parse YAML or env vars → config dict
  build_weekly_email_body()  — compose subject + body from report files
  send_weekly_email()        — send via SMTP with TLS/SSL support
  resolve_plot_paths()       — resolve attach_plots config to existing files
"""

from __future__ import annotations

import html as html_mod
import logging
import os
import re
import smtplib
import ssl
from datetime import date
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import trading_crab_lib as _tcl
from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)

# Keys expected in the config dict (matching email.example.yaml).
# Code uses from_address / to_address (GSD convention).
REQUIRED_KEYS = frozenset(
    {"smtp_host", "smtp_port", "username", "password", "from_address", "to_address"}
)

# Env var prefix for email config overrides.
_ENV_MAP: dict[str, str] = {
    "TC_SMTP_HOST": "smtp_host",
    "TC_SMTP_PORT": "smtp_port",
    "TC_SMTP_USER": "username",
    "TC_SMTP_PASSWORD": "password",
    "TC_EMAIL_FROM": "from_address",
    "TC_EMAIL_TO": "to_address",
    "TC_EMAIL_USE_TLS": "use_tls",
    "TC_EMAIL_USE_SSL": "use_ssl",
}


def _env_overrides() -> dict:
    """Read email config values from TC_* environment variables."""
    cfg: dict = {}
    for env_key, cfg_key in _ENV_MAP.items():
        val = os.getenv(env_key)
        if val is not None:
            # Coerce port to int, booleans for tls/ssl
            if cfg_key == "smtp_port":
                try:
                    cfg[cfg_key] = int(val)
                except ValueError:
                    log.warning("TC_SMTP_PORT=%r is not an integer, ignoring", val)
            elif cfg_key in ("use_tls", "use_ssl"):
                cfg[cfg_key] = val.lower() in ("true", "1", "yes")
            else:
                cfg[cfg_key] = val
    return cfg


def load_email_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load email configuration from a YAML file or environment variables.

    Resolution order (later wins):
      1. config/email.yaml (or config/email.local.yaml as fallback)
      2. TC_* environment variables override individual keys

    Expected keys (matching config/email.example.yaml):
        smtp_host:      SMTP server hostname
        smtp_port:      port (587 for TLS, 465 for SSL)
        username:       SMTP username
        password:       SMTP password (or app-specific password)
        from_address:   sender email address
        to_address:     recipient email address (or comma-separated list)
        use_tls:        true for STARTTLS (port 587)
        use_ssl:        true for implicit SSL (port 465)

    Returns empty dict if file is missing/malformed AND no env vars set.
    """
    cfg: dict = {}

    # ── YAML file ──────────────────────────────────────────────────────
    if config_path is None:
        config_path = _tcl.CONFIG_DIR / "email.yaml"
        if not config_path.exists():
            config_path = _tcl.CONFIG_DIR / "email.local.yaml"

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                cfg = loaded
            else:
                log.warning("Email config is not a dict: %s", config_path)
        except (OSError, yaml.YAMLError) as exc:
            log.warning("Failed to load email config: %s", exc)

    # ── Env var overrides ──────────────────────────────────────────────
    env_cfg = _env_overrides()
    if env_cfg:
        cfg.update(env_cfg)
        log.debug("Email config: applied %d env var override(s)", len(env_cfg))

    # ── Strict validation (fail-fast) ─────────────────────────────────
    if cfg:
        missing = sorted(REQUIRED_KEYS - set(cfg))
        if missing:
            log.error("Email config missing required keys: %s", missing)
            return {}

    if not cfg:
        log.warning(
            "Email config not found at %s and no TC_* env vars set", config_path
        )

    return cfg


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
        df = pd.read_csv(dashboard_file)
        body = f"Dashboard summary ({len(df)} assets):\n\n{df.to_string(index=False)}"
        return subject, body

    return subject, "(No report files found)"


def resolve_plot_paths(
    plot_names: list[str],
    plots_dir: Path | None = None,
) -> list[Path]:
    """
    Resolve a list of plot filenames to existing file paths.

    Args:
        plot_names: list of filenames (e.g. ["03_regime_pca_scatter.png"])
        plots_dir:  directory containing plots (default: outputs/plots/)

    Returns:
        List of Path objects for files that exist (missing files are logged
        at WARNING level and skipped).
    """
    if plots_dir is None:
        plots_dir = OUTPUT_DIR / "plots"

    resolved: list[Path] = []
    for name in plot_names:
        path = plots_dir / name
        if path.exists():
            resolved.append(path)
        else:
            log.warning("Plot file not found, skipping: %s", path)
    return resolved


def _markdown_to_html(text: str) -> str:
    """
    Lightweight markdown → HTML conversion for the weekly report.

    Handles the subset of markdown used by write_weekly_report_md():
    - ## headings → <h2>
    - # headings  → <h1>
    - **bold**    → <strong>
    - - list item → <li> wrapped in <ul>
    - blank lines → paragraph breaks

    Uses stdlib only (no external markdown library required).
    """
    lines = text.splitlines()
    html_lines: list[str] = []
    in_ul = False

    for line in lines:
        # Headings
        if line.startswith("## "):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            content = html_mod.escape(line[3:])
            html_lines.append(f"<h2>{content}</h2>")
        elif line.startswith("# "):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            content = html_mod.escape(line[2:])
            html_lines.append(f"<h1>{content}</h1>")
        # Unordered list items
        elif line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            content = html_mod.escape(line[2:])
            # inline **bold**
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html_lines.append(f"<li>{content}</li>")
        # Blank line
        elif line.strip() == "":
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            html_lines.append("<br>")
        # Normal paragraph line
        else:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            content = html_mod.escape(line)
            # inline **bold**
            content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
            html_lines.append(f"<p>{content}</p>")

    if in_ul:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _build_html_body_with_plots(plain_text: str, plot_paths: list[Path]) -> str:
    """
    Build an HTML email body: markdown-rendered report body followed by
    inline plot images referencing Content-ID attachments.
    """
    html_body = _markdown_to_html(plain_text)
    parts = [
        "<html><body>",
        html_body,
    ]
    if plot_paths:
        parts += ["<hr>", "<h3>Key Plots</h3>"]
        for i, path in enumerate(plot_paths):
            cid = f"plot_{i}"
            stem = html_mod.escape(path.stem)
            parts.append(
                f'<p><strong>{stem}</strong></p>'
                f'<img src="cid:{cid}" alt="{stem}" '
                f'style="max-width: 100%; height: auto;">'
            )
    parts.append("</body></html>")
    return "\n".join(parts)


def send_weekly_email(
    config: dict[str, Any],
    subject: str,
    body: str,
    plot_paths: list[Path] | None = None,
) -> bool:
    """
    Send an email via SMTP.

    Args:
        config:      dict from load_email_config()
        subject:     email subject line
        body:        email body (plain text)
        plot_paths:  optional list of PNG paths to embed as inline images.
                     When provided, the email is sent as multipart/related
                     with an HTML body containing <img> references.

    Returns:
        True if sent successfully, False otherwise.
    """
    # Validate required keys
    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        log.error("Email config missing required keys: %s", missing)
        return False

    from_addr = config["from_address"]
    to_addr = config["to_address"]

    # Normalize to_address to a list
    if isinstance(to_addr, str):
        recipients = [a.strip() for a in to_addr.split(",") if a.strip()]
    elif isinstance(to_addr, list):
        recipients = to_addr
    else:
        recipients = [str(to_addr)]

    if not recipients:
        log.error("No recipients configured")
        return False

    # Build MIME message
    if plot_paths:
        # Multipart/related with HTML body + inline images
        msg = MIMEMultipart("related")
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        # Alternative part: plain text + HTML
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(body, "plain"))

        html_body = _build_html_body_with_plots(body, plot_paths)
        alt.attach(MIMEText(html_body, "html"))
        msg.attach(alt)

        # Attach images with Content-ID
        for i, path in enumerate(plot_paths):
            cid = f"plot_{i}"
            try:
                img_data = path.read_bytes()
                img = MIMEImage(img_data, name=path.name)
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header(
                    "Content-Disposition", "inline", filename=path.name
                )
                msg.attach(img)
                log.debug("Attached plot: %s (cid:%s)", path.name, cid)
            except OSError as exc:
                log.warning("Failed to read plot %s: %s", path, exc)

        log.info("Built HTML email with %d inline plot(s)", len(plot_paths))
    else:
        # Multipart/alternative: plain text + HTML (no inline images)
        msg = MIMEMultipart("alternative")
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        html_body = _build_html_body_with_plots(body, [])
        msg.attach(MIMEText(html_body, "html"))

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
        server.sendmail(from_addr, recipients, msg.as_string())
        server.quit()
        log.info("Weekly email sent to %s", recipients)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        log.error("Failed to send email: %s", exc)
        return False
