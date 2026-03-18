#!/usr/bin/env python3
"""
Weekly report runner — one command to run the pipeline and archive the report.

Usage (from repo root):
  python scripts/run_weekly_report.py              # steps 2–7 (cached ingest)
  python scripts/run_weekly_report.py --full      # steps 1–7 (full refresh)
  python scripts/run_weekly_report.py --plots     # also generate figures
  python scripts/run_weekly_report.py --verbose   # DEBUG logging

After the pipeline completes, this script:
  - Copies outputs/reports/weekly_report.md → outputs/reports/weekly_YYYY-MM-DD.md
  - Writes outputs/reports/email_body.txt (plain-text body for email paste or sendmail)

Cron example (e.g. Monday 9am): 0 9 * * 1 cd /path/to/repo && python scripts/run_weekly_report.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
WEEKLY_REPORT = "weekly_report.md"

# Allow importing the market_regime package when run as a script.
sys.path.insert(0, str(REPO_ROOT / "src"))

from market_regime.email import (  # noqa: E402
    build_weekly_email_body,
    load_email_config,
    send_weekly_email,
)


def archive_weekly_report(reports_dir: Path) -> None:
    """
    Copy weekly_report.md to weekly_YYYY-MM-DD.md and write email_body.txt.
    No-op if weekly_report.md does not exist. Caller can pass a custom path for tests.
    """
    report_path = reports_dir / WEEKLY_REPORT
    if not report_path.exists():
        return
    today = date.today().isoformat()
    stamped = reports_dir / f"weekly_{today}.md"
    shutil.copy2(report_path, stamped)
    print(f"Archived report → {stamped}")
    email_body_path = reports_dir / "email_body.txt"
    body = report_path.read_text(encoding="utf-8")
    email_body_path.write_text(body, encoding="utf-8")
    print(f"Email body → {email_body_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Trading-Crab pipeline for weekly report (steps 2–7 or 1–7).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run steps 1–7 (full ingest refresh). Default is 2–7 (cached ingest).",
    )
    parser.add_argument("--plots", action="store_true", help="Generate and save figures.")
    parser.add_argument("--verbose", action="store_true", help="Set logging to DEBUG.")
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send the weekly report via SMTP using config/email.local.yaml.",
    )
    args = parser.parse_args()

    steps = "1,2,3,4,5,6,7" if args.full else "2,3,4,5,6,7"
    argv = [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--steps", steps]
    if args.plots:
        argv.append("--plots")
    if args.verbose:
        argv.append("--verbose")

    result = subprocess.run(argv, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode

    report_path = REPORTS_DIR / WEEKLY_REPORT
    if report_path.exists():
        archive_weekly_report(REPORTS_DIR)
    else:
        print(f"No {WEEKLY_REPORT} at {report_path} — skip archive/email body.")

    if args.send_email:
        cfg = load_email_config()
        if not cfg:
            print("Email config not found or invalid; skipping send.")
        else:
            subject, body = build_weekly_email_body(REPORTS_DIR)
            ok = send_weekly_email(cfg, subject, body)
            if ok:
                print("Weekly report email sent.")
            else:
                print("Weekly report email failed to send (see logs).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
