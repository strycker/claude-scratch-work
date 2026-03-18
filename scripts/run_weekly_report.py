#!/usr/bin/env python3
"""
Weekly report automation script.

Runs the full pipeline, archives the report, and optionally sends an email.

Usage:
    python scripts/run_weekly_report.py                  # run pipeline + archive
    python scripts/run_weekly_report.py --email          # also send email
    python scripts/run_weekly_report.py --full            # run with --refresh
    python scripts/run_weekly_report.py --steps 5,6,7    # only specified steps
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_regime import OUTPUT_DIR, CONFIG_DIR

log = logging.getLogger(__name__)

REPORTS_DIR = OUTPUT_DIR / "reports"
ARCHIVES_DIR = REPORTS_DIR / "archive"


def archive_weekly_report(reports_dir: Path | None = None) -> Path | None:
    """
    Create a timestamped archive of the current weekly report.

    Copies dashboard.csv and weekly_report.md (if they exist) into
    reports/archive/YYYY-MM-DD/.

    Also extracts a plain-text email_body.txt suitable for email delivery.

    Returns the archive directory path, or None if no files to archive.
    """
    if reports_dir is None:
        reports_dir = REPORTS_DIR

    today = date.today().isoformat()
    archive_dir = (reports_dir / "archive" / today)
    archive_dir.mkdir(parents=True, exist_ok=True)

    files_archived = 0
    for filename in ["dashboard.csv", "weekly_report.md"]:
        src = reports_dir / filename
        if src.exists():
            shutil.copy2(src, archive_dir / filename)
            files_archived += 1

    if files_archived == 0:
        log.warning("No report files found to archive in %s", reports_dir)
        return None

    # Create email body extract
    email_body = archive_dir / "email_body.txt"
    report_md = reports_dir / "weekly_report.md"
    dashboard_csv = reports_dir / "dashboard.csv"

    if report_md.exists():
        email_body.write_text(report_md.read_text(encoding="utf-8"), encoding="utf-8")
    elif dashboard_csv.exists():
        email_body.write_text(
            f"Weekly dashboard ({today}):\n\n{dashboard_csv.read_text(encoding='utf-8')}",
            encoding="utf-8",
        )

    log.info("Archived %d report files to %s", files_archived, archive_dir)
    return archive_dir


def build_pipeline_command(
    full: bool = False,
    steps: str | None = None,
    plots: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Build the run_pipeline.py command-line argument list."""
    cmd = [sys.executable, str(PROJECT_ROOT / "run_pipeline.py")]

    if full:
        cmd.extend(["--refresh", "--recompute"])
    else:
        cmd.append("--recompute")

    if steps:
        cmd.extend(["--steps", steps])

    if plots:
        cmd.append("--plots")

    if verbose:
        cmd.append("--verbose")

    return cmd


def run_weekly_report(
    full: bool = False,
    steps: str | None = None,
    send_email: bool = False,
    plots: bool = True,
    verbose: bool = False,
) -> int:
    """
    Run the weekly report pipeline end-to-end.

    1. Execute run_pipeline.py with the specified flags
    2. Archive the report
    3. Optionally send email

    Returns 0 on success, non-zero on failure.
    """
    cmd = build_pipeline_command(full=full, steps=steps, plots=plots, verbose=verbose)
    log.info("Running pipeline: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            log.error("Pipeline failed (exit %d):\n%s", result.returncode, result.stderr)
            return result.returncode
        log.info("Pipeline completed successfully")
    except subprocess.TimeoutExpired:
        log.error("Pipeline timed out after 30 minutes")
        return 1
    except Exception as exc:
        log.error("Pipeline execution failed: %s", exc)
        return 1

    # Archive
    archive_dir = archive_weekly_report()
    if archive_dir is None:
        log.warning("No report to archive — pipeline may not have produced output")

    # Email
    if send_email:
        from market_regime.email import (
            build_weekly_email_body,
            load_email_config,
            send_weekly_email,
        )

        email_cfg = load_email_config()
        if not email_cfg:
            log.warning("Email config not found — skipping email delivery")
        else:
            report_dir = archive_dir if archive_dir else REPORTS_DIR
            subject, body = build_weekly_email_body(report_dir)
            if send_weekly_email(email_cfg, subject, body):
                log.info("Email sent successfully")
            else:
                log.error("Email delivery failed")
                return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run weekly market regime report")
    parser.add_argument("--full", action="store_true", help="Full refresh (re-scrape)")
    parser.add_argument("--steps", type=str, default=None, help="Pipeline steps (e.g. 5,6,7)")
    parser.add_argument("--email", action="store_true", help="Send report via email")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate plots")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    plots = not args.no_plots
    rc = run_weekly_report(
        full=args.full,
        steps=args.steps,
        send_email=args.email,
        plots=plots,
        verbose=args.verbose,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
