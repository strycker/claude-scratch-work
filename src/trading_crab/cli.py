"""
CLI entry points for the trading-crab application.

Entry points registered in pyproject.toml:
    tradingcrab         → run_pipeline()
    tradingcrab-setup   → setup()
    tradingcrab-publish → publish_notebooks()
"""

from __future__ import annotations


def run_pipeline() -> None:
    """Run the full market-regime pipeline.

    Invoked as ``tradingcrab`` from the command line after ``pip install trading-crab``.
    Delegates to ``trading_crab.pipeline.main()``.
    """
    from trading_crab.pipeline import main
    main()


def setup() -> None:
    """Interactive configuration setup for trading-crab.

    Generates ``config/settings.yaml`` from the example template
    and prompts for FRED API key and email configuration.

    .. note:: Stub — full implementation deferred to Phase E.
    """
    print("tradingcrab-setup: interactive configuration setup")
    print("(Stub — not yet implemented. See NEXT_STEPS.md Phase E.)")
    print()
    print("For now, manually:")
    print("  1. cp .env.example .env  && edit .env (add FRED_API_KEY)")
    print("  2. cp config/email.example.yaml config/email.local.yaml")
    print("  3. pip install -e src/trading_crab_lib/[all]")


def publish_notebooks() -> None:
    """Export notebooks to HTML/PDF for sharing.

    .. note:: Stub — full implementation deferred to Phase F.
    """
    print("tradingcrab-publish: notebook publishing")
    print("(Stub — not yet implemented. See NEXT_STEPS.md Phase F.)")
