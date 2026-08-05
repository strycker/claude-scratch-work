#!/usr/bin/env python
"""Throwaway diagnostic — why is yfinance rate-limited on every chunk?

Run on a residential connection:  python scripts/diagnose_yfinance.py

Yahoo returned YFRateLimitError immediately for all 22 tickers, in every
chunk, on the first attempt — not after sustained traffic. That pattern
suggests the cookie/crumb handshake yfinance performs before any download is
failing, rather than genuine throttling. The prime suspect is the custom
curl_cffi session we hand to yf.download: recent yfinance versions manage
their own impersonating session and fetch a crumb through it, and supplying a
foreign session can bypass that setup.

Delete this file once the answer is known.
"""

from __future__ import annotations

import warnings

TICKERS = ["SPY", "AGG"]
START, END = "2024-01-01", "2024-03-01"


def _shape(raw: object) -> str:
    if raw is None:
        return "None"
    try:
        return f"{len(raw)} rows x {len(raw.columns)} cols"  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return repr(raw)[:60]


def _attempt(label: str, **kwargs: object) -> None:
    import yfinance as yf

    print(f"\n--- {label} ---")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=TICKERS, start=START, end=END,
                interval="1d", auto_adjust=True, progress=False,
                **kwargs,
            )
        empty = raw is None or getattr(raw, "empty", True)
        print(f"  result  : {_shape(raw)}")
        print(f"  verdict : {'EMPTY ❌' if empty else 'DATA ✅'}")
    except Exception as exc:  # noqa: BLE001 — diagnostic: report any failure shape
        print(f"  verdict : RAISED ❌ {type(exc).__name__}: {str(exc)[:100]}")


def main() -> int:
    import yfinance as yf

    print(f"yfinance version: {getattr(yf, '__version__', 'unknown')}")

    # 1. No session at all — let yfinance manage its own cookie/crumb flow.
    _attempt("1. yf.download with NO session (yfinance manages its own)")

    # 2. The session we currently pass: curl_cffi, verify=False, impersonated.
    from trading_crab_lib.ingestion.assets import _ssl_bypass_curl_session

    session = _ssl_bypass_curl_session()
    if session is None:
        print("\n--- 2. skipped: curl_cffi unavailable ---")
    else:
        _attempt("2. yf.download with the SSL-bypass curl_cffi session (current)", session=session)

    # 3. A verified-TLS impersonating session from the new shared helper.
    from trading_crab_lib.ingestion.http import browser_session

    verified = browser_session()
    if verified is None:
        print("\n--- 3. skipped: no HTTP client ---")
    else:
        _attempt("3. yf.download with a verify=True impersonating session", session=verified)

    print("\nIf 1 is the only DATA -> drop the custom session; yfinance handles it.")
    print("If 1 and 3 are DATA but 2 is not -> verify=False is breaking the crumb fetch.")
    print("If all fail -> Yahoo is genuinely throttling this IP; wait and retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
