#!/usr/bin/env python
"""Throwaway diagnostic — why does Stooq still serve a challenge page?

Run on a residential connection:  python scripts/diagnose_stooq.py

Discriminates between three hypotheses:
  A. our BROWSER_HEADERS override defeats curl_cffi's impersonation
  B. the CSV endpoint needs a cookie the challenge page sets first
  C. impersonation is simply insufficient for Stooq

Delete this file once the answer is known.
"""

from __future__ import annotations

CSV_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"
QUOTE_URL = "https://stooq.com/q/d/?s=spy.us"


def _verdict(text: str) -> str:
    return "CSV ✅" if text.lstrip().lower().startswith("date,") else "challenge/HTML ❌"


def _report(label: str, resp: object, session: object = None) -> None:
    text = getattr(resp, "text", "") or ""
    status = getattr(resp, "status_code", "?")
    print(f"\n--- {label} ---")
    print(f"  status  : {status}")
    print(f"  verdict : {_verdict(text)}")
    print(f"  length  : {len(text)}")
    print(f"  first80 : {text[:80]!r}")
    ctype = getattr(resp, "headers", {}) or {}
    print(f"  ctype   : {ctype.get('content-type', '?')}")
    if session is not None:
        cookies = getattr(session, "cookies", None)
        try:
            names = list(cookies.keys()) if cookies else []
        except (AttributeError, TypeError):
            names = []
        print(f"  cookies : {names}")


def main() -> int:
    try:
        from curl_cffi import requests as cr
    except ImportError:
        print("curl_cffi not installed — cannot test impersonation.")
        return 2

    from trading_crab_lib.ingestion.http import BROWSER_HEADERS

    # 1. Pure impersonation, NO header override (hypothesis A: this one works)
    s1 = cr.Session(impersonate="chrome")
    _report("1. impersonate=chrome, NO custom headers", s1.get(CSV_URL, timeout=30), s1)

    # 2. Impersonation + our BROWSER_HEADERS (what ships today)
    s2 = cr.Session(impersonate="chrome")
    _report(
        "2. impersonate=chrome + BROWSER_HEADERS (current behavior)",
        s2.get(CSV_URL, headers=dict(BROWSER_HEADERS), timeout=30),
        s2,
    )

    # 3. Visit the quote page first, then the CSV on the same session
    #    (hypothesis B: a cookie gates the CSV endpoint)
    s3 = cr.Session(impersonate="chrome")
    warm = s3.get(QUOTE_URL, timeout=30)
    print(f"\n[warm-up] {QUOTE_URL} -> status {getattr(warm, 'status_code', '?')}")
    _report("3. quote page first, then CSV (same session)", s3.get(CSV_URL, timeout=30), s3)

    # 4. Safari fingerprint — some checks allowlist differently per browser
    try:
        s4 = cr.Session(impersonate="safari")
        _report("4. impersonate=safari, no custom headers", s4.get(CSV_URL, timeout=30), s4)
    except (ValueError, RuntimeError) as exc:
        print(f"\n--- 4. safari --- unavailable: {exc}")

    print("\nIf 1 says CSV and 2 says challenge -> the header override is the bug.")
    print("If 3 is the only CSV -> the endpoint needs a cookie; add a warm-up request.")
    print("If all say challenge -> impersonation is insufficient; Stooq needs another route.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
