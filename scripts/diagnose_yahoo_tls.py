#!/usr/bin/env python
"""Throwaway diagnostic — is Yahoo's failure TLS, or a real rate limit?

Run on your machine:  python scripts/diagnose_yahoo_tls.py

diagnose_yfinance.py showed that yfinance fails two different ways:
  - with verify=True  -> "self signed certificate in certificate chain" (P6)
  - with verify=False -> YFRateLimitError

But stooq.com verifies FINE through the same curl_cffi with verify=True, so
this is unlikely to be a machine-wide TLS interception. The likelier cause is
that curl_cffi's own bundled CA store lacks a root in Yahoo's chain.

That distinction decides the fix:
  - if verify=certifi.where() works, P22's verify=False is unnecessary and we
    can restore real certificate verification instead of disabling it;
  - and only then can we tell whether the 429 is genuinely Yahoo throttling,
    since today it is only ever observed on the verification-disabled path.

Delete this file once the answer is known.
"""

from __future__ import annotations

import os

# Yahoo's actual data endpoint — what yfinance calls under the hood.
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"
STOOQ_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"


def _probe(label: str, url: str, **session_kwargs: object) -> None:
    from curl_cffi import requests as cr

    print(f"\n--- {label} ---")
    print(f"  verify  : {session_kwargs.get('verify', '(default)')}")
    try:
        session = cr.Session(impersonate="chrome", **session_kwargs)  # type: ignore[arg-type]
        resp = session.get(url, timeout=30)
        body = resp.text or ""
        print(f"  status  : {resp.status_code}")
        print(f"  length  : {len(body)}")
        print(f"  first120: {body[:120]!r}")
        if resp.status_code == 429:
            print("  >> genuine 429 FROM YAHOO (body above is Yahoo's, not a proxy's)")
        elif resp.status_code == 200 and '"chart"' in body:
            print("  >> DATA ✅ Yahoo answered normally")
    except Exception as exc:  # noqa: BLE001 — diagnostic: report any failure shape
        print(f"  ERROR   : {type(exc).__name__}: {str(exc)[:130]}")


def main() -> int:
    try:
        import certifi
        from curl_cffi import requests as cr  # noqa: F401 — availability probe
    except ImportError as exc:
        print(f"missing dependency: {exc}")
        return 2

    print(f"certifi bundle : {certifi.where()}")
    for var in ("SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY"):
        print(f"{var:18}: {os.environ.get(var, '(unset)')}")

    # 1. Control: does anything verify cleanly with the default CA store?
    _probe("1. CONTROL stooq.com, default verify", STOOQ_URL)

    # 2. Yahoo with the default (curl_cffi bundled) CA store — expected to fail.
    _probe("2. Yahoo chart API, default verify", CHART_URL)

    # 3. Yahoo with certifi's CA bundle explicitly. THE KEY TEST: if this
    #    verifies, curl_cffi CAN take a custom CA and P22's verify=False is
    #    unnecessary.
    _probe("3. Yahoo chart API, verify=certifi.where()", CHART_URL, verify=certifi.where())

    # 4. Yahoo with verification disabled — reproduces today's behavior and
    #    shows whether the 429 body really comes from Yahoo.
    _probe("4. Yahoo chart API, verify=False (current behavior)", CHART_URL, verify=False)

    print("\nRead:")
    print("  3 works            -> curl_cffi accepts a CA path; fix P22 with certifi, not verify=False.")
    print("  3 fails, 4 gives 200 -> Yahoo's chain genuinely is not in any local bundle; keep bypass, isolated.")
    print("  4 shows a real 429 -> Yahoo is throttling this IP; try a different network before more code.")
    print("  1 fails too        -> machine-wide TLS interception after all; find the intercepting CA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
