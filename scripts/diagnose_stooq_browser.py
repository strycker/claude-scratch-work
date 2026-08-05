#!/usr/bin/env python
"""Throwaway diagnostic — Stooq answers headless Chromium with "Access denied".

Run on a residential connection:  python scripts/diagnose_stooq_browser.py

The JS challenge is being solved (we no longer get the 796-byte challenge page),
but Stooq now returns "Access denied" — it is detecting the automation itself.
This tests, in one run, which of four things is true:

  A. automation flags are the tell   -> stealth launch args fix it
  B. the CSV endpoint rejects non-navigation fetches -> page.goto works, request API doesn't
  C. headless itself is the tell     -> headful works, headless does not
  D. the IP/session is denied outright -> nothing works, and even the plain
     HTML quote page is denied, not just the CSV endpoint

Variant 0 is the control: if the ordinary quote PAGE is also denied, the block
is not endpoint-specific and stealth work is pointless.

Delete this file once the answer is known.
"""

from __future__ import annotations

import os

CSV_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"
QUOTE_URL = "https://stooq.com/q/d/?s=spy.us"

# Suppresses the CDP "navigator.webdriver = true" tell that automation exposes.
STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
]

# Removes the residual webdriver property before any page script observes it.
STEALTH_INIT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"

EXE = os.environ.get("TC_CHROMIUM_PATH")


def _verdict(text: str) -> str:
    head = text.lstrip().lower()
    if head.startswith("date,"):
        return "CSV ✅"
    if "access denied" in head[:400]:
        return "ACCESS DENIED ❌"
    if "requires javascript" in head[:800]:
        return "JS CHALLENGE ❌"
    return f"other ❌ ({len(text)} bytes)"


def _launch(p, *, headless: bool, stealth: bool):
    kwargs: dict = {"headless": headless}
    if EXE:
        kwargs["executable_path"] = EXE
    if stealth:
        kwargs["args"] = STEALTH_ARGS
    return p.chromium.launch(**kwargs)


def _run(p, label: str, *, headless: bool, stealth: bool, navigate: bool, url: str = CSV_URL) -> None:
    print(f"\n--- {label} ---")
    browser = None
    try:
        browser = _launch(p, headless=headless, stealth=stealth)
        context = browser.new_context()
        if stealth:
            context.add_init_script(STEALTH_INIT)
        page = context.new_page()

        # Warm-up navigation so the challenge JS runs and sets its cookie.
        page.goto(QUOTE_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)

        if navigate:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=45000)
            text = page.content()
            status = resp.status if resp else "?"
        else:
            resp = context.request.get(url, timeout=45000)
            text = resp.text()
            status = resp.status

        print(f"  status  : {status}")
        print(f"  verdict : {_verdict(text)}")
        print(f"  first90 : {text[:90]!r}")
    except Exception as exc:  # noqa: BLE001 — diagnostic: report any failure shape
        print(f"  verdict : RAISED ❌ {type(exc).__name__}: {str(exc)[:110]}")
    finally:
        if browser is not None:
            browser.close()


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: uv pip install playwright && playwright install chromium")
        return 2

    print(f"TC_CHROMIUM_PATH: {EXE or '(unset — using Playwright default)'}")

    with sync_playwright() as p:
        # 0. CONTROL — is the plain quote page itself denied?
        _run(p, "0. CONTROL: plain quote PAGE, headless, no stealth",
             headless=True, stealth=False, navigate=True, url=QUOTE_URL)

        # 1. Baseline — what ships today.
        _run(p, "1. headless, no stealth, context.request.get (current)",
             headless=True, stealth=False, navigate=False)

        # 2. Stealth flags + init script.
        _run(p, "2. headless + STEALTH, context.request.get",
             headless=True, stealth=True, navigate=False)

        # 3. Stealth + real page navigation instead of the request API.
        _run(p, "3. headless + STEALTH, page.goto navigation",
             headless=True, stealth=True, navigate=True)

        # 4. Headful — the strongest signal available locally.
        _run(p, "4. HEADFUL + STEALTH, page.goto navigation",
             headless=False, stealth=True, navigate=True)

    print("\nRead:")
    print("  variant 0 denied too      -> IP/session block; stealth cannot help. Abandon Stooq.")
    print("  2 or 3 gives CSV          -> automation flags were the tell; fold that into browser.py.")
    print("  only 4 gives CSV          -> headless is the tell; costs a visible window per run.")
    print("  all denied but 0 fine     -> the CSV endpoint specifically refuses bots. Abandon Stooq.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
