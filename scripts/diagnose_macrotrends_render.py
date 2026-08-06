#!/usr/bin/env python
"""Throwaway diagnostic — why does read_html see no table in our rendered HTML?

Run on your machine:  python scripts/diagnose_macrotrends_render.py

Two hypotheses, and the log alone cannot separate them:

  A. The code being imported is NOT the code in the repo. `pip install .`
     (no -e) copies the package into site-packages, so source edits are
     ignored. Section 1 settles this by printing where the module is loaded
     from and whether the state="attached" fix is present in the loaded file.

  B. There genuinely is no <table> in the DOM at the moment we look, even
     though a standalone diagnostic saw one (class='table', 501 rows) four
     seconds after domcontentloaded. Sections 2-4 compare our production
     fetch against that standalone approach on the same page, and report what
     the returned HTML actually contains.

Delete this file once the answer is known.
"""

from __future__ import annotations

import os
import re
from io import StringIO

URL = "https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart"
EXE = os.environ.get("TC_CHROMIUM_PATH")


def _describe(label: str, html: str | None) -> None:
    print(f"\n  --- {label} ---")
    if not html:
        print("    (no HTML returned)")
        return
    import pandas as pd

    n_table_tags = len(re.findall(r"<table", html, re.I))
    interstitial = "just a moment" in html[:3000].lower() or "cf-browser-verification" in html.lower()
    print(f"    length          : {len(html)}")
    print(f"    '<table' count  : {n_table_tags}")
    print(f"    interstitial?   : {interstitial}")
    print(f"    has <iframe>    : {bool(re.search(r'<iframe', html, re.I))}")
    try:
        tables = pd.read_html(StringIO(html))
        print(f"    read_html       : {len(tables)} table(s)")
        for t in tables[:3]:
            print(f"      shape={t.shape} cols={list(t.columns)[:4]}")
    except Exception as exc:  # noqa: BLE001 — diagnostic
        print(f"    read_html       : RAISED {type(exc).__name__}: {str(exc)[:80]}")


def main() -> int:
    print("=" * 62)
    print(" 1. IS THE REPO CODE ACTUALLY WHAT'S IMPORTED?")
    print("=" * 62)
    import trading_crab_lib
    from trading_crab_lib.ingestion import browser as browser_mod

    print(f"  trading_crab_lib : {trading_crab_lib.__file__}")
    print(f"  browser module   : {browser_mod.__file__}")
    editable = "site-packages" not in (browser_mod.__file__ or "")
    print(f"  editable install : {editable}")
    try:
        import inspect

        src = inspect.getsource(browser_mod)
        has_fix = 'state="attached"' in src
        print(f"  has state=attached fix : {has_fix}")
        if not has_fix:
            print("  >> STALE INSTALL. Re-run:")
            print("     pip install -e 'src/trading_crab_lib/[all,dev]' && pip install -e '.[dev]'")
            return 1
    except OSError:
        print("  (source unavailable — almost certainly a non-editable install)")
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed.")
        return 2

    launch: dict = {"headless": True, "args": ["--disable-blink-features=AutomationControlled"]}
    if EXE:
        launch["executable_path"] = EXE

    print()
    print("=" * 62)
    print(" 2. WHAT OUR PRODUCTION fetch_page_html RETURNS")
    print("=" * 62)
    from trading_crab_lib.ingestion.browser import fetch_page_html

    _describe("fetch_page_html(wait_for_selector='table', require_selector=False)",
              fetch_page_html(URL, wait_for_selector="table", require_selector=False))

    print()
    print("=" * 62)
    print(" 3. SAME PAGE, the standalone approach that DID find a table")
    print("=" * 62)
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        try:
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
            page = context.new_page()
            page.goto(URL, wait_until="domcontentloaded", timeout=45000)

            # Poll the LIVE DOM once a second so we can see WHEN a table appears.
            print("\n  table count in live DOM, per second:")
            for sec in range(1, 16):
                page.wait_for_timeout(1000)
                n = page.evaluate("() => document.querySelectorAll('table').length")
                frames = page.evaluate("() => document.querySelectorAll('iframe').length")
                if sec in (1, 2, 3, 4, 5, 8, 10, 12, 15):
                    print(f"    t={sec:2}s  tables={n}  iframes={frames}")
                if n and sec >= 5:
                    break

            _describe("page.content() after polling", page.content())

            # Does the table live in a child FRAME rather than the main document?
            print("\n  --- per-frame table counts ---")
            for fr in page.frames:
                try:
                    n = fr.evaluate("() => document.querySelectorAll('table').length")
                    if n:
                        print(f"    frame {fr.url[:70]!r}: {n} table(s)")
                except Exception:  # noqa: BLE001 — cross-origin frames refuse evaluate
                    pass

            # And what the toggle button looks like, if present.
            print("\n  --- candidate toggle controls ---")
            ctrls = page.evaluate("""() => Array.from(
                document.querySelectorAll('a,button,[role=button],i,span')
            ).filter(e => {
                const hay = [e.className, e.id, e.title, e.getAttribute('aria-label')].join(' ');
                return /table/i.test(hay);
            }).slice(0, 8).map(e => ({tag: e.tagName, cls: e.className, id: e.id, title: e.title}))""")
            for c in ctrls:
                print(f"    {c}")
            browser.close()
        except Exception as exc:  # noqa: BLE001 — diagnostic
            print(f"  ERROR {type(exc).__name__}: {str(exc)[:160]}")
            browser.close()
            return 1

    print()
    print("Read:")
    print("  1 says stale install    -> reinstall editable; nothing else here matters.")
    print("  3 shows tables>0 at t=Ns -> our fetch returns too early or too late; align it.")
    print("  3 shows tables==0 always -> the earlier sighting needed something we are not doing.")
    print("  per-frame shows a table  -> it is in an IFRAME; page.content() cannot see it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
