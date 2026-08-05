#!/usr/bin/env python
"""Throwaway diagnostic — what does the macrotrends page actually contain?

Run on your machine:  python scripts/diagnose_macrotrends.py

Cloudflare 403s every non-browser client we have, including WebFetch, so the
page structure cannot be inspected from the dev container. But a real browser
loads it fine, and Playwright is already installed for the Stooq path.

The chart/table toggle does not change the URL, which means the series is
already in the page and JavaScript merely re-renders it. This reports WHICH
JavaScript variable holds it, so the scraper can read the variable directly
via page.evaluate() instead of regex-matching HTML — and confirms whether
Cloudflare is even passed when a real browser asks.

Delete this file once the answer is known.
"""

from __future__ import annotations

import json
import os
import re

URL = "https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart"
EXE = os.environ.get("TC_CHROMIUM_PATH")

# What macrotrends.py currently looks for, so we can see if it would have matched.
CURRENT_PATTERN = re.compile(r"var\s+\w*[Dd]ata\s*=\s*(\[\s*\{.*?\}\s*\])\s*;", re.DOTALL)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed. Run: uv pip install playwright && playwright install chromium")
        return 2

    launch: dict = {"headless": True, "args": ["--disable-blink-features=AutomationControlled"]}
    if EXE:
        launch["executable_path"] = EXE

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        try:
            context = browser.new_context()
            context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
            page = context.new_page()
            resp = page.goto(URL, wait_until="networkidle", timeout=60000)
            html = page.content()

            print("=" * 62)
            print(" 1. Did a real browser get past Cloudflare?")
            print("=" * 62)
            print(f"  status : {resp.status if resp else '?'}")
            print(f"  length : {len(html)}")
            blocked = "just a moment" in html[:3000].lower() or "cf-browser-verification" in html.lower()
            print(f"  verdict: {'CLOUDFLARE INTERSTITIAL ❌' if blocked else 'REAL PAGE ✅'}")
            print(f"  title  : {page.title()!r}")

            print()
            print("=" * 62)
            print(" 2. Would macrotrends.py's current regex have matched?")
            print("=" * 62)
            m = CURRENT_PATTERN.search(html)
            print(f"  match  : {'YES' if m else 'NO'}")
            if m:
                print(f"  first120: {m.group(1)[:120]!r}")

            print()
            print("=" * 62)
            print(" 3. Which JS globals hold array-of-object data?")
            print("=" * 62)
            # Ask the live JS context rather than guessing at variable names.
            found = page.evaluate("""() => {
                const out = [];
                for (const k of Object.getOwnPropertyNames(window)) {
                    let v;
                    try { v = window[k]; } catch (e) { continue; }
                    if (Array.isArray(v) && v.length > 20 && v[0] && typeof v[0] === 'object') {
                        out.push({name: k, length: v.length, keys: Object.keys(v[0]), sample: v[0]});
                    }
                }
                return out;
            }""")
            if not found:
                print("  (none — the data may live in a closure or be fetched via XHR)")
            for entry in found:
                print(f"  var {entry['name']}: {entry['length']} rows, keys={entry['keys']}")
                print(f"      sample: {json.dumps(entry['sample'])[:160]}")

            print()
            print("=" * 62)
            print(" 4. Any <table> in the DOM? (the 'table view' toggle)")
            print("=" * 62)
            tables = page.evaluate("""() => Array.from(document.querySelectorAll('table')).map(t => ({
                id: t.id, cls: t.className, rows: t.rows.length,
                head: Array.from(t.querySelectorAll('th')).slice(0,6).map(h => h.innerText.trim())
            }))""")
            print(f"  {len(tables)} table(s)")
            for t in tables:
                print(f"    id={t['id']!r} class={t['cls']!r} rows={t['rows']} head={t['head']}")

            print()
            print("=" * 62)
            print(" 5. Network requests that returned data")
            print("=" * 62)
            print("  (re-navigating with a response listener)")
            hits: list[str] = []
            page.on("response", lambda r: hits.append(f"{r.status} {r.url}")
                    if any(s in r.url for s in (".php", ".json", ".csv", "/api/", "assets")) else None)
            page.goto(URL, wait_until="networkidle", timeout=60000)
            for h in hits[:25]:
                print(f"    {h}")

            browser.close()
        except Exception as exc:  # noqa: BLE001 — diagnostic
            print(f"ERROR {type(exc).__name__}: {str(exc)[:200]}")
            browser.close()
            return 1

    print()
    print("Read:")
    print("  1 REAL PAGE + 3 finds a var -> read that var via page.evaluate; no URL hunt needed.")
    print("  1 REAL PAGE + 4 finds a table -> click the toggle and parse the DOM table.")
    print("  1 INTERSTITIAL -> even a real browser is blocked; macrotrends needs another source.")
    print("  5 shows a .php/.json hit -> that endpoint may be fetchable directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
