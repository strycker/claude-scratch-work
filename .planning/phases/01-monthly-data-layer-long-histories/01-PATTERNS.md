# Phase 1: Monthly Data Layer & Long Histories - Pattern Map

**Mapped:** 2026-07-10
**Files analyzed:** 12 (new subpackage files + config + tests + docs)
**Analogs found:** 12 / 12 (all role-match or exact; incumbent is FROZEN, so every
new file is an "analog, not an edit-in-place")

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|-----------------|---------------|
| `src/trading_crab_lib/platform/__init__.py` | config/marker | — | `src/trading_crab_lib/ingestion/__init__.py` | role-match |
| `src/trading_crab_lib/platform/ingestion/alfred.py` | service (ingestion) | request-response (bulk API pull) | `src/trading_crab_lib/ingestion/fred.py` | exact (same client, new resample rule + vintage logic) |
| `src/trading_crab_lib/platform/ingest_monthly.py` | service (orchestrator) | request-response / batch | `src/trading_crab_lib/ingestion/fred.py::fetch_all()` | exact (parallel-fetch pattern reused) |
| `src/trading_crab_lib/platform/splice.py` | utility (transform) | transform | `src/trading_crab_lib/transforms.py` (`_fill_column`, ratio helpers) | role-match |
| `src/trading_crab_lib/platform/taxonomy.py` | config/utility | transform | `src/trading_crab_lib/config.py::validate_config()` | role-match (declarative classification + validation) |
| `src/trading_crab_lib/platform/transforms_monthly.py` | service (transform) | batch/transform | `src/trading_crab_lib/transforms.py::engineer_all()` | role-match (orchestrates ordered steps) |
| `src/trading_crab_lib/platform/checkpoints.py` | utility (persistence) | CRUD | `src/trading_crab_lib/checkpoints.py::CheckpointManager` | exact (reuse class, new `checkpoint_dir`) |
| `config/platform_settings.yaml` | config | — | `config/settings.yaml` | exact |
| `docs/splicing_rules.md` | docs | — | (none — new doc; format follows CLAUDE.md ADR style) | no analog |
| `tests/unit/test_platform_ingest.py` | test | request-response (mocked) | `tests/unit/test_ingestion.py` | exact |
| `tests/unit/test_platform_splice.py` | test | transform | `tests/unit/test_transforms.py` | role-match |
| `tests/unit/test_platform_alfred.py` | test | request-response (mocked) | `tests/unit/test_ingestion.py` (FRED mock section) | exact |
| `tests/unit/test_platform_taxonomy.py` | test | config validation | `tests/unit/test_config.py` | exact |
| `tests/unit/test_platform_paid_provider_stubs.py` | test | stub/no-op | (none — trivial `NotImplementedError` stub test) | no analog needed |

## Pattern Assignments

### `src/trading_crab_lib/platform/ingestion/alfred.py` (service, request-response)

**Analog:** `src/trading_crab_lib/ingestion/fred.py` (full file, 109 lines — read in one pass)

**Imports pattern** (fred.py lines 17-34):
```python
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd

try:
    from fredapi import Fred
except ImportError as _err:
    raise ImportError(
        "fredapi is required for FRED data ingestion. "
        "Install with: pip install 'trading-crab-lib[ingestion]'"
    ) from _err

log = logging.getLogger(__name__)
_MAX_WORKERS = 8
```
Copy verbatim except lower `_MAX_WORKERS` per RESEARCH Pitfall 3 (all-releases payloads
are larger — use a smaller cap, e.g. `_MAX_WORKERS = 3`, or fetch serially).

**Core fetch pattern — adapt, don't reuse verbatim** (fred.py lines 41-47):
```python
def _fetch_one(fred: Fred, series_id: str, start: str, end: str, shift: bool) -> pd.Series:
    """Pull one FRED series, resample to QE, optionally apply publication lag."""
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    quarterly = raw.resample("QE").last()
    if shift:
        quarterly = quarterly.shift(1)
    return quarterly
```
New `alfred.py` needs a **different** primitive — `get_series_all_releases()` (bulk,
one call, no resample inside — see RESEARCH.md Pattern 2 for `value_as_of()` shape) —
plus a monthly-resample analog for non-vintage series (`.resample("ME")` not `"QE"`,
per RESEARCH Pitfall 1). Do not import/monkeypatch `fred.py`'s private `_fetch_one` —
duplicate the client-construction + parallel-fetch skeleton only.

**Error handling pattern** (fred.py lines 81-92, inside `_fetch_task` closure):
```python
def _fetch_task(series_id: str, meta: dict) -> tuple[str, pd.Series | None]:
    friendly_name = meta["name"]
    ...
    try:
        s = _fetch_one(fred, series_id, start, end, shift)
        s.name = friendly_name
        return friendly_name, s
    except Exception as exc:  # noqa: BLE001 — fredapi raises various types
        log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
        return friendly_name, None
```
Copy this try/except-and-WARNING shape exactly — same graceful-degradation convention
used everywhere in `ingestion/`.

**Config-driven series list pattern** (fred.py docstring, lines 57-68):
```
Config shape expected:
    fred:
      series:
        GDP:
          name:  "fred_gdp"
          shift: true
```
`platform_settings.yaml` should mirror this shape with a `vintage: true` flag per
D-06 series (see RESEARCH.md Pitfall 4 — `vintage: true` is primary control,
`shift` is the pre-vintage-era fallback only).

---

### `src/trading_crab_lib/platform/ingest_monthly.py` (service, orchestrator)

**Analog:** `src/trading_crab_lib/ingestion/fred.py::fetch_all()` (lines 50-108)

**Core orchestration pattern** (fred.py lines 94-108):
```python
frames: dict[str, pd.Series] = {}
with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(series_cfg))) as pool:
    futures = {
        pool.submit(_fetch_task, sid, meta): sid
        for sid, meta in series_cfg.items()
    }
    for future in as_completed(futures):
        friendly_name, series = future.result()
        if series is not None:
            frames[friendly_name] = series

df = pd.DataFrame(frames)
df.index.name = "date"
log.info("FRED fetch complete: %d quarters, %d series", len(df), len(df.columns))
return df
```
Same skeleton, but `ingest_monthly.py` merges across **multiple existing fetchers**
(fred, multpl, macrotrends, assets — imported, not modified per D-01) plus the new
`alfred.py`. Use `pd.concat([...], axis=1)` (outer join) to merge per RESEARCH.md
Pitfall 5 — never `pd.merge()`/`DataFrame.join()` defaults, which silently drop rows
for short-history satellites (DATA-05 NULL-tolerance requirement).

**NULL-tolerant merge pattern** — see `macrotrends.py::fetch_all()` lines 242-243:
```python
df = pd.concat(frames, axis=1)
df.index.name = "date"
```
This is the exact merge primitive to standardize on for the platform subpackage.

---

### `src/trading_crab_lib/platform/splice.py` (utility, transform)

**No direct analog for the splice math itself** (novel to this phase — see
RESEARCH.md "Code Examples" for the `ratio_splice()` reference implementation, copy
that pseudocode as the starting point). The closest *structural* analog for how a
transform module is organized/tested is `src/trading_crab_lib/transforms.py`.

**Docstring + module structure convention** (transforms.py style — inferred from
CLAUDE.md conventions, module-level docstring explaining *why*, `# ── Name ──`
section dividers, `from __future__ import annotations` first).

**Error handling / validation convention** — mirror `config.py::validate_config()`'s
fail-fast style (collect errors, raise once) for the "assert splice continuity at
join date" check called out in RESEARCH.md test map (`test_splice_continuity_at_join`).

---

### `src/trading_crab_lib/platform/taxonomy.py` (config/utility)

**Analog:** `src/trading_crab_lib/config.py` (full file, 200 lines — read in one pass)

**Validation pattern to copy** (config.py lines 71-114, `validate_config()`):
```python
def validate_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            errors.append(f"Missing required section '{section}' in settings.yaml. ...")
    ...
    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"settings.yaml has {len(errors)} validation error(s):\n{bullet_list}")
```
`taxonomy.py`'s "every feature has exactly one tag" check (DATA-04 test requirement)
should follow this same collect-all-errors-then-raise-once shape, not fail on the
first bad feature.

**Config-loading pattern** (config.py lines 117-158, `load()`):
```python
def load(settings_path: dict[str, Any] | Path | str | None = None) -> dict[str, Any]:
    load_dotenv()
    if isinstance(settings_path, dict):
        cfg: dict[str, Any] = settings_path
    else:
        path = Path(settings_path) if settings_path is not None else CONFIG_DIR / "settings.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    validate_config(cfg)
    ...
    return cfg
```
Copy the dict/Path/str tri-mode acceptance for a `platform.config.load()` if the
new subpackage needs its own loader for `platform_settings.yaml` — keeps parity
with the incumbent's D48-established convention, and keeps the two config schemas
independent (D-02 requirement: never touch the frozen `validate_config()` schema).

---

### `src/trading_crab_lib/platform/transforms_monthly.py` (service, transform)

**Analog:** `src/trading_crab_lib/transforms.py::engineer_all()` — orchestrates
ordered steps (per CLAUDE.md ADR #4: cross-ratios → log → select → gap-fill →
derivatives → select). Not read directly this session (large file); RESEARCH.md
already documents the exact ordering invariant and confirms gap-fill must stay
AFTER log transform. Mirror that same "fixed step order, each step a named helper
function" structure for the monthly resample + agency-alignment pipeline
(resample → align-quarterly-with-lag-or-vintage → taxonomy-tag).

---

### `src/trading_crab_lib/platform/checkpoints.py` (utility, persistence)

**Analog:** `src/trading_crab_lib/checkpoints.py::CheckpointManager` (full file,
283 lines — read in one pass)

**Reuse-not-reimplement pattern** (checkpoints.py lines 114-121, `__init__`):
```python
def __init__(self, checkpoint_dir: Path | None = None) -> None:
    if checkpoint_dir is not None:
        self.dir = checkpoint_dir
    else:
        env_override = os.environ.get("TC_CHECKPOINT_DIR")
        self.dir = Path(env_override) if env_override else CHECKPOINT_DIR
    self.dir.mkdir(parents=True, exist_ok=True)
```
The class already accepts an arbitrary `checkpoint_dir` — per RESEARCH.md's own
recommendation, **do not subclass or fork this class**. `platform/checkpoints.py`
should be a one-line-of-substance module:
```python
from trading_crab_lib.checkpoints import CheckpointManager
from trading_crab_lib import DATA_DIR

PLATFORM_CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "platform"

def get_platform_checkpoint_manager() -> CheckpointManager:
    return CheckpointManager(checkpoint_dir=PLATFORM_CHECKPOINT_DIR)
```
→ skips reimplementing save/load/is_fresh/clear entirely; only a directory-scoping
factory function is needed.

---

### `config/platform_settings.yaml` (config)

**Analog:** `config/settings.yaml` — same top-level shape conventions (per-series
`name`/`shift` dicts for FRED, `datasets` lists for scrapers). New top-level keys
per RESEARCH.md "Feature Taxonomy Config Design": `taxonomy:` block (fast/slow/agency
lists), `splice:` block (per-asset-class source/join_date/method), `universe:` block
(core/satellites/holdings/watchlist per D-08…D-13). Keep this file **entirely
separate** from `settings.yaml` — D-02 requires the frozen incumbent's
`validate_config()` schema untouched.

---

### Test files

**Analog:** `tests/unit/test_ingestion.py` (mock pattern, lines 1-58 read)

**HTTP/API mock pattern to copy** (test_ingestion.py lines 38-59):
```python
class _FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self.content = content.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

@patch("trading_crab_lib.ingestion.multpl.time.sleep")
@patch("trading_crab_lib.ingestion.multpl.requests.get")
def test_multpl_scrape_raw_rows(mock_get, mock_sleep):
    from trading_crab_lib.ingestion.multpl import _scrape_raw_rows
    mock_get.return_value = _FakeResponse(SAMPLE_MULTPL_HTML)
    rows = _scrape_raw_rows("https://example.com/table")
    assert len(rows) == 3
```
For `test_platform_alfred.py`, mock `fredapi.Fred.get_series_all_releases` directly
(return a synthetic DataFrame with `date`/`realtime_start`/`realtime_end`/`value`
columns per RESEARCH.md Pattern 2) rather than mocking HTTP — `fredapi` already
wraps the HTTP layer, same as the incumbent's `test_ingestion.py` FRED section does
(not shown above but same `@patch("trading_crab_lib.ingestion.fred.Fred")` shape —
confirm exact mock target by grepping that file's FRED section if a 6th analog read
is needed at implementation time).

## Shared Patterns

### Graceful degradation on network/scrape failure
**Source:** `src/trading_crab_lib/ingestion/fred.py` lines 90-92 and
`macrotrends.py` lines 234-236 (identical shape in both):
```python
except Exception as exc:  # noqa: BLE001 — fredapi/network raises various types
    log.warning("Failed to fetch %s (%s): %s", friendly_name, series_id, exc)
    return friendly_name, None
```
**Apply to:** every new fetcher in `platform/ingestion/` (`alfred.py` and any
Shiller/WTISPLC cross-check fetcher).

### Checkpoint reuse (not reimplementation)
**Source:** `src/trading_crab_lib/checkpoints.py::CheckpointManager.__init__`
(already accepts `checkpoint_dir: Path | None`)
**Apply to:** `platform/checkpoints.py` — factory function only, no subclass.

### NULL-tolerant outer-join merge
**Source:** `macrotrends.py::fetch_all()` line 242 — `pd.concat(frames, axis=1)`
**Apply to:** `ingest_monthly.py`'s cross-source merge (DATA-05 requirement);
never use `pd.merge()`/`.join()` defaults.

### Config-driven series lists, never hardcoded IDs
**Source:** `fred.py` docstring config-shape example (lines 57-68) and
`macrotrends.py::fetch_all()` docstring (lines 195-208)
**Apply to:** `platform_settings.yaml`'s `fred_vintage_series`, `splice`, and
`universe` blocks — same "list-of-dicts or dict-of-dicts in YAML, never a Python
constant" convention.

### Fail-fast config validation (collect-all-errors)
**Source:** `config.py::validate_config()` lines 71-114
**Apply to:** `platform/taxonomy.py`'s "every feature tagged exactly once" check.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `docs/splicing_rules.md` | docs | — | No existing per-asset splice-documentation file in the codebase; format should follow the ADR style already established in root `CLAUDE.md` (Context/Decision/Rationale/Tradeoff shape) since that's the project's only precedent for "document a locked methodology decision." |
| `src/trading_crab_lib/platform/splice.py` core math | utility | transform | Splicing (ratio-scale-at-join, par-bond repricing) is genuinely new domain logic — RESEARCH.md's "Code Examples" section is the reference implementation to start from, not a codebase analog. |

## Metadata

**Analog search scope:** `src/trading_crab_lib/ingestion/` (fred.py, macrotrends.py,
assets.py), `src/trading_crab_lib/checkpoints.py`, `src/trading_crab_lib/config.py`,
`tests/unit/test_ingestion.py`
**Files scanned:** 6 read in full, plus CONTEXT.md/RESEARCH.md for the file list
**Pattern extraction date:** 2026-07-10
